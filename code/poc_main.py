"""
PoC: Empirical Positioning Uncertainty Extraction from Taxi GPS Data
======================================================================

Purpose:
    For each Chinese city's one-day taxi GPS dataset, this script:
    1. Reads all parquet files in city_data/<city_name>/
    2. Identifies stationary segments (parked taxis)
    3. Computes per-segment positioning uncertainty (sigma_pos)
    4. Aggregates spatially (100m, 250m grids) and temporally (6 time bins)
    5. Outputs everything we need to continue the research without re-running

Workflow:
    coding/poc_main.py  -->  city_data/<city>/  -->  output/poc_<city>/

Run:
    python poc_main.py --city beijing
    python poc_main.py --city beijing --data-root /path/to/city_data --out-root /path/to/output

Author: [Your Team]
"""

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Plotting (saved to file, no display needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ============================================================
# CONFIGURATION (v2 - tuned based on diagnostic findings)
# ============================================================

# Stationary segment detection
SPEED_THRESHOLD_KMH = 2.0          # speed below this = potentially stationary
MIN_SEGMENT_DURATION_S = 180       # minimum 3 minutes (was 5min - too strict)
MAX_SEGMENT_DISPLACEMENT_M = 50    # was 30m, slightly relaxed
MIN_POINTS_PER_SEGMENT = 8         # was 10, relaxed for shorter sampling intervals
MAX_TIME_GAP_WITHIN_SEGMENT_S = 120  # gaps > 2 min split a segment

# v2 NEW: coordinate diversity filter (the most important addition)
# Diagnostic showed many segments have unique_coords/n_points < 0.3, indicating
# map-matching artifacts. We require segments to preserve coordinate variation.
MIN_UNIQUENESS_RATIO = 0.5        # at least 50% of points must have distinct coords
MIN_UNIQUE_COORDS = 5             # at least 5 distinct (lon, lat) pairs

# v2 NEW: long-segment splitting
# Long stops (>15min) often have map-matching kicked in part-way through.
# Split them into chunks and evaluate each separately.
MAX_SEGMENT_DURATION_S = 900       # 15 minutes; split longer segments
SUBSEGMENT_DURATION_S = 600        # split into 10-minute sub-segments

# Spatial aggregation grids (in meters)
GRID_SIZES_M = [50, 100, 250]     # main analysis: 100m. 50m & 250m for sensitivity check

# Temporal binning (6 bins, 4 hours each)
TIME_BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
TIME_BIN_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]

# Earth constants for lat/lon to meter conversion
M_PER_DEG_LAT = 111320.0           # approximate, varies <1% by latitude

# Diagnostic / sanity-check thresholds
SIGMA_POS_REASONABLE_RANGE_M = (0.1, 100.0)

# ============================================================
# LOGGING
# ============================================================

def setup_logger(out_dir: Path):
    """Set up a logger that writes both to stdout and to a file in output."""
    log_path = out_dir / "run.log"
    logger = logging.getLogger("poc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ============================================================
# DATA LOADING
# ============================================================

def load_city_data(city_dir: Path, logger) -> pd.DataFrame:
    """Read all parquet files in a city directory, concat them.

    Expected schema (based on the sample data):
        vehicle_id, lon, lat, speed_kmh, heading, status, gps_time,
        recv_time, veh_type, admin_code, road_name, road_class
    """
    parquet_files = sorted(city_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {city_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files in {city_dir}")
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)
        logger.info(f"  Loaded {f.name}: {len(df):,} rows")

    df = pd.concat(dfs, ignore_index=True)

    # Parse types
    df["gps_time"] = pd.to_datetime(df["gps_time"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")

    # Drop bad rows
    n0 = len(df)
    df = df.dropna(subset=["vehicle_id", "lon", "lat", "speed_kmh", "gps_time"])
    df = df[(df["lon"] > 70) & (df["lon"] < 140) & (df["lat"] > 15) & (df["lat"] < 55)]
    df = df[df["speed_kmh"] >= 0]
    df = df[df["speed_kmh"] < 200]  # exceeding 200 km/h is taxi GPS error

    # Sort once: this is required for segmentation
    df = df.sort_values(["vehicle_id", "gps_time"]).reset_index(drop=True)
    logger.info(f"Total records after cleaning: {len(df):,} (dropped {n0-len(df):,})")
    logger.info(f"Unique vehicles: {df['vehicle_id'].nunique():,}")
    logger.info(f"Time range: {df['gps_time'].min()} -> {df['gps_time'].max()}")
    logger.info(f"Spatial extent: lon [{df['lon'].min():.4f}, {df['lon'].max():.4f}], "
                f"lat [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    return df


# ============================================================
# STATIONARY SEGMENT DETECTION
# ============================================================

def detect_stationary_segments(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Identify stationary segments per vehicle.

    A stationary segment is a maximal run of consecutive records (per vehicle,
    sorted by time) where speed <= SPEED_THRESHOLD_KMH and consecutive time
    gap < MAX_TIME_GAP.

    Returns a long-format DataFrame with one row per (segment, point inside segment).
    """
    logger.info("Detecting stationary segments...")

    # Ensure sorted (defensive — caller already sorts but be sure)
    df = df.sort_values(["vehicle_id", "gps_time"]).reset_index(drop=True)
    df["is_slow"] = df["speed_kmh"] <= SPEED_THRESHOLD_KMH

    # Compute time gap to previous record (per vehicle)
    df["prev_time"] = df.groupby("vehicle_id")["gps_time"].shift(1)
    df["dt_s"] = (df["gps_time"] - df["prev_time"]).dt.total_seconds()

    # Build segment_id using a numpy-based approach that's easy to verify.
    # A segment continues if: same vehicle AND current is_slow AND prev was slow
    #                          AND time gap is small.
    # Otherwise a new boundary is drawn.
    n = len(df)
    is_slow = df["is_slow"].to_numpy()
    vehicle = df["vehicle_id"].to_numpy()
    dt_s = df["dt_s"].to_numpy()

    # An "in-segment continuation" condition:
    #   row i extends segment of row i-1 iff:
    #   - same vehicle as i-1
    #   - i-1 was slow (so row i-1 belongs to a segment)
    #   - i is slow
    #   - time gap is OK
    continues = np.zeros(n, dtype=bool)
    if n > 1:
        same_vehicle = (vehicle[1:] == vehicle[:-1])
        prev_slow = is_slow[:-1]
        cur_slow = is_slow[1:]
        small_gap = np.nan_to_num(dt_s[1:], nan=1e9) <= MAX_TIME_GAP_WITHIN_SEGMENT_S
        continues[1:] = same_vehicle & prev_slow & cur_slow & small_gap

    # Segment ID: increments at every "non-continuation among slow rows"
    # i.e., a new segment begins at every slow row that does NOT continue from previous slow row.
    new_segment_starts = is_slow & ~continues
    segment_id_full = np.where(new_segment_starts, 1, 0).cumsum()
    # Non-slow rows: segment_id = -1
    segment_id_full = np.where(is_slow, segment_id_full, -1)
    df["segment_id"] = segment_id_full

    # Filter to only segment rows
    seg_df = df[df["segment_id"] >= 0].copy()
    n_segments = seg_df["segment_id"].nunique() if len(seg_df) > 0 else 0
    logger.info(f"  Slow-speed points: {len(seg_df):,}")
    logger.info(f"  Provisional segments: {n_segments:,}")

    return seg_df


# ============================================================
# SIGMA_POS COMPUTATION
# ============================================================

def compute_segment_sigma_pos(seg_df: pd.DataFrame, logger) -> tuple:
    """For each stationary segment, compute sigma_pos (positioning uncertainty).

    v2 changes:
    - Splits long segments (>15 min) into 10-min sub-segments. Map-matching
      tends to engage after a few minutes of stationarity, so early portions
      of a long stop are often raw while later portions are snapped. Splitting
      lets us use the raw early portions and reject the snapped later ones.
    - Adds uniqueness_ratio filter: requires segment internal coordinate
      diversity > MIN_UNIQUENESS_RATIO. This is the key filter to exclude
      map-matched segments (where most points share the same coords).
    - Tracks why each segment was rejected (filter funnel statistics).

    sigma_pos is computed using MAD-based estimator:
        sigma_pos = sqrt(MAD_x^2 + MAD_y^2) * 1.4826
    """
    logger.info("Computing per-segment sigma_pos (v2 with sub-segmenting and uniqueness filter)...")
    grouped = seg_df.groupby("segment_id")

    # Filter funnel counters
    funnel = {
        "total_provisional": 0,
        "subsegments_created": 0,
        "rejected_too_few_points": 0,
        "rejected_too_short": 0,
        "rejected_too_displaced": 0,
        "rejected_low_uniqueness": 0,
        "rejected_too_few_unique_coords": 0,
        "passed_all_filters": 0,
    }

    rows = []
    for sid, g in grouped:
        funnel["total_provisional"] += 1
        # Sort by time to be safe
        g = g.sort_values("gps_time").reset_index(drop=True)
        full_duration = (g["gps_time"].max() - g["gps_time"].min()).total_seconds()

        # Decide if we split this segment
        if full_duration > MAX_SEGMENT_DURATION_S:
            # Split into sub-segments based on time
            t0 = g["gps_time"].iloc[0]
            g["sub_idx"] = ((g["gps_time"] - t0).dt.total_seconds() //
                             SUBSEGMENT_DURATION_S).astype(int)
            sub_groups = [(f"{sid}_{si}", sub) for si, sub in g.groupby("sub_idx")]
            funnel["subsegments_created"] += len(sub_groups)
        else:
            sub_groups = [(str(sid), g)]

        # Process each (sub-)segment
        for sub_sid, sg in sub_groups:
            n_pts = len(sg)
            duration = (sg["gps_time"].max() - sg["gps_time"].min()).total_seconds()

            # Filter 1: minimum points
            if n_pts < MIN_POINTS_PER_SEGMENT:
                funnel["rejected_too_few_points"] += 1
                continue
            # Filter 2: minimum duration
            if duration < MIN_SEGMENT_DURATION_S:
                funnel["rejected_too_short"] += 1
                continue

            lon = sg["lon"].values
            lat = sg["lat"].values
            lat_mean = float(np.mean(lat))

            # Coordinate diversity (NEW filter)
            n_unique_coords = len(set(zip(lon.tolist(), lat.tolist())))
            uniqueness_ratio = n_unique_coords / n_pts

            # Filter 3: minimum unique coordinates count
            if n_unique_coords < MIN_UNIQUE_COORDS:
                funnel["rejected_too_few_unique_coords"] += 1
                continue
            # Filter 4: uniqueness ratio (most important - excludes map-matched data)
            if uniqueness_ratio < MIN_UNIQUENESS_RATIO:
                funnel["rejected_low_uniqueness"] += 1
                continue

            # Convert to meters relative to median location
            lon_m_per_deg = M_PER_DEG_LAT * np.cos(np.radians(lat_mean))
            x_m = (lon - np.median(lon)) * lon_m_per_deg
            y_m = (lat - np.median(lat)) * M_PER_DEG_LAT

            # Filter 5: max displacement (vehicle should be approximately stationary)
            max_disp = float(np.sqrt(x_m**2 + y_m**2).max())
            if max_disp > MAX_SEGMENT_DISPLACEMENT_M:
                funnel["rejected_too_displaced"] += 1
                continue

            # Robust sigma estimation via MAD
            mad_x = np.median(np.abs(x_m))
            mad_y = np.median(np.abs(y_m))
            sigma_x = 1.4826 * mad_x
            sigma_y = 1.4826 * mad_y
            sigma_pos = float(np.sqrt(sigma_x**2 + sigma_y**2))

            # Non-robust std for comparison
            std_pos = float(np.sqrt(np.var(x_m) + np.var(y_m)))

            funnel["passed_all_filters"] += 1
            rows.append({
                "segment_id": sub_sid,
                "vehicle_id": sg["vehicle_id"].iloc[0],
                "n_points": int(n_pts),
                "n_unique_coords": int(n_unique_coords),
                "uniqueness_ratio": float(uniqueness_ratio),
                "duration_s": float(duration),
                "median_lon": float(np.median(lon)),
                "median_lat": float(np.median(lat)),
                "max_disp_m": float(max_disp),
                "sigma_x_m": float(sigma_x),
                "sigma_y_m": float(sigma_y),
                "sigma_pos_m": sigma_pos,
                "sigma_pos_std_m": std_pos,
                "start_time": sg["gps_time"].min(),
                "end_time": sg["gps_time"].max(),
                "hour_start": sg["gps_time"].min().hour,
            })

    seg_summary = pd.DataFrame(rows)

    # Print filter funnel
    logger.info(f"  Filter funnel:")
    logger.info(f"    Total provisional segments:      {funnel['total_provisional']:,}")
    logger.info(f"    Sub-segments created (long ones split): {funnel['subsegments_created']:,}")
    logger.info(f"    Rejected: too few points:        {funnel['rejected_too_few_points']:,}")
    logger.info(f"    Rejected: too short duration:    {funnel['rejected_too_short']:,}")
    logger.info(f"    Rejected: too few unique coords: {funnel['rejected_too_few_unique_coords']:,}")
    logger.info(f"    Rejected: low uniqueness ratio:  {funnel['rejected_low_uniqueness']:,}")
    logger.info(f"    Rejected: too displaced:         {funnel['rejected_too_displaced']:,}")
    logger.info(f"    Passed all filters:              {funnel['passed_all_filters']:,}")

    if len(seg_summary) == 0:
        logger.warning("  No valid segments found! Check thresholds.")
        return seg_summary, funnel

    logger.info(f"  sigma_pos: median={seg_summary['sigma_pos_m'].median():.2f}m, "
                f"mean={seg_summary['sigma_pos_m'].mean():.2f}m, "
                f"p95={seg_summary['sigma_pos_m'].quantile(0.95):.2f}m")
    logger.info(f"  uniqueness: median={seg_summary['uniqueness_ratio'].median():.3f}, "
                f"min={seg_summary['uniqueness_ratio'].min():.3f}")
    return seg_summary, funnel


# ============================================================
# SPATIAL AGGREGATION
# ============================================================

def aggregate_to_grid(seg_summary: pd.DataFrame, grid_size_m: int,
                      logger) -> pd.DataFrame:
    """Aggregate segment-level sigma_pos into spatial grid cells."""
    if len(seg_summary) == 0:
        return pd.DataFrame()

    # Reference point for projection (city center approximation)
    lat0 = seg_summary["median_lat"].median()
    lon0 = seg_summary["median_lon"].median()

    lon_m_per_deg = M_PER_DEG_LAT * np.cos(np.radians(lat0))

    # Project to local meters
    seg = seg_summary.copy()
    seg["x_m"] = (seg["median_lon"] - lon0) * lon_m_per_deg
    seg["y_m"] = (seg["median_lat"] - lat0) * M_PER_DEG_LAT

    # Bin into grid
    seg["grid_x"] = (seg["x_m"] // grid_size_m).astype(int)
    seg["grid_y"] = (seg["y_m"] // grid_size_m).astype(int)

    # Aggregate
    agg = seg.groupby(["grid_x", "grid_y"]).agg(
        n_segments=("sigma_pos_m", "count"),
        n_unique_vehicles=("vehicle_id", "nunique"),
        sigma_pos_median=("sigma_pos_m", "median"),
        sigma_pos_mean=("sigma_pos_m", "mean"),
        sigma_pos_p25=("sigma_pos_m", lambda x: x.quantile(0.25)),
        sigma_pos_p75=("sigma_pos_m", lambda x: x.quantile(0.75)),
        sigma_pos_p95=("sigma_pos_m", lambda x: x.quantile(0.95)),
        center_lon=("median_lon", "mean"),
        center_lat=("median_lat", "mean"),
    ).reset_index()

    # Convert grid back to coords (cell centers)
    agg["cell_lon"] = lon0 + (agg["grid_x"] + 0.5) * grid_size_m / lon_m_per_deg
    agg["cell_lat"] = lat0 + (agg["grid_y"] + 0.5) * grid_size_m / M_PER_DEG_LAT
    agg["grid_size_m"] = grid_size_m

    logger.info(f"  Grid {grid_size_m}m: {len(agg):,} cells, "
                f"median segments/cell={agg['n_segments'].median():.1f}")
    return agg


# ============================================================
# TEMPORAL AGGREGATION
# ============================================================

def aggregate_temporal(seg_summary: pd.DataFrame, logger) -> pd.DataFrame:
    """Aggregate sigma_pos by time of day."""
    if len(seg_summary) == 0:
        return pd.DataFrame()

    rows = []
    for (h0, h1), label in zip(TIME_BINS, TIME_BIN_LABELS):
        sub = seg_summary[(seg_summary["hour_start"] >= h0) &
                          (seg_summary["hour_start"] < h1)]
        if len(sub) == 0:
            rows.append({"time_bin": label, "n_segments": 0,
                         "sigma_pos_median": np.nan, "sigma_pos_mean": np.nan,
                         "sigma_pos_p25": np.nan, "sigma_pos_p75": np.nan,
                         "sigma_pos_p95": np.nan})
        else:
            rows.append({
                "time_bin": label,
                "n_segments": len(sub),
                "sigma_pos_median": sub["sigma_pos_m"].median(),
                "sigma_pos_mean": sub["sigma_pos_m"].mean(),
                "sigma_pos_p25": sub["sigma_pos_m"].quantile(0.25),
                "sigma_pos_p75": sub["sigma_pos_m"].quantile(0.75),
                "sigma_pos_p95": sub["sigma_pos_m"].quantile(0.95),
            })
    out = pd.DataFrame(rows)
    return out


# ============================================================
# DATA QUALITY DIAGNOSTICS
# ============================================================

def compute_data_quality(df: pd.DataFrame, seg_summary: pd.DataFrame, logger) -> dict:
    """Diagnostics about the input data quality."""
    # Sampling interval
    df_sorted = df.sort_values(["vehicle_id", "gps_time"])
    dt = df_sorted.groupby("vehicle_id")["gps_time"].diff().dt.total_seconds()
    dt_valid = dt.dropna()

    quality = {
        "n_total_records": int(len(df)),
        "n_unique_vehicles": int(df["vehicle_id"].nunique()),
        "time_span_h": float((df["gps_time"].max() - df["gps_time"].min()).total_seconds() / 3600),
        "lon_min": float(df["lon"].min()),
        "lon_max": float(df["lon"].max()),
        "lat_min": float(df["lat"].min()),
        "lat_max": float(df["lat"].max()),
        "spatial_span_ew_km": float((df["lon"].max() - df["lon"].min()) *
                                     M_PER_DEG_LAT * np.cos(np.radians(df["lat"].mean())) / 1000),
        "spatial_span_ns_km": float((df["lat"].max() - df["lat"].min()) * M_PER_DEG_LAT / 1000),
        "sampling_interval_p25_s": float(dt_valid.quantile(0.25)),
        "sampling_interval_p50_s": float(dt_valid.quantile(0.50)),
        "sampling_interval_p75_s": float(dt_valid.quantile(0.75)),
        "frac_records_speed_zero": float((df["speed_kmh"] == 0).mean()),
        "frac_records_speed_lt5": float((df["speed_kmh"] < 5).mean()),
        "n_valid_segments": int(len(seg_summary)),
    }
    if len(seg_summary) > 0:
        quality.update({
            "sigma_pos_median_m": float(seg_summary["sigma_pos_m"].median()),
            "sigma_pos_mean_m": float(seg_summary["sigma_pos_m"].mean()),
            "sigma_pos_p95_m": float(seg_summary["sigma_pos_m"].quantile(0.95)),
            "sigma_pos_min_m": float(seg_summary["sigma_pos_m"].min()),
            "sigma_pos_max_m": float(seg_summary["sigma_pos_m"].max()),
            "frac_segments_in_reasonable_range": float(
                ((seg_summary["sigma_pos_m"] >= SIGMA_POS_REASONABLE_RANGE_M[0]) &
                 (seg_summary["sigma_pos_m"] <= SIGMA_POS_REASONABLE_RANGE_M[1])).mean()
            ),
        })
    return quality


# ============================================================
# VISUALIZATION
# ============================================================

def make_figures(seg_summary: pd.DataFrame, grid_dfs: dict,
                 temporal_df: pd.DataFrame, fig_dir: Path, city: str,
                 logger, funnel: dict = None):
    """Generate diagnostic figures saved as PNG."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Always plot filter funnel if available, even if no segments survived
    if funnel is not None:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            stages = [
                ("Provisional", funnel["total_provisional"]),
                ("Sub-segments\n(after split)",
                    funnel["total_provisional"] + funnel["subsegments_created"]
                    - sum(1 for _ in [funnel["total_provisional"]] if _)),  # informational
                ("Pass: enough pts",
                    funnel["total_provisional"] + funnel["subsegments_created"]
                    - funnel["rejected_too_few_points"]),
                ("Pass: long enough",
                    funnel["total_provisional"] + funnel["subsegments_created"]
                    - funnel["rejected_too_few_points"]
                    - funnel["rejected_too_short"]),
                ("Pass: enough\nunique coords",
                    funnel["total_provisional"] + funnel["subsegments_created"]
                    - funnel["rejected_too_few_points"]
                    - funnel["rejected_too_short"]
                    - funnel["rejected_too_few_unique_coords"]),
                ("Pass: high\nuniqueness",
                    funnel["total_provisional"] + funnel["subsegments_created"]
                    - funnel["rejected_too_few_points"]
                    - funnel["rejected_too_short"]
                    - funnel["rejected_too_few_unique_coords"]
                    - funnel["rejected_low_uniqueness"]),
                ("FINAL: passed\ndisplacement",
                    funnel["passed_all_filters"]),
            ]
            labels = [s[0] for s in stages]
            counts = [s[1] for s in stages]
            colors = ["#4393C3"] * (len(stages) - 1) + ["#2166AC"]
            bars = ax.bar(labels, counts, color=colors, edgecolor="white")
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{c:,}", ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("Number of segments")
            ax.set_title(f"({city}) Filter funnel: provisional → valid")
            ax.tick_params(axis="x", rotation=0, labelsize=8)
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig6_filter_funnel.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not draw funnel: {e}")

    if len(seg_summary) == 0:
        logger.warning("No segments to plot.")
        return

    # ---- Fig 1: sigma_pos distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    sigma = seg_summary["sigma_pos_m"].clip(0, 50)
    ax.hist(sigma, bins=60, color="#4393C3", edgecolor="white")
    ax.axvline(sigma.median(), color="red", linestyle="--",
               label=f"Median = {sigma.median():.2f}m")
    ax.axvline(sigma.mean(), color="orange", linestyle="--",
               label=f"Mean = {sigma.mean():.2f}m")
    ax.set_xlabel("sigma_pos (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"({city}) Segment-level sigma_pos distribution\n"
                 f"N = {len(seg_summary):,} stationary segments")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(np.log10(seg_summary["sigma_pos_m"].clip(lower=0.01)),
            bins=60, color="#4393C3", edgecolor="white")
    ax.set_xlabel("log10(sigma_pos)  [m]")
    ax.set_ylabel("Count")
    ax.set_title("Log scale for tail visibility")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_sigma_pos_distribution.png", dpi=150)
    plt.close()

    # ---- Fig 2: spatial heatmap (100m grid) ----
    if 100 in grid_dfs and len(grid_dfs[100]) > 0:
        agg = grid_dfs[100]
        fig, ax = plt.subplots(figsize=(9, 8))
        # Use only cells with enough segments
        agg_plot = agg[agg["n_segments"] >= 3]
        if len(agg_plot) > 0:
            sc = ax.scatter(agg_plot["cell_lon"], agg_plot["cell_lat"],
                            c=agg_plot["sigma_pos_median"].clip(0, 20),
                            s=10, cmap="RdYlGn_r", marker="s")
            plt.colorbar(sc, ax=ax, label="Median sigma_pos (m)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"({city}) 100m grid: median sigma_pos\n"
                         f"({len(agg_plot):,} cells with ≥3 segments)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig2_spatial_heatmap_100m.png", dpi=150)
        plt.close()

    # ---- Fig 3: temporal pattern ----
    if len(temporal_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(temporal_df))
        ax.plot(x, temporal_df["sigma_pos_median"], "o-", color="#2166AC",
                linewidth=2, markersize=8, label="Median")
        ax.fill_between(x, temporal_df["sigma_pos_p25"],
                        temporal_df["sigma_pos_p75"],
                        alpha=0.25, color="#2166AC", label="IQR")
        ax.plot(x, temporal_df["sigma_pos_p95"], "s--", color="#B2182B",
                alpha=0.7, label="P95")
        ax.set_xticks(x)
        ax.set_xticklabels(temporal_df["time_bin"])
        ax.set_xlabel("Time of day (hour)")
        ax.set_ylabel("sigma_pos (m)")
        ax.set_title(f"({city}) Temporal pattern of sigma_pos")
        ax.legend()
        ax.grid(alpha=0.3)
        # Add segment counts as secondary info
        for i, n in enumerate(temporal_df["n_segments"]):
            ax.annotate(f"n={n}", (i, temporal_df["sigma_pos_median"].iloc[i]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center", color="gray")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig3_temporal_pattern.png", dpi=150)
        plt.close()

    # ---- Fig 4: segment quality (n_points vs duration) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.scatter(seg_summary["n_points"], seg_summary["sigma_pos_m"].clip(0, 30),
               alpha=0.3, s=10)
    ax.set_xlabel("Number of GPS points in segment")
    ax.set_ylabel("sigma_pos (m, clipped at 30)")
    ax.set_title("Segment size vs. estimated sigma_pos")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    ax = axes[1]
    ax.scatter(seg_summary["duration_s"] / 60, seg_summary["sigma_pos_m"].clip(0, 30),
               alpha=0.3, s=10)
    ax.set_xlabel("Segment duration (minutes)")
    ax.set_ylabel("sigma_pos (m, clipped at 30)")
    ax.set_title("Duration vs. estimated sigma_pos")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_segment_quality.png", dpi=150)
    plt.close()

    # ---- Fig 5: robust vs non-robust comparison ----
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = seg_summary[(seg_summary["sigma_pos_m"] < 50) &
                        (seg_summary["sigma_pos_std_m"] < 50)]
    ax.scatter(valid["sigma_pos_m"], valid["sigma_pos_std_m"],
               alpha=0.3, s=10)
    lim = max(valid["sigma_pos_m"].max(), valid["sigma_pos_std_m"].max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("sigma_pos (MAD-based, robust)")
    ax.set_ylabel("sigma_pos (std-based, non-robust)")
    ax.set_title("Robust vs non-robust sigma estimate\n(divergence indicates outliers/jumps)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_robust_vs_std.png", dpi=150)
    plt.close()

    logger.info(f"  Figures saved to {fig_dir}")


# ============================================================
# MAIN
# ============================================================

def run_poc(city: str, data_root: Path, out_root: Path):
    """Run full PoC pipeline for one city."""
    city_dir = data_root / city
    out_dir = out_root / f"poc_{city}"

    # Auto-cleanup: remove old output for this city to ensure fresh results.
    # Only this city's poc_<city> folder is removed; other cities and the
    # diag_<city> folders are untouched.
    if out_dir.exists():
        print(f"[INFO] Removing existing {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "intermediate").mkdir(exist_ok=True)

    logger = setup_logger(out_dir)
    logger.info(f"========== PoC starting for city: {city} ==========")
    logger.info(f"Data dir: {city_dir}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Run started at {datetime.now().isoformat()}")

    # Step 1: Load data
    df = load_city_data(city_dir, logger)

    # Step 2: Detect stationary segments
    seg_df = detect_stationary_segments(df, logger)

    # Step 3: Compute sigma_pos per segment (returns funnel stats too)
    seg_summary, funnel = compute_segment_sigma_pos(seg_df, logger)

    # Save filter funnel even if no segments survived (helpful for debugging)
    with open(out_dir / "stats" / "filter_funnel.json", "w") as f:
        json.dump(funnel, f, indent=2)

    if len(seg_summary) == 0:
        logger.error("No valid segments found - check thresholds or data quality")
        quality = compute_data_quality(df, seg_summary, logger)
        with open(out_dir / "stats" / "data_quality.json", "w") as f:
            json.dump(quality, f, indent=2, default=str)
        return

    # Step 4: Save segment-level data (intermediate, anonymized - vehicle_id hashed)
    seg_summary_anon = seg_summary.copy()
    seg_summary_anon["vehicle_id"] = seg_summary_anon["vehicle_id"].apply(
        lambda v: f"v{hash(str(v)) % 10**8:08d}"  # 8-digit hash, no reverse possible
    )
    seg_summary_anon.to_parquet(out_dir / "intermediate" / "segments.parquet",
                                 index=False)
    logger.info(f"  Saved {len(seg_summary_anon):,} segments to intermediate/")

    # Step 5: Spatial aggregation at multiple grid sizes
    grid_dfs = {}
    for gs in GRID_SIZES_M:
        agg = aggregate_to_grid(seg_summary, gs, logger)
        agg.to_parquet(out_dir / "intermediate" / f"grid_{gs}m.parquet",
                       index=False)
        grid_dfs[gs] = agg

    # Step 6: Temporal aggregation
    temporal_df = aggregate_temporal(seg_summary, logger)
    temporal_df.to_csv(out_dir / "stats" / "temporal_pattern.csv", index=False)

    # Step 7: Data quality and summary statistics
    quality = compute_data_quality(df, seg_summary, logger)
    with open(out_dir / "stats" / "data_quality.json", "w") as f:
        json.dump(quality, f, indent=2, default=str)

    # Sigma_pos quantile summary
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    sigma_quantiles = {f"q{int(q*100):02d}": float(seg_summary["sigma_pos_m"].quantile(q))
                       for q in quantiles}
    with open(out_dir / "stats" / "sigma_pos_quantiles.json", "w") as f:
        json.dump(sigma_quantiles, f, indent=2)

    # Step 8: Figures
    make_figures(seg_summary, grid_dfs, temporal_df,
                 out_dir / "figures", city, logger, funnel=funnel)

    # Step 9: Final summary printout
    logger.info("\n" + "="*60)
    logger.info(f"FINAL SUMMARY for {city}:")
    logger.info(f"  Records:  {quality['n_total_records']:,}")
    logger.info(f"  Vehicles: {quality['n_unique_vehicles']:,}")
    logger.info(f"  Stationary segments (valid): {quality['n_valid_segments']:,}")
    if quality['n_valid_segments'] > 0:
        logger.info(f"  sigma_pos median:  {quality['sigma_pos_median_m']:.2f}m")
        logger.info(f"  sigma_pos p95:     {quality['sigma_pos_p95_m']:.2f}m")
        logger.info(f"  Reasonable range:  {quality['frac_segments_in_reasonable_range']*100:.1f}%")
    logger.info(f"  Output:   {out_dir}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="PoC: extract sigma_pos from taxi GPS. "
                    "Without --city, processes all subfolders in data-root.")
    parser.add_argument("--city", default=None,
                        help="(Optional) City folder name. "
                             "If omitted, processes ALL subfolders in city_data/.")
    parser.add_argument("--data-root", default="../city_data",
                        help="Root dir containing city subdirs")
    parser.add_argument("--out-root", default="../output",
                        help="Root dir for output")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not data_root.exists():
        sys.exit(f"data-root {data_root} does not exist")

    # Determine list of cities to run
    if args.city is not None:
        cities = [args.city]
    else:
        # Auto-discover: every subfolder of data_root that contains parquet files
        cities = []
        for sub in sorted(data_root.iterdir()):
            if sub.is_dir() and any(sub.glob("*.parquet")):
                cities.append(sub.name)
        if not cities:
            sys.exit(f"No city subfolders with parquet files found in {data_root}")
        print(f"\n[BATCH MODE] Found {len(cities)} cities: {cities}\n")

    failed = []
    for i, city in enumerate(cities):
        if len(cities) > 1:
            print(f"\n{'#'*70}")
            print(f"# [{i+1}/{len(cities)}] Processing city: {city}")
            print(f"{'#'*70}")
        try:
            run_poc(city, data_root, out_root)
        except Exception as e:
            failed.append(city)
            print(f"[BATCH] {city} failed with: {e}")
            import traceback
            traceback.print_exc()
            print(f"[BATCH] Continuing with next city...")

    if len(cities) > 1:
        print(f"\n{'='*70}")
        print(f"BATCH SUMMARY: {len(cities)-len(failed)}/{len(cities)} cities succeeded")
        if failed:
            print(f"  Failed: {failed}")
        print('='*70)


if __name__ == "__main__":
    main()
