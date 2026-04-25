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
# CONFIGURATION
# ============================================================

# Stationary segment detection
SPEED_THRESHOLD_KMH = 2.0          # speed below this = potentially stationary
MIN_SEGMENT_DURATION_S = 300       # minimum 5 minutes of stationarity
MAX_SEGMENT_DISPLACEMENT_M = 30    # if max displacement > 30m, not truly stationary
MIN_POINTS_PER_SEGMENT = 10        # need at least 10 GPS points for reliable sigma estimate
MAX_TIME_GAP_WITHIN_SEGMENT_S = 120  # gaps > 2 min split a segment

# Spatial aggregation grids (in meters)
GRID_SIZES_M = [50, 100, 250]     # main analysis: 100m. 50m & 250m for sensitivity check

# Temporal binning (6 bins, 4 hours each)
TIME_BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
TIME_BIN_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]

# Earth constants for lat/lon to meter conversion
M_PER_DEG_LAT = 111320.0           # approximate, varies <1% by latitude

# Diagnostic / sanity-check thresholds
SIGMA_POS_REASONABLE_RANGE_M = (0.1, 100.0)  # sigma_pos outside this is suspicious

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

def compute_segment_sigma_pos(seg_df: pd.DataFrame, logger) -> pd.DataFrame:
    """For each stationary segment, compute sigma_pos (positioning uncertainty).

    sigma_pos is computed using median absolute deviation (MAD)-based estimator,
    which is robust to outliers/jumps common in urban canyon GPS:

        sigma_pos = sqrt(MAD_x^2 + MAD_y^2) * 1.4826
    
    where MAD_x is in meters (longitude scaled by cos(lat)).
    """
    logger.info("Computing per-segment sigma_pos...")
    grouped = seg_df.groupby("segment_id")

    rows = []
    n_total = grouped.ngroups
    for sid, g in grouped:
        n_pts = len(g)
        duration = (g["gps_time"].max() - g["gps_time"].min()).total_seconds()

        # Filter: minimum points and duration
        if n_pts < MIN_POINTS_PER_SEGMENT or duration < MIN_SEGMENT_DURATION_S:
            continue

        lon = g["lon"].values
        lat = g["lat"].values
        lat_mean = np.mean(lat)

        # Convert to meters relative to median location
        lon_m_per_deg = M_PER_DEG_LAT * np.cos(np.radians(lat_mean))
        x_m = (lon - np.median(lon)) * lon_m_per_deg
        y_m = (lat - np.median(lat)) * M_PER_DEG_LAT

        # Reject if max displacement too large (vehicle was actually moving slowly)
        max_disp = np.sqrt(x_m**2 + y_m**2).max()
        if max_disp > MAX_SEGMENT_DISPLACEMENT_M:
            continue

        # Robust sigma estimation via MAD
        # MAD = median(|x - median(x)|), and sigma ≈ 1.4826 * MAD for Gaussian
        mad_x = np.median(np.abs(x_m))
        mad_y = np.median(np.abs(y_m))
        sigma_x = 1.4826 * mad_x
        sigma_y = 1.4826 * mad_y
        sigma_pos = np.sqrt(sigma_x**2 + sigma_y**2)

        # Also compute non-robust std for comparison
        std_pos = np.sqrt(np.var(x_m) + np.var(y_m))

        rows.append({
            "segment_id": sid,
            "vehicle_id": g["vehicle_id"].iloc[0],
            "n_points": n_pts,
            "duration_s": duration,
            "median_lon": float(np.median(lon)),
            "median_lat": float(np.median(lat)),
            "max_disp_m": float(max_disp),
            "sigma_x_m": float(sigma_x),
            "sigma_y_m": float(sigma_y),
            "sigma_pos_m": float(sigma_pos),
            "sigma_pos_std_m": float(std_pos),
            "start_time": g["gps_time"].min(),
            "end_time": g["gps_time"].max(),
            "hour_start": g["gps_time"].min().hour,
        })

    seg_summary = pd.DataFrame(rows)
    logger.info(f"  Valid segments: {len(seg_summary):,} (after duration/points/displacement filters)")
    if len(seg_summary) == 0:
        logger.warning("  No valid segments found! Check thresholds.")
        return seg_summary

    logger.info(f"  sigma_pos: median={seg_summary['sigma_pos_m'].median():.2f}m, "
                f"mean={seg_summary['sigma_pos_m'].mean():.2f}m, "
                f"p95={seg_summary['sigma_pos_m'].quantile(0.95):.2f}m")
    return seg_summary


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
                 logger):
    """Generate diagnostic figures saved as PNG."""
    fig_dir.mkdir(parents=True, exist_ok=True)
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

    # Step 3: Compute sigma_pos per segment
    seg_summary = compute_segment_sigma_pos(seg_df, logger)

    if len(seg_summary) == 0:
        logger.error("No valid segments found - check thresholds or data quality")
        # Still save data quality report
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
                 out_dir / "figures", city, logger)

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
    parser = argparse.ArgumentParser(description="PoC: extract sigma_pos from taxi GPS")
    parser.add_argument("--city", required=True,
                        help="City folder name under data-root (e.g. 'beijing')")
    parser.add_argument("--data-root", default="../city_data",
                        help="Root dir containing city subdirs")
    parser.add_argument("--out-root", default="../output",
                        help="Root dir for output")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not data_root.exists():
        sys.exit(f"data-root {data_root} does not exist")

    run_poc(args.city, data_root, out_root)


if __name__ == "__main__":
    main()
