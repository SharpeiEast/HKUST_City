"""
HKUST GPS Data Diagnostic (v2)
================================

Updates from v1:
- Auto-cleans old `output/diag_<city>/` before running
- Captures full terminal output to a log file
- Detects coordinate reference system (WGS-84 vs GCJ-02)
- More verbose progress messages for junior collaborators

Goal: Diagnose whether HKUST taxi GPS data is suitable for empirical
positioning uncertainty research. Examines five aspects:
1. Spatial coverage (CBD vs. suburbs)
2. Map-matching artifacts (3 independent signals)
3. Vehicle subset structure (raw vs processed mix)
4. Sampling characteristics
5. Coordinate reference system

Output is fully aggregated and anonymized — no raw GPS records leave HKUST.

Usage (from inside coding/ folder):
    python data_diagnostic.py --city beijing
    python data_diagnostic.py --city shanghai
    python data_diagnostic.py --city guangzhou
    python data_diagnostic.py --city zhengzhou   # optional: as a control case

Default folder layout:
    project_root/
    ├── coding/
    │   └── data_diagnostic.py    ← run from here
    ├── city_data/
    │   ├── beijing/*.parquet
    │   ├── shanghai/*.parquet
    │   ├── guangzhou/*.parquet
    │   └── zhengzhou/*.parquet   (optional control case)
    └── output/                   ← auto-cleaned and recreated per city
        ├── diag_beijing/
        ├── diag_shanghai/
        └── diag_guangzhou/
"""
import argparse
import json
import logging
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

CITY_CBDS = {
    "beijing":   (116.40, 39.91),
    "shanghai":  (121.50, 31.24),
    "guangzhou": (113.32, 23.13),
    "shenzhen":  (114.06, 22.54),
    "zhengzhou": (113.62, 34.75),  # control case
}

CITY_BBOX = {
    "beijing":   (115.40, 39.40, 117.50, 40.60),
    "shanghai":  (120.85, 30.65, 122.00, 31.55),
    "guangzhou": (112.95, 22.95, 113.95, 23.95),
    "shenzhen":  (113.70, 22.40, 114.65, 22.86),
    "zhengzhou": (113.30, 34.55, 114.10, 34.95),
}

SPEED_THRESHOLD_KMH = 2.0
MIN_SEGMENT_DURATION_S = 300
MAX_SEGMENT_DISPLACEMENT_M = 30
MIN_POINTS_PER_SEGMENT = 10
MAX_TIME_GAP_WITHIN_SEGMENT_S = 120
M_PER_DEG_LAT = 111320.0


# ============================================================
# OUTPUT FOLDER MANAGEMENT (with auto-cleanup)
# ============================================================

def setup_output_dir(out_root: Path, city: str) -> Path:
    """Create a clean output dir, removing any existing one."""
    out_dir = out_root / f"diag_{city}"
    if out_dir.exists():
        print(f"[INFO] Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "intermediate").mkdir(exist_ok=True)
    return out_dir


# ============================================================
# LOGGING (captures to file AND stdout)
# ============================================================

class TeeStream:
    """Duplicates stdout/stderr to both terminal and file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            try:
                s.write(msg)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_logger_and_capture(out_dir: Path):
    """Set up logger AND redirect stdout/stderr to also log files.

    This way, ANY print() or unhandled exception trace is captured —
    very useful when a junior collaborator runs the code and we can't
    see their terminal directly.
    """
    log_path = out_dir / "diagnostic.log"
    terminal_capture = out_dir / "terminal_output.txt"

    # File handler for the python `logging` module
    logger = logging.getLogger("diag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Tee stdout / stderr to a separate raw capture file
    capture_file = open(terminal_capture, "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, capture_file)
    sys.stderr = TeeStream(sys.__stderr__, capture_file)

    return logger, capture_file


# ============================================================
# DATA LOADING
# ============================================================

def load_city_data(city_dir: Path, logger) -> pd.DataFrame:
    files = sorted(city_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {city_dir}")
    logger.info(f"Found {len(files)} parquet files in {city_dir}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
        logger.info(f"  Loaded {f.name}: {len(df):,} rows, "
                    f"columns = {list(df.columns)[:8]}{'...' if len(df.columns)>8 else ''}")

    df = pd.concat(dfs, ignore_index=True)

    df["gps_time"] = pd.to_datetime(df["gps_time"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["speed_kmh"] = pd.to_numeric(df["speed_kmh"], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["vehicle_id", "lon", "lat", "speed_kmh", "gps_time"])
    df = df[(df["lon"] > 70) & (df["lon"] < 140) &
            (df["lat"] > 15) & (df["lat"] < 55)]
    df = df[(df["speed_kmh"] >= 0) & (df["speed_kmh"] < 200)]
    df = df.sort_values(["vehicle_id", "gps_time"]).reset_index(drop=True)

    logger.info(f"Records after cleaning: {len(df):,} "
                f"(dropped {n0-len(df):,}, {100*(n0-len(df))/n0:.2f}%)")
    logger.info(f"Vehicles: {df['vehicle_id'].nunique():,}")
    return df


# ============================================================
# DIAGNOSTIC 0: COORDINATE SYSTEM DETECTION
# ============================================================

def detect_coordinate_system(df: pd.DataFrame, city: str, logger) -> dict:
    """
    Detect whether GPS coordinates are WGS-84 or GCJ-02 (Mars coordinates,
    used by Chinese commercial maps).

    Method: WGS-84 and GCJ-02 differ by 50-500m at any given point in
    mainland China, with a known nonlinear offset. We compute the offset
    from the city's nominal CBD (assumed WGS-84) to the median observed
    GPS location. If offset is consistent with GCJ-02 transform, flag it.
    """
    logger.info("\n--- Diagnostic 0: Coordinate system check ---")

    if city not in CITY_CBDS:
        logger.info(f"  [SKIP] No reference CBD known for '{city}'")
        return {"note": "no reference CBD"}

    cbd_lon, cbd_lat = CITY_CBDS[city]

    # Compute median GPS location near the assumed CBD (within 5km in WGS-84 sense)
    # We use a generous 10km box to account for GCJ-02 offset
    search_box = 0.1  # ~11km
    near = df[(df.lon > cbd_lon - search_box) & (df.lon < cbd_lon + search_box) &
              (df.lat > cbd_lat - search_box) & (df.lat < cbd_lat + search_box)]
    if len(near) < 100:
        logger.info(f"  [WARN] Only {len(near)} records near assumed CBD "
                    f"({cbd_lon}, {cbd_lat}); coordinate check unreliable")
        return {"note": "insufficient records near CBD"}

    # Compute GCJ-02 offset that WGS-84 (cbd_lon, cbd_lat) would have.
    # This is a simplified GCJ-02 transform constant for China:
    # Approximate offset at China (lat ~35°): roughly +0.0035° lat, +0.0085° lon
    # The exact transform requires the official Chinese encryption algorithm,
    # but the offset has known typical magnitudes:
    #   beijing:   dlon ≈ +0.0061, dlat ≈ +0.0011  (GCJ-02 = WGS-84 + this)
    #   shanghai:  dlon ≈ +0.0061, dlat ≈ +0.0017
    #   guangzhou: dlon ≈ +0.0048, dlat ≈ +0.0014
    typical_gcj_offset = {
        "beijing":   (0.0061, 0.0011),
        "shanghai":  (0.0061, 0.0017),
        "guangzhou": (0.0048, 0.0014),
        "shenzhen":  (0.0050, 0.0015),
        "zhengzhou": (0.0058, 0.0017),
    }

    # Median GPS deviation from WGS-84 CBD (in meters)
    med_lon = near.lon.median()
    med_lat = near.lat.median()
    dlon_obs = med_lon - cbd_lon
    dlat_obs = med_lat - cbd_lat

    expected_dlon, expected_dlat = typical_gcj_offset.get(city, (0.006, 0.0015))

    logger.info(f"  Assumed CBD (WGS-84): ({cbd_lon}, {cbd_lat})")
    logger.info(f"  Observed median near CBD: ({med_lon:.6f}, {med_lat:.6f})")
    logger.info(f"  Observed offset: dlon={dlon_obs:+.6f}, dlat={dlat_obs:+.6f} "
                f"({dlon_obs*111000:+.0f}m, {dlat_obs*111000:+.0f}m)")
    logger.info(f"  Expected GCJ-02 offset: dlon={expected_dlon:+.6f}, "
                f"dlat={expected_dlat:+.6f}")

    # Decision: if observed offset matches GCJ-02 within 200m, flag as GCJ-02
    match_gcj = (
        abs(dlon_obs - expected_dlon) < 0.002 and
        abs(dlat_obs - expected_dlat) < 0.001
    )
    match_wgs = abs(dlon_obs) < 0.002 and abs(dlat_obs) < 0.002

    if match_gcj:
        verdict = "Likely GCJ-02 (Chinese encrypted coordinates)"
        recommendation = "GPS data is in GCJ-02. Must convert to WGS-84 before joining with CMAB (which is WGS-84/EPSG:3857)."
    elif match_wgs:
        verdict = "Likely WGS-84"
        recommendation = "GPS and CMAB share WGS-84 reference; no conversion needed."
    else:
        verdict = "Inconclusive — offset doesn't match either reference"
        recommendation = ("Offset is anomalous. Possible explanations: (a) GPS data "
                          "represents a different geographic area than expected, (b) "
                          "data is in BD-09 or another exotic CRS, (c) CBD reference is wrong.")

    logger.info(f"  Verdict: {verdict}")
    return {
        "city": city,
        "assumed_cbd_wgs84": [cbd_lon, cbd_lat],
        "observed_median_near_cbd": [float(med_lon), float(med_lat)],
        "observed_offset_deg": [float(dlon_obs), float(dlat_obs)],
        "observed_offset_m": [float(dlon_obs * 111000), float(dlat_obs * 111000)],
        "expected_gcj02_offset_deg": [expected_dlon, expected_dlat],
        "verdict": verdict,
        "recommendation": recommendation,
    }


# ============================================================
# DIAGNOSTIC 1: SPATIAL COVERAGE
# ============================================================

def diagnose_spatial_coverage(df: pd.DataFrame, city: str, out_dir: Path,
                                logger) -> dict:
    logger.info("\n--- Diagnostic 1: Spatial coverage ---")

    cbd_lon, cbd_lat = CITY_CBDS.get(city, (df.lon.median(), df.lat.median()))
    bbox = CITY_BBOX.get(city, (df.lon.min(), df.lat.min(),
                                  df.lon.max(), df.lat.max()))

    dx_km = (df.lon.values - cbd_lon) * 111 * np.cos(np.radians(cbd_lat))
    dy_km = (df.lat.values - cbd_lat) * 111
    dist_km = np.sqrt(dx_km ** 2 + dy_km ** 2)

    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    dist_quantiles = {f"q{int(q*100):02d}": float(np.quantile(dist_km, q))
                       for q in quantiles}
    logger.info(f"  Distance from CBD (km): {dist_quantiles}")

    lon_step = 1.0 / (111 * np.cos(np.radians(cbd_lat)))
    lat_step = 1.0 / 111
    grid_lon_idx = np.floor((df.lon - bbox[0]) / lon_step).astype(int)
    grid_lat_idx = np.floor((df.lat - bbox[1]) / lat_step).astype(int)
    grid_df = pd.DataFrame({"grid_lon_idx": grid_lon_idx,
                             "grid_lat_idx": grid_lat_idx})
    cell_counts = grid_df.groupby(["grid_lon_idx", "grid_lat_idx"]).size()
    cell_counts.name = "n_records"
    cell_counts = cell_counts.reset_index()
    cell_counts["cell_lon"] = bbox[0] + (cell_counts["grid_lon_idx"] + 0.5) * lon_step
    cell_counts["cell_lat"] = bbox[1] + (cell_counts["grid_lat_idx"] + 0.5) * lat_step
    cell_counts.to_parquet(out_dir / "intermediate" / "spatial_density_1km.parquet",
                            index=False)

    coverage_thresholds = [10, 100, 1000, 10000]
    coverage = {f"cells_with_ge_{t}_records": int((cell_counts.n_records >= t).sum())
                for t in coverage_thresholds}
    sorted_counts = cell_counts.n_records.sort_values(ascending=False).reset_index(drop=True)
    total = sorted_counts.sum()
    concentration = {
        "top_1_cell_pct": float(sorted_counts.iloc[0] / total * 100) if len(sorted_counts) > 0 else 0,
        "top_10_cells_pct": float(sorted_counts.head(10).sum() / total * 100) if len(sorted_counts) >= 10 else 0,
        "top_100_cells_pct": float(sorted_counts.head(100).sum() / total * 100) if len(sorted_counts) >= 100 else 0,
        "n_total_cells": int(len(cell_counts)),
        "median_records_per_cell": float(cell_counts.n_records.median()),
    }
    logger.info(f"  Coverage: {coverage}")
    logger.info(f"  Concentration: {concentration}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    lon_min, lat_min, lon_max, lat_max = bbox
    nx = int(np.ceil((lon_max - lon_min) / lon_step))
    ny = int(np.ceil((lat_max - lat_min) / lat_step))
    grid = np.zeros((ny, nx), dtype=float)
    for _, r in cell_counts.iterrows():
        ix, iy = int(r.grid_lon_idx), int(r.grid_lat_idx)
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[iy, ix] = r.n_records
    im = ax.imshow(np.log10(grid + 1), origin="lower",
                    extent=[lon_min, lon_max, lat_min, lat_max],
                    cmap="hot_r", aspect="auto")
    ax.scatter([cbd_lon], [cbd_lat], c="cyan", s=120, marker="*",
                edgecolors="black", linewidths=1.5, label="Assumed CBD",
                zorder=5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"({city}) GPS density (log10), 1km grid\n"
                 f"{len(cell_counts):,} cells with data")
    plt.colorbar(im, ax=ax, label="log10(records + 1)")
    ax.legend()

    ax = axes[1]
    sorted_dist = np.sort(dist_km)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    ax.plot(sorted_dist, cdf, color="#2166AC", linewidth=2)
    for q, val in dist_quantiles.items():
        ax.axvline(val, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Distance from CBD (km)")
    ax.set_ylabel("Cumulative fraction of records")
    ax.set_title(f"Distance-to-CBD distribution\n"
                 f"50% within {dist_quantiles['q50']:.1f}km, "
                 f"95% within {dist_quantiles['q95']:.1f}km")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(100, sorted_dist[-1]))
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "diag1_spatial_coverage.png", dpi=120)
    plt.close()

    return {
        "cbd": [cbd_lon, cbd_lat],
        "bbox": list(bbox),
        "distance_from_cbd_km": dist_quantiles,
        "coverage": coverage,
        "concentration": concentration,
    }


# ============================================================
# DIAGNOSTIC 2: MAP-MATCHING DETECTION
# ============================================================

def diagnose_map_matching(df: pd.DataFrame, out_dir: Path, logger) -> dict:
    logger.info("\n--- Diagnostic 2: Map-matching detection ---")
    results = {}

    # Signal A: coordinate precision
    sample_size = min(100_000, len(df))
    sample = df.sample(sample_size, random_state=42)
    lon_decs = sample.lon.astype(str).str.split(".").str[1].str.len().fillna(0)
    lat_decs = sample.lat.astype(str).str.split(".").str[1].str.len().fillna(0)
    results["signal_A_precision"] = {
        "lon_decimal_places_median": float(lon_decs.median()),
        "lon_decimal_places_p95": float(lon_decs.quantile(0.95)),
        "lat_decimal_places_median": float(lat_decs.median()),
        "lat_decimal_places_p95": float(lat_decs.quantile(0.95)),
    }
    logger.info(f"  Signal A — coordinate precision: lon med={lon_decs.median()} decimals")

    # Signal B: identical consecutive coordinates for stopped vehicles
    df_sorted = df.sort_values(["vehicle_id", "gps_time"])
    stopped = df_sorted[df_sorted.speed_kmh < 1.0].copy()
    if len(stopped) > 100:
        stopped["prev_lon"] = stopped.groupby("vehicle_id")["lon"].shift(1)
        stopped["prev_lat"] = stopped.groupby("vehicle_id")["lat"].shift(1)
        valid_pairs = stopped.dropna(subset=["prev_lon", "prev_lat"])
        identical = (valid_pairs.lon == valid_pairs.prev_lon) & (valid_pairs.lat == valid_pairs.prev_lat)
        results["signal_B_consecutive_identical"] = {
            "n_stopped_pairs_examined": int(len(valid_pairs)),
            "frac_consecutive_identical": float(identical.mean()),
            "interpretation": (
                "If >50%: data is heavily map-matched. "
                "If 0-5%: data preserves natural GPS jitter. "
                "If 5-50%: partially processed."
            ),
        }
        logger.info(f"  Signal B — consec-identical (stopped): "
                    f"{100*identical.mean():.1f}% (n={len(valid_pairs):,})")
    else:
        results["signal_B_consecutive_identical"] = {"note": "Insufficient stopped data"}

    # Signal C: within-segment coordinate diversity
    df_seg = df_sorted.copy()
    df_seg["is_slow"] = df_seg["speed_kmh"] <= SPEED_THRESHOLD_KMH
    df_seg["prev_time"] = df_seg.groupby("vehicle_id")["gps_time"].shift(1)
    df_seg["dt_s"] = (df_seg["gps_time"] - df_seg["prev_time"]).dt.total_seconds()

    n = len(df_seg)
    is_slow = df_seg["is_slow"].to_numpy()
    vehicle = df_seg["vehicle_id"].to_numpy()
    dt_s = df_seg["dt_s"].to_numpy()
    continues = np.zeros(n, dtype=bool)
    if n > 1:
        same_v = (vehicle[1:] == vehicle[:-1])
        ps = is_slow[:-1]
        cs = is_slow[1:]
        gp = np.nan_to_num(dt_s[1:], nan=1e9) <= MAX_TIME_GAP_WITHIN_SEGMENT_S
        continues[1:] = same_v & ps & cs & gp
    new_seg = is_slow & ~continues
    seg_id = np.where(new_seg, 1, 0).cumsum()
    seg_id = np.where(is_slow, seg_id, -1)
    df_seg["segment_id"] = seg_id

    segs = df_seg[df_seg.segment_id >= 0]
    grouped = segs.groupby("segment_id").agg(
        n_points=("lon", "size"),
        n_unique=("lon", lambda x: len(set(zip(x, segs.loc[x.index, "lat"])))),
        duration_s=("gps_time", lambda x: (x.max() - x.min()).total_seconds())
    )
    long_segs = grouped[(grouped.n_points >= MIN_POINTS_PER_SEGMENT) &
                         (grouped.duration_s >= MIN_SEGMENT_DURATION_S)].copy()
    if len(long_segs) > 0:
        long_segs["uniqueness_ratio"] = long_segs.n_unique / long_segs.n_points
        results["signal_C_coordinate_diversity"] = {
            "n_qualifying_segments": int(len(long_segs)),
            "uniqueness_ratio_min": float(long_segs.uniqueness_ratio.min()),
            "uniqueness_ratio_p25": float(long_segs.uniqueness_ratio.quantile(0.25)),
            "uniqueness_ratio_median": float(long_segs.uniqueness_ratio.median()),
            "uniqueness_ratio_p75": float(long_segs.uniqueness_ratio.quantile(0.75)),
            "frac_segments_all_identical": float((long_segs.uniqueness_ratio < 0.05).mean()),
            "frac_segments_fully_diverse": float((long_segs.uniqueness_ratio > 0.9).mean()),
            "interpretation": (
                "Raw GPS: typically >0.8 per segment. "
                "Map-matched: clusters near 0 or specific values like 0.5."
            ),
        }
        logger.info(f"  Signal C — coord-diversity median: "
                    f"{long_segs.uniqueness_ratio.median():.3f}, "
                    f"frac all-identical: {100*(long_segs.uniqueness_ratio<0.05).mean():.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.hist(long_segs.uniqueness_ratio, bins=50, color="#4393C3",
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="0.5 reference")
        ax.set_xlabel("Uniqueness ratio (unique coords / n_points)")
        ax.set_ylabel("Number of segments")
        ax.set_title(f"Coordinate diversity within segments (N={len(long_segs):,})")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[1]
        ax.scatter(long_segs.n_points, long_segs.uniqueness_ratio,
                    alpha=0.3, s=15, c="#4393C3")
        ax.set_xlabel("Number of points in segment")
        ax.set_ylabel("Uniqueness ratio")
        ax.set_title("Diversity vs segment size")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "diag2_coordinate_diversity.png", dpi=120)
        plt.close()
    else:
        results["signal_C_coordinate_diversity"] = {"note": "Insufficient long segments"}

    # Combined score
    score = 0
    score_components = 0
    if "frac_consecutive_identical" in results.get("signal_B_consecutive_identical", {}):
        score += results["signal_B_consecutive_identical"]["frac_consecutive_identical"]
        score_components += 1
    if "frac_segments_all_identical" in results.get("signal_C_coordinate_diversity", {}):
        score += results["signal_C_coordinate_diversity"]["frac_segments_all_identical"]
        score_components += 1
    score = score / score_components if score_components > 0 else 0
    results["overall_map_matching_score"] = {
        "value": float(score),
        "interpretation": (
            "0.0 = pure raw GPS. 0.5 = partial. 1.0 = fully map-matched. "
            "Threshold for our research: must be < 0.3."
        ),
    }
    logger.info(f"  Overall map-matching score: {score:.3f}")
    return results


# ============================================================
# DIAGNOSTIC 3: VEHICLE SUBSETS
# ============================================================

def diagnose_vehicle_subsets(df: pd.DataFrame, out_dir: Path, logger) -> dict:
    logger.info("\n--- Diagnostic 3: Vehicle subsets ---")
    veh_stats = df.groupby("vehicle_id").agg(
        n_records=("lon", "size"),
        n_unique_coords=("lon", lambda x: len(set(zip(x, df.loc[x.index, "lat"])))),
    )
    veh_stats["uniqueness_per_vehicle"] = veh_stats.n_unique_coords / veh_stats.n_records
    veh_stats = veh_stats[veh_stats.n_records >= 100]

    if len(veh_stats) > 50:
        results = {
            "n_vehicles_analyzed": int(len(veh_stats)),
            "uniqueness_distribution": {
                "min": float(veh_stats.uniqueness_per_vehicle.min()),
                "p10": float(veh_stats.uniqueness_per_vehicle.quantile(0.10)),
                "median": float(veh_stats.uniqueness_per_vehicle.median()),
                "p90": float(veh_stats.uniqueness_per_vehicle.quantile(0.90)),
                "max": float(veh_stats.uniqueness_per_vehicle.max()),
            },
            "n_vehicles_high_diversity": int((veh_stats.uniqueness_per_vehicle > 0.8).sum()),
            "n_vehicles_low_diversity": int((veh_stats.uniqueness_per_vehicle < 0.2).sum()),
            "n_vehicles_intermediate": int(((veh_stats.uniqueness_per_vehicle >= 0.2) &
                                            (veh_stats.uniqueness_per_vehicle <= 0.8)).sum()),
            "interpretation": (
                "Raw GPS vehicles: high diversity (>0.8). "
                "Map-matched vehicles: low diversity (<0.2). "
                "If two peaks visible: data mix two sources."
            ),
        }
        logger.info(f"  Vehicles by diversity: high>0.8: {results['n_vehicles_high_diversity']}, "
                    f"low<0.2: {results['n_vehicles_low_diversity']}, "
                    f"middle: {results['n_vehicles_intermediate']}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(veh_stats.uniqueness_per_vehicle, bins=40, color="#4393C3",
                 edgecolor="white", linewidth=0.3)
        ax.axvline(0.2, color="red", linestyle="--", alpha=0.5, label="map-matched")
        ax.axvline(0.8, color="green", linestyle="--", alpha=0.5, label="raw")
        ax.set_xlabel("Per-vehicle uniqueness ratio")
        ax.set_ylabel("Number of vehicles")
        ax.set_title(f"Vehicle data quality distribution (N={len(veh_stats):,})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "diag3_vehicle_diversity.png", dpi=120)
        plt.close()
    else:
        results = {"note": "Too few vehicles"}
    return results


# ============================================================
# DIAGNOSTIC 4: SAMPLING
# ============================================================

def diagnose_sampling(df: pd.DataFrame, out_dir: Path, logger) -> dict:
    logger.info("\n--- Diagnostic 4: Sampling ---")
    df_s = df.sort_values(["vehicle_id", "gps_time"])
    dt = df_s.groupby("vehicle_id")["gps_time"].diff().dt.total_seconds()
    dt = dt.dropna()
    dt = dt[(dt > 0) & (dt < 7200)]

    interval_q = {f"q{int(q*100):02d}": float(dt.quantile(q))
                  for q in [0.05, 0.25, 0.50, 0.75, 0.95]}
    interval_buckets = {
        "le_5s_pct":      float((dt <= 5).mean()) * 100,
        "5_to_15s_pct":   float(((dt > 5) & (dt <= 15)).mean()) * 100,
        "15_to_30s_pct":  float(((dt > 15) & (dt <= 30)).mean()) * 100,
        "30_to_60s_pct":  float(((dt > 30) & (dt <= 60)).mean()) * 100,
        "gt_60s_pct":     float((dt > 60).mean()) * 100,
    }
    df_h = df.copy()
    df_h["hour"] = df_h["gps_time"].dt.hour
    hourly = df_h.groupby("hour").size()

    results = {
        "interval_quantiles_s": interval_q,
        "interval_buckets_pct": interval_buckets,
        "hourly_record_counts": {int(h): int(c) for h, c in hourly.items()},
    }
    logger.info(f"  Sampling interval median: {interval_q['q50']:.0f}s")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(dt[dt <= 120], bins=60, color="#4393C3", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Sampling interval (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Inter-record gap (med={interval_q['q50']:.0f}s)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(hourly.index, hourly.values, color="#4393C3", edgecolor="white")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Number of records")
    ax.set_title("Records by hour")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "diag4_sampling.png", dpi=120)
    plt.close()
    return results


# ============================================================
# DIAGNOSTIC 5: SEGMENT INVENTORY
# ============================================================

def diagnose_segments(df: pd.DataFrame, out_dir: Path, logger) -> dict:
    logger.info("\n--- Diagnostic 5: Segment inventory ---")
    df_s = df.sort_values(["vehicle_id", "gps_time"])
    df_s["is_slow"] = df_s["speed_kmh"] <= SPEED_THRESHOLD_KMH
    df_s["prev_time"] = df_s.groupby("vehicle_id")["gps_time"].shift(1)
    df_s["dt_s"] = (df_s["gps_time"] - df_s["prev_time"]).dt.total_seconds()

    n = len(df_s)
    is_slow = df_s["is_slow"].to_numpy()
    vehicle = df_s["vehicle_id"].to_numpy()
    dt_s = df_s["dt_s"].to_numpy()
    continues = np.zeros(n, dtype=bool)
    if n > 1:
        same_v = (vehicle[1:] == vehicle[:-1])
        ps = is_slow[:-1]
        cs = is_slow[1:]
        gp = np.nan_to_num(dt_s[1:], nan=1e9) <= MAX_TIME_GAP_WITHIN_SEGMENT_S
        continues[1:] = same_v & ps & cs & gp
    new_seg = is_slow & ~continues
    seg_id = np.where(new_seg, 1, 0).cumsum()
    seg_id = np.where(is_slow, seg_id, -1)
    df_s["segment_id"] = seg_id

    segs = df_s[df_s.segment_id >= 0]
    grouped = segs.groupby("segment_id").agg(
        n_points=("lon", "size"),
        duration_s=("gps_time", lambda x: (x.max() - x.min()).total_seconds())
    )
    inventory = {
        "n_total_provisional_segments": int(len(grouped)),
        "n_segments_ge_5min": int((grouped.duration_s >= 300).sum()),
        "n_segments_ge_10min": int((grouped.duration_s >= 600).sum()),
        "n_segments_ge_30min": int((grouped.duration_s >= 1800).sum()),
        "n_segments_ge_10pts": int((grouped.n_points >= 10).sum()),
        "n_segments_ge_30pts": int((grouped.n_points >= 30).sum()),
        "n_segments_passing_basic_filter": int(((grouped.n_points >= MIN_POINTS_PER_SEGMENT) &
                                                  (grouped.duration_s >= MIN_SEGMENT_DURATION_S)).sum()),
    }
    logger.info(f"  Inventory: {inventory}")
    return inventory


# ============================================================
# MAIN
# ============================================================

def run_diagnostic(city: str, data_root: Path, out_root: Path):
    out_dir = setup_output_dir(out_root, city)
    logger, capture_file = setup_logger_and_capture(out_dir)

    print(f"========== DATA DIAGNOSTIC v2: {city} ==========")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working dir: {Path.cwd()}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {out_dir}")

    try:
        city_dir = data_root / city
        if not city_dir.exists():
            raise FileNotFoundError(f"City data folder not found: {city_dir}")

        df = load_city_data(city_dir, logger)

        summary = {
            "city": city,
            "diagnostic_version": "v2",
            "run_time": datetime.now().isoformat(),
            "data_size": {
                "n_records": int(len(df)),
                "n_vehicles": int(df.vehicle_id.nunique()),
                "time_min": str(df.gps_time.min()),
                "time_max": str(df.gps_time.max()),
            },
            "diag0_coordinate_system": detect_coordinate_system(df, city, logger),
            "diag1_spatial": diagnose_spatial_coverage(df, city, out_dir, logger),
            "diag2_map_matching": diagnose_map_matching(df, out_dir, logger),
            "diag3_vehicle_subsets": diagnose_vehicle_subsets(df, out_dir, logger),
            "diag4_sampling": diagnose_sampling(df, out_dir, logger),
            "diag5_segments": diagnose_segments(df, out_dir, logger),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print(f"FINAL SUMMARY for {city}")
        print("=" * 60)
        print(f"  Records:                  {summary['data_size']['n_records']:,}")
        print(f"  Vehicles:                 {summary['data_size']['n_vehicles']:,}")
        crs = summary["diag0_coordinate_system"]
        if "verdict" in crs:
            print(f"  Coordinate system:        {crs['verdict']}")
        s1 = summary["diag1_spatial"]
        print(f"  Distance from CBD med:    {s1['distance_from_cbd_km']['q50']:.1f} km")
        print(f"  Records in top 10 cells:  {s1['concentration']['top_10_cells_pct']:.1f}%")
        s2 = summary["diag2_map_matching"]
        sb = s2.get("signal_B_consecutive_identical", {})
        if "frac_consecutive_identical" in sb:
            print(f"  Stopped consec-identical: {100*sb['frac_consecutive_identical']:.1f}%")
        sc = s2.get("signal_C_coordinate_diversity", {})
        if "uniqueness_ratio_median" in sc:
            print(f"  Coord-uniqueness median:  {sc['uniqueness_ratio_median']:.3f}")
        print(f"  Map-matching score:       {s2['overall_map_matching_score']['value']:.3f}")
        s5 = summary["diag5_segments"]
        print(f"  Provisional segments:     {s5['n_total_provisional_segments']:,}")
        print(f"  Passing basic filter:     {s5['n_segments_passing_basic_filter']:,}")
        print("=" * 60)
        print(f"\nSUCCESS. Output saved to {out_dir}")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        print(f"\n!!! ERROR !!!: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise so batch caller can decide whether to continue
        raise
    finally:
        # Restore original stdout/stderr so batch-mode subsequent cities have clean state
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        try:
            capture_file.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(
        description="Run GPS data diagnostic for one or all cities. "
                    "Without --city, scans data-root and runs every subfolder.")
    p.add_argument("--city", default=None,
                    help="(Optional) Run only this city. "
                         "If omitted, runs ALL subfolders in city_data/.")
    p.add_argument("--data-root", default="../city_data")
    p.add_argument("--out-root", default="../output")
    args = p.parse_args()

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
        print(f"\n[BATCH MODE] Found {len(cities)} cities to process: {cities}\n")

    # Run each city; on failure of one, continue with the rest
    failed = []
    for i, city in enumerate(cities):
        print(f"\n{'#'*70}")
        print(f"# [{i+1}/{len(cities)}] Processing city: {city}")
        print(f"{'#'*70}")
        try:
            run_diagnostic(city, data_root, out_root)
        except SystemExit:
            failed.append(city)
            print(f"[BATCH] {city} failed, continuing with next city...")
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
