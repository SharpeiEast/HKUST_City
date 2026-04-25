# PoC: Empirical Positioning Uncertainty Extraction from Taxi GPS

## Purpose

This PoC pipeline extracts empirical positioning uncertainty (`sigma_pos`)
from one-day taxi GPS trajectory data. It identifies parked taxis,
computes their location jitter, and aggregates results spatially and
temporally. Output is fully anonymized — no individual vehicle identifiers
or trajectories are exposed.

## Folder Layout

```
project_root/
├── coding/                      ← THIS folder. Contains the script.
│   ├── poc_main.py
│   └── README.md  (this file)
├── city_data/                   ← Input. One subfolder per city.
│   ├── beijing/
│   │   ├── *.parquet            (any number of parquet files)
│   ├── shanghai/
│   │   ├── *.parquet
│   └── shenzhen/
│       └── *.parquet
└── output/                      ← Output written here. Send back to us.
    ├── poc_beijing/
    ├── poc_shanghai/
    └── poc_shenzhen/
```

## Expected Input Schema

Each parquet file should have at minimum these columns (string/float types as in
the sample data):

| column      | type    | description                                  |
|-------------|---------|----------------------------------------------|
| vehicle_id  | string  | unique vehicle identifier                    |
| lon         | float   | longitude (decimal degrees, WGS84 or GCJ-02) |
| lat         | float   | latitude                                     |
| speed_kmh   | float   | instantaneous speed in km/h                  |
| gps_time    | datetime/string | GPS timestamp (parseable)            |

Optional columns (used if present, ignored if not):
`heading`, `status`, `recv_time`, `veh_type`, `admin_code`, `road_name`,
`road_class`.

## Environment

Python 3.9+ with:
```
pandas >= 1.5
numpy >= 1.20
pyarrow >= 10.0
matplotlib >= 3.5
```

Install:
```
pip install pandas numpy pyarrow matplotlib
```

No GPU, no internet, no other system dependencies required.

## Running

From inside `coding/`:
```
python poc_main.py --city beijing
python poc_main.py --city shanghai
python poc_main.py --city shenzhen
```

Default paths:
- input  = `../city_data/<city>/`
- output = `../output/poc_<city>/`

If your folder layout is different, override:
```
python poc_main.py --city beijing \
    --data-root /absolute/path/to/city_data \
    --out-root /absolute/path/to/output
```

## Expected Runtime

For a typical city × 1 day (≈10–50 million records, after our other tests):
- Loading data: 1–5 minutes
- Stationary segment detection: 2–10 minutes
- sigma_pos computation: 1–5 minutes  (segment-by-segment loop)
- Aggregation + figures: <1 minute
- **Total: 5–20 minutes per city**

Memory: peaks around 2–5× the size of input parquet files.
For very large cities (Beijing/Shanghai), expect peak ~16–32 GB RAM.

## Output Contents (what to send back)

For each city, `output/poc_<city>/` contains:

```
poc_<city>/
├── run.log                              ← Full execution log
├── stats/
│   ├── data_quality.json                ← Input data diagnostics
│   ├── sigma_pos_quantiles.json         ← Distribution summary
│   └── temporal_pattern.csv             ← Hourly aggregation
├── intermediate/                        ← Anonymized aggregated data
│   ├── segments.parquet                 ← Per-segment summary (vehicle_id hashed)
│   ├── grid_50m.parquet                 ← 50m grid aggregation
│   ├── grid_100m.parquet                ← 100m grid aggregation (main analysis)
│   └── grid_250m.parquet                ← 250m grid aggregation (sensitivity)
└── figures/
    ├── fig1_sigma_pos_distribution.png
    ├── fig2_spatial_heatmap_100m.png
    ├── fig3_temporal_pattern.png
    ├── fig4_segment_quality.png
    └── fig5_robust_vs_std.png
```

**The entire `output/` folder is what we need back.** No raw GPS records,
no individual trajectories, no original vehicle IDs are included.

## Privacy Notes

1. `vehicle_id` is hashed (8-digit hash) in `segments.parquet`.
2. Only segments are exported, not individual GPS points.
3. Spatial grids aggregate across segments — no single-vehicle trajectory
   can be reconstructed from the output.

## Quick Sanity Check

After running, please verify in `run.log` that:
- "Records: ###,###" matches your expectation for the city
- "Stationary segments (valid): ###,###" is at least a few hundred
- "sigma_pos median" is in a reasonable range (0.5–10 m)

If "Stationary segments (valid)" is 0 or very small (<100), there may
be a data format issue — please check `data_quality.json` and contact us.

## Configuration Constants (in `poc_main.py`)

If results look odd, these are the main thresholds to inspect:

```python
SPEED_THRESHOLD_KMH = 2.0          # what counts as "stopped"
MIN_SEGMENT_DURATION_S = 300       # at least 5 minutes
MAX_SEGMENT_DISPLACEMENT_M = 30    # reject moving-but-slow segments
MIN_POINTS_PER_SEGMENT = 10        # need enough points for sigma estimate
```

These should NOT be changed for the production run without consulting us,
to keep results comparable across cities.

---
Contact: [your email]
