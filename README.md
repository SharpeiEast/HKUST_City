# HKUST GPS Analysis Package

Two scripts in this folder. Both support batch mode (no `--city` argument).

## Folder Structure

```
project_root/
├── coding/
│   ├── poc_main.py            ← σ_pos extraction (updated to support batch)
│   ├── data_diagnostic.py     ← data quality diagnostic
│   └── README.md
├── city_data/
│   ├── beijing/*.parquet
│   ├── shanghai/*.parquet
│   ├── guangzhou/*.parquet
│   └── zhengzhou/*.parquet    ← please add this if possible
└── output/                    ← results land here, send back to us
    ├── poc_<city>/            ← from poc_main.py
    └── diag_<city>/           ← from data_diagnostic.py
```

## How to Run (recommended: batch mode)

From inside `coding/`:

```bash
pip install pandas numpy pyarrow matplotlib

# Run the diagnostic on all cities
python data_diagnostic.py

# Run the σ_pos PoC on all cities
python poc_main.py
```

That's it. Both scripts auto-detect every subfolder in `city_data/` and process them in sequence. If one city fails, the script continues with the next.

## Single-city mode (optional, for debugging)

```bash
python data_diagnostic.py --city beijing
python poc_main.py --city beijing
```

## What's New

- **Batch mode**: no `--city` argument needed; all cities processed automatically.
- **`data_diagnostic.py`**: new script that detects map-matching artifacts, coordinate system, and spatial coverage. Auto-cleans its own output folder before each city. Captures full terminal output to `terminal_output.txt`.
- **`poc_main.py`**: same as before (extract σ_pos from stationary segments), now also supports batch mode.

## Special Request: Zhengzhou as Control Case

We've previously analyzed Zhengzhou data (admin code 411002) and it appeared to preserve raw GPS jitter — much different from the Beijing/Shanghai/Guangzhou results. To confirm this difference is real, please add Zhengzhou data to `city_data/zhengzhou/` if accessible. Both scripts will then automatically include it.

## Output

For each city, you'll get two folders:
- `output/poc_<city>/` — σ_pos analysis (segments, grids, figures)
- `output/diag_<city>/` — data quality diagnostic (5 diagnostics, includes coordinate system check)

The terminal output is also saved to `output/diag_<city>/terminal_output.txt`. Please send back the entire `output/` folder.

## Privacy

All output is aggregated. No raw GPS records, no individual trajectories, no original vehicle IDs. Vehicle IDs are hashed in `poc_main.py`'s segment outputs.

## Common Errors

- `No city subfolders with parquet files found`: `city_data/` is empty or wrong location.
- `MemoryError`: data file too large; contact us.

If anything fails, please send us:
1. The error message from the terminal
2. `output/diag_<city>/terminal_output.txt` (if it was created)
3. `output/diag_<city>/diagnostic.log` (if it was created)
