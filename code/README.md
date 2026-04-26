# HKUST GPS Analysis Package

## Folder Structure

```
project_root/
├── coding/                    ← code goes here
│   ├── poc_main.py
│   ├── data_diagnostic.py
│   └── README.md (this file)
├── city_data/                 ← input data
│   ├── beijing/*.parquet
│   ├── shanghai/*.parquet
│   ├── guangzhou/*.parquet
│   └── zhengzhou/*.parquet    ← please add this if accessible
└── output/                    ← results, send back to us
                                  (auto-created, no need to make it manually)
```

## Setup

```bash
pip install pandas numpy pyarrow matplotlib
```

That's all. No internet, no GPU.

## How to Run

From inside `coding/`:

```bash
# 1. Run the diagnostic on all cities
python data_diagnostic.py

# 2. Run the PoC on all cities
python poc_main.py
```

That's it. Both scripts:
- Auto-discover all city subfolders in `city_data/`
- Process them one by one
- Continue with the next city if one fails
- Auto-create `output/` if it doesn't exist
- Auto-cleanup the relevant city subfolder before re-running

You don't need to manually create or delete any folders.

## Output Layout

After running both scripts, you'll have:

```
output/
├── diag_beijing/           ← from data_diagnostic.py
├── diag_shanghai/
├── diag_guangzhou/
├── diag_zhengzhou/         ← if zhengzhou data was provided
├── poc_beijing/            ← from poc_main.py
├── poc_shanghai/
├── poc_guangzhou/
└── poc_zhengzhou/
```

**Send back the entire `output/` folder.**

## What Each Script Touches (Important)

| Script | What it deletes | What it creates |
|--------|----------------|-----------------|
| `data_diagnostic.py --city beijing` | Only `output/diag_beijing/` | `output/diag_beijing/` |
| `data_diagnostic.py` (batch) | Each `output/diag_<city>/` (one at a time) | All `output/diag_<city>/` |
| `poc_main.py --city beijing` | Only `output/poc_beijing/` | `output/poc_beijing/` |
| `poc_main.py` (batch) | Each `output/poc_<city>/` (one at a time) | All `output/poc_<city>/` |

**`diag_*` and `poc_*` folders never interfere with each other.** Running
`poc_main.py` after `data_diagnostic.py` does NOT delete the diagnostic
output. Running them in any order is safe.

## Single-City Mode (Optional)

If you want to debug or re-run only one city:

```bash
python data_diagnostic.py --city beijing
python poc_main.py --city beijing
```

## Special Request: Zhengzhou as Control Case

We previously analyzed Zhengzhou data (admin_code 411002) locally and it
appeared to preserve raw GPS jitter, very different from the
Beijing/Shanghai/Guangzhou data you ran earlier. Most stationary segments
in those three cities had zero coordinate jitter, suggesting map-matching
has been applied somewhere in the pipeline.

To definitively distinguish "data is processed differently per city" from
"Zhengzhou is just an outlier", we'd appreciate if you could:

1. Add Zhengzhou data (any one day in 2019) to `city_data/zhengzhou/`
2. Re-run both scripts (no extra command needed; batch mode picks it up)

If Zhengzhou results differ significantly from the other three cities,
we'll know the difference is real and not random.

## Privacy

All output is aggregated. No raw GPS coordinates, no individual
trajectories, no original vehicle IDs leave the HKUST environment.
Vehicle IDs in PoC segments are hashed (8-digit irreversible).

## If Something Goes Wrong

The diagnostic captures full terminal output to
`output/diag_<city>/terminal_output.txt`. If a script crashes:

1. Check `output/diag_<city>/terminal_output.txt`
2. Check `output/diag_<city>/diagnostic.log`
3. Send both files to us

Common errors:

- `No city subfolders with parquet files found`: `city_data/` is empty or
  the parquet files are not under city subfolders.
- `MemoryError`: data file too large; tell us, we'll discuss splitting.

---

Contact: [Your email]
