# Nightglow Tables

Utilities for assembling MAVEN/IUVS aurora quicklook results into a single,
classified master table.

The main workflow combines scan-level CSV files produced by
`periapse_quicklooks`, adds orbit-level classifications from manual review,
applies scan-level false-detection corrections, and removes duplicate scan
identifiers.

## Master-table workflow

`make_master_table.py` performs the complete workflow:

1. Find and concatenate the quicklook CSV files.
2. Merge the manually reviewed classification tables by `orbit_number`.
3. Apply false-detection overrides by `nick_number`.
4. Keep the first row for each unique `nick_number`.
5. Write the completed master CSV.

Rows without a classification are excluded. The script also stops if the
classification files assign more than one `type` to the same orbit.

### Inputs

The paths and filename patterns are configured at the top of
`make_master_table.py`.

| Input | Default pattern or path | Required columns |
| --- | --- | --- |
| Quicklook tables | `E:/nightglow/tables_cnn/*short_cnn.csv` | `orbit_number`, `nick_number`, plus the desired scan data |
| Classification tables | `E:/nightglow/class_tables/aurora_* - Sheet1*.csv` | `orbit_number`, `type` |
| False-detection overrides | `D:/nightglow_tables/false_detections - Sheet1.csv` | `nick_number`, `type` |

`nick_number` is the scan identifier. It is read as text so identifiers such
as `8001.01` retain their formatting.

Example input structures are provided in:

- `example_ql_table.csv`
- `example_class.csv`
- `example_false.csv`

### Output

By default, the completed table is written to:

```text
E:/nightglow/master_cnn.csv
```

The output path is controlled by `MASTER_CSV` in `make_master_table.py`.
An existing file at that path is replaced when the script starts.

## Setup and usage

Python 3 with pandas is sufficient for the master-table workflow. To install
all dependencies used by the repository's additional utilities:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Before running the pipeline, edit `DATA_DIR`, `LOOKUP_DIR`, `MASTER_CSV`, and
`OVERRIDE_CSV` near the top of `make_master_table.py` to match your local
directory layout. Then run:

```powershell
python make_master_table.py
```

## Scripts

| Script | Purpose |
| --- | --- |
| `make_master_table.py` | Runs the complete master-table workflow. This is the recommended entry point. |
| `combine_ql_tables.py` | Legacy standalone version of the quicklook-table concatenation step. |
| `merge_class_tables.py` | Legacy standalone version of the classification merge step. |
| `drop_false.py` | Legacy standalone version of the false-detection override step. |
| `all_detections.py` | Builds `all_scans.csv` directly from L1C FITS files. It processes every available scan and may take hours. |

The three legacy scripts use their own hard-coded paths. If they are run
individually, use them in this order:

1. `combine_ql_tables.py`
2. `merge_class_tables.py`
3. `drop_false.py`


## Data provenance

The quicklook CSVs consumed by this project are generated outside this
repository by `periapse_quicklooks`. The master table is then used by other
aurora-analysis projects, including `aurora_tools` and `cosmic_ray_CNN`.

Generated tables, local virtual environments, and the local `PyUVS` copy are
excluded through `.gitignore` because they are either large, machine-specific,
or reproducible.
