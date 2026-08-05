import pandas as pd
import glob
import os

LOOKUP_DIR = "D:/nightglow/class_tables"   # folder with aurora_####_#### - Sheet1.csv files
MASTER_CSV = "D:/nightglow_tables/master.csv"               # your master (orbit + scan)
PATTERN = os.path.join(LOOKUP_DIR, "aurora_* - Sheet1*.csv")  # handles " - Sheet1" and " - Sheet1 (1)" etc.

ORBIT_COL = "orbit_number"   # change if needed
TYPE_COL = "type"     # change if needed

# 1) Load + stack all lookup CSVs
files = sorted(glob.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"No files matched: {PATTERN}")

lookup_frames = []
for f in files:
    df = pd.read_csv(f)

    # Keep only orbit + type (robust to extra columns)
    if ORBIT_COL not in df.columns or TYPE_COL not in df.columns:
        raise ValueError(f"{os.path.basename(f)} missing required columns: {ORBIT_COL}, {TYPE_COL}")

    df = df[[ORBIT_COL, TYPE_COL]].copy()
    df["source_file"] = os.path.basename(f)  # optional, helps debugging
    lookup_frames.append(df)

lookup = pd.concat(lookup_frames, ignore_index=True)

# 2) Clean + dedupe to ONE type per orbit
lookup[ORBIT_COL] = pd.to_numeric(lookup[ORBIT_COL], errors="coerce")
lookup = lookup.dropna(subset=[ORBIT_COL])
lookup[ORBIT_COL] = lookup[ORBIT_COL].astype(int)

# If you want to detect conflicting types for the same orbit, fail fast:
conflicts = lookup.groupby(ORBIT_COL)[TYPE_COL].nunique()
bad_orbits = conflicts[conflicts > 1]
if not bad_orbits.empty:
    # show a small sample of the conflicting orbits
    sample = bad_orbits.index.tolist()[:15]
    raise ValueError(
        f"Conflicting '{TYPE_COL}' values found for {len(bad_orbits)} orbit(s). Sample: {sample}\n"
        f"Resolve conflicts or change the rule for which one to keep."
    )

# Otherwise, if you're sure there are no conflicts (or you want to pick one),
# you can replace the conflict check above with:
# lookup = lookup.drop_duplicates(subset=[ORBIT_COL], keep="last")

lookup = lookup.drop_duplicates(subset=[ORBIT_COL], keep="last")

# 3) Merge onto master: (orbit, scan, ...) + type
master = pd.read_csv(MASTER_CSV, dtype={"orbit_number": str})

if ORBIT_COL not in master.columns:
    raise ValueError(f"Master CSV missing '{ORBIT_COL}' column")

master[ORBIT_COL] = pd.to_numeric(master[ORBIT_COL], errors="coerce")

# left join so you keep all master rows
out = master.merge(
    lookup[[ORBIT_COL, TYPE_COL]],
    on=ORBIT_COL,
    how="left",
    suffixes=("", "_new"),
)

# If master ALREADY has a 'type' column and you want to overwrite it with the lookup:
if f"{TYPE_COL}_new" in out.columns:
    if TYPE_COL in out.columns:
        # overwrite master type when lookup type is non-null; otherwise keep master
        out[TYPE_COL] = out[f"{TYPE_COL}_new"].where(
            out[f"{TYPE_COL}_new"].notna(),
            out[TYPE_COL]
        )
        out = out.drop(columns=[f"{TYPE_COL}_new"])
    else:
        out = out.rename(columns={f"{TYPE_COL}_new": TYPE_COL})

# Drop rows where type column has no entry
out = out.dropna(subset=[TYPE_COL])

out.to_csv(MASTER_CSV, index=False)
print(f"Updated {MASTER_CSV} using {len(files)} lookup file(s).")