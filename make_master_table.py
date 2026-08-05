import pandas as pd
import glob
import os

# -----------------------
# Paths
# -----------------------
DATA_DIR = "D:/nightglow/tables"
LOOKUP_DIR = "D:/nightglow/class_tables"

MASTER_CSV = "D:/nightglow_tables/master.csv"
OVERRIDE_CSV = "D:/nightglow_tables/false_detections - Sheet1.csv"

SHORT_PATTERN = os.path.join(DATA_DIR, "*short_1.csv")
CLASS_PATTERN = os.path.join(LOOKUP_DIR, "aurora_* - Sheet1*.csv")

ORBIT_COL = "orbit_number"
TYPE_COL = "type"
NICK_COL = "nick_number"


# -----------------------
# 1) Combine quicklook tables
# -----------------------
if os.path.exists(MASTER_CSV):
    os.remove(MASTER_CSV)

short_files = sorted(glob.glob(SHORT_PATTERN))

if not short_files:
    raise FileNotFoundError(f"No files matched: {SHORT_PATTERN}")

dfs = [pd.read_csv(f) for f in short_files]
master = pd.concat(dfs, ignore_index=True)

master.to_csv(MASTER_CSV, index=False)
print(f"Created master table from {len(short_files)} quicklook file(s).")


# -----------------------
# 2) Merge classification tables
# -----------------------
class_files = sorted(glob.glob(CLASS_PATTERN))

if not class_files:
    raise FileNotFoundError(f"No files matched: {CLASS_PATTERN}")

lookup_frames = []

for f in class_files:
    df = pd.read_csv(f)

    if ORBIT_COL not in df.columns or TYPE_COL not in df.columns:
        raise ValueError(
            f"{os.path.basename(f)} missing required columns: "
            f"{ORBIT_COL}, {TYPE_COL}"
        )

    df = df[[ORBIT_COL, TYPE_COL]].copy()
    df["source_file"] = os.path.basename(f)
    lookup_frames.append(df)

lookup = pd.concat(lookup_frames, ignore_index=True)

lookup[ORBIT_COL] = pd.to_numeric(lookup[ORBIT_COL], errors="coerce")
lookup = lookup.dropna(subset=[ORBIT_COL])
lookup[ORBIT_COL] = lookup[ORBIT_COL].astype(int)

conflicts = lookup.groupby(ORBIT_COL)[TYPE_COL].nunique()
bad_orbits = conflicts[conflicts > 1]

if not bad_orbits.empty:
    sample = bad_orbits.index.tolist()[:15]
    raise ValueError(
        f"Conflicting '{TYPE_COL}' values found for "
        f"{len(bad_orbits)} orbit(s). Sample: {sample}\n"
        f"Resolve conflicts or change the rule for which one to keep."
    )

lookup = lookup.drop_duplicates(subset=[ORBIT_COL], keep="last")

master = pd.read_csv(MASTER_CSV, dtype={NICK_COL: str})

if ORBIT_COL not in master.columns:
    raise ValueError(f"Master CSV missing '{ORBIT_COL}' column")

master[ORBIT_COL] = pd.to_numeric(master[ORBIT_COL], errors="coerce")

out = master.merge(
    lookup[[ORBIT_COL, TYPE_COL]],
    on=ORBIT_COL,
    how="left",
    suffixes=("", "_new"),
)

if f"{TYPE_COL}_new" in out.columns:
    if TYPE_COL in out.columns:
        out[TYPE_COL] = out[f"{TYPE_COL}_new"].where(
            out[f"{TYPE_COL}_new"].notna(),
            out[TYPE_COL]
        )
        out = out.drop(columns=[f"{TYPE_COL}_new"])
    else:
        out = out.rename(columns={f"{TYPE_COL}_new": TYPE_COL})

out = out.dropna(subset=[TYPE_COL])

out.to_csv(MASTER_CSV, index=False)
print(f"Merged classifications from {len(class_files)} class table(s).")


# -----------------------
# 3) Apply false-detection overrides
# -----------------------
master = pd.read_csv(MASTER_CSV, dtype={NICK_COL: str})
overrides = pd.read_csv(OVERRIDE_CSV, dtype={NICK_COL: str})

out = master.merge(
    overrides,
    on=NICK_COL,
    how="left",
    suffixes=("", "_override")
)

out[TYPE_COL] = out[f"{TYPE_COL}_override"].combine_first(out[TYPE_COL])
out = out.drop(columns=[f"{TYPE_COL}_override"])


# -----------------------
# 4) Drop duplicate nick_numbers
# -----------------------
before = len(out)

out = out.drop_duplicates(
    subset=[NICK_COL],
    keep="first"
)

after = len(out)

out.to_csv(MASTER_CSV, index=False)

print(f"Applied false-detection overrides.")
print(f"Dropped {before - after} duplicate '{NICK_COL}' row(s).")
print(f"Final master table saved to: {MASTER_CSV}")