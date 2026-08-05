import pandas as pd

MASTER_CSV = "D:/nightglow_tables/master.csv"
OVERRIDE_CSV = "D:/nightglow_tables/false_detections - Sheet1.csv"

# Read master, force nick_number to string
master = pd.read_csv(
    MASTER_CSV,
    dtype={"nick_number": str}
)

# Read overrides, force nick_number to string
overrides = pd.read_csv(
    OVERRIDE_CSV,
    dtype={"nick_number": str}
)

# Merge overrides onto master
out = master.merge(
    overrides,
    on="nick_number",
    how="left",
    suffixes=("", "_override")
)

# Override wins where present
out["type"] = out["type_override"].combine_first(out["type"])

# Clean up
out = out.drop(columns=["type_override"])

# Optional: track which rows were overridden
#out["type_was_overridden"] = out["nick_number"].isin(overrides["nick_number"])

out.to_csv(MASTER_CSV, index=False)

#print(f"Applied {out['type_was_overridden'].sum()} nick-level overrides.")
