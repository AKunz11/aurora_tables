import pandas as pd
import glob
import os

DATA_DIR = "D:/nightglow/tables"
MASTER_CSV = "D:/nightglow_tables/master.csv"

# Delete master.csv if it exists
if os.path.exists(MASTER_CSV):
    os.remove(MASTER_CSV)

files = glob.glob(os.path.join(DATA_DIR, "*short_1.csv"))

dfs = [pd.read_csv(f) for f in files]
new_data = pd.concat(dfs, ignore_index=True)

new_data.to_csv(MASTER_CSV, index=False)