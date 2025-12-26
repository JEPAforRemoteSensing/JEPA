import pandas as pd
import os

ben_serbia = pd.read_parquet('BEN_full/metadata.parquet')

ben_serbia = ben_serbia[ben_serbia['country'] == 'Serbia']

ben_14k = ben_serbia[ben_serbia['patch_id'].str[11:19].astype(int) <= 20170831]

s1_names = set(ben_14k['s1_name'])
s2_names = set(ben_14k['patch_id'])

ben_full_s1_path = "BEN_full/BigEarthNet-S1"
ben_full_s2_path = "BEN_full/BigEarthNet-S2"

deleted_count = 0
total_count = 0
for folder in os.scandir(ben_full_s1_path):
    for subfolder in os.scandir(folder):
        total_count += 1
        if subfolder.name not in s1_names:
            deleted_count += 1

print(f"\nTotal: {total_count}")
print(f"Remaining: {total_count - deleted_count}")


# Dates from 20170612 to 20170831 are summer (yyyymmdd)
