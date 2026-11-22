import pandas as pd

old = pd.read_csv("data/UCMF_cleaned.csv")
new = pd.read_csv("data/UCMF_cleaned_new.csv")

print(f"Existing cleaned data: {old.shape[0]} rows, {old.shape[1]} columns")
print(f"New cleaned data:      {new.shape[0]} rows, {new.shape[1]} columns")

if old.shape[0] == new.shape[0]:
    print("Same number of rows; comparing common columns by position.")
    for column in old.columns:
        if column not in new.columns:
            continue
        diff = (
            old[column].fillna("NA").astype(str)
            != new[column].fillna("NA").astype(str)
        ).sum()
        if diff:
            print(f"{column}: {diff} differences")
else:
    print("Row counts differ, so position-based row comparison is not valid.")

old_only = sorted(set(old.columns) - set(new.columns))
new_only = sorted(set(new.columns) - set(old.columns))

if old_only:
    print(f"Columns only in existing data: {old_only}")
if new_only:
    print(f"Columns only in new data: {new_only}")

print("\nColumn summary (null count):")
for column in sorted(set(old.columns) & set(new.columns)):
    old_nulls = int(old[column].isna().sum())
    new_nulls = int(new[column].isna().sum())
    print(f"{column}: old={old_nulls}, new={new_nulls}")
    

