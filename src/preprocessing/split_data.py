'''separate the rows that have the target variable from those that don't, for later use in training and testing'''
import pandas as pd

df = pd.read_csv("data/UCMF_cleaned.csv")

has_target = df["normal_x_anormal"].notna()
df[has_target].to_csv("data/UCMF_labelled.csv", index=False)
df[~has_target].to_csv("data/UCMF_unlabelled.csv", index=False)

print(f"Labelled:   {has_target.sum()} rows")
print(f"Unlabelled: {(~has_target).sum()} rows")
