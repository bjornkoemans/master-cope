"""
Step 1: Convert BPI Challenge 2012 XES to raw CSV.

Usage:
    python xes_to_csv.py

Input:  raw/BPI_Challenge_2012.xes
Output: intermediate/bpic_2012_raw.csv
"""
import pm4py
import pandas as pd

xes_path = "raw/BPI_Challenge_2012.xes"
csv_path = "intermediate/bpic_2012_raw.csv"

print(f"Loading XES from {xes_path}...")
log = pm4py.read_xes(xes_path)
df = pm4py.convert_to_dataframe(log)

# BPIC 2012 specific: only keep W_ events (work items)
df = df[df["concept:name"].str.startswith("W")]

# Rename columns for consistency with csv_to_activity_log.py
df = df.rename(columns={
    "case:concept:name": "case_id",
    "concept:name": "activity",
    "org:resource": "resource",
    "time:timestamp": "timestamp",
})

df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} events to {csv_path}")
print(f"Cases: {df['case_id'].nunique()}")
print(f"Lifecycle transitions: {df['lifecycle:transition'].value_counts().to_dict()}")
