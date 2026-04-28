"""
Step 1: Convert CVS Pharmacy XES to raw CSV.

Usage:
    python xes_to_csv.py

Input:  raw/cvs_pharmacy.xes
Output: intermediate/NEW_cvs_pharmacy_raw.csv
"""
from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd

xes_path = "data/cvs_pharmacy/raw/cvs_pharmacy.xes"
csv_path = "data/cvs_pharmacy/intermediate/cvs_pharmacy_raw.csv"

print(f"Loading XES from {xes_path}...")
log = xes_importer.apply(xes_path)

df = pd.DataFrame([
    {
        "case_id": trace.attributes.get("concept:name"),
        "activity": event.get("concept:name"),
        "resource": event.get("org:resource"),
        "timestamp": event.get("time:timestamp"),
        **{k: v for k, v in event.items()
           if k not in ["concept:name", "org:resource", "time:timestamp"]}
    }
    for trace in log
    for event in trace
])

df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} events to {csv_path}")
print(f"Cases: {df['case_id'].nunique()}")
print(f"Lifecycle transitions: {df['lifecycle:transition'].value_counts().to_dict()}")
