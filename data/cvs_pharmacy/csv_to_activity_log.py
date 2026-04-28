"""
Step 2: Convert raw CVS Pharmacy CSV to activity log.

Groups assign/start/complete lifecycle events by (case_id, elementId)
into single activity instances with assign_timestamp, start_timestamp,
and end_timestamp.

Usage:
    python csv_to_activity_log.py

Input:  intermediate/cvs_pharmacy_raw.csv
Output: processed/cvs_pharmacy.csv
"""
import pandas as pd

INPUT_PATH = "data/cvs_pharmacy/intermediate/cvs_pharmacy_raw.csv"
OUTPUT_PATH = "data/cvs_pharmacy/processed/cvs_pharmacy.csv"

# ── Load input ─────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

# Normalize lifecycle:transition to lowercase
df["lifecycle:transition"] = df["lifecycle:transition"].str.lower()

print(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases")
print(f"Lifecycle transitions: {df['lifecycle:transition'].value_counts().to_dict()}")

# ── Build activity instances using elementId matching ──────────
# Each activity instance has a unique elementId shared across
# assign, start, and complete events.
rows = []
counter = 0

for (case_id, element_id), group in df.groupby(["case_id", "elementId"]):
    group = group.sort_values("timestamp")

    # Split by lifecycle type
    assign_events = group[group["lifecycle:transition"] == "assign"]
    start_events = group[group["lifecycle:transition"] == "start"]
    complete_events = group[group["lifecycle:transition"] == "complete"]

    # Skip if no complete event (incomplete activity)
    if complete_events.empty:
        continue

    # Extract timestamps (take first of each type)
    assign_ts = assign_events.iloc[0]["timestamp"] if not assign_events.empty else None
    start_ts = start_events.iloc[0]["timestamp"] if not start_events.empty else None
    complete_ts = complete_events.iloc[0]["timestamp"]

    # Get activity name and resource from the complete event (most reliable)
    complete_row = complete_events.iloc[0]
    activity = complete_row["activity"]
    resource = complete_row["resource"] if pd.notna(complete_row["resource"]) else None

    # Fallback: if no start, use complete timestamp
    if start_ts is None:
        start_ts = complete_ts

    # Fallback: if no assign, use start timestamp
    if assign_ts is None:
        assign_ts = start_ts

    rows.append({
        "elementId": element_id,
        "processId": "cvs_pharmacy",
        "activity_name": activity,
        "lifecycle:transition": "COMPLETE",
        "assign_timestamp": assign_ts,
        "start_timestamp": start_ts,
        "end_timestamp": complete_ts,
        "case_id": case_id,
        "resource": resource,
    })
    counter += 1

# ── Create output ──────────────────────────────────────────────
out = pd.DataFrame(rows)

# Remove gateway/system activities without a resource (e.g. "Prescription received",
# "Prescription fulfilled") — these cannot be assigned to agents
n_before = len(out)
out = out[out["resource"].notna()].reset_index(drop=True)
n_dropped = n_before - len(out)
if n_dropped > 0:
    print(f"Removed {n_dropped} gateway/system activities without resource")

# Sort by case_id and assign_timestamp (chronological order)
out = out.sort_values(["case_id", "assign_timestamp"]).reset_index(drop=True)

# Agent IDs (0-indexed)
out["agent"] = out["resource"].astype("category").cat.codes

out.to_csv(OUTPUT_PATH, index=False)

# ── Summary ────────────────────────────────────────────────────
named_resources = out["resource"].dropna().unique()
nan_count = out["resource"].isna().sum()
print(f"\nWritten {len(out)} activity instances to {OUTPUT_PATH}")
print(f"Cases: {out['case_id'].nunique()}")
print(f"Unique agents: {len(named_resources)} — {sorted(named_resources)}")
if nan_count > 0:
    print(f"  (+ {nan_count} events without resource, e.g. gateway activities)")
print(f"Unique activities: {out['activity_name'].nunique()}")
print(f"Date range: {out['assign_timestamp'].min()} to {out['end_timestamp'].max()}")

# ── Queue time analysis ────────────────────────────────────────
out["assign_timestamp"] = pd.to_datetime(out["assign_timestamp"])
out["start_timestamp"] = pd.to_datetime(out["start_timestamp"])
out["end_timestamp"] = pd.to_datetime(out["end_timestamp"])

queue_time = (out["start_timestamp"] - out["assign_timestamp"]).dt.total_seconds() / 3600
print(f"\nQueue time (assign → start):")
print(f"  Zero queue:  {(queue_time == 0).sum()} ({(queue_time == 0).sum()/len(out)*100:.1f}%)")
print(f"  > 0 hours:   {(queue_time > 0).sum()} ({(queue_time > 0).sum()/len(out)*100:.1f}%)")
print(f"  Median:      {queue_time.median():.2f}h")
print(f"  Max:         {queue_time.max():.1f}h ({queue_time.max()/24:.1f} days)")

# Cross-day check (assign to start)
cross_day = out[out["assign_timestamp"].dt.date != out["start_timestamp"].dt.date]
print(f"\nCross-day tasks (assign on different day than start): {len(cross_day)} ({len(cross_day)/len(out)*100:.2f}%)")
if len(cross_day) > 0:
    print("Activities with cross-day queue:")
    print(cross_day["activity_name"].value_counts().to_string())
