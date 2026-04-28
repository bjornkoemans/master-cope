"""
Step 2: Convert raw BPI Challenge 2012 CSV to activity log.

Matches START and COMPLETE lifecycle events into activity instances
with start_timestamp and end_timestamp. Resource IDs are mapped to
two-letter codes (AA, AB, ...) for readability.

Usage:
    python csv_to_activity_log.py

Input:  intermediate/bpic_2012_raw.csv
Output: processed/bpi_challenge_2012.csv
"""
import pandas as pd
import string

INPUT_PATH = "intermediate/bpic_2012_raw.csv"
OUTPUT_PATH = "processed/bpi_challenge_2012.csv"


def generate_letter_codes(n):
    """Generate AA, AB, ..., AZ, BA, BB, ... letter codes."""
    letters = string.ascii_uppercase
    codes = []
    i = 0
    while len(codes) < n:
        codes.append(letters[i // 26] + letters[i % 26])
        i += 1
    return codes


# ── Load input ─────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

# Normalize lifecycle:transition to lowercase
df["lifecycle:transition"] = df["lifecycle:transition"].str.lower()

# Drop entire cases that contain missing resources
valid_cases = (
    df.groupby("case_id")["resource"]
    .apply(lambda x: x.notna().all())
)
df = df[df["case_id"].isin(valid_cases[valid_cases].index)]

print(f"Loaded {len(df)} events from {df['case_id'].nunique()} cases")
print(f"Lifecycle transitions: {df['lifecycle:transition'].value_counts().to_dict()}")

# ── Build activity instances (match start -> complete) ─────────
rows = []
counter = 0

for case_id, case_df in df.groupby("case_id"):
    case_df = case_df.sort_values("timestamp")
    open_acts = {}

    for _, r in case_df.iterrows():
        activity = r["activity"]
        resource = r["resource"]
        key = (activity, resource if pd.notna(resource) else "__unassigned__")

        if r["lifecycle:transition"] == "start":
            open_acts.setdefault(key, []).append(r)

        elif r["lifecycle:transition"] == "complete":
            if key in open_acts and open_acts[key]:
                start = open_acts[key].pop(0)["timestamp"]
            else:
                # Fallback: match by activity only
                alt_key = None
                for k in open_acts:
                    if k[0] == activity and open_acts[k]:
                        alt_key = k
                        break
                if alt_key and open_acts[alt_key]:
                    start = open_acts[alt_key].pop(0)["timestamp"]
                else:
                    start = r["timestamp"]

            rows.append({
                "elementId": f"{case_id}_{activity}_{counter}",
                "processId": "bpi_challenge_2012",
                "activity_name": activity,
                "lifecycle:transition": "COMPLETE",
                "start_timestamp": start,
                "end_timestamp": r["timestamp"],
                "case_id": case_id,
                "resource": resource,
            })
            counter += 1
        # Skip "assign" and other lifecycle events

# ── Create output ──────────────────────────────────────────────
out = pd.DataFrame(rows)

# Map numeric resource IDs to letter codes for readability
unique_resources = sorted(out["resource"].unique())
letter_codes = generate_letter_codes(len(unique_resources))
resource_to_code = dict(zip(unique_resources, letter_codes))
out["resource"] = out["resource"].map(resource_to_code)

# Agent IDs (0-indexed)
out["agent"] = out["resource"].astype("category").cat.codes

out.to_csv(OUTPUT_PATH, index=False)

# ── Summary ────────────────────────────────────────────────────
print(f"\nWritten {len(out)} activity instances to {OUTPUT_PATH}")
print(f"Cases: {out['case_id'].nunique()}")
print(f"Unique agents: {out['resource'].nunique()}")
print(f"Unique activities: {out['activity_name'].nunique()}")
print(f"Date range: {out['start_timestamp'].min()} to {out['end_timestamp'].max()}")

# Cross-day check
out["start_timestamp"] = pd.to_datetime(out["start_timestamp"])
out["end_timestamp"] = pd.to_datetime(out["end_timestamp"])
cross_day = out[out["start_timestamp"].dt.date != out["end_timestamp"].dt.date]
print(f"\nCross-day tasks: {len(cross_day)} ({len(cross_day)/len(out)*100:.2f}%)")
if len(cross_day) > 0:
    print("Activities with cross-day tasks:")
    print(cross_day["activity_name"].value_counts().to_string())
