import pandas as pd

# load input
df = pd.read_csv("BPI_Challenge_2012.csv")

# parse timestamps (mixed precision, timezone-aware)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)

# drop entire cases that contain missing resources
valid_cases = (
    df.groupby("case_id")["resource"]
      .apply(lambda x: x.notna().all())
)

df = df[df["case_id"].isin(valid_cases[valid_cases].index)]

rows = []
counter = 0

# build activity instances
for case_id, case_df in df.groupby("case_id"):
    case_df = case_df.sort_values("timestamp")
    open_acts = {}

    for _, r in case_df.iterrows():
        key = (r["activity"], r["resource"])

        if r["lifecycle:transition"] == "START":
            open_acts.setdefault(key, []).append(r)

        elif r["lifecycle:transition"] == "COMPLETE":
            if key in open_acts and open_acts[key]:
                start = open_acts[key].pop(0)["timestamp"]
            else:
                start = r["timestamp"]

            rows.append({
                "Unnamed: 0": counter,
                "elementId": f"{case_id}_{r['activity']}_{counter}",
                "processId": "loan_process",
                "activity_name": r["activity"],
                "lifecycle:transition": "COMPLETE",
                "start_timestamp": start,
                "end_timestamp": r["timestamp"],
                "case_id": case_id,
                "resourceId": r["resource"],
                "resource": r["resource"],
                "resourceCost": None
            })
            counter += 1

# create dataframe
out = pd.DataFrame(rows)

# create agent ids (0..n), deterministic
unique_resources = sorted(out["resource"].unique())
resource_to_agent = {res: i for i, res in enumerate(unique_resources)}
out["agent"] = out["resource"].map(resource_to_agent).astype(int)

# write output
output_path = "activity_log.csv"
out.to_csv(output_path, index=False)

print(f"Written {len(out)} activity instances to {output_path}")
print(f"Remaining cases: {out['case_id'].nunique()}")
