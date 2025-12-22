import pandas as pd
import string

# helper: generate AA, AB, ..., AZ, BA, BB, ...
def generate_letter_codes(n):
    letters = string.ascii_uppercase
    codes = []
    i = 0
    while len(codes) < n:
        codes.append(letters[i // 26] + letters[i % 26])
        i += 1
    return codes


# load input
df = pd.read_csv("BPI_Challenge_2012.csv")

# parse timestamps
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
                "resource": r["resource"],
                "resourceCost": None
            })
            counter += 1

# create dataframe
out = pd.DataFrame(rows)

# create deterministic resource → letter-code mapping
unique_resources = sorted(out["resource"].unique())
letter_codes = generate_letter_codes(len(unique_resources))
resource_to_code = dict(zip(unique_resources, letter_codes))

out["resource"] = out["resource"].map(resource_to_code)

# create agent ids (0..n)
out["agent"] = out["resource"].astype("category").cat.codes

# write output
output_path = "activity_log.csv"
out.to_csv(output_path, index=False)

print(f"Written {len(out)} activity instances to {output_path}")
print(f"Remaining cases: {out['case_id'].nunique()}")
print(f"Unique agents: {out['agent'].nunique()}")
