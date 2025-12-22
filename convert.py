from pm4py.objects.log.importer.xes import importer as xes_importer
import pandas as pd

# Input and output paths
xes_path = "BPI_Challenge_2012.xes"
csv_path = "BPI_Challenge_2012.csv"

# Load XES log
log = xes_importer.apply(xes_path)

# Convert to DataFrame
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

# Save to CSV
df.to_csv(csv_path, index=False)

print(f"Saved CSV to {csv_path}")
