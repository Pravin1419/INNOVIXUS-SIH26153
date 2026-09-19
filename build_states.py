import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path(
    r".\data\processed_14_02.csv"
)

OUTPUT_FILE = Path(
    r".\data\network_states_10s.csv"
)

CHUNK_SIZE = 100_000
WINDOW_SECONDS = 10

# The 19 features created during preprocessing
FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Pkt Len Mean",
    "Pkt Len Std",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "ACK Flag Cnt",
    "PSH Flag Cnt",
    "URG Flag Cnt",
    "Down/Up Ratio",
]

# --------------------------------------------------
# STORAGE FOR TEMPORARY AGGREGATION
# --------------------------------------------------

print("=" * 60)
print("INNOVIXUS - NETWORK STATE BUILDER")
print("=" * 60)

print("\nReading dataset in chunks...")
print(f"Chunk size: {CHUNK_SIZE:,} flows")
print(f"Time window: {WINDOW_SECONDS} seconds")

# We store sums and counts so that if one time window
# crosses a chunk boundary, its final mean is still correct.

window_sums = {}
window_counts = {}
window_flow_counts = {}
window_attack_counts = {}

total_rows = 0

# --------------------------------------------------
# READ DATA CHUNK BY CHUNK
# --------------------------------------------------

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        usecols=["Timestamp", "Label"] + FEATURES
    ),
    start=1
):

    total_rows += len(chunk)

    print(
        f"\rProcessing chunk {chunk_number} | "
        f"flows processed: {total_rows:,}",
        end=""
    )

    # Correct timestamp format
    chunk["Timestamp"] = pd.to_datetime(
        chunk["Timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Remove invalid timestamps
    chunk = chunk.dropna(subset=["Timestamp"])

    # Create 10-second temporal window
    chunk["Window"] = chunk["Timestamp"].dt.floor(
        f"{WINDOW_SECONDS}s"
    )

    # Identify malicious flows
    chunk["IsAttack"] = (
        chunk["Label"].str.strip().str.lower() != "benign"
    ).astype(int)

    # Aggregate each chunk
    grouped = chunk.groupby("Window")

    sums = grouped[FEATURES].sum()
    counts = grouped[FEATURES].count()

    flow_counts = grouped.size()
    attack_counts = grouped["IsAttack"].sum()

    # Merge chunk results into global dictionaries
    for window in sums.index:

        if window not in window_sums:
            window_sums[window] = sums.loc[window].copy()
            window_counts[window] = counts.loc[window].copy()
            window_flow_counts[window] = int(flow_counts.loc[window])
            window_attack_counts[window] = int(attack_counts.loc[window])

        else:
            window_sums[window] += sums.loc[window]
            window_counts[window] += counts.loc[window]
            window_flow_counts[window] += int(flow_counts.loc[window])
            window_attack_counts[window] += int(attack_counts.loc[window])

print("\n\nAll flows processed!")

# --------------------------------------------------
# BUILD FINAL NETWORK STATES
# --------------------------------------------------

print("Building final temporal states...")

states = []

for window in sorted(window_sums.keys()):

    # Calculate mean feature value for this time window
    feature_means = (
        window_sums[window] /
        window_counts[window].replace(0, np.nan)
    )

    feature_means = feature_means.fillna(0)

    row = {
        "Timestamp": window,
        "Flow_Count": window_flow_counts[window],
        "Attack_Count": window_attack_counts[window],
    }

    # Add the 19 network-state features
    for feature in FEATURES:
        row[feature] = feature_means[feature]

    # Attack ratio in this time window
    if window_flow_counts[window] > 0:
        row["Attack_Ratio"] = (
            window_attack_counts[window]
            / window_flow_counts[window]
        )
    else:
        row["Attack_Ratio"] = 0.0

    # Binary state label
    row["State_Label"] = (
        "Attack"
        if window_attack_counts[window] > 0
        else "Benign"
    )

    states.append(row)

# Convert to DataFrame
states_df = pd.DataFrame(states)

# Sort chronologically
states_df = states_df.sort_values(
    "Timestamp"
).reset_index(drop=True)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

states_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("NETWORK STATE BUILDING COMPLETE")
print("=" * 60)

print(f"\nOriginal flows : {total_rows:,}")
print(f"Network states : {len(states_df):,}")
print(f"Features/state : {len(FEATURES)}")
print(f"Time window    : {WINDOW_SECONDS} seconds")

print("\nState labels:")
print(states_df["State_Label"].value_counts())

print("\nAttack ratio:")
print(states_df["Attack_Ratio"].describe())

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 states:")
print(
    states_df[
        ["Timestamp", "Flow_Count", "Attack_Count", "Attack_Ratio", "State_Label"]
    ].head()
)