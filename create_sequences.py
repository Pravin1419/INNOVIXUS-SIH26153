import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_FILE = Path(r".\data\network_states_10s.csv")
OUTPUT_FILE = Path(r".\data\lstm_sequences.npz")

SEQUENCE_LENGTH = 10

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

print("=" * 60)
print("INNOVIXUS - LSTM SEQUENCE BUILDER")
print("=" * 60)

# --------------------------------------------------
# LOAD NETWORK STATES
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values("Timestamp").reset_index(drop=True)

print("\nNetwork states loaded:", len(df))

# --------------------------------------------------
# CREATE FEATURE MATRIX
# --------------------------------------------------

X = df[FEATURES].values.astype(np.float32)

# Attack ratio is our target signal
attack_ratio = df["Attack_Ratio"].values.astype(np.float32)

# Binary attack state
attack_state = (
    df["State_Label"]
    .eq("Attack")
    .astype(np.float32)
    .values
)

# --------------------------------------------------
# BUILD SEQUENCES
# --------------------------------------------------

X_sequences = []
y_next_state = []
y_next_attack_ratio = []

for i in range(len(X) - SEQUENCE_LENGTH):

    sequence = X[i:i + SEQUENCE_LENGTH]

    next_state = X[i + SEQUENCE_LENGTH]

    next_attack_ratio = attack_ratio[
        i + SEQUENCE_LENGTH
    ]

    X_sequences.append(sequence)
    y_next_state.append(next_state)
    y_next_attack_ratio.append(next_attack_ratio)

X_sequences = np.array(X_sequences, dtype=np.float32)
y_next_state = np.array(y_next_state, dtype=np.float32)
y_next_attack_ratio = np.array(
    y_next_attack_ratio,
    dtype=np.float32
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

np.savez_compressed(
    OUTPUT_FILE,
    X=X_sequences,
    y_state=y_next_state,
    y_attack_ratio=y_next_attack_ratio,
)

print("\n" + "=" * 60)
print("SEQUENCE BUILDING COMPLETE")
print("=" * 60)

print("\nSequence shape:", X_sequences.shape)
print("Target state shape:", y_next_state.shape)
print("Attack ratio shape:", y_next_attack_ratio.shape)

print("\nSequence length:", SEQUENCE_LENGTH)
print("Features per state:", len(FEATURES))

print("\nSaved to:")
print(OUTPUT_FILE)