import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Dataset path
DATA_PATH = r".\data\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded!")
print("Original shape:", df.shape)

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"].astype(str).str.strip(),
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

print("\nTimestamp check:")
print(df["Timestamp"].head())
print("Timestamp range:", df["Timestamp"].min(), "to", df["Timestamp"].max())

# Remove invalid timestamps
df = df.dropna(subset=["Timestamp"])

# Keep only timestamps from the actual capture date
df = df[
    df["Timestamp"].dt.date ==
    pd.Timestamp("2018-02-14").date()
].copy()

print("\nTimestamp validation:")
print("Valid rows:", len(df))
print("Timestamp range:", df["Timestamp"].min(), "to", df["Timestamp"].max())

# Sort chronologically
df = df.sort_values("Timestamp").reset_index(drop=True)

# Replace infinite values
df = df.replace([np.inf, -np.inf], np.nan)

# Selected network-state features
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

# Keep only features that actually exist
available_features = [f for f in FEATURES if f in df.columns]

print("\nSelected features:")
for feature in available_features:
    print(" -", feature)

# Convert selected features to numeric
for feature in available_features:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

# Fill missing values using median
df[available_features] = df[available_features].fillna(
    df[available_features].median()
)

# Create feature matrix
X = df[available_features].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nPreprocessing complete!")
print("Feature matrix shape:", X_scaled.shape)
print("Labels:", df["Label"].value_counts().to_dict())

# Save processed data
processed = pd.DataFrame(
    X_scaled,
    columns=available_features
)

processed["Timestamp"] = df["Timestamp"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

processed["Label"] = df["Label"].values
processed.to_csv(
    r".\data\processed_14_02.csv",
    index=False
)

print("\nSaved:")
print(r".\data\processed_14_02.csv")