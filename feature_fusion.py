import pandas as pd
import numpy as np
from pathlib import Path


FLOW_FILE = Path(r".\data\network_states_10s.csv")
PACKET_FILE = Path(r".\data\packet_features.csv")
OUTPUT_FILE = Path(r".\data\fused_network_states.csv")


FLOW_FEATURES = [
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


PACKET_FEATURES = [
    "TTL",
    "Packet_Length",
    "Payload_Size",
    "TCP_Window",
    "SYN",
    "ACK",
    "FIN",
    "RST",
    "PSH",
    "URG",
    "Fragment_Offset",
    "More_Fragments",
    "Retransmission",
]


print("=" * 60)
print("INNOVIXUS - FEATURE FUSION ENGINE")
print("=" * 60)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading flow states...")
flow_df = pd.read_csv(FLOW_FILE)

print("Flow states:", len(flow_df))

print("\nLoading packet features...")
packet_df = pd.read_csv(PACKET_FILE)

print("Packets:", len(packet_df))


# --------------------------------------------------
# TIMESTAMP CONVERSION
# --------------------------------------------------

flow_df["Timestamp"] = pd.to_datetime(
    flow_df["Timestamp"]
)

packet_df["Timestamp"] = pd.to_datetime(
    packet_df["Timestamp"]
)


# --------------------------------------------------
# CREATE 10-SECOND WINDOWS
# --------------------------------------------------

packet_df["Time_Window"] = (
    packet_df["Timestamp"]
    .dt.floor("10s")
)


# --------------------------------------------------
# AGGREGATE PACKET FEATURES
# --------------------------------------------------

packet_aggregation = {}

for feature in PACKET_FEATURES:

    if feature in [
        "SYN",
        "ACK",
        "FIN",
        "RST",
        "PSH",
        "URG",
        "More_Fragments",
        "Retransmission"
    ]:
        packet_aggregation[feature] = "sum"

    else:
        packet_aggregation[feature] = "mean"


packet_states = (
    packet_df
    .groupby("Time_Window")
    .agg(packet_aggregation)
    .reset_index()
)


packet_counts = (
    packet_df
    .groupby("Time_Window")
    .size()
    .reset_index(name="Packet_Count")
)


packet_states = packet_states.merge(
    packet_counts,
    on="Time_Window",
    how="left"
)


# --------------------------------------------------
# MATCH FLOW STATES WITH PACKET STATES
# --------------------------------------------------

flow_df["Time_Window"] = (
    flow_df["Timestamp"]
    .dt.floor("10s")
)


fused_df = flow_df.merge(
    packet_states,
    on="Time_Window",
    how="left",
    suffixes=("", "_Packet")
)


# --------------------------------------------------
# FILL MISSING PACKET DATA
# --------------------------------------------------

for feature in PACKET_FEATURES + ["Packet_Count"]:

    if feature in fused_df.columns:
        fused_df[feature] = (
            fused_df[feature]
            .fillna(0)
        )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

fused_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE FUSION COMPLETE")
print("=" * 60)

print("\nFused states :", len(fused_df))

print(
    "Flow features :",
    len(FLOW_FEATURES)
)

print(
    "Packet features :",
    len(PACKET_FEATURES)
)

print(
    "Total columns :",
    len(fused_df.columns)
)

print("\nPacket-derived features added:")

for feature in PACKET_FEATURES:
    print(" -", feature)

print(" - Packet_Count")

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst rows:")

print(
    fused_df[
        [
            "Timestamp",
            "Time_Window",
            "TTL",
            "Packet_Length",
            "Payload_Size",
            "TCP_Window",
            "Retransmission",
            "Packet_Count"
        ]
    ].head(10).to_string(index=False)
)