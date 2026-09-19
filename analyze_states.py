import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path(r".\data\network_states_10s.csv")

print("=" * 60)
print("INNOVIXUS - ATTACK STATE ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values("Timestamp").reset_index(drop=True)

# --------------------------------------------------
# BASIC INFORMATION
# --------------------------------------------------

print("\nTotal network states:", len(df))

print("\nState distribution:")
print(df["State_Label"].value_counts())

# --------------------------------------------------
# ATTACK RATIO DISTRIBUTION
# --------------------------------------------------

benign = df.loc[
    df["State_Label"] == "Benign",
    "Attack_Ratio"
]

attack = df.loc[
    df["State_Label"] == "Attack",
    "Attack_Ratio"
]

print("\n" + "-" * 60)
print("BENIGN STATE ATTACK-RATIO")
print("-" * 60)

print(benign.describe())

print("\n" + "-" * 60)
print("ATTACK STATE ATTACK-RATIO")
print("-" * 60)

print(attack.describe())

# --------------------------------------------------
# PERCENTILES
# --------------------------------------------------

print("\n" + "-" * 60)
print("IMPORTANT PERCENTILES")
print("-" * 60)

percentiles = [50, 75, 90, 95, 99]

print("\nBenign:")
for p in percentiles:
    print(
        f"{p}th percentile:",
        round(np.percentile(benign, p), 4)
    )

print("\nAttack:")
for p in percentiles:
    print(
        f"{p}th percentile:",
        round(np.percentile(attack, p), 4)
    )

# --------------------------------------------------
# THRESHOLD ANALYSIS
# --------------------------------------------------

print("\n" + "-" * 60)
print("THRESHOLD ANALYSIS")
print("-" * 60)

thresholds = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

for threshold in thresholds:

    predicted_attack = (
        df["Attack_Ratio"] >= threshold
    )

    actual_attack = (
        df["State_Label"] == "Attack"
    )

    tp = (
        predicted_attack &
        actual_attack
    ).sum()

    fp = (
        predicted_attack &
        ~actual_attack
    ).sum()

    fn = (
        ~predicted_attack &
        actual_attack
    ).sum()

    tn = (
        ~predicted_attack &
        ~actual_attack
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.4f}"
    )

# --------------------------------------------------
# ATTACK TIMELINE
# --------------------------------------------------

print("\n" + "-" * 60)
print("ATTACK TIMELINE")
print("-" * 60)

attack_states = df[
    df["State_Label"] == "Attack"
]

if len(attack_states) > 0:

    print(
        "\nFirst attack state:",
        attack_states.iloc[0]["Timestamp"]
    )

    print(
        "Last attack state:",
        attack_states.iloc[-1]["Timestamp"]
    )

    print(
        "Total attack states:",
        len(attack_states)
    )

# --------------------------------------------------
# ATTACK TRANSITIONS
# --------------------------------------------------

print("\n" + "-" * 60)
print("STATE TRANSITIONS")
print("-" * 60)

previous = df["State_Label"].shift(1)

transitions = pd.crosstab(
    previous,
    df["State_Label"]
)

print(transitions)

print("\nAnalysis complete.")