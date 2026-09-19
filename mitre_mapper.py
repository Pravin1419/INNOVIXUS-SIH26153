import pandas as pd
from pathlib import Path


INPUT_FILE = Path(r".\data\infiltration_predictions.csv")
OUTPUT_FILE = Path(r".\data\mitre_predictions.csv")


# --------------------------------------------------
# CIC-IDS2018 - 14 FEBRUARY 2018 ATTACK TIMELINE
# --------------------------------------------------

FTP_START = pd.Timestamp("2018-02-14 10:32:00")
FTP_END = pd.Timestamp("2018-02-14 12:09:00")

SSH_START = pd.Timestamp("2018-02-14 14:01:00")
SSH_END = pd.Timestamp("2018-02-14 15:31:00")


# --------------------------------------------------
# MITRE ATT&CK MAPPING
# --------------------------------------------------

def map_attack_scenario(timestamp, probability):

    # FTP Brute Force
    if FTP_START <= timestamp <= FTP_END:

        return (
            "FTP-BruteForce",
            "T1110 - Brute Force",
            "Credential Access",
            "Credential Access",
            "Prediction occurs within the documented FTP brute-force attack window."
        )

    # SSH Brute Force
    elif SSH_START <= timestamp <= SSH_END:

        return (
            "SSH-Bruteforce",
            "T1110 - Brute Force",
            "Credential Access",
            "Credential Access",
            "Prediction occurs within the documented SSH brute-force attack window."
        )

    # High predicted risk outside documented attack window
    elif probability >= 0.80:

        return (
            "Unknown / Unclassified",
            "Not assigned",
            "Suspicious Activity",
            "Requires Investigation",
            "High predicted attack probability, but no documented attack scenario overlaps this timestamp."
        )

    # Moderate risk
    elif probability >= 0.50:

        return (
            "Unknown / Unclassified",
            "Not assigned",
            "Suspicious Activity",
            "Requires Investigation",
            "Moderate predicted attack probability; additional evidence is required."
        )

    # Low risk
    else:

        return (
            "None",
            "None",
            "None",
            "Monitoring",
            "No strong malicious progression signal."
        )


# --------------------------------------------------
# LOAD PREDICTIONS
# --------------------------------------------------

print("=" * 65)
print("INNOVIXUS - MITRE ATT&CK TIMELINE MAPPER")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

print("\nPredictions loaded:", len(df))


# --------------------------------------------------
# APPLY TIMELINE MAPPING
# --------------------------------------------------

mapped_results = df.apply(
    lambda row: map_attack_scenario(
        row["Timestamp"],
        row["Infiltration_Probability"]
    ),
    axis=1
)


mapped_df = pd.DataFrame(
    mapped_results.tolist(),
    columns=[
        "Attack_Scenario",
        "MITRE_Technique",
        "MITRE_Tactic",
        "MITRE_Stage",
        "Stage_Reason"
    ],
    index=df.index
)


df = pd.concat(
    [df, mapped_df],
    axis=1
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("MITRE MAPPING RESULTS")
print("=" * 65)

print("\nAttack scenario distribution:")

print(
    df["Attack_Scenario"].value_counts()
)


print("\nMITRE technique distribution:")

print(
    df["MITRE_Technique"].value_counts()
)


print("\nMITRE tactic distribution:")

print(
    df["MITRE_Tactic"].value_counts()
)


print("\nMITRE stage distribution:")

print(
    df["MITRE_Stage"].value_counts()
)


# --------------------------------------------------
# SHOW ATTACK PREDICTIONS
# --------------------------------------------------

attack_rows = df[
    df["Attack_Scenario"].isin(
        [
            "FTP-BruteForce",
            "SSH-Bruteforce"
        ]
    )
]


print("\nDocumented attack-window predictions:")

if len(attack_rows) > 0:

    print(
        attack_rows[
            [
                "Timestamp",
                "Infiltration_Probability",
                "Attack_Scenario",
                "MITRE_Technique",
                "MITRE_Tactic",
                "MITRE_Stage"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print("No predictions fall inside the documented attack windows.")


print("\nSaved to:")
print(OUTPUT_FILE)

print("\nMITRE timeline mapping complete.")