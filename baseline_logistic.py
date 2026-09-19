import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# FILE
# --------------------------------------------------

STATE_FILE = Path(r".\data\network_states_10s.csv")


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

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
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(STATE_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values("Timestamp").reset_index(drop=True)

X = df[FEATURES]

y = df["State_Label"].eq("Attack").astype(int)


# --------------------------------------------------
# TEMPORAL TRAIN / TEST SPLIT
# --------------------------------------------------

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("=" * 60)
print("INNOVIXUS - LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print("\nTotal states :", len(df))
print("Training     :", len(X_train))
print("Testing      :", len(X_test))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


print("\nTraining baseline...")

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# METRICS
# --------------------------------------------------

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(f"\nPrecision       : {precision:.4f}")
print(f"Recall          : {recall:.4f}")
print(f"F1 Score        : {f1:.4f}")
print(f"ROC-AUC         : {auc:.4f}")
print(f"False Positive Rate : {false_positive_rate:.4f}")

print("\nConfusion Matrix:")
print(cm)


print("\n" + "=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)
