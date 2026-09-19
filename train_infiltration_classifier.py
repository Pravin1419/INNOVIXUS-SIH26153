import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

STATE_FILE = Path(r".\data\network_states_10s.csv")
MODEL_FILE = Path(r".\data\world_model_lstm.pth")
OUTPUT_FILE = Path(r".\data\infiltration_predictions.csv")

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LSTM MODEL
# --------------------------------------------------

class WorldModelLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        final_state = output[:, -1, :]

        return self.output_layer(final_state)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("INNOVIXUS - EVALUATED INFILTRATION CLASSIFIER")
print("=" * 60)

df = pd.read_csv(STATE_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

X = df[FEATURES].values.astype(np.float32)

y = (
    df["State_Label"]
    .eq("Attack")
    .astype(int)
    .values
)

print("\nTotal states:", len(df))
print("Attack states:", y.sum())
print("Benign states:", len(y) - y.sum())


# --------------------------------------------------
# LOAD WORLD MODEL
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model = WorldModelLSTM(
    checkpoint["input_size"],
    checkpoint["hidden_size"],
    checkpoint["num_layers"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nWorld Model loaded.")


# --------------------------------------------------
# GENERATE WORLD-MODEL PREDICTIONS
# --------------------------------------------------

predicted_states = []

print("\nGenerating predicted future states...")

with torch.no_grad():

    for i in range(SEQUENCE_LENGTH, len(X)):

        sequence = X[
            i - SEQUENCE_LENGTH:i
        ]

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        prediction = model(
            tensor
        ).cpu().numpy()[0]

        predicted_states.append(
            prediction
        )

predicted_states = np.array(
    predicted_states,
    dtype=np.float32
)

# Corresponding ground truth
target_y = y[SEQUENCE_LENGTH:]

timestamps = df[
    "Timestamp"
].iloc[SEQUENCE_LENGTH:].reset_index(drop=True)


# --------------------------------------------------
# TEMPORAL TRAIN / TEST SPLIT
# --------------------------------------------------

split = int(
    len(predicted_states) * 0.8
)

X_train = predicted_states[:split]
y_train = target_y[:split]

X_test = predicted_states[split:]
y_test = target_y[split:]

test_timestamps = timestamps.iloc[
    split:
].reset_index(drop=True)

print("\nWorld-model predictions:", len(predicted_states))

print("Classifier training:", len(X_train))
print("Classifier testing :", len(X_test))


# --------------------------------------------------
# CLASSIFIER
# --------------------------------------------------

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

classifier.fit(
    X_train,
    y_train
)

print("\nInfiltration classifier trained.")


# --------------------------------------------------
# TEST PREDICTIONS
# --------------------------------------------------

probabilities = classifier.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


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

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print("HELD-OUT TEST RESULTS")
print("=" * 60)

print(
    f"\nPrecision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Benign",
            "Attack"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "Timestamp": test_timestamps,
    "Actual_State": np.where(
        y_test == 1,
        "Attack",
        "Benign"
    ),
    "Infiltration_Probability": probabilities,
    "Predicted_State": np.where(
        predictions == 1,
        "Attack",
        "Benign"
    )
})

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPredictions saved to:")
print(OUTPUT_FILE)