import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# FILES
# --------------------------------------------------

STATE_FILE = Path(r".\data\network_states_10s.csv")
RESULT_FILE = Path(r".\data\benchmark_results.csv")

SEQUENCE_LENGTH = 10
TRAIN_RATIO = 0.80

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LSTM WORLD MODEL
# --------------------------------------------------

class WorldModelLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=64, num_layers=2):
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
# LOAD STATES
# --------------------------------------------------

print("=" * 65)
print("INNOVIXUS - FAIR TEMPORAL BENCHMARK")
print("=" * 65)

df = pd.read_csv(STATE_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

X = df[FEATURES].values.astype(np.float32)

y = df["State_Label"].eq(
    "Attack"
).astype(int).values

total_states = len(df)

split = int(total_states * TRAIN_RATIO)

print("\nTotal states :", total_states)
print("Training     :", split)
print("Testing      :", total_states - split)

print("\nTest begins at:")
print(df["Timestamp"].iloc[split])


# --------------------------------------------------
# TRAIN LSTM ONLY ON TRAINING PERIOD
# --------------------------------------------------

world_model = WorldModelLSTM(
    input_size=len(FEATURES)
).to(DEVICE)

optimizer = torch.optim.Adam(
    world_model.parameters(),
    lr=0.001
)

loss_function = nn.MSELoss()

train_sequences = []
train_targets = []

for i in range(
    SEQUENCE_LENGTH,
    split
):

    train_sequences.append(
        X[i - SEQUENCE_LENGTH:i]
    )

    train_targets.append(
        X[i]
    )

train_sequences = np.array(
    train_sequences,
    dtype=np.float32
)

train_targets = np.array(
    train_targets,
    dtype=np.float32
)

train_tensor = torch.tensor(
    train_sequences
).to(DEVICE)

target_tensor = torch.tensor(
    train_targets
).to(DEVICE)


print("\nTraining World Model...")
print(
    "Training sequences:",
    len(train_sequences)
)

world_model.train()

for epoch in range(20):

    optimizer.zero_grad()

    predictions = world_model(
        train_tensor
    )

    loss = loss_function(
        predictions,
        target_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch {epoch + 1:02d}/20 "
            f"| Loss: {loss.item():.6f}"
        )

world_model.eval()

print("World Model training complete.")


# --------------------------------------------------
# GENERATE FUTURE STATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating future-state predictions...")

predicted_states = []
valid_indices = []

with torch.no_grad():

    for i in range(
        SEQUENCE_LENGTH,
        total_states
    ):

        sequence = X[
            i - SEQUENCE_LENGTH:i
        ]

        tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        prediction = world_model(
            tensor
        ).cpu().numpy()[0]

        predicted_states.append(
            prediction
        )

        valid_indices.append(i)

predicted_states = np.array(
    predicted_states,
    dtype=np.float32
)

valid_indices = np.array(
    valid_indices
)

future_labels = y[valid_indices]


# --------------------------------------------------
# TEMPORAL SPLIT
# --------------------------------------------------

train_mask = valid_indices < split
test_mask = valid_indices >= split

wm_train_X = predicted_states[train_mask]
wm_test_X = predicted_states[test_mask]

wm_train_y = future_labels[train_mask]
wm_test_y = future_labels[test_mask]


# --------------------------------------------------
# INNOVIXUS CLASSIFIER
# --------------------------------------------------

print("\nTraining INNOVIXUS future-state classifier...")

innovixus_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

innovixus_model.fit(
    wm_train_X,
    wm_train_y
)

innovixus_pred = innovixus_model.predict(
    wm_test_X
)

innovixus_prob = innovixus_model.predict_proba(
    wm_test_X
)[:, 1]


# --------------------------------------------------
# TRADITIONAL TEMPORAL BASELINE
# --------------------------------------------------
#
# Current state -> next state's attack status
#
# To predict state i, use state i-1.
# This aligns exactly with the World Model target.
# --------------------------------------------------

baseline_X = []
baseline_y = []

for i in range(
    SEQUENCE_LENGTH,
    total_states
):

    baseline_X.append(
        X[i - 1]
    )

    baseline_y.append(
        y[i]
    )

baseline_X = np.array(
    baseline_X,
    dtype=np.float32
)

baseline_y = np.array(
    baseline_y
)


baseline_train_mask = valid_indices < split
baseline_test_mask = valid_indices >= split

baseline_train_X = baseline_X[
    baseline_train_mask
]

baseline_test_X = baseline_X[
    baseline_test_mask
]

baseline_train_y = baseline_y[
    baseline_train_mask
]

baseline_test_y = baseline_y[
    baseline_test_mask
]


print("Training Logistic Regression baseline...")

baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

baseline_model.fit(
    baseline_train_X,
    baseline_train_y
)

baseline_pred = baseline_model.predict(
    baseline_test_X
)

baseline_prob = baseline_model.predict_proba(
    baseline_test_X
)[:, 1]


# --------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------

def calculate_metrics(
    actual,
    predicted,
    probabilities
):

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    auc = roc_auc_score(
        actual,
        probabilities
    )

    cm = confusion_matrix(
        actual,
        predicted
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    return (
        precision,
        recall,
        f1,
        auc,
        fpr,
        cm
    )


# --------------------------------------------------
# CALCULATE RESULTS
# --------------------------------------------------

baseline_metrics = calculate_metrics(
    baseline_test_y,
    baseline_pred,
    baseline_prob
)

innovixus_metrics = calculate_metrics(
    wm_test_y,
    innovixus_pred,
    innovixus_prob
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n")
print("=" * 65)
print("FAIR HELD-OUT TEST RESULTS")
print("=" * 65)

print("\nLogistic Regression Baseline")
print("-" * 40)

print(
    f"Precision : {baseline_metrics[0]:.4f}"
)

print(
    f"Recall    : {baseline_metrics[1]:.4f}"
)

print(
    f"F1 Score  : {baseline_metrics[2]:.4f}"
)

print(
    f"ROC-AUC   : {baseline_metrics[3]:.4f}"
)

print(
    f"FPR       : {baseline_metrics[4]:.4f}"
)

print("\nConfusion Matrix:")
print(baseline_metrics[5])


print("\nINNOVIXUS - LSTM World Model")
print("-" * 40)

print(
    f"Precision : {innovixus_metrics[0]:.4f}"
)

print(
    f"Recall    : {innovixus_metrics[1]:.4f}"
)

print(
    f"F1 Score  : {innovixus_metrics[2]:.4f}"
)

print(
    f"ROC-AUC   : {innovixus_metrics[3]:.4f}"
)

print(
    f"FPR       : {innovixus_metrics[4]:.4f}"
)

print("\nConfusion Matrix:")
print(innovixus_metrics[5])


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame({

    "Model": [
        "Logistic Regression Baseline",
        "INNOVIXUS LSTM World Model"
    ],

    "Precision": [
        baseline_metrics[0],
        innovixus_metrics[0]
    ],

    "Recall": [
        baseline_metrics[1],
        innovixus_metrics[1]
    ],

    "F1_Score": [
        baseline_metrics[2],
        innovixus_metrics[2]
    ],

    "ROC_AUC": [
        baseline_metrics[3],
        innovixus_metrics[3]
    ],

    "False_Positive_Rate": [
        baseline_metrics[4],
        innovixus_metrics[4]
    ]
})

results.to_csv(
    RESULT_FILE,
    index=False
)

print("\n")
print("=" * 65)
print("Benchmark saved to:")
print(RESULT_FILE)
print("=" * 65)