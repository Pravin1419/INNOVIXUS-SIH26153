import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

STATE_FILE = Path(r".\data\network_states_10s.csv")
MODEL_FILE = Path(r".\data\world_model_lstm.pth")
OUTPUT_FILE = Path(r".\data\feature_explanation.csv")

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
# LSTM WORLD MODEL
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
print("INNOVIXUS - EXPLAINABILITY ENGINE")
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

print("\nStates:", len(df))


# --------------------------------------------------
# LOAD WORLD MODEL
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

world_model = WorldModelLSTM(
    checkpoint["input_size"],
    checkpoint["hidden_size"],
    checkpoint["num_layers"]
).to(DEVICE)

world_model.load_state_dict(
    checkpoint["model_state_dict"]
)

world_model.eval()

print("World Model loaded.")


# --------------------------------------------------
# GENERATE PREDICTED STATES
# --------------------------------------------------

predicted_states = []

with torch.no_grad():

    for i in range(
        SEQUENCE_LENGTH,
        len(X)
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

predicted_states = np.array(
    predicted_states,
    dtype=np.float32
)

target_y = y[SEQUENCE_LENGTH:]


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


# --------------------------------------------------
# TRAIN SAME INFILTRATION CLASSIFIER
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

print("Infiltration classifier trained.")


# --------------------------------------------------
# SELECT A TEST EXAMPLE
# --------------------------------------------------

# Choose the test example with the highest predicted
# attack probability.

test_probabilities = classifier.predict_proba(
    X_test
)[:, 1]

example_index = np.argmax(
    test_probabilities
)

example = X_test[
    example_index:example_index + 1
]

example_probability = test_probabilities[
    example_index
]

example_actual = y_test[
    example_index
]

test_timestamp = df[
    "Timestamp"
].iloc[
    SEQUENCE_LENGTH + split + example_index
]

print("\nSelected test example:")
print("Timestamp:", test_timestamp)
print(
    "Actual:",
    "Attack" if example_actual == 1 else "Benign"
)
print(
    "Predicted attack probability:",
    round(example_probability, 4)
)


# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------

print("\nCalculating SHAP explanation...")

# Logistic regression is a linear model, so LinearExplainer
# provides direct feature attribution.

background = X_train[
    :min(200, len(X_train))
]

explainer = shap.LinearExplainer(
    classifier,
    background
)

shap_values = explainer(
    example
)

values = shap_values.values

# Handle SHAP's possible output shape
if values.ndim == 2:
    values = values[0]

# --------------------------------------------------
# BUILD EXPLANATION TABLE
# --------------------------------------------------

explanation = pd.DataFrame({
    "Feature": FEATURES,
    "SHAP_Value": values
})

explanation[
    "Absolute_SHAP"
] = explanation[
    "SHAP_Value"
].abs()

explanation[
    "Direction"
] = np.where(
    explanation["SHAP_Value"] > 0,
    "Increases attack probability",
    "Decreases attack probability"
)

explanation = explanation.sort_values(
    "Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

# Keep top 10
top_features = explanation.head(10)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

top_features.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP CONTRIBUTING FEATURES")
print("=" * 60)

print(
    top_features[
        [
            "Feature",
            "SHAP_Value",
            "Direction"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nExplainability complete.")