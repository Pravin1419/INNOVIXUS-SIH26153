import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

STATE_FILE = Path(r".\data\network_states_10s.csv")
MODEL_FILE = Path(r".\data\world_model_lstm.pth")
OUTPUT_FILE = Path(r".\data\infiltration_forecast.csv")

SEQUENCE_LENGTH = 10
FORECAST_STEPS = 10

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
# MODEL
# --------------------------------------------------

class WorldModelLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
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
print("INNOVIXUS - INFILTRATION PREDICTION ENGINE")
print("=" * 60)

df = pd.read_csv(STATE_FILE)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

X = df[FEATURES].values.astype(np.float32)

actual_attack_ratio = df[
    "Attack_Ratio"
].values.astype(np.float32)

print("\nTotal states:", len(df))


# --------------------------------------------------
# LOAD MODEL
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

print("World model loaded.")
print("Device:", DEVICE)


# --------------------------------------------------
# CALIBRATION
# --------------------------------------------------

# Use only the training portion to determine
# the scale of predicted attack intensity.

train_end = int(len(X) * 0.8)

training_attack_ratios = actual_attack_ratio[
    :train_end
]

attack_reference = np.percentile(
    training_attack_ratios,
    95
)

attack_reference = max(
    attack_reference,
    0.01
)

print(
    "\nTraining attack reference:",
    round(attack_reference, 4)
)


# --------------------------------------------------
# START FROM HELD-OUT TEST PERIOD
# --------------------------------------------------

start = train_end

history = X[
    start - SEQUENCE_LENGTH:start
].copy()

current_time = df.iloc[
    start - 1
]["Timestamp"]

print(
    "\nForecast starting from:",
    current_time
)

print(
    "Forecast horizon:",
    FORECAST_STEPS * 10,
    "seconds"
)


# --------------------------------------------------
# K-STEP PREDICTION
# --------------------------------------------------

results = []

with torch.no_grad():

    for step in range(1, FORECAST_STEPS + 1):

        input_tensor = torch.tensor(
            history[-SEQUENCE_LENGTH:],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        predicted_state = model(
            input_tensor
        ).cpu().numpy()[0]

        # --------------------------------------------------
        # PREDICTED ATTACK INTENSITY
        # --------------------------------------------------

        # The predicted state contains normalized features.
        # Use the magnitude of the predicted state as an
        # anomaly/intensity signal.

        intensity = np.mean(
            np.abs(predicted_state)
        )

        # Convert intensity into a bounded 0-1 score.
        infiltration_score = (
            intensity /
            (intensity + attack_reference)
        )

        infiltration_score = float(
            np.clip(
                infiltration_score,
                0.0,
                1.0
            )
        )

        # --------------------------------------------------
        # STAGE ESTIMATION
        # --------------------------------------------------

        if infiltration_score < 0.25:

            stage = "Monitoring"

        elif infiltration_score < 0.50:

            stage = "Reconnaissance"

        elif infiltration_score < 0.70:

            stage = "Initial Access"

        elif infiltration_score < 0.85:

            stage = "Lateral Movement / C2"

        else:

            stage = "Exfiltration Risk"

        future_time = (
            current_time +
            pd.Timedelta(seconds=10)
        )

        results.append(
            {
                "Step": step,
                "Timestamp": future_time,
                "Infiltration_Score":
                    infiltration_score,
                "Predicted_Stage": stage,
                "Predicted_Intensity":
                    float(intensity),
            }
        )

        # --------------------------------------------------
        # FEEDBACK LOOP
        # --------------------------------------------------

        history = np.vstack(
            [history, predicted_state]
        )

        current_time = future_time


# --------------------------------------------------
# SAVE
# --------------------------------------------------

forecast_df = pd.DataFrame(results)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 60)
print("INFILTRATION FORECAST COMPLETE")
print("=" * 60)

print(
    forecast_df.to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)