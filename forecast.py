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
OUTPUT_FILE = Path(r".\data\future_forecast.csv")

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
print("INNOVIXUS - K-STEP WORLD MODEL FORECAST")
print("=" * 60)

df = pd.read_csv(STATE_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"]
)

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

print("\nNetwork states:", len(df))
print("Device:", DEVICE)


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

print("World model loaded successfully.")


# --------------------------------------------------
# START FROM LAST TEST WINDOW
# --------------------------------------------------

# Use the last 10 states from the held-out period.
# This keeps the demonstration separate from the training
# sequence region.

test_start = int(len(df) * 0.8)

history = df.iloc[
    test_start - SEQUENCE_LENGTH:test_start
][FEATURES].values.astype(np.float32)

current_time = df.iloc[
    test_start - 1
]["Timestamp"]

print("\nStarting forecast from:")
print(current_time)

print(
    "\nForecasting",
    FORECAST_STEPS,
    "future states..."
)


# --------------------------------------------------
# K-STEP ROLLOUT
# --------------------------------------------------

forecast_rows = []

with torch.no_grad():

    for step in range(1, FORECAST_STEPS + 1):

        input_tensor = torch.tensor(
            history[-SEQUENCE_LENGTH:],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        predicted_state = model(
            input_tensor
        ).cpu().numpy()[0]

        # Advance by 10 seconds
        future_time = (
            current_time +
            pd.Timedelta(seconds=10)
        )

        row = {
            "Step": step,
            "Timestamp": future_time,
        }

        for i, feature in enumerate(FEATURES):
            row[feature] = predicted_state[i]

        forecast_rows.append(row)

        # Feed prediction back into model
        history = np.vstack(
            [history, predicted_state]
        )

        current_time = future_time


# --------------------------------------------------
# SAVE FORECAST
# --------------------------------------------------

forecast_df = pd.DataFrame(
    forecast_rows
)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 60)
print("K-STEP FORECAST COMPLETE")
print("=" * 60)

print("\nFuture states:")

print(
    forecast_df[
        ["Step", "Timestamp"]
    ].to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)