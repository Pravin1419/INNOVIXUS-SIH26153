import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_FILE = Path(r".\data\lstm_sequences.npz")
MODEL_FILE = Path(r".\data\world_model_lstm.pth")

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

INPUT_SIZE = 19
HIDDEN_SIZE = 64
NUM_LAYERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("INNOVIXUS - LSTM WORLD MODEL TRAINING")
print("=" * 60)

print("\nDevice:", DEVICE)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = np.load(DATA_FILE)

X = data["X"]
y = data["y_state"]

print("\nInput shape :", X.shape)
print("Target shape:", y.shape)

# --------------------------------------------------
# TEMPORAL TRAIN / TEST SPLIT
# --------------------------------------------------

split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# --------------------------------------------------
# PYTORCH TENSORS
# --------------------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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

        # Use the final temporal state
        final_state = output[:, -1, :]

        prediction = self.output_layer(
            final_state
        )

        return prediction


model = WorldModelLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS
).to(DEVICE)

# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {average_loss:.6f}"
    )

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

model.eval()

with torch.no_grad():

    X_test_device = X_test.to(DEVICE)
    y_test_device = y_test.to(DEVICE)

    predictions = model(
        X_test_device
    )

    test_loss = criterion(
        predictions,
        y_test_device
    )

print("\n" + "=" * 60)
print("WORLD MODEL TRAINING COMPLETE")
print("=" * 60)

print(f"\nTest MSE: {test_loss.item():.6f}")

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "sequence_length": 10,
    },
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)