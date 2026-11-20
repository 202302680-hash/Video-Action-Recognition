import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.cnn_lstm_7class import CNNLSTM7
from src.dataset_resnet_7class import create_resnet7_dataloaders


# =========================
# CONFIGURATION
# =========================

SEED = 42
BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "cnn_lstm_7class_tuned_best.pth"
HISTORY_PATH = EXPERIMENT_DIR / "cnn_lstm_7class_tuned_training.csv"


# =========================
# REPRODUCIBILITY
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DATA
# =========================

train_loader, val_loader, test_loader = create_resnet7_dataloaders(
    batch_size=BATCH_SIZE
)

print("Device:", DEVICE)
print("Train videos:", len(train_loader.dataset))
print("Validation videos:", len(val_loader.dataset))
print("Test videos:", len(test_loader.dataset))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)


# =========================
# MODEL
# =========================

model = CNNLSTM7(num_classes=7).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# =========================
# TRAINING
# =========================

best_val_loss = float("inf")
best_val_accuracy = 0.0
epochs_without_improvement = 0

history = []

for epoch in range(EPOCHS):

    # -------- TRAIN --------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for videos, labels in train_loader:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # -------- VALIDATION --------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.7f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })


    # -------- SAVE BEST MODEL --------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        epochs_without_improvement = 0

        print("Best model saved!")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


# =========================
# SAVE TRAINING HISTORY
# =========================

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


# =========================
# FINAL SUMMARY
# =========================

print("\n===== CNN + LSTM 7-CLASS TRAINING COMPLETED =====")
print("Best Validation Accuracy:", round(best_val_accuracy, 4))
print("Best Validation Loss:", round(best_val_loss, 4))
print("Checkpoint:", CHECKPOINT_PATH)
print("Training history:", HISTORY_PATH)
