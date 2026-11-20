
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset_resnet import create_resnet_dataloaders
from src.models.cnn_lstm_resnet import CNNLSTMResNet


# =========================
# 1. Configuration
# =========================

SEED = 42
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)


# =========================
# 2. Reproducibility
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# 3. Device and Dataset
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, val_loader, test_loader = (
    create_resnet_dataloaders(batch_size=BATCH_SIZE)
)


# =========================
# 4. Model
# =========================

model = CNNLSTMResNet(
    num_classes=10,
    hidden_size=128,
    freeze_backbone=True
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6
)


# =========================
# 5. Training Setup
# =========================

best_val_accuracy = 0.0
best_val_loss = float("inf")
epochs_without_improvement = 0

history = []

print("\n===== RESNET18 + LSTM TRAINING =====")
print("Device:", device)
print("Training videos:", len(train_loader.dataset))
print("Validation videos:", len(val_loader.dataset))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Backbone: Frozen ResNet18")

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# =========================
# 6. Training Loop
# =========================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for videos, labels in train_loader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
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


    # =========================
    # 7. Validation
    # =========================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(device)
            labels = labels.to(device)

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


    # =========================
    # 8. Print Results
    # =========================

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })


    # =========================
    # 9. Save Best Model
    # =========================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "resnet_lstm_best.pth"
        )

        print("Best model saved!")

    else:

        epochs_without_improvement += 1


    # =========================
    # 10. Early Stopping
    # =========================

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =========================
# 11. Save Training History
# =========================

history_path = (
    EXPERIMENT_DIR / "resnet_lstm_training.csv"
)

with open(
    history_path,
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
# 12. Final Results
# =========================

print("\n===== TRAINING COMPLETED =====")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Validation Loss at Best Accuracy: "
    f"{best_val_loss:.4f}"
)

print(
    "Checkpoint: checkpoints/resnet_lstm_best.pth"
)

print("History:", history_path)
