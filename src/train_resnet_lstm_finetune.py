
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


# =====================================
# 1. Configuration
# =====================================

SEED = 42
BATCH_SIZE = 2
EPOCHS = 15
PATIENCE = 5

WEIGHT_DECAY = 0.0001

BACKBONE_LR = 0.00001
LSTM_LR = 0.0003

CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)


# =====================================
# 2. Reproducibility
# =====================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================
# 3. Device and Data
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

train_loader, val_loader, _ = create_resnet_dataloaders(
    batch_size=BATCH_SIZE
)


# =====================================
# 4. Model
# =====================================

model = CNNLSTMResNet(
    num_classes=10,
    hidden_size=128,
    freeze_backbone=True
).to(device)

# Unfreeze only ResNet Layer 4
for param in model.cnn[7].parameters():
    param.requires_grad = True


# =====================================
# 5. Loss Function
# =====================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)


# =====================================
# 6. Optimizer
# =====================================

optimizer = AdamW(
    [
        {
            "params": model.cnn[7].parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.lstm.parameters(),
            "lr": LSTM_LR
        },
        {
            "params": model.fc.parameters(),
            "lr": LSTM_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)


# =====================================
# 7. Learning Rate Scheduler
# =====================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=[1e-7, 1e-6, 1e-6]
)


# =====================================
# 8. Training Setup
# =====================================

best_val_accuracy = 0.0
best_val_loss = float("inf")

epochs_without_improvement = 0

history = []

print("\n===== RESNET18 + LSTM WITH AUGMENTATION =====")
print("Device:", device)
print("Training videos:", len(train_loader.dataset))
print("Validation videos:", len(val_loader.dataset))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Backbone: Layer 4 trainable")
print("Backbone LR:", BACKBONE_LR)
print("LSTM LR:", LSTM_LR)
print("Weight decay:", WEIGHT_DECAY)
print("Label smoothing: 0.05")
print("Data augmentation: Enabled for training only")


# =====================================
# 9. Training Loop
# =====================================

for epoch in range(EPOCHS):

    model.train()

    # Keep frozen ResNet layers in evaluation mode
    model.cnn[:7].eval()

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
            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),
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


    # =================================
    # 10. Validation
    # =================================

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

    current_backbone_lr = optimizer.param_groups[0]["lr"]
    current_lstm_lr = optimizer.param_groups[1]["lr"]

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Backbone LR: {current_backbone_lr:.7f} | "
        f"LSTM LR: {current_lstm_lr:.6f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "backbone_lr": current_backbone_lr,
        "lstm_lr": current_lstm_lr
    })


    # =================================
    # 11. Save Best Model
    # =================================

    improved = (
        val_accuracy > best_val_accuracy
        or (
            val_accuracy == best_val_accuracy
            and val_loss < best_val_loss
        )
    )

    if improved:

        best_val_accuracy = val_accuracy
        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "resnet_lstm_augmented_best.pth"
        )

        print("Best augmented model saved!")

    else:

        epochs_without_improvement += 1


    # =================================
    # 12. Early Stopping
    # =================================

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =====================================
# 13. Save Training History
# =====================================

history_path = (
    EXPERIMENT_DIR / "resnet_lstm_augmented_training.csv"
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
            "backbone_lr",
            "lstm_lr"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


# =====================================
# 14. Final Results
# =====================================

print("\n===== AUGMENTED FINE-TUNING COMPLETED =====")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    "Checkpoint: "
    "checkpoints/resnet_lstm_augmented_best.pth"
)

print("History:", history_path)
