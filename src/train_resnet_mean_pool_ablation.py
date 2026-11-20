
import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.resnet_mean_pool_ablation import ResNetMeanPoolAblation
from src.dataset_resnet_7class import UCF101ResNet7Dataset


# =========================
# Reproducibility
# =========================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# Configuration
# =========================

BATCH_SIZE = 2
EPOCHS = 15
PATIENCE = 5

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001

NUM_CLASSES = 7

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# =========================
# Dataset
# =========================

train_dataset = UCF101ResNet7Dataset(
    "train",
    root_dir="experiments"
)

val_dataset = UCF101ResNet7Dataset(
    "val",
    root_dir="experiments"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# Model
# =========================

model = ResNetMeanPoolAblation(
    num_classes=NUM_CLASSES,
    freeze_backbone=True,
    dropout=0.4
).to(device)

# Keep pretrained CNN frozen
model.cnn.eval()


# =========================
# Loss and Optimizer
# =========================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

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
    patience=2
)


# =========================
# Output Files
# =========================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

checkpoint_path = (
    "checkpoints/resnet_mean_pool_ablation_best.pth"
)

csv_path = (
    "experiments/resnet_mean_pool_ablation_training.csv"
)


# =========================
# Training Variables
# =========================

best_val_loss = float("inf")
best_val_acc = 0.0

epochs_without_improvement = 0

history = []


# =========================
# Training Loop
# =========================

for epoch in range(EPOCHS):

    model.train()

    # Keep frozen CNN in evaluation mode
    model.cnn.eval()

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

        train_correct += (
            outputs.argmax(1) == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total


    # =========================
    # Validation
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

            val_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total


    # =========================
    # Scheduler
    # =========================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # =========================
    # Print Results
    # =========================

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {current_lr:.7f}"
    )


    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr
    })


    # =========================
    # Save Best Model
    # =========================

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        epochs_without_improvement = 0

        print("Best model saved!")

    else:

        epochs_without_improvement += 1


    # =========================
    # Early Stopping
    # =========================

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


# =========================
# Save Training History
# =========================

with open(csv_path, "w", newline="") as f:

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
# Final Summary
# =========================

print("\nTraining completed.")

print(
    "Best Validation Loss:",
    round(best_val_loss, 4)
)

print(
    "Best Validation Accuracy:",
    round(best_val_acc * 100, 2),
    "%"
)

print("Checkpoint:", checkpoint_path)
print("History:", csv_path)
