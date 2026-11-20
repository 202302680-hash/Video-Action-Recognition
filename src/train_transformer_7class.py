import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.cnn_transformer_resnet import CNNTransformerResNet
from src.dataset_resnet_7class import create_resnet7_dataloaders


# Configuration
SEED = 42
BATCH_SIZE = 2
MAX_EPOCHS = 15
PATIENCE = 5
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "transformer_resnet_7class_best.pth"
)

HISTORY_PATH = (
    EXPERIMENT_DIR / "transformer_resnet_7class_training.csv"
)


# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# Data
train_loader, val_loader, test_loader = (
    create_resnet7_dataloaders(
        batch_size=BATCH_SIZE
    )
)

print("Train videos:", len(train_loader.dataset))
print("Validation videos:", len(val_loader.dataset))
print("Test videos:", len(test_loader.dataset))


# Model
model = CNNTransformerResNet(
    num_classes=7,
    num_frames=16,
    d_model=512,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    freeze_backbone=True
).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = AdamW(
    [
        p for p in model.parameters()
        if p.requires_grad
    ],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

print("\nModel: ResNet18 + Transformer")
print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# Training function
def train_one_epoch():

    model.train()

    # Keep frozen ResNet in evaluation mode
    model.cnn.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in train_loader:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return total_loss / total, correct / total


# Validation function
def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return total_loss / total, correct / total


# Training loop
best_val_loss = float("inf")
best_val_accuracy = 0.0
epochs_without_improvement = 0

history = []

print("\n========== TRANSFORMER TRAINING STARTED ==========")

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"LR: {current_lr:.7f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": 7,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "seed": SEED
            },
            BEST_MODEL_PATH
        )

        print("Best Transformer model saved!")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# Save history
with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


print("\n========== TRAINING COMPLETED ==========")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
print("Checkpoint:", BEST_MODEL_PATH)
print("History:", HISTORY_PATH)
