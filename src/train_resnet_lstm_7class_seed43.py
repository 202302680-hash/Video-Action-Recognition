import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.cnn_lstm_resnet import CNNLSTMResNet
from src.dataset_resnet_7class import create_resnet7_dataloaders


# =========================
# CONFIGURATION
# =========================

SEED = 43
BATCH_SIZE = 2
MAX_EPOCHS = 15
PATIENCE = 5

BACKBONE_LR = 0.00001
HEAD_LR = 0.0003
WEIGHT_DECAY = 0.0001

NUM_CLASSES = 7
HIDDEN_SIZE = 128

CHECKPOINT_DIR = Path("checkpoints")
EXPERIMENT_DIR = Path("experiments")

CHECKPOINT_DIR.mkdir(exist_ok=True)
EXPERIMENT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "resnet_lstm_7class_seed43_best.pth"
HISTORY_PATH = EXPERIMENT_DIR / "resnet_lstm_7class_seed43_training.csv"


# =========================
# REPRODUCIBILITY
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# DATA LOADERS
# =========================

train_loader, val_loader, test_loader = (
    create_resnet7_dataloaders(
        batch_size=BATCH_SIZE
    )
)

print("\nTrain videos:", len(train_loader.dataset))
print("Validation videos:", len(val_loader.dataset))
print("Test videos:", len(test_loader.dataset))


# =========================
# MODEL
# =========================

model = CNNLSTMResNet(
    num_classes=NUM_CLASSES,
    hidden_size=HIDDEN_SIZE,
    freeze_backbone=True
)

# Unfreeze ResNet18 Layer4 only
for param in model.cnn[7].parameters():
    param.requires_grad = True

model = model.to(device)

backbone_parameters = [
    p for p in model.cnn[7].parameters()
    if p.requires_grad
]

head_parameters = [
    p for p in model.lstm.parameters()
    if p.requires_grad
]

head_parameters += [
    p for p in model.fc.parameters()
    if p.requires_grad
]

trainable_parameters = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nModel: ResNet18 + LSTM")
print("Number of classes:", NUM_CLASSES)
print("Trainable parameters:", trainable_parameters)
print("Fine-tuned layer: ResNet18 Layer4")


# =========================
# LOSS AND OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)

optimizer = AdamW(
    [
        {
            "params": backbone_parameters,
            "lr": BACKBONE_LR
        },
        {
            "params": head_parameters,
            "lr": HEAD_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# =========================
# TRAINING FUNCTION
# =========================

def train_one_epoch():

    model.train()

    # Keep frozen backbone layers in evaluation mode
    model.cnn[:7].eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in train_loader:

        videos = videos.to(device)
        labels = labels.to(device)

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


# =========================
# VALIDATION FUNCTION
# =========================

def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return total_loss / total, correct / total


# =========================
# TRAINING LOOP
# =========================

best_val_loss = float("inf")
best_val_accuracy = 0.0
epochs_without_improvement = 0

history = []

print("\n========== 7-CLASS TRAINING STARTED ==========")

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    scheduler.step(val_loss)

    current_backbone_lr = optimizer.param_groups[0]["lr"]
    current_head_lr = optimizer.param_groups[1]["lr"]

    print(
        f"Epoch {epoch}/{MAX_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Backbone LR: {current_backbone_lr:.7f} | "
        f"Head LR: {current_head_lr:.7f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "backbone_lr": current_backbone_lr,
        "head_lr": current_head_lr
    })

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "hidden_size": HIDDEN_SIZE,
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "seed": SEED
            },
            BEST_MODEL_PATH
        )

        print("Best 7-class model saved!")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
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
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


print("\n========== TRAINING COMPLETED ==========")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
print("Checkpoint:", BEST_MODEL_PATH)
print("Training history:", HISTORY_PATH)
