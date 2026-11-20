
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.dataset import UCF101Dataset
from src.models.cnn_lstm import CNNLSTM


# =========================
# CONFIGURATION
# =========================

SEED = 42
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("experiments")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# REPRODUCIBILITY
# =========================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# DATA LOADERS
# =========================

train_dataset = UCF101Dataset("train")
val_dataset = UCF101Dataset("val")

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
# VIDEO AUGMENTATION
# TRAINING ONLY
# =========================

def augment_videos(videos):

    videos = videos.clone()

    batch_size = videos.size(0)

    # Random horizontal flip per video
    flip_mask = torch.rand(
        batch_size, 1, 1, 1, 1,
        device=videos.device
    ) < 0.5

    flipped = torch.flip(videos, dims=[-1])

    videos = torch.where(flip_mask, flipped, videos)

    # Random brightness
    brightness = (
        0.9 + 0.2 * torch.rand(
            batch_size, 1, 1, 1, 1,
            device=videos.device
        )
    )

    # Random contrast
    contrast = (
        0.9 + 0.2 * torch.rand(
            batch_size, 1, 1, 1, 1,
            device=videos.device
        )
    )

    mean = videos.mean(
        dim=(1, 2, 3, 4),
        keepdim=True
    )

    videos = (videos - mean) * contrast + mean
    videos = videos * brightness

    return videos.clamp(0.0, 1.0)


# =========================
# MODEL
# =========================

model = CNNLSTM(num_classes=10).to(DEVICE)

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
    patience=2,
    min_lr=1e-6
)


print("\n===== IMPROVED CNN + LSTM TRAINING =====")
print("Device:", DEVICE)
print("Training videos:", len(train_dataset))
print("Validation videos:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)


# =========================
# TRAINING
# =========================

best_val_accuracy = 0.0
best_val_loss = float("inf")
epochs_without_improvement = 0

history = []

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for videos, labels in train_loader:

        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        # Augmentation only during training
        videos = augment_videos(videos)

        optimizer.zero_grad()

        outputs = model(videos)
        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping
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
    # VALIDATION
    # NO AUGMENTATION
    # =========================

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


    # =========================
    # SCHEDULER
    # =========================

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # =========================
    # PRINT RESULTS
    # =========================

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
    print(f"Learning Rate: {current_lr:.7f}")


    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })


    # =========================
    # SAVE BEST ACCURACY MODEL
    # =========================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "seed": SEED
        }, CHECKPOINT_DIR / "cnn_lstm_improved_best.pth")

        print("Best accuracy model saved!")


    # =========================
    # EARLY STOPPING ON VAL LOSS
    # =========================

    if val_loss < best_val_loss - 0.001:

        best_val_loss = val_loss
        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")
        break


# =========================
# SAVE HISTORY
# =========================

history_path = (
    RESULTS_DIR / "cnn_lstm_improved_training.csv"
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
# FINAL SUMMARY
# =========================

print("\n===== IMPROVED TRAINING COMPLETED =====")
print("Best Validation Accuracy:", round(best_val_accuracy, 4))
print("Best Validation Loss:", round(best_val_loss, 4))

print(
    "Checkpoint:",
    CHECKPOINT_DIR / "cnn_lstm_improved_best.pth"
)

print("Training history:", history_path)
