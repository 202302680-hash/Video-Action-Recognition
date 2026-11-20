import torch
import torch.nn as nn
import torch.optim as optim
import csv
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.cnn_transformer_resnet_ablation import CNNTransformerResNetAblation
from src.dataset_resnet_7class import UCF101ResNet7Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 2
EPOCHS = 15
PATIENCE = 5
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001
DROPOUT = 0.4

print("Device:", DEVICE)
print("Experiment: Ablation 2 - Dropout 0.4")
print("Transformer Layers: 1")
print("Weight Decay:", WEIGHT_DECAY)

train_dataset = UCF101ResNet7Dataset("train")
val_dataset = UCF101ResNet7Dataset("val")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

model = CNNTransformerResNetAblation(
    num_classes=7,
    num_frames=16,
    d_model=512,
    num_heads=8,
    num_layers=1,
    dropout=DROPOUT,
    freeze_backbone=True
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

Path("checkpoints").mkdir(exist_ok=True)
Path("experiments").mkdir(exist_ok=True)

best_val_loss = float("inf")
best_val_accuracy = 0.0
epochs_without_improvement = 0
history = []

for epoch in range(EPOCHS):

    model.train()
    model.cnn.eval()

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
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total

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
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            "checkpoints/transformer_ablation_dropout04_best.pth"
        )
        print("Best model saved!")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

with open(
    "experiments/transformer_ablation_dropout04_training.csv",
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy", "learning_rate"
        ]
    )
    writer.writeheader()
    writer.writerows(history)

print("\nTraining completed.")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Validation Accuracy at Best Loss: {best_val_accuracy*100:.2f}%")
print("Checkpoint: checkpoints/transformer_ablation_dropout04_best.pth")
print("History: experiments/transformer_ablation_dropout04_training.csv")
