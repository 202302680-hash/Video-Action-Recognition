import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
import csv
from pathlib import Path

from src.models.cnn_transformer_resnet_ablation import CNNTransformerResNetAblation
from src.dataset_resnet_7class import UCF101ResNet7Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "ApplyEyeMakeup",
    "Biking",
    "CliffDiving",
    "Diving",
    "PlayingGuitar",
    "PlayingViolin",
    "SoccerJuggling"
]

test_dataset = UCF101ResNet7Dataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

model = CNNTransformerResNetAblation(
    num_classes=7,
    num_frames=16,
    d_model=512,
    num_heads=8,
    num_layers=1,
    dropout=0.2,
    freeze_backbone=True
).to(DEVICE)

checkpoint = torch.load(
    "checkpoints/transformer_ablation_1layer_best.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

criterion = nn.CrossEntropyLoss()

all_labels = []
all_predictions = []
total_loss = 0

with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

test_loss = total_loss / len(test_dataset)

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, average="macro", zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, average="macro", zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, average="macro", zero_division=0
)

print("\n========== ABLATION TEST RESULTS ==========")
print("Test Videos:", len(test_dataset))
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {f1:.4f}")

print("\n========== PER-CLASS RESULTS ==========")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(7)),
    target_names=CLASS_NAMES,
    zero_division=0
))

Path("experiments").mkdir(exist_ok=True)

with open(
    "experiments/transformer_ablation_1layer_test_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "test_loss", "accuracy", "macro_precision",
        "macro_recall", "macro_f1"
    ])
    writer.writerow([
        test_loss, accuracy, precision, recall, f1
    ])

matrix = confusion_matrix(
    all_labels, all_predictions, labels=list(range(7))
)

with open(
    "experiments/transformer_ablation_1layer_confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["Actual/Predicted"] + CLASS_NAMES)
    for name, row in zip(CLASS_NAMES, matrix):
        writer.writerow([name] + row.tolist())

print("\nAblation evaluation saved successfully.")
