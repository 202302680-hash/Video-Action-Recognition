import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import csv
from pathlib import Path

from src.models.resnet_mean_pool_ablation import ResNetMeanPoolAblation
from src.dataset_resnet_7class import UCF101ResNet7Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2

CLASS_NAMES = [
    "ApplyEyeMakeup",
    "Biking",
    "CliffDiving",
    "Diving",
    "PlayingGuitar",
    "PlayingViolin",
    "SoccerJuggling"
]

# Load the same 7-class test split used by the other models
test_dataset = UCF101ResNet7Dataset("test", root_dir="experiments")
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = ResNetMeanPoolAblation(
    num_classes=7,
    freeze_backbone=True,
    dropout=0.4
).to(DEVICE)

checkpoint = torch.load(
    "checkpoints/resnet_mean_pool_ablation_best.pth",
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

all_labels = []
all_predictions = []
total_loss = 0.0

criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for videos, labels in test_loader:
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(outputs, dim=1)

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

print("\n========== RESNET MEAN POOLING ABLATION TEST RESULTS ==========")
print("Test Videos:", len(test_dataset))
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {f1:.4f}")

print("\n========== PER-CLASS RESULTS ==========")
report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(7)),
    target_names=CLASS_NAMES,
    zero_division=0
)
print(report)

output_dir = Path("experiments")
output_dir.mkdir(exist_ok=True)

matrix = confusion_matrix(
    all_labels, all_predictions, labels=list(range(7))
)

with open(
    output_dir / "resnet_mean_pool_ablation_confusion_matrix.csv",
    "w", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["Actual/Predicted"] + CLASS_NAMES)
    for name, row in zip(CLASS_NAMES, matrix):
        writer.writerow([name] + row.tolist())

with open(
    output_dir / "resnet_mean_pool_ablation_test_results.csv",
    "w", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "test_loss", "accuracy", "macro_precision",
        "macro_recall", "macro_f1"
    ])
    writer.writerow([test_loss, accuracy, precision, recall, f1])

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(7)),
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)

with open(
    output_dir / "resnet_mean_pool_ablation_per_class_metrics.csv",
    "w", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["class", "precision", "recall", "f1_score", "support"])
    for class_name in CLASS_NAMES:
        row = report_dict[class_name]
        writer.writerow([
            class_name,
            row["precision"],
            row["recall"],
            row["f1-score"],
            row["support"]
        ])

print("\nEvaluation files saved successfully:")
print("experiments/resnet_mean_pool_ablation_test_results.csv")
print("experiments/resnet_mean_pool_ablation_confusion_matrix.csv")
print("experiments/resnet_mean_pool_ablation_per_class_metrics.csv")
