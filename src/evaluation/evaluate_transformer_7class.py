import csv
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.models.cnn_transformer_resnet import CNNTransformerResNet
from src.dataset_resnet_7class import UCF101ResNet7Dataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASSES = [
    "ApplyEyeMakeup",
    "Biking",
    "CliffDiving",
    "Diving",
    "PlayingGuitar",
    "PlayingViolin",
    "SoccerJuggling"
]

CHECKPOINT_PATH = Path(
    "checkpoints/transformer_resnet_7class_best.pth"
)

OUTPUT_DIR = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)


# Test dataset
test_dataset = UCF101ResNet7Dataset("test")

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# Load model
model = CNNTransformerResNet(
    num_classes=7,
    num_frames=16,
    d_model=512,
    num_heads=8,
    num_layers=2,
    dropout=0.2,
    freeze_backbone=True
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# Evaluate
all_labels = []
all_predictions = []
total_loss = 0.0

criterion = torch.nn.CrossEntropyLoss()

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


# Metrics
y_true = np.array(all_labels)
y_pred = np.array(all_predictions)

test_loss = total_loss / len(test_dataset)

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true, y_pred,
    average="macro",
    zero_division=0
)


print("\n========== TRANSFORMER TEST RESULTS ==========")
print("Test Videos:", len(test_dataset))
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {f1:.4f}")


# Per-class metrics
print("\n========== PER-CLASS RESULTS ==========")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(7)),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=CLASSES,
        zero_division=0
    )
)


with open(
    OUTPUT_DIR / "transformer_7class_per_class_metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "support"
    ])

    for class_name in CLASSES:

        row = report[class_name]

        writer.writerow([
            class_name,
            row["precision"],
            row["recall"],
            row["f1-score"],
            int(row["support"])
        ])


# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(7))
)

with open(
    OUTPUT_DIR / "transformer_7class_confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(["Actual / Predicted"] + CLASSES)

    for i, class_name in enumerate(CLASSES):
        writer.writerow([class_name] + cm[i].tolist())


# Summary
with open(
    OUTPUT_DIR / "transformer_7class_test_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(["metric", "value"])
    writer.writerow(["test_videos", len(test_dataset)])
    writer.writerow(["test_loss", test_loss])
    writer.writerow(["accuracy", accuracy])
    writer.writerow(["macro_precision", precision])
    writer.writerow(["macro_recall", recall])
    writer.writerow(["macro_f1", f1])


print("\nTransformer evaluation completed.")
print("Results saved in experiments/")
