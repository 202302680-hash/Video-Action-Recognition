
import csv
from pathlib import Path

import numpy as np
import torch

from src.dataset_resnet import create_resnet_dataloaders
from src.models.cnn_lstm_resnet import CNNLSTMResNet


# =====================================
# 1. Configuration
# =====================================

BATCH_SIZE = 4
NUM_CLASSES = 10

CHECKPOINT_PATH = Path(
    "checkpoints/resnet_lstm_augmented_best.pth"
)

EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "ApplyEyeMakeup",
    "Basketball",
    "Biking",
    "CliffDiving",
    "CricketBowling",
    "Diving",
    "PlayingGuitar",
    "PlayingViolin",
    "SoccerJuggling",
    "TennisSwing"
]


# =====================================
# 2. Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n===== AUGMENTED RESNET18 + LSTM TEST EVALUATION =====")
print("Device:", device)


# =====================================
# 3. Load Test Data
# =====================================

_, _, test_loader = create_resnet_dataloaders(
    batch_size=BATCH_SIZE
)

print("Test videos:", len(test_loader.dataset))


# =====================================
# 4. Load Best Model
# =====================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

model = CNNLSTMResNet(
    num_classes=NUM_CLASSES,
    hidden_size=128,
    freeze_backbone=True
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Augmented checkpoint loaded successfully.")


# =====================================
# 5. Test Predictions
# =====================================

all_predictions = []
all_targets = []

total_loss = 0.0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():

    for batch_idx, (videos, labels) in enumerate(test_loader):

        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_targets.extend(
            labels.cpu().numpy().tolist()
        )

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Processed batches: "
                f"{batch_idx + 1}/{len(test_loader)}"
            )


# =====================================
# 6. Calculate Metrics
# =====================================

y_true = np.array(all_targets)
y_pred = np.array(all_predictions)

confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

for actual, predicted in zip(y_true, y_pred):
    confusion[actual, predicted] += 1

accuracy = np.mean(y_true == y_pred)

total = len(y_true)
test_loss = total_loss / total

class_precision = []
class_recall = []
class_f1 = []
class_accuracy = []

for i in range(NUM_CLASSES):

    tp = confusion[i, i]
    fp = confusion[:, i].sum() - tp
    fn = confusion[i, :].sum() - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    support = confusion[i, :].sum()

    per_class_accuracy = (
        tp / support
        if support > 0
        else 0.0
    )

    class_precision.append(precision)
    class_recall.append(recall)
    class_f1.append(f1)
    class_accuracy.append(per_class_accuracy)


macro_precision = np.mean(class_precision)
macro_recall = np.mean(class_recall)
macro_f1 = np.mean(class_f1)


# =====================================
# 7. Print Overall Results
# =====================================

print("\n========== OVERALL TEST RESULTS ==========")

print("Test videos:", total)
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")


# =====================================
# 8. Per-Class Results
# =====================================

print("\n========== PER-CLASS RESULTS ==========")

for i, name in enumerate(CLASS_NAMES):

    support = int(confusion[i, :].sum())

    print(
        f"{name:20s} | "
        f"Accuracy: {class_accuracy[i]:.4f} | "
        f"Precision: {class_precision[i]:.4f} | "
        f"Recall: {class_recall[i]:.4f} | "
        f"F1: {class_f1[i]:.4f} | "
        f"Support: {support}"
    )


# =====================================
# 9. Save Confusion Matrix
# =====================================

confusion_path = (
    EXPERIMENT_DIR /
    "resnet_lstm_augmented_confusion_matrix.csv"
)

with open(
    confusion_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        ["Actual / Predicted"] + CLASS_NAMES
    )

    for i, name in enumerate(CLASS_NAMES):
        writer.writerow(
            [name] + confusion[i].tolist()
        )


# =====================================
# 10. Save Per-Class Metrics
# =====================================

metrics_path = (
    EXPERIMENT_DIR /
    "resnet_lstm_augmented_per_class_metrics.csv"
)

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "support"
    ])

    for i, name in enumerate(CLASS_NAMES):

        writer.writerow([
            name,
            class_accuracy[i],
            class_precision[i],
            class_recall[i],
            class_f1[i],
            int(confusion[i, :].sum())
        ])


# =====================================
# 11. Save Overall Metrics
# =====================================

overall_path = (
    EXPERIMENT_DIR /
    "resnet_lstm_augmented_test_results.csv"
)

with open(
    overall_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(["metric", "value"])

    writer.writerows([
        ["test_videos", total],
        ["test_loss", test_loss],
        ["accuracy", accuracy],
        ["macro_precision", macro_precision],
        ["macro_recall", macro_recall],
        ["macro_f1", macro_f1]
    ])


# =====================================
# 12. Final Summary
# =====================================

print("\n========== EVALUATION COMPLETED ==========")

print("Overall results:", overall_path)
print("Per-class metrics:", metrics_path)
print("Confusion matrix:", confusion_path)

print("\nTest evaluation finished successfully.")
