import csv
from collections import Counter
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
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

TRAIN_FILE = "data/splits_7class/train.csv"
TEST_FILE = "data/splits_7class/test.csv"

def read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

train_data = read_csv(TRAIN_FILE)
test_data = read_csv(TEST_FILE)

# Find the most frequent training class
counts = Counter(int(row["label"]) for row in train_data)
majority_class = counts.most_common(1)[0][0]

# Predict the same class for every test video
y_true = [int(row["label"]) for row in test_data]
y_pred = [majority_class] * len(y_true)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n========== MAJORITY CLASS BASELINE ==========")
print("Training videos:", len(train_data))
print("Test videos:", len(test_data))
print("Training class counts:", counts)
print("Predicted class:", CLASSES[majority_class])
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1: {f1:.4f}")

print("\n========== PER-CLASS RESULTS ==========")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
))

Path("experiments").mkdir(exist_ok=True)

with open(
    "experiments/majority_baseline_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "predicted_class", "accuracy",
        "macro_precision", "macro_recall", "macro_f1"
    ])
    writer.writerow([
        "Majority Class Baseline",
        CLASSES[majority_class],
        accuracy, precision, recall, f1
    ])

matrix = confusion_matrix(
    y_true, y_pred, labels=list(range(len(CLASSES)))
)

with open(
    "experiments/majority_baseline_confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["Actual/Predicted"] + CLASSES)
    for name, row in zip(CLASSES, matrix):
        writer.writerow([name] + row.tolist())

print("\nBaseline results saved successfully.")
