
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments"
OUTPUT = ROOT / "report" / "Error_Analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "CNN_LSTM": {
        42: "cnn_lstm_7class_confusion_matrix.csv",
        43: "cnn_lstm_7class_seed43_confusion_matrix.csv",
        44: "cnn_lstm_7class_seed44_confusion_matrix.csv",
    },
    "ResNet18_LSTM": {
        42: "resnet_lstm_7class_confusion_matrix.csv",
        43: "resnet_lstm_7class_seed43_confusion_matrix.csv",
        44: "resnet_lstm_7class_seed44_confusion_matrix.csv",
    },
    "ResNet18_Transformer": {
        42: "transformer_7class_improved_confusion_matrix.csv",
        43: "transformer_7class_seed43_confusion_matrix.csv",
        44: "transformer_7class_seed44_confusion_matrix.csv",
    },
}


def read_matrix(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    rows = [
        [cell.strip() for cell in row]
        for row in rows
        if any(cell.strip() for cell in row)
    ]

    if len(rows) < 2:
        raise ValueError("Confusion matrix is empty.")

    class_names = rows[0][1:]
    actual_names = []
    matrix = []

    for row in rows[1:]:
        if len(row) < len(class_names) + 1:
            continue

        actual_names.append(row[0])
        matrix.append([
            int(float(value))
            for value in row[1:len(class_names) + 1]
        ])

    if not matrix or len(matrix) != len(class_names):
        raise ValueError("Unexpected confusion matrix format.")

    return class_names, actual_names, matrix


def analyze(file_path, model, seed):
    classes, actual_names, matrix = read_matrix(file_path)

    total = sum(sum(row) for row in matrix)
    correct = 0
    errors = []

    for i, actual in enumerate(actual_names):
        for j, predicted in enumerate(classes):
            count = matrix[i][j]

            if actual == predicted:
                correct += count
            elif count > 0:
                errors.append({
                    "model": model,
                    "seed": seed,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "error_count": count,
                })

    wrong = total - correct

    summary = {
        "model": model,
        "seed": seed,
        "total_samples": total,
        "correct_predictions": correct,
        "errors": wrong,
        "accuracy_pct": round(correct / total * 100, 2) if total else 0,
        "error_rate_pct": round(wrong / total * 100, 2) if total else 0,
    }

    return summary, errors


def save_csv(filename, data, columns):
    path = OUTPUT / filename

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)

    print(f"Created: {path}")


summaries = []
all_errors = []

for model, seed_files in MODELS.items():
    for seed, filename in seed_files.items():
        file_path = EXPERIMENTS / filename

        if not file_path.exists():
            print(f"Missing file: {filename}")
            continue

        try:
            summary, errors = analyze(file_path, model, seed)
            summaries.append(summary)
            all_errors.extend(errors)

            print(
                f"{model} | Seed {seed} | "
                f"Accuracy: {summary['accuracy_pct']}% | "
                f"Errors: {summary['errors']}"
            )

        except Exception as e:
            print(f"Could not analyze {filename}: {e}")


if summaries:
    save_csv(
        "error_summary.csv",
        summaries,
        [
            "model", "seed", "total_samples",
            "correct_predictions", "errors",
            "accuracy_pct", "error_rate_pct"
        ]
    )

if all_errors:
    all_errors.sort(
        key=lambda item: item["error_count"],
        reverse=True
    )

    save_csv(
        "misclassification_details.csv",
        all_errors,
        [
            "model", "seed", "actual_class",
            "predicted_class", "error_count"
        ]
    )

print("\nERROR ANALYSIS FINISHED")
print(f"Results folder: {OUTPUT}")
