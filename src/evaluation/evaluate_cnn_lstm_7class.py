
import os
import csv
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.cnn_lstm_7class_tuned import CNNLSTM7Tuned
from src.dataset_resnet_7class import create_resnet7_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    checkpoint_path = "checkpoints/cnn_lstm_7class_tuned_best.pth"

    class_names = [
        "ApplyEyeMakeup",
        "Biking",
        "CliffDiving",
        "Diving",
        "PlayingGuitar",
        "PlayingViolin",
        "SoccerJuggling"
    ]

    print("Device:", device)

    _, _, test_loader = create_resnet7_dataloaders(
        batch_size=batch_size
    )

    model = CNNLSTM7Tuned(num_classes=7).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    test_loss = total_loss / len(test_loader.dataset)
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

    print("\n========== TUNED CNN + LSTM TEST RESULTS ==========")
    print("Test Videos:", len(y_true))
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")

    print("\n========== PER-CLASS RESULTS ==========")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    os.makedirs("experiments", exist_ok=True)

    with open(
        "experiments/cnn_lstm_7class_tuned_test_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_videos",
            "test_loss",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1"
        ])
        writer.writerow([
            len(y_true),
            test_loss,
            accuracy,
            precision,
            recall,
            f1
        ])

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(7))
    )

    with open(
        "experiments/cnn_lstm_7class_tuned_confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Actual/Predicted"] + class_names)

        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name] + row.tolist())

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
