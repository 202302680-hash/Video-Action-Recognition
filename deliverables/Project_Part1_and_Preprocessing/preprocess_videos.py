
import csv
import cv2
import numpy as np
from pathlib import Path

# =========================
# PREPROCESSING CONFIGURATION
# =========================

NUM_FRAMES = 16
IMAGE_SIZE = 112

SPLITS = ["train", "val", "test"]

OUTPUT_ROOT = Path("data/processed/frames")
REPORT_PATH = Path("data/processed/preprocessing_report.csv")


# =========================
# VIDEO PREPROCESSING
# =========================

def preprocess_video(video_path, output_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return "FAILED_TO_OPEN"

    frames = []

    # Read video sequentially
    while True:

        success, frame = cap.read()

        if not success:
            break

        # Resize frame
        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Convert BGR to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)

    cap.release()

    # Check available frames
    if len(frames) < NUM_FRAMES:
        return "FAILED_NOT_ENOUGH_FRAMES"

    # Uniformly sample 16 frames
    indices = np.linspace(
        0,
        len(frames) - 1,
        NUM_FRAMES,
        dtype=int
    )

    video_array = np.stack(
        [frames[index] for index in indices]
    ).astype(np.uint8)

    # Create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed video
    np.save(output_path, video_array)

    # Verify saved output
    saved = np.load(output_path)

    if saved.shape != (16, 112, 112, 3):
        return "FAILED_VALIDATION"

    if saved.dtype != np.uint8:
        return "FAILED_VALIDATION"

    return "SUCCESS"


# =========================
# MAIN PREPROCESSING
# =========================

def main():

    results = []

    for split in SPLITS:

        csv_path = Path(f"data/splits/{split}.csv")

        with open(
            csv_path,
            newline="",
            encoding="utf-8"
        ) as f:

            rows = list(csv.DictReader(f))

        success_count = 0
        failed_count = 0

        print(f"\n===== PROCESSING {split.upper()} =====")
        print("Videos:", len(rows))

        for i, row in enumerate(rows, 1):

            video_path = Path(row["filepath"])
            class_name = row["class_name"]

            output_path = (
                OUTPUT_ROOT
                / split
                / class_name
                / (video_path.stem + ".npy")
            )

            status = preprocess_video(
                video_path,
                output_path
            )

            if status == "SUCCESS":

                success_count += 1
                output_value = str(output_path)

            else:

                failed_count += 1
                output_value = ""

            results.append({
                "split": split,
                "class_name": class_name,
                "video": str(video_path),
                "status": status,
                "output": output_value
            })

            if i % 50 == 0 or i == len(rows):

                print(
                    f"Progress: {i}/{len(rows)} | "
                    f"Success: {success_count} | "
                    f"Failed: {failed_count}"
                )

        print(f"{split.upper()} finished.")
        print("Success:", success_count)
        print("Failed:", failed_count)

    # Save preprocessing report
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fields = [
            "split",
            "class_name",
            "video",
            "status",
            "output"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n===== FINAL PREPROCESSING SUMMARY =====")

    print("Total videos:", len(results))

    print(
        "Successful:",
        sum(r["status"] == "SUCCESS" for r in results)
    )

    print(
        "Failed:",
        sum(r["status"] != "SUCCESS" for r in results)
    )

    print("Output:", OUTPUT_ROOT)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
