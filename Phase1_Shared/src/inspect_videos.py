
import csv
import cv2
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "UCF-101"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "train.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "video_inspection.csv"

random.seed(42)

with open(SPLIT_FILE, "r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

classes = sorted(set(row["class_name"] for row in rows))
selected = []

for class_name in classes:
    class_rows = [r for r in rows if r["class_name"] == class_name]
    selected.extend(random.sample(class_rows, min(10, len(class_rows))))

results = []

for i, row in enumerate(selected, 1):
    video_path = DATA_ROOT / row["filepath"]

    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()

    if opened:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        success, frame = cap.read()
        readable = success and frame is not None
    else:
        fps = frame_count = width = height = 0
        readable = False

    cap.release()

    results.append({
        "filepath": row["filepath"],
        "class_name": row["class_name"],
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "readable": readable
    })

    print(f"[{i}/{len(selected)}] {row['filepath']} | Readable={readable}")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n===== INSPECTION SUMMARY =====")
print("Videos inspected:", len(results))
print("Readable:", sum(r["readable"] for r in results))
print("Unreadable:", sum(not r["readable"] for r in results))
print("Report saved to:", OUTPUT_FILE)
