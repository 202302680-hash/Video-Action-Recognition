import csv
import random
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("data/raw/UCF-101")
SPLIT_DIR = Path("data/splits/official/ucfTrainTestlist")
OUTPUT_DIR = Path("data/splits")

CLASSES = [
    "ApplyEyeMakeup", "Basketball", "Biking", "CliffDiving",
    "CricketBowling", "Diving", "PlayingGuitar", "PlayingViolin",
    "SoccerJuggling", "TennisSwing"
]

def read_list(filename, has_label):
    rows = []
    for line in (SPLIT_DIR / filename).read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue

        relative_path = parts[0]
        class_name = relative_path.split("/")[0]

        if class_name not in CLASSES:
            continue

        match = re.search(r"_g(\d+)_c\d+\.avi$", relative_path)
        if not match:
            raise ValueError(f"Cannot extract group ID: {relative_path}")

        full_path = ROOT / Path(relative_path)
        if not full_path.is_file():
            raise FileNotFoundError(full_path)

        rows.append({
            "filepath": relative_path,
            "class_name": class_name,
            "label": CLASSES.index(class_name),
            "group": match.group(1)
        })

    return rows

official_train = read_list("trainlist01.txt", True)
test_rows = read_list("testlist01.txt", False)

train_rows = []
val_rows = []

rng = random.Random(42)

for class_name in CLASSES:
    class_rows = [
        row for row in official_train
        if row["class_name"] == class_name
    ]

    groups = sorted(set(row["group"] for row in class_rows))
    rng.shuffle(groups)

    val_count = max(1, round(len(groups) * 0.20))
    val_groups = set(groups[:val_count])

    for row in class_rows:
        if row["group"] in val_groups:
            val_rows.append(row)
        else:
            train_rows.append(row)

def check_overlap(a, b, name):
    groups_a = {(r["class_name"], r["group"]) for r in a}
    groups_b = {(r["class_name"], r["group"]) for r in b}
    overlap = groups_a & groups_b
    print(f"{name} group overlap: {len(overlap)}")
    if overlap:
        raise RuntimeError(f"Group leakage detected: {overlap}")

check_overlap(train_rows, val_rows, "Train / Validation")
check_overlap(train_rows, test_rows, "Train / Test")
check_overlap(val_rows, test_rows, "Validation / Test")

def save_csv(filename, rows):
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "class_name", "label", "group"]
        )
        writer.writeheader()
        writer.writerows(rows)

save_csv("train.csv", train_rows)
save_csv("val.csv", val_rows)
save_csv("test.csv", test_rows)

print("\n===== FINAL SPLIT =====")
print("Train:", len(train_rows))
print("Validation:", len(val_rows))
print("Test:", len(test_rows))
print("Total:", len(train_rows) + len(val_rows) + len(test_rows))

print("\n===== PER CLASS =====")
for class_name in CLASSES:
    tr = sum(r["class_name"] == class_name for r in train_rows)
    va = sum(r["class_name"] == class_name for r in val_rows)
    te = sum(r["class_name"] == class_name for r in test_rows)
    print(f"{class_name}: Train={tr}, Val={va}, Test={te}")

print("\nSaved: train.csv, val.csv, test.csv")
