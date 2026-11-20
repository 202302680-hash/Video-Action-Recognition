import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(".")
SOURCE = ROOT / "data" / "splits"
DEST = ROOT / "data" / "splits_7class"

CLASSES = [
    "ApplyEyeMakeup",
    "Biking",
    "CliffDiving",
    "Diving",
    "PlayingGuitar",
    "PlayingViolin",
    "SoccerJuggling",
]

CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASSES)}
SPLITS = ["train", "val", "test"]

DEST.mkdir(parents=True, exist_ok=True)

groups_by_split = {}
counts_by_split = {}

for split in SPLITS:
    source_file = SOURCE / f"{split}.csv"
    output_file = DEST / f"{split}.csv"

    if not source_file.exists():
        raise FileNotFoundError(f"Missing split file: {source_file}")

    with open(source_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    required = {"class_name", "label", "filepath"}
    if not required.issubset(set(fieldnames or [])):
        raise ValueError(
            f"{split}.csv must contain columns: {sorted(required)}"
        )

    filtered_rows = []
    class_counts = Counter()
    class_groups = {name: set() for name in CLASSES}

    for row in rows:
        class_name = row["class_name"]

        if class_name not in CLASS_TO_LABEL:
            continue

        row["label"] = str(CLASS_TO_LABEL[class_name])
        filtered_rows.append(row)
        class_counts[class_name] += 1

        match = re.search(r"_g(\d+)_", row["filepath"])
        if match:
            class_groups[class_name].add(match.group(1))
        else:
            raise ValueError(
                f"Could not identify group ID: {row['filepath']}"
            )

    missing_classes = [
        name for name in CLASSES if class_counts[name] == 0
    ]
    if missing_classes:
        raise ValueError(
            f"Missing classes in {split}: {missing_classes}"
        )

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    groups_by_split[split] = class_groups
    counts_by_split[split] = class_counts

    print(f"\n{split.upper()}: {len(filtered_rows)} videos")
    for name in CLASSES:
        print(f"  {name:20s}: {class_counts[name]}")

# Verify no group leakage between splits, separately for each class
print("\nCHECKING GROUP LEAKAGE...")

for class_name in CLASSES:
    train_groups = groups_by_split["train"][class_name]
    val_groups = groups_by_split["val"][class_name]
    test_groups = groups_by_split["test"][class_name]

    train_val = train_groups & val_groups
    train_test = train_groups & test_groups
    val_test = val_groups & test_groups

    if train_val or train_test or val_test:
        raise RuntimeError(
            f"Group leakage found in {class_name}: "
            f"train-val={train_val}, "
            f"train-test={train_test}, "
            f"val-test={val_test}"
        )

print("No group leakage detected.")

# Save the new class mapping
mapping_file = DEST / "class_mapping_7class.csv"
with open(mapping_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "class_name"])
    for name, label in CLASS_TO_LABEL.items():
        writer.writerow([label, name])

total = sum(sum(counts.values()) for counts in counts_by_split.values())

print("\n" + "=" * 45)
print("7-CLASS SPLITS CREATED SUCCESSFULLY")
print(f"Total videos: {total}")
print(f"Output folder: {DEST}")
print("Original 10-class split files were not changed.")
print("=" * 45)
