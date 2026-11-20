
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent

INPUT_FILE = ROOT / "report" / "Model_Comparison" / "final_3seed_model_comparison.csv"
OUTPUT_DIR = ROOT / "report" / "Model_Comparison"
OUTPUT_FILE = OUTPUT_DIR / "model_comparison_ranked.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

# Read comparison results
with open(INPUT_FILE, "r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Sort by mean accuracy
rows.sort(
    key=lambda row: float(row["accuracy_mean_pct"]),
    reverse=True
)

# Add ranking
for rank, row in enumerate(rows, start=1):
    row["rank"] = rank

# Save ranked results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = ["rank"] + [
    name for name in rows[0].keys() if name != "rank"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# Display comparison
print("\n========== MODEL COMPARISON ==========\n")

for row in rows:
    print(f"Rank: {row['rank']}")
    print(f"Model: {row['model']}")
    print(f"Accuracy: {float(row['accuracy_mean_pct']):.2f}%")
    print(f"Macro F1: {float(row['macro_f1_mean_pct']):.2f}%")
    print("-" * 40)

best = rows[0]

print("\n========== BEST MODEL ==========\n")
print(f"Best Model: {best['model']}")
print(f"Mean Accuracy: {float(best['accuracy_mean_pct']):.2f}%")
print(f"Mean Macro F1: {float(best['macro_f1_mean_pct']):.2f}%")

print(f"\nRanked results saved to: {OUTPUT_FILE}")
