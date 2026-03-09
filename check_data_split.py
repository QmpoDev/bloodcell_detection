"""
Validates data_split folder: structure and image counts.
Default: expects 4000 subset (2800 train / 600 val / 600 test).
Use --full to only report counts (for full-dataset split).
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SPLIT = PROJECT_ROOT / "data_split"

CLASSES = ("Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil")
SPLITS = ("TRAIN", "VAL", "TEST")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Expected for 4000 subset
EXPECTED_SUBSET = {
    "Eosinophil": (700, 150, 150),
    "Lymphocyte": (700, 150, 150),
    "Monocyte": (700, 150, 150),
    "Neutrophil": (700, 150, 150),
}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return -1
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_subset() -> bool:
    all_ok = True
    split_totals = [0, 0, 0]
    if not DATA_SPLIT.exists():
        print(f"ERROR: data_split folder not found: {DATA_SPLIT}")
        return False
    print("Checking data_split (expected: 4000 images, 70% train / 15% val / 15% test)\n")
    print(f"{'Class':<12} {'TRAIN':>6} {'VAL':>6} {'TEST':>6}  {'Status':<8}")
    print("-" * 45)
    for class_name in CLASSES:
        exp = EXPECTED_SUBSET[class_name]
        row_ok = True
        counts = []
        for i, (split, exp_n) in enumerate(zip(SPLITS, exp)):
            n = count_images(DATA_SPLIT / split / class_name)
            if n < 0:
                n = 0
                row_ok = all_ok = False
            counts.append(n)
            split_totals[i] += n
            if n != exp_n:
                row_ok = all_ok = False
        print(f"{class_name:<12} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}  {'OK' if row_ok else 'MISMATCH':<8}")
    print("-" * 45)
    total_ok = sum(split_totals) == 4000 and split_totals == [2800, 600, 600]
    if not total_ok:
        all_ok = False
    print(f"{'TOTAL':<12} {split_totals[0]:>6} {split_totals[1]:>6} {split_totals[2]:>6}  {'OK' if total_ok else 'MISMATCH'}")
    return all_ok


def check_full() -> bool:
    if not DATA_SPLIT.exists():
        print(f"ERROR: data_split folder not found: {DATA_SPLIT}")
        return False
    print("Data_split counts (full dataset)\n")
    print(f"{'Class':<12} {'TRAIN':>6} {'VAL':>6} {'TEST':>6}")
    print("-" * 40)
    split_totals = [0, 0, 0]
    for class_name in CLASSES:
        counts = []
        for i, split in enumerate(SPLITS):
            n = count_images(DATA_SPLIT / split / class_name)
            if n < 0:
                n = 0
            counts.append(n)
            split_totals[i] += n
        print(f"{class_name:<12} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}")
    print("-" * 40)
    print(f"{'TOTAL':<12} {split_totals[0]:>6} {split_totals[1]:>6} {split_totals[2]:>6}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Report counts only (full dataset)")
    args = parser.parse_args()
    if args.full:
        check_full()
        print("\nDone.")
        exit(0)
    ok = check_subset()
    print("\n" + ("Data split is correct." if ok else "Data split has errors. Run prepare_data.py to fix."))
    exit(0 if ok else 1)
