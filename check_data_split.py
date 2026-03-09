"""
Validates data_split folder: structure and image counts.
Expected: 4000 total (1000 per class), TRAIN 2800 / VAL 600 / TEST 600.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_SPLIT = PROJECT_ROOT / "data_split"

# Expected (train, val, test) per class — must match prepare_data.py
EXPECTED = {
    "Eosinophil": (700, 150, 150),
    "Lymphocyte": (700, 150, 150),
    "Monocyte": (700, 150, 150),
    "Neutrophil": (700, 150, 150),
}

SPLITS = ("TRAIN", "VAL", "TEST")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def count_images(folder: Path) -> int:
    """Count image files in folder (by extension)."""
    if not folder.exists():
        return -1
    return sum(
        1 for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_data_split() -> bool:
    all_ok = True
    split_totals = [0, 0, 0]

    if not DATA_SPLIT.exists():
        print(f"ERROR: data_split folder not found: {DATA_SPLIT}")
        return False

    print("Checking data_split (expected: 4000 images, 70% train / 15% val / 15% test)\n")
    print(f"{'Class':<12} {'TRAIN':>6} {'VAL':>6} {'TEST':>6}  {'Status':<8}")
    print("-" * 45)

    for class_name, (exp_train, exp_val, exp_test) in EXPECTED.items():
        row_ok = True
        counts = []
        for i, (split, exp) in enumerate(zip(SPLITS, (exp_train, exp_val, exp_test))):
            folder = DATA_SPLIT / split / class_name
            n = count_images(folder)
            if n < 0:
                print(f"  ERROR: missing folder {folder}")
                row_ok = False
                all_ok = False
                n = 0
            counts.append(n)
            split_totals[i] += n
            if n != exp:
                row_ok = False
                all_ok = False

        status = "OK" if row_ok else "MISMATCH"
        print(f"{class_name:<12} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}  {status:<8}")

    print("-" * 45)
    total_count = sum(split_totals)
    total_ok = total_count == 4000 and split_totals == [2800, 600, 600]
    if not total_ok:
        all_ok = False
    print(f"{'TOTAL':<12} {split_totals[0]:>6} {split_totals[1]:>6} {split_totals[2]:>6}  {'OK' if total_ok else 'MISMATCH'}")

    return all_ok


if __name__ == "__main__":
    ok = check_data_split()
    print("\n" + ("Data split is correct." if ok else "Data split has errors. Run prepare_data.py to fix."))
    exit(0 if ok else 1)
