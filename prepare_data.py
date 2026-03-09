"""
Laboratory 2 - Data Preparation
Default: 4000 images (1000 per class), 70% train / 15% val / 15% test.
Full dataset: run with --full to use all images from TRAIN (70% train, 30% val) and TEST (100% test).
"""

import argparse
import shutil
import random
from pathlib import Path

# Reproducible splits
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Paths (project root = folder containing this script)
PROJECT_ROOT = Path(__file__).resolve().parent
ORIGINAL = PROJECT_ROOT / "data_raw" / "dataset2-master" / "dataset2-master" / "images"
DEST = PROJECT_ROOT / "data_split"

# Class names: source folders are UPPERCASE, destination use Title Case
CLASS_MAP = {
    "EOSINOPHIL": "Eosinophil",
    "LYMPHOCYTE": "Lymphocyte",
    "MONOCYTE": "Monocyte",
    "NEUTROPHIL": "Neutrophil",
}

# Subset (4000 images): per-class split 700 train, 150 val, 150 test
SPLIT_PER_CLASS_SUBSET = {
    "EOSINOPHIL": (700, 150, 150),
    "LYMPHOCYTE": (700, 150, 150),
    "MONOCYTE": (700, 150, 150),
    "NEUTROPHIL": (700, 150, 150),
}

TRAIN_SRC = ORIGINAL / "TRAIN"
TEST_SRC = ORIGINAL / "TEST"


def list_images(folder: Path) -> list[Path]:
    """Return list of image files in folder (common extensions)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.upper() in {e.upper() for e in exts}]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_subset() -> None:
    """Prepare 4000-image subset (1000 per class, 70/15/15)."""
    if not TRAIN_SRC.exists():
        raise FileNotFoundError(f"TRAIN folder not found: {TRAIN_SRC}")
    if not TEST_SRC.exists():
        raise FileNotFoundError(f"TEST folder not found: {TEST_SRC}")

    for split in ("TRAIN", "VAL", "TEST"):
        for dest_class in CLASS_MAP.values():
            d = DEST / split / dest_class
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()

    total_copied = 0
    for src_name, dest_name in CLASS_MAP.items():
        n_train, n_val, n_test = SPLIT_PER_CLASS_SUBSET[src_name]
        train_src_dir = TRAIN_SRC / src_name
        test_src_dir = TEST_SRC / src_name
        train_images = list_images(train_src_dir)
        test_images = list_images(test_src_dir)

        if len(train_images) < n_train + n_val:
            raise RuntimeError(f"Class {src_name}: TRAIN has {len(train_images)} images, need {n_train + n_val}")
        if len(test_images) < n_test:
            raise RuntimeError(f"Class {src_name}: TEST has {len(test_images)} images, need {n_test}")

        random.shuffle(train_images)
        random.shuffle(test_images)
        train_selected = train_images[:n_train]
        val_selected = train_images[n_train : n_train + n_val]
        test_selected = test_images[:n_test]

        ensure_dir(DEST / "TRAIN" / dest_name)
        ensure_dir(DEST / "VAL" / dest_name)
        ensure_dir(DEST / "TEST" / dest_name)
        for path in train_selected:
            shutil.copy2(path, DEST / "TRAIN" / dest_name / path.name)
            total_copied += 1
        for path in val_selected:
            shutil.copy2(path, DEST / "VAL" / dest_name / path.name)
            total_copied += 1
        for path in test_selected:
            shutil.copy2(path, DEST / "TEST" / dest_name / path.name)
            total_copied += 1
        print(f"  {dest_name}: train={len(train_selected)}, val={len(val_selected)}, test={len(test_selected)}")

    print(f"\nDone. Total images copied: {total_copied} (expected 4000).")


def prepare_full() -> None:
    """Use full dataset: 70% of TRAIN → train, 30% of TRAIN → val, 100% of TEST → test."""
    if not TRAIN_SRC.exists():
        raise FileNotFoundError(f"TRAIN folder not found: {TRAIN_SRC}")
    if not TEST_SRC.exists():
        raise FileNotFoundError(f"TEST folder not found: {TEST_SRC}")

    for split in ("TRAIN", "VAL", "TEST"):
        for dest_class in CLASS_MAP.values():
            d = DEST / split / dest_class
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()

    total_copied = 0
    for src_name, dest_name in CLASS_MAP.items():
        train_src_dir = TRAIN_SRC / src_name
        test_src_dir = TEST_SRC / src_name
        train_images = list_images(train_src_dir)
        test_images = list_images(test_src_dir)

        random.shuffle(train_images)
        n_train = int(0.70 * len(train_images))
        n_val = len(train_images) - n_train
        train_selected = train_images[:n_train]
        val_selected = train_images[n_train:]
        test_selected = test_images  # use all

        ensure_dir(DEST / "TRAIN" / dest_name)
        ensure_dir(DEST / "VAL" / dest_name)
        ensure_dir(DEST / "TEST" / dest_name)
        for path in train_selected:
            shutil.copy2(path, DEST / "TRAIN" / dest_name / path.name)
            total_copied += 1
        for path in val_selected:
            shutil.copy2(path, DEST / "VAL" / dest_name / path.name)
            total_copied += 1
        for path in test_selected:
            shutil.copy2(path, DEST / "TEST" / dest_name / path.name)
            total_copied += 1
        print(f"  {dest_name}: train={len(train_selected)}, val={len(val_selected)}, test={len(test_selected)}")

    print(f"\nDone. Total images copied: {total_copied} (full dataset, ~12.5k).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data_split from data_raw.")
    parser.add_argument("--full", action="store_true", help="Use full dataset (~12.5k) instead of 4000 subset")
    args = parser.parse_args()

    print("Data preparation - Laboratory 2")
    print(f"Source: {ORIGINAL}")
    print(f"Destination: {DEST}")
    print("Mode: FULL dataset" if args.full else "Mode: 4000 subset")
    print()
    if args.full:
        prepare_full()
    else:
        prepare_subset()
