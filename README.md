# EventDP — Blood Cell Image Classification

Event-Driven Programming lab project: **image classification of blood cell types** (Eosinophil, Lymphocyte, Monocyte, Neutrophil) using the [Blood Cell Images / BCCD](https://github.com/Shenggan/BCCD_Dataset) dataset (dataset2-master). The pipeline prepares data, trains a CNN (TensorFlow/Keras), and classifies single images.

**Repo:** [github.com/QmpoDev/bloodcell_detection](https://github.com/QmpoDev/bloodcell_detection)

---

## Project structure

```
bloodcell_detection/
├── data_raw/                    # [YOU ADD] Original dataset; not in repo. Place dataset2-master here.
│   └── dataset2-master/
│       └── dataset2-master/
│           └── images/
│               ├── TRAIN/       # EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL
│               └── TEST/        # same 4 class folders
│
├── data_split/                  # [GENERATED] Created by prepare_data.py. Do not commit.
│   ├── TRAIN/
│   ├── VAL/
│   └── TEST/                    # each with Eosinophil, Lymphocyte, Monocyte, Neutrophil
│
├── saved_model/                 # [GENERATED] Trained model saved here by training.py. Do not commit.
│   └── blood_cell_model.keras
│
├── prepare_data.py              # Builds data_split from data_raw.
│                                # Default: 4000 images (1000/class). --full: use all ~12.5k.
├── check_data_split.py          # Validates data_split. Default: expect 4000. --full: report counts only.
├── training.py                  # Trains CNN; auto-detects train/val size (works for 4000 or full).
├── classify.py                  # Classify one image: python classify.py path/to/image.jpg
├── augment.py                   # Optional: balance another dataset (class imbalance). Not used in main pipeline.
├── README.md
└── Paul Emmanuelle Quimpo - Laboratory 2 - Data Preparation.pdf   # Lab instructions
```

**Script summary**

| Script | Purpose |
|--------|--------|
| `prepare_data.py` | Copy images from `data_raw` into `data_split` with 70% train / 15% val / 15% test (subset) or 70% train / 30% val from TRAIN + 100% TEST (full). Seed 42 for reproducibility. |
| `check_data_split.py` | Check folder structure and image counts. Without `--full`: expect 2800/600/600. With `--full`: print counts only. |
| `training.py` | Load data from `data_split`, train CNN (early stopping, dropout, LR 1e-4), save best model to `saved_model/blood_cell_model.keras`. |
| `classify.py` | Load saved model and predict class + confidence for one image. |
| `augment.py` | Optional; for other datasets with class imbalance. Not used for this project’s training. |

---

## Dataset

- **Source:** Blood Cell Images (dataset2-master). Images are **already augmented** (rotated, scaled, flipped) in the source.
- **Classes:** Eosinophil, Lymphocyte, Monocyte, Neutrophil (4 classes).
- **No data leakage:** Training and validation are sampled from the original **TRAIN** folder; the **TEST** set comes only from the original **TEST** folder so evaluation is on unseen data.

### Two modes for data_split

1. **Subset (default)** — 4000 images total  
   - 1000 per class.  
   - Split: 700 train / 150 val / 150 test per class → **2800 train, 600 val, 600 test**.  
   - Use when you want faster iteration or limited data.

2. **Full dataset** — `python prepare_data.py --full`  
   - Uses **all** images in `data_raw`: 70% of TRAIN → train, 30% of TRAIN → val, 100% of TEST → test (~12.5k total: ~6.9k train, ~3k val, ~2.5k test).  
   - Use for best accuracy; training takes longer.

### Required layout in data_raw

After you add the dataset, this path must exist:

```
data_raw/dataset2-master/dataset2-master/images/
├── TRAIN/
│   ├── EOSINOPHIL/
│   ├── LYMPHOCYTE/
│   ├── MONOCYTE/
│   └── NEUTROPHIL/
└── TEST/
    ├── EOSINOPHIL/
    ├── LYMPHOCYTE/
    ├── MONOCYTE/
    └── NEUTROPHIL/
```

The dataset is not in the repo; obtain it from the Blood Cell Images / BCCD source.

---

## Design choices (for context)

- **Lab background:** Lab 2 (Data Preparation) suggested 500–700 images; this repo supports a 4000 subset and a full-dataset mode.
- **Overfitting:** With small data, the model can hit 100% training accuracy but ~25% test (random). The training script uses **early stopping** (patience 8, restore best weights), **dropout** (0.25 after conv, 0.5 before output), and **lower learning rate** (1e-4) to improve generalization.
- **Augmentation:** Training images are already augmented in dataset2-master; no extra augmentation script is run in the main pipeline. `augment.py` exists for balancing other datasets (class imbalance) only.
- **Training size:** `training.py` **auto-detects** the number of train/val images from `data_split`, so the same script works for both 4000 and full dataset.

---

## Setup

1. **Python:** 3.10+ recommended.
2. **Dependencies:** TensorFlow and Pillow (Keras uses Pillow to load images).
   ```bash
   pip install tensorflow pillow
   ```
3. **Dataset:** Place dataset2-master so `data_raw/dataset2-master/dataset2-master/images/` exists with `TRAIN/` and `TEST/` and the four class folders as above.
4. **(Optional)** Virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate   # Linux / macOS
   ```

**Note:** On Windows, TensorFlow ≥ 2.11 does not use GPU (CPU only unless using WSL2 or DirectML). Training is slower but works.

---

## Cloning and setting up on another PC

```bash
git clone https://github.com/QmpoDev/bloodcell_detection.git
cd bloodcell_detection

python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux / macOS

pip install tensorflow pillow
```

Then add the dataset under `data_raw/` as above, and run the usage steps below.

---

## Usage

### 1. Build data_split (choose one)

**4000-image subset:**
```bash
python prepare_data.py
```

**Full dataset (~12.5k):**
```bash
python prepare_data.py --full
```

Running again overwrites `data_split/`. Seed is fixed (42) so the same command gives the same split.

### 2. Verify data_split (optional)

**If you used the 4000 subset:**
```bash
python check_data_split.py
```
Exits 0 if counts are 2800 / 600 / 600.

**If you used --full:**
```bash
python check_data_split.py --full
```
Prints per-class and total counts (no pass/fail).

### 3. Train the model

```bash
python training.py
```

- Reads from `data_split/TRAIN` and `data_split/VAL` (sizes auto-detected).
- Saves the best model (by validation loss) to `saved_model/blood_cell_model.keras`.
- Reports test accuracy on `data_split/TEST` at the end.

### 4. Classify an image

```bash
python classify.py path/to/image.jpg
```

Or run `python classify.py` and enter the path when prompted. Output: predicted class (Eosinophil, Lymphocyte, Monocyte, Neutrophil) and confidence.

### 5. Optional: augment.py

Only for **other** datasets where some classes have fewer images. Not used in this project’s training (our data are already augmented and balanced in data_split).

```bash
pip install pillow   # if not already
python augment.py
```
Enter dataset path and type (`camera` for blood/microscope, `xray` for X-ray). Use `camera` for blood cells.

---

## Troubleshooting

- **"Could not import PIL.Image"** when running `training.py` → install Pillow: `pip install pillow`.
- **"No images in data_split/TRAIN or VAL"** → run `prepare_data.py` (or `prepare_data.py --full`) first.
- **Test accuracy ~25%** with 4000 images → expected with small data; try `prepare_data.py --full` and retrain for better accuracy.
- **Training very slow** → On Windows, TensorFlow uses CPU only; normal. Use fewer epochs or the 4000 subset for quicker runs.

---

## Reference

- **Lab instructions:** *Paul Emmanuelle Quimpo - Laboratory 2 - Data Preparation.pdf* (in repo).
- **Dataset:** [Blood Cell Images / BCCD (dataset2-master)](https://github.com/Shenggan/BCCD_Dataset).
- **Repo:** [github.com/QmpoDev/bloodcell_detection](https://github.com/QmpoDev/bloodcell_detection).
