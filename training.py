import os
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Paths and config (works for 4000 subset or full dataset)
train_dir = "data_split/TRAIN"
val_dir = "data_split/VAL"
test_dir = "data_split/TEST"
batch_size = 32
image_size = (150, 150)
num_classes = 4  # Eosinophil, Lymphocyte, Monocyte, Neutrophil

def _count_images(dir_path: str) -> int:
    p = Path(dir_path)
    if not p.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts)

# Step 2: Load as tf.data.Dataset (avoids "ran out of data" with repeat)
train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True,
    seed=42,
)
val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False,
)
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False,
)

# Rescale pixel values to [0, 1]
normalize = lambda img, label: (img / 255.0, label)
train_ds = train_ds.map(normalize).repeat()
val_ds = val_ds.map(normalize)
test_ds = test_ds.map(normalize)

# Step 3: Model (Input + Dropout to reduce overfitting)
model = keras.Sequential([
    layers.Input(shape=(*image_size, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

# Step 4: Compile (lower LR helps when model stays at random accuracy)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Step 5: Train with early stopping (auto-detect sizes for 4000 subset or full dataset)
train_samples = _count_images(train_dir)
val_samples = _count_images(val_dir)
steps_per_epoch = train_samples // batch_size
validation_steps = max(1, val_samples // batch_size)
if train_samples == 0 or val_samples == 0:
    raise RuntimeError("No images in data_split/TRAIN or data_split/VAL. Run prepare_data.py first.")

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=[early_stop],
)

# Step 6: Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print("\nTest accuracy:", test_acc)

# Step 7: Save model
directory = "saved_model"
os.makedirs(directory, exist_ok=True)
model.save(os.path.join(directory, "blood_cell_model.keras"))
print("Model saved successfully.")
