"""
Classify a single image using the trained blood cell model.
Usage: python classify.py [path_to_image]
  If no path is given, prompts for one.
"""
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Paths and labels (must match training.py)
MODEL_PATH = "saved_model/blood_cell_model.keras"
IMAGE_SIZE = (150, 150)
# Class order matches flow_from_directory (alphabetical folder names)
CLASS_LABELS = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


def classify_image(img_path: str) -> None:
    model = tf.keras.models.load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    predicted_label = CLASS_LABELS[predicted_class]
    confidence = float(predictions[0][predicted_class])

    print("Predicted class:", predicted_label)
    print("Confidence: {:.2f}%".format(confidence * 100))


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter image path: ").strip()

    if not img_path:
        print("No image path provided.")
        sys.exit(1)

    classify_image(img_path)
