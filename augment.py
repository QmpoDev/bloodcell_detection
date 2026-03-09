"""
Optional data-augmentation script (professor-provided).

NOT used for this project's training: our images already come from an augmented
dataset (dataset2-master: rotated, scaled, flipped). training.py does not use this.

Use augment.py only when you have a different dataset where:
- Some classes have fewer images than others (class imbalance), and
- You want to balance by creating extra augmented copies (flips, rotations, color)
  until every class has as many images as the largest class.

Modes: 'camera' (flips, 90° rotations, color tweaks) or 'xray' (small rotations only).
Requires: pip install pillow
"""
import os
import random
from PIL import Image, ImageEnhance

dataset_path = input("Enter dataset path: ")
mode = input("Image type (camera/xray): ").lower()

classes = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c))]

# count images
counts = {}
for c in classes:
    path = os.path.join(dataset_path, c)
    counts[c] = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

max_count = max(counts.values())


def augment(img):
    if mode == "camera":
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            img = img.rotate(random.choice([90, 180, 270]))

        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    elif mode == "xray":
        # minimal rotation only
        img = img.rotate(random.uniform(-5, 5))

    return img


for c in classes:
    class_path = os.path.join(dataset_path, c)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    while len(images) < max_count:
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)

        img = Image.open(img_path)
        new_img = augment(img)

        new_name = f"aug_{random.randint(10000, 99999)}.jpg"
        new_img.save(os.path.join(class_path, new_name))

        images.append(new_name)

print("Dataset balanced.")
