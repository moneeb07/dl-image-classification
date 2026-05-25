"""
Download a few STL-10 test images per class and save them to test_images/
so you can upload them into the Streamlit app.

Run once:
    python get_test_images.py
"""
import os
import tensorflow_datasets as tfds
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
N_PER_CLASS = 3   # how many test images to save per class

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

os.makedirs(OUT_DIR, exist_ok=True)
for c in CLASS_NAMES:
    os.makedirs(os.path.join(OUT_DIR, c), exist_ok=True)

print("Downloading STL-10 test split via TensorFlow Datasets...")
ds = tfds.load("stl10", split="test", as_supervised=True)

counts = {i: 0 for i in range(len(CLASS_NAMES))}
saved = 0
for img, lbl in ds:
    c = int(lbl.numpy())
    if counts[c] >= N_PER_CLASS:
        if all(v >= N_PER_CLASS for v in counts.values()):
            break
        continue
    counts[c] += 1
    cls = CLASS_NAMES[c]
    path = os.path.join(OUT_DIR, cls, f"{cls}_{counts[c]}.png")
    Image.fromarray(img.numpy()).save(path)
    saved += 1

print(f"\nSaved {saved} images to: {OUT_DIR}")
print("Folder structure:")
for c in CLASS_NAMES:
    files = sorted(os.listdir(os.path.join(OUT_DIR, c)))
    print(f"  {c}/  ({len(files)} images)")
