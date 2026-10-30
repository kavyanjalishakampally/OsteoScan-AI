
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(BASE_DIR, "XRAY2.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_crops")

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = Image.open(IMAGE_PATH).convert("RGB")
width, height = image.size

print("Original image size:", image.size)

# Crop the knee on the left side of the displayed X-ray
image_left = image.crop((
    int(width * 0.08),
    int(height * 0.15),
    int(width * 0.49),
    int(height * 0.90)
))

# Crop the knee on the right side of the displayed X-ray
image_right = image.crop((
    int(width * 0.51),
    int(height * 0.15),
    int(width * 0.92),
    int(height * 0.90)
))

image_left.save(os.path.join(OUTPUT_DIR, "image_left_knee.png"))
image_right.save(os.path.join(OUTPUT_DIR, "image_right_knee.png"))

print("Saved both knee crops in:", OUTPUT_DIR)
print("Next, open test_crops and inspect both images.")
