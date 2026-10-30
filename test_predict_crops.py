
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "osteoporosis_efficientnet_best_v2.keras"
)

CROP_DIR = os.path.join(BASE_DIR, "test_crops")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")


def predict_knee(image_path, label):
    if not os.path.exists(image_path):
        print(f"{label}: Image not found: {image_path}")
        return

    img = load_img(image_path, target_size=(224, 224), color_mode="rgb")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    probability = float(model.predict(img_array, verbose=0)[0][0])

    if probability >= 0.5:
        prediction = "Osteoporosis"
        confidence = probability * 100
    else:
        prediction = "Normal"
        confidence = (1 - probability) * 100

    print(f"{label}")
    print(f"  Prediction: {prediction}")
    print(f"  Model confidence: {confidence:.2f}%")
    print()


predict_knee(
    os.path.join(CROP_DIR, "image_left_knee.png"),
    "Image-left knee"
)

predict_knee(
    os.path.join(CROP_DIR, "image_right_knee.png"),
    "Image-right knee"
)
