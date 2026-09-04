"""
predict.py
----------
Run inference on a single blood-group image using a trained model.

Usage
-----
    python src/predict.py --image path/to/fingerprint.bmp \
                           --model models/final_model.keras \
                           --labels models/class_names.json
"""

import json
import argparse
import numpy as np
from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_fingerprint, IMG_SIZE
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Predict blood group from an image")
    p.add_argument("--image", type=str, required=True, help="Path to an image")
    p.add_argument("--model", type=str, default="models/final_model.keras")
    p.add_argument("--labels", type=str, default="models/class_names.json")
    return p.parse_args()


def predict_blood_group(image_path: str, model_path: str, labels_path: str):
    with open(labels_path) as f:
        class_names = json.load(f)

    model = load_model(model_path)

    channels = model.input_shape[-1]
    read_mode = cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(image_path, read_mode)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    processed = preprocess_fingerprint(img, IMG_SIZE, channels=channels)
    batch = np.expand_dims(processed, axis=0)  # (1, H, W, 1)

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    ranked = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)

    return {
        "predicted_blood_group": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {name: float(p) for name, p in ranked},
    }


if __name__ == "__main__":
    args = parse_args()
    result = predict_blood_group(args.image, args.model, args.labels)

    print(f"\nPredicted blood group: {result['predicted_blood_group']}")
    print(f"Confidence: {result['confidence']*100:.2f}%\n")
    print("All class probabilities:")
    for name, p in result["all_probabilities"].items():
        print(f"  {name:>4s}: {p*100:6.2f}%")
