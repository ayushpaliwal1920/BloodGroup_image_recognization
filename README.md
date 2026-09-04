# Blood Group Classification using Image Processing (CNN)

A machine learning project that predicts blood groups from images using a
Convolutional Neural Network.

Images are preprocessed (resized and contrast-enhanced) and classified into
eight blood-group categories. The deployed app uses ONNX Runtime so it does
not require TensorFlow on Streamlit Cloud.

## Project Structure

```
blood_group_detection/
├── dataset/                     # blood-group dataset root
│   ├── train/                   # training class folders
│   └── val/                     # evaluation class folders
├── src/
│   ├── data_preprocessing.py    # loading, CLAHE enhancement, splitting, augmentation
│   ├── model.py                 # CNN architecture (+ optional MobileNetV2 transfer model)
│   ├── train.py                 # full training + evaluation pipeline
│   └── predict.py               # single-image inference
├── models/                      # saved trained models & class label map
├── outputs/                     # training curves, confusion matrix, report
├── requirements.txt
├── requirements-train.txt      # optional local TensorFlow training dependencies
├── app.py                      # Streamlit frontend
└── README.md
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select `ayushpaliwal1920/BloodGroup_image_recognization` and the `main` branch.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

The app loads the committed ONNX model from `models/final_model.onnx` and the
labels from `models/class_names.json`. Streamlit Cloud installs the lightweight
inference dependencies from `requirements.txt`; no secrets or environment
variables are required. TensorFlow is kept separate in `requirements-train.txt`
for local retraining and is not installed by Streamlit Cloud.

## 1. Get a Dataset

The loader expects `train/` and `val/` directories containing class folders
such as `A Positive`, `A Negative`, `AB Positive`, and `O Negative`.

## 2. Install Local Training Dependencies

```bash
pip install -r requirements-train.txt
```

## 3. Train the Model

```bash
python src/train.py --data_dir dataset --model_type transfer --epochs 20 --batch_size 16
```

This will:

- Load and preprocess blood-group images (RGB, resize to 128x128, CLAHE contrast enhancement)
- Train a pretrained MobileNetV2 model with balanced class weighting
- Apply early stopping and learning-rate reduction on plateau
- Save the Keras model to `models/final_model.keras` and export `models/final_model.onnx` for deployment
- Save `models/class_names.json` (label index → blood-group mapping)
- Save `outputs/training_curves.png`, `outputs/confusion_matrix.png`, and `outputs/classification_report.txt`

## 4. Predict on a New Image

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```
Predicted blood group: O+
Confidence: 87.42%

All class probabilities:
   O+:  87.42%
   A+:   6.15%
   B+:   3.20%
   ...
```

## Model Architecture

- 4x [Conv2D → BatchNorm → MaxPool] blocks (32 → 64 → 128 → 256 filters)
- MobileNetV2 backbone with GlobalAveragePooling2D
- Dense(256) → Dropout(0.4) → Dense(8, softmax)
- Optimizer: Adam, Loss: categorical cross-entropy

The original grayscale CNN remains available with `--model_type cnn`, but the
transfer model is the recommended option for this small dataset.

## Notes & Limitations

- **Research status**: This project is educational and is not a validated
  diagnostic tool.
- **Not a medical device**: predictions from this model should never
  replace serological blood typing for transfusion, transplant, or
  clinical decisions.
- **Data quality matters most**: accuracy depends heavily on image quality,
  dataset size, and label correctness. A small or noisy
  dataset will overfit quickly despite the augmentation/dropout/early
  stopping in this pipeline.
- To improve results: increase dataset size, try the MobileNetV2 transfer
  model, tune `IMG_SIZE`/augmentation strength, or ensemble multiple CNN runs.
