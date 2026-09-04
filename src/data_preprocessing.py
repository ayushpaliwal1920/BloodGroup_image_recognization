"""
data_preprocessing.py
----------------------
Handles loading, cleaning, augmenting and splitting the blood-group image
dataset.

Expected dataset layout:

    dataset/
        train/
            A Positive/  *.jpg
            ...
        val/
            A Positive/  *.jpg
            ...

Each folder name is converted to a compact blood-group label, such as
"A Positive" -> "A+".
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
IMG_SIZE = 128          # images are resized to IMG_SIZE x IMG_SIZE
VALID_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
LABEL_NAMES = {
    "A Positive": "A+",
    "A Negative": "A-",
    "B Positive": "B+",
    "B Negative": "B-",
    "AB Positive": "AB+",
    "AB Negative": "AB-",
    "O Positive": "O+",
    "O Negative": "O-",
}


def load_dataset(dataset_dir: str, img_size: int = IMG_SIZE, channels: int = 1):
    """
    Load images from class folders, resize them, normalize pixel values and
    pair them with an integer blood-group label.

    Returns
    -------
    X : np.ndarray  shape (N, img_size, img_size, 1)  float32, scaled 0-1
    y : np.ndarray  shape (N,)                        integer class ids
    class_names : list[str]  sorted list of class labels found in the folders
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: '{dataset_dir}'")

    class_dirs = [directory for directory in root.iterdir() if directory.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found inside '{dataset_dir}'. "
            "Expected folders such as 'A Positive' and 'O Negative'."
        )

    records = []
    for class_dir in class_dirs:
        class_name = LABEL_NAMES.get(class_dir.name, class_dir.name)
        for image_path in class_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                records.append((image_path, class_name))

    if not records:
        raise ValueError(
            f"No images found in class folders under '{dataset_dir}'."
        )

    class_names = sorted({class_name for _, class_name in records})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    images, labels = [], []
    loaded_counts = Counter()
    for image_path, class_name in records:
        read_mode = cv2.IMREAD_COLOR if channels == 3 else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(str(image_path), read_mode)
        if img is None:
            continue
        images.append(preprocess_fingerprint(img, img_size, channels=channels))
        labels.append(class_to_index[class_name])
        loaded_counts[class_name] += 1

    for class_name in class_names:
        print(f"  Loaded {loaded_counts[class_name]:5d} images for class '{class_name}'")

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="int64")
    return X, y, class_names


def preprocess_fingerprint(
    img: np.ndarray, img_size: int = IMG_SIZE, channels: int = 1
) -> np.ndarray:
    """
    Apply the standard image pre-processing pipeline:
      1. Resize to a fixed square shape
      2. CLAHE contrast enhancement to sharpen ridge patterns
      3. Normalize pixel intensities to [0, 1]
      4. Add channel dimension (grayscale => 1 channel)
    """
    img = cv2.resize(img, (img_size, img_size))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if channels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        img = clahe.apply(img)

    img = img.astype("float32") / 255.0
    if channels == 1:
        img = np.expand_dims(img, axis=-1)
    return img


def prepare_data(
    dataset_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    channels: int = 1,
):
    """
    Load, split (train / val / test) and one-hot encode the dataset.

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test, class_names
    """
    print(f"Loading dataset from: {dataset_dir}")
    root = Path(dataset_dir)
    train_dir = root / "train"
    provided_val_dir = root / "val"

    if train_dir.is_dir() and provided_val_dir.is_dir():
        X_train_all, y_train_all, class_names = load_dataset(str(train_dir), channels=channels)
        X_test, y_test, test_class_names = load_dataset(str(provided_val_dir), channels=channels)
        if set(class_names) != set(test_class_names):
            raise ValueError("Train and val folders must contain the same class folders")
        class_to_index = {name: index for index, name in enumerate(class_names)}
        y_test = np.array([class_to_index[test_class_names[index]] for index in y_test], dtype="int64")
        X_train_val, y_train_val = X_train_all, y_train_all
        test_size = 0
    else:
        X, y, class_names = load_dataset(dataset_dir, channels=channels)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )

    num_classes = len(class_names)

    # Use the supplied val split as test data; otherwise split off test data above.
    val_fraction_of_remainder = val_size if test_size == 0 else val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction_of_remainder,
        stratify=y_train_val,
        random_state=seed,
    )

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}  Classes: {class_names}")

    return {
        "X_train": X_train, "y_train": y_train_cat,
        "X_val": X_val, "y_val": y_val_cat,
        "X_test": X_test, "y_test": y_test_cat,
        "class_names": class_names,
    }


def get_augmentation_generator():
    """
    Light augmentation appropriate for fingerprint ridge images.
    Avoid heavy geometric distortion since ridge topology is
    label-relevant information.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.08,
        fill_mode="nearest",
    )
