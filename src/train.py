"""
train.py
--------
End-to-end training pipeline:
    1. Load & preprocess the blood-group image dataset
  2. Build the CNN
  3. Train with augmentation, early stopping and checkpointing
  4. Evaluate on the held-out test set
  5. Save the trained model, label map and training curves

Usage
-----
    python src/train.py --data_dir dataset --epochs 40 --batch_size 32
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_preprocessing import prepare_data, get_augmentation_generator, IMG_SIZE
from model import build_cnn, build_transfer_model


def parse_args():
    p = argparse.ArgumentParser(description="Train blood-group image classification CNN")
    p.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root folder")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--model_type", choices=("cnn", "transfer"), default="transfer")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--model_dir", type=str, default="models")
    return p.parse_args()


def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Blood Group Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Data
    channels = 3 if args.model_type == "transfer" else 1
    data = prepare_data(args.data_dir, channels=channels)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]
    num_classes = len(class_names)
    train_labels = np.argmax(y_train, axis=1)
    raw_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=train_labels,
    )
    class_weights = np.minimum(np.sqrt(raw_class_weights), 3.0)
    class_weights /= class_weights.mean()
    class_weight_map = dict(enumerate(class_weights))
    print(f"Class weights: {dict(zip(class_names, np.round(class_weights, 2)))}")

    with open(os.path.join(args.model_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # 2. Model
    if args.model_type == "transfer":
        model = build_transfer_model(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            num_classes=num_classes,
            learning_rate=min(args.learning_rate, 1e-4),
        )
    else:
        model = build_cnn(
            input_shape=(IMG_SIZE, IMG_SIZE, 1),
            num_classes=num_classes,
            learning_rate=args.learning_rate,
        )
    model.summary()

    # 3. Callbacks
    checkpoint_path = os.path.join(args.model_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    # 4. Train (with light augmentation on the training set only)
    datagen = get_augmentation_generator()
    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        validation_data=(X_val, y_val),
        class_weight=class_weight_map,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    print(report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\nTest loss: {test_loss:.4f}\n\n")
        f.write(report)

    # 6. Plots
    plot_history(history, args.out_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, args.out_dir)

    # 7. Save final model
    model.save(os.path.join(args.model_dir, "final_model.keras"))
    print(f"\nSaved model to {args.model_dir}/final_model.keras")
    print(f"Saved evaluation artifacts to {args.out_dir}/")


if __name__ == "__main__":
    main()
