"""
model.py
--------
Defines the CNN architecture used to classify blood-group images.

The architecture follows a standard deep CNN pattern: stacked
Conv-BatchNorm-ReLU-Pool blocks for visual feature extraction, followed by
dense layers with dropout for classification.
"""

from tensorflow.keras import layers, models, optimizers


def build_cnn(input_shape=(128, 128, 1), num_classes=8, learning_rate=1e-3):
    """
    Build and compile a CNN for blood-group image classification.
    """
    model = models.Sequential(name="blood_group_cnn")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Classifier head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(input_shape=(128, 128, 3), num_classes=8, learning_rate=1e-4):
    """
    Optional stronger baseline: MobileNetV2 backbone pretrained on ImageNet
    with a custom classification head. Requires 3-channel input.
    Useful if the from-scratch CNN in build_cnn() underfits on a small dataset.
    """
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # freeze for initial training; unfreeze later to fine-tune

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="blood_group_mobilenetv2_transfer")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
