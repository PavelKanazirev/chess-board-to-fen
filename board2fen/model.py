"""A small convolutional neural network (CNN) that classifies one 32x32 grayscale square."""

import keras
from keras import layers

from board2fen.labels import NUM_CLASSES


def build_model(img_size: int = 32, num_classes: int = NUM_CLASSES) -> keras.Model:
    """Input: raw pixels 0..255, shape (batch, img_size, img_size, 1). Output: class probabilities.

    Three blocks of [convolution -> batch normalization -> ReLU] x 2 + max pooling turn the
    32x32 image into 64 feature maps of 4x4. Global average pooling and one dense layer then
    pick the class.
    """
    inputs = keras.Input((img_size, img_size, 1), name="square")
    x = layers.Rescaling(1 / 255)(inputs)  # pixels 0..255 -> 0..1, done inside the model
    for filters in (16, 32, 64):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)  # halve width and height: 32 -> 16 -> 8 -> 4
    x = layers.GlobalAveragePooling2D()(x)  # 64 feature maps -> 64 numbers
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    return keras.Model(inputs, outputs, name="square_classifier")


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Attach the optimizer, the loss to minimise and the metric to report."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
