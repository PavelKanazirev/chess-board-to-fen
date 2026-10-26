"""Feeding the dataset to Keras: batching, shuffling and random augmentation."""

import keras
import numpy as np
import tensorflow as tf
from keras import layers


def augmentation(seed: int = 0) -> keras.Sequential:
    """Random changes applied to every training batch, so no two epochs see identical squares.

    Used only while training. The model itself stays a plain classifier, which keeps it simple
    to export to TensorFlow Lite and ONNX later.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),  # real knights face either way
            layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed),  # a slightly-off cut
            layers.RandomContrast(0.3, seed=seed),
            layers.RandomBrightness(0.2, value_range=(0, 255), seed=seed),
        ],
        name="augmentation",
    )


def make_dataset(
    squares: np.ndarray, labels: np.ndarray, batch_size: int = 256, training: bool = False, seed: int = 0
) -> tf.data.Dataset:
    """A tf.data pipeline of (batch of squares, batch of labels) for ``model.fit``.

    Squares stay uint8 until a batch is prepared, which needs 4x less memory than float32.
    Training data is shuffled and augmented; validation data is passed through unchanged.
    """
    dataset = tf.data.Dataset.from_tensor_slices((squares, labels))
    if training:
        dataset = dataset.shuffle(min(len(labels), 50_000), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    augment = augmentation(seed) if training else None

    def prepare(x, y):
        x = tf.cast(x, tf.float32)[..., tf.newaxis]  # (batch, 32, 32) uint8 -> (batch, 32, 32, 1)
        if augment is not None:
            x = augment(x, training=True)
            x += tf.random.normal(tf.shape(x), stddev=4.0)  # sensor noise, in grey levels
            x = tf.clip_by_value(x, 0.0, 255.0)
        return x, tf.cast(y, tf.int32)

    return dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
