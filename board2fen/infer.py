"""Running a trained square classifier, independent of the runtime that executes it.

Today the only backend is Keras. TensorFlow Lite and ONNX Runtime backends come next; code that
uses a ``SquareClassifier`` won't have to change.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class SquareClassifier(ABC):
    """Turns square images into class probabilities."""

    img_size: int  # the side length of the squares the model expects

    @abstractmethod
    def predict(self, squares: np.ndarray) -> np.ndarray:
        """uint8 squares of shape (N, img_size, img_size) -> probabilities of shape (N, classes)."""


class KerasClassifier(SquareClassifier):
    """Runs a ``.keras`` model with TensorFlow."""

    def __init__(self, path: Path):
        import keras  # imported here, so the rest of board2fen works without TensorFlow

        self.model = keras.models.load_model(path)
        self.img_size = int(self.model.input_shape[1])

    def predict(self, squares: np.ndarray) -> np.ndarray:
        x = squares[..., None].astype(np.float32)  # add the channel axis Keras expects
        return np.asarray(self.model(x, training=False))


def load_classifier(path: str | Path) -> SquareClassifier:
    """Load a model file, choosing the backend from its extension."""
    path = Path(path)
    if path.suffix == ".keras":
        return KerasClassifier(path)
    raise ValueError(f"unsupported model format {path.suffix!r} (expected .keras)")
