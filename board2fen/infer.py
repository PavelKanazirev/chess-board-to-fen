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


class TFLiteClassifier(SquareClassifier):
    """Runs a ``.tflite`` model with LiteRT, the TensorFlow Lite runtime. No TensorFlow needed."""

    def __init__(self, path: Path, num_threads: int | None = None):
        from ai_edge_litert.interpreter import Interpreter  # the small runtime package

        self.interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.img_size = int(self.input["shape"][1])
        self.batch_size = int(self.input["shape"][0])

    def predict(self, squares: np.ndarray) -> np.ndarray:
        if len(squares) != self.batch_size:  # the file is exported for a batch of 1
            shape = [len(squares), self.img_size, self.img_size, 1]
            self.interpreter.resize_tensor_input(self.input["index"], shape)
            self.interpreter.allocate_tensors()
            self.batch_size = len(squares)
        x = squares[..., None]
        if self.input["dtype"] == np.float32:
            x = x.astype(np.float32)
        else:  # quantized input: stored value = real value / scale + zero_point
            scale, zero_point = self.input["quantization"]
            limits = np.iinfo(self.input["dtype"])
            x = np.clip(np.round(x / scale + zero_point), limits.min, limits.max).astype(self.input["dtype"])
        self.interpreter.set_tensor(self.input["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.output["index"])
        if self.output["dtype"] != np.float32:  # quantized output: back to real probabilities
            scale, zero_point = self.output["quantization"]
            y = (y.astype(np.float32) - zero_point) * scale
        return y


class OnnxClassifier(SquareClassifier):
    """Runs a ``.onnx`` model with ONNX Runtime. No TensorFlow needed."""

    def __init__(self, path: Path, num_threads: int | None = None):
        import onnxruntime as ort  # the runtime package

        options = ort.SessionOptions()
        if num_threads:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = int(self.session.get_inputs()[0].shape[1])

    def predict(self, squares: np.ndarray) -> np.ndarray:
        x = squares[..., None].astype(np.float32)
        return self.session.run(None, {self.input_name: x})[0]


def load_classifier(path: str | Path) -> SquareClassifier:
    """Load a model file, choosing the backend from its extension."""
    path = Path(path)
    if path.suffix == ".keras":
        return KerasClassifier(path)
    if path.suffix == ".tflite":
        return TFLiteClassifier(path)
    if path.suffix == ".onnx":
        return OnnxClassifier(path)

    raise ValueError(f"unsupported model format {path.suffix!r} (expected .keras, .tflite or .onnx)")
