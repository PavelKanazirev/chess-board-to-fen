"""Converting the trained Keras model into files for edge devices."""

import time
from pathlib import Path

import numpy as np

from board2fen.infer import SquareClassifier

QUANTIZATIONS = ("fp32", "fp16", "int8")


def export_tflite(
    model, path: Path, quantization: str = "fp32", calibration: np.ndarray | None = None
) -> Path:
    """Write a Keras ``model`` as a TensorFlow Lite file.

    ``quantization``:
      "fp32"  32-bit floats, the same numbers as Keras
      "fp16"  weights stored as 16-bit floats: half the size, still computed in 32-bit
      "int8"  weights and activations as 8-bit integers, uint8 input and output. Needs
              ``calibration``: typical squares, used to measure the range of every value.
    """
    import tensorflow as tf  # only needed to convert, not to run the result

    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {quantization!r}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        if calibration is None:
            raise ValueError("int8 quantization needs calibration squares")

        def representative_dataset():
            for square in calibration:  # one square at a time, shaped like a batch of 1
                yield [square[None, :, :, None].astype(np.float32)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # raw camera pixels go straight in
        converter.inference_output_type = tf.uint8
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(converter.convert())
    return path


def export_onnx(model, path: Path, opset: int = 17) -> Path:
    """Write a Keras ``model`` as an ONNX file (32-bit floats).

    The input keeps a variable batch size, so one call can classify all 64 squares of a board.
    ``opset`` is the version of the ONNX operator set; 17 is supported by every current runtime.
    """
    import tensorflow as tf  # only needed to convert
    import tf2onnx

    img_size = model.input_shape[1]
    signature = (tf.TensorSpec((None, img_size, img_size, 1), tf.float32, name="square"),)

    @tf.function(input_signature=signature)
    def classify(square):
        return {"probabilities": model(square, training=False)}

    path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(classify, input_signature=signature, opset=opset, output_path=str(path))
    return path


def quantize_onnx(fp32_path: Path, path: Path, calibration: np.ndarray) -> Path:
    """Write an int8 version of an ONNX model, calibrated on typical squares.

    Uses QDQ format: QuantizeLinear/DequantizeLinear pairs around each layer, which runtimes and
    accelerators fuse into integer kernels. The input and output stay float32, so the model is a
    drop-in replacement for the fp32 one.

    ``reduce_range`` stores weights in 7 instead of 8 bits. On x86 CPUs with AVX2 but without VNNI
    (e.g. Intel 10th generation laptops), ONNX Runtime's uint8 x int8 kernels can saturate in
    16-bit intermediate sums; measured here, accuracy drops from 100% to 96.8% without it.
    """
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class SquareReader(CalibrationDataReader):
        def __init__(self):
            squares = calibration[..., None].astype(np.float32)
            self.batches = iter(np.array_split(squares, max(1, len(squares) // 32)))

        def get_next(self):
            batch = next(self.batches, None)
            return None if batch is None else {"square": batch}

    path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        onnx.load(str(fp32_path)),  # an in-memory model: avoids temporary files next to the input
        str(path),
        SquareReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
    )
    return path


def ms_per_board(classifier: SquareClassifier, squares: np.ndarray, runs: int = 50) -> float:
    """Median time to classify one board (64 squares), after a few warm-up runs."""
    board = squares[:64]
    for _ in range(5):
        classifier.predict(board)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        classifier.predict(board)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))
