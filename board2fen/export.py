"""Converting the trained Keras model into files for edge devices."""

from pathlib import Path

import numpy as np

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
