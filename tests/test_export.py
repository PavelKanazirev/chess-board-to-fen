from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")  # converting needs TensorFlow ("train" extra)
pytest.importorskip("ai_edge_litert")  # running needs LiteRT ("tflite" extra)

from board2fen.dataset import generate  # noqa: E402
from board2fen.evaluation import ms_per_board  # noqa: E402
from board2fen.export import export_tflite  # noqa: E402
from board2fen.infer import load_classifier  # noqa: E402
from board2fen.model import build_model  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "square_classifier.keras"


@pytest.fixture(scope="module")
def small_model():
    keras.utils.set_random_seed(0)
    return build_model()  # untrained: enough to compare the formats with each other


@pytest.fixture(scope="module")
def squares():
    squares, _labels = generate(range(2))
    return squares


def test_fp32_tflite_gives_the_same_probabilities_as_keras(small_model, squares, tmp_path):
    path = export_tflite(small_model, tmp_path / "model.tflite")
    tflite = load_classifier(path)
    expected = np.asarray(small_model(squares[..., None].astype(np.float32), training=False))
    assert np.allclose(tflite.predict(squares), expected, atol=1e-4)


def test_fp16_file_is_about_half_the_size(small_model, tmp_path):
    fp32 = export_tflite(small_model, tmp_path / "fp32.tflite")
    fp16 = export_tflite(small_model, tmp_path / "fp16.tflite", "fp16")
    assert fp16.stat().st_size < 0.6 * fp32.stat().st_size


def test_int8_takes_raw_pixels_and_is_the_smallest(small_model, squares, tmp_path):
    fp16 = export_tflite(small_model, tmp_path / "fp16.tflite", "fp16")
    int8 = export_tflite(small_model, tmp_path / "int8.tflite", "int8", calibration=squares)
    classifier = load_classifier(int8)
    assert classifier.input["dtype"] == np.uint8
    assert int8.stat().st_size < fp16.stat().st_size
    probabilities = classifier.predict(squares)
    assert probabilities.shape == (len(squares), 13)


def test_int8_needs_calibration_data(small_model, tmp_path):
    with pytest.raises(ValueError):
        export_tflite(small_model, tmp_path / "int8.tflite", "int8")


def test_unknown_quantization_is_rejected(small_model, tmp_path):
    with pytest.raises(ValueError):
        export_tflite(small_model, tmp_path / "x.tflite", "int4")


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model in models/")
def test_int8_model_agrees_with_the_trained_keras_model(tmp_path):
    model = keras.models.load_model(MODEL)
    calibration, _labels = generate(range(10))
    int8 = load_classifier(export_tflite(model, tmp_path / "int8.tflite", "int8", calibration))
    test_squares, _labels = generate(range(2_000_000, 2_000_010))
    keras_probabilities = model(test_squares[..., None].astype(np.float32), training=False)
    keras_classes = np.asarray(keras_probabilities).argmax(axis=1)
    int8_classes = int8.predict(test_squares).argmax(axis=1)
    assert (int8_classes == keras_classes).mean() > 0.98


def test_ms_per_board_times_one_board():
    class CountingClassifier:
        img_size = 32
        calls = 0

        def predict(self, squares):
            self.calls += 1
            assert len(squares) == 64  # always exactly one board
            return np.zeros((len(squares), 13))

    classifier = CountingClassifier()
    ms = ms_per_board(classifier, np.zeros((100, 32, 32), np.uint8), runs=10)
    assert ms >= 0
    assert classifier.calls == 5 + 10  # warm-up runs + timed runs
