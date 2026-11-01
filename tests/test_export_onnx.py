import numpy as np
import pytest

keras = pytest.importorskip("keras")  # converting needs TensorFlow ("train" extra)
pytest.importorskip("tf2onnx")
pytest.importorskip("onnxruntime")  # running needs ONNX Runtime ("onnxruntime" extra)

from board2fen.dataset import generate  # noqa: E402
from board2fen.export import export_onnx  # noqa: E402
from board2fen.infer import load_classifier  # noqa: E402
from board2fen.model import build_model  # noqa: E402


@pytest.fixture(scope="module")
def small_model():
    keras.utils.set_random_seed(0)
    return build_model()  # untrained: enough to compare the formats with each other


@pytest.fixture(scope="module")
def squares():
    squares, _labels = generate(range(2))
    return squares


def test_onnx_gives_the_same_probabilities_as_keras(small_model, squares, tmp_path):
    onnx_model = load_classifier(export_onnx(small_model, tmp_path / "model.onnx"))
    expected = np.asarray(small_model(squares[..., None].astype(np.float32), training=False))
    assert onnx_model.img_size == 32
    assert np.allclose(onnx_model.predict(squares), expected, atol=1e-4)


def test_onnx_accepts_any_batch_size(small_model, squares, tmp_path):
    onnx_model = load_classifier(export_onnx(small_model, tmp_path / "model.onnx"))
    assert onnx_model.predict(squares[:1]).shape == (1, 13)
    assert onnx_model.predict(squares[:64]).shape == (min(64, len(squares)), 13)
