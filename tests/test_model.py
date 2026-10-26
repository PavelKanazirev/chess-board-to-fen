import numpy as np
import pytest

keras = pytest.importorskip("keras")  # skip these tests if the "train" extra isn't installed

from board2fen.dataset import generate  # noqa: E402
from board2fen.labels import NUM_CLASSES  # noqa: E402
from board2fen.model import build_model, compile_model  # noqa: E402


def test_output_is_a_probability_per_class():
    model = build_model()
    squares = np.random.default_rng(0).integers(0, 256, (5, 32, 32, 1)).astype(np.float32)
    probabilities = model.predict(squares, verbose=0)
    assert probabilities.shape == (5, NUM_CLASSES)
    assert (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_model_is_small_enough_for_an_edge_device():
    assert build_model().count_params() < 100_000


def test_model_can_learn_a_tiny_dataset():
    keras.utils.set_random_seed(0)
    squares, labels = generate(range(3))
    x = squares[..., None].astype(np.float32)
    model = compile_model(build_model())
    history = model.fit(x, labels, epochs=40, batch_size=32, verbose=0)
    assert history.history["accuracy"][-1] > 0.9
