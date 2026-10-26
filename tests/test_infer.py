from pathlib import Path

import numpy as np
import pytest
from board2fen.infer import load_classifier

from board2fen.labels import NUM_CLASSES

MODEL = Path(__file__).resolve().parent.parent / "models" / "square_classifier.keras"


def test_unknown_model_formats_are_rejected():
    with pytest.raises(ValueError):
        load_classifier("model.pt")


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model in models/")
def test_keras_classifier_returns_probabilities():
    pytest.importorskip("keras")
    classifier = load_classifier(MODEL)
    assert classifier.img_size == 32
    squares = np.random.default_rng(0).integers(0, 256, (64, 32, 32), dtype=np.uint8)
    probabilities = classifier.predict(squares)
    assert probabilities.shape == (64, NUM_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
