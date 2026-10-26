"""Regression test for the trained model committed in models/.

If a later change (renderer, detector, training) makes the committed model worse on fresh data,
this test fails in CI.
"""

from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")  # skip if the "train" extra isn't installed

from board2fen.dataset import generate  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "square_classifier.keras"


@pytest.mark.skipif(not MODEL.exists(), reason="no trained model in models/")
def test_trained_model_recognises_unseen_squares():
    model = keras.models.load_model(MODEL)
    squares, labels = generate(range(2_000_000, 2_000_020))  # test seeds, never trained on
    predicted = model.predict(squares[..., None].astype(np.float32), verbose=0).argmax(axis=1)
    assert (predicted == labels).mean() > 0.98
