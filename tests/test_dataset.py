import numpy as np
from board2fen.dataset import EMPTY, scene_squares

from board2fen.labels import NUM_CLASSES


def test_scene_squares_shapes_and_types():
    squares, labels = scene_squares(0)
    assert squares.dtype == np.uint8 and labels.dtype == np.uint8
    assert squares.shape == (len(labels), 32, 32)
    assert 0 < len(labels) <= 64
    assert labels.max() < NUM_CLASSES


def test_scene_squares_is_reproducible():
    first, first_labels = scene_squares(3)
    again, again_labels = scene_squares(3)
    assert np.array_equal(first, again) and np.array_equal(first_labels, again_labels)


def test_keep_empty_controls_the_empty_squares():
    _squares, all_labels = scene_squares(1, keep_empty=1.0)
    assert len(all_labels) == 64
    _squares, labels = scene_squares(1, keep_empty=0.0)
    assert EMPTY not in labels
    assert len(labels) == np.count_nonzero(all_labels != EMPTY)
