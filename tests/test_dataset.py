import numpy as np

from board2fen.dataset import EMPTY, generate, load, save, scene_squares
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


def test_generate_gives_the_same_data_with_any_number_of_workers():
    serial = generate(range(4), workers=1)
    parallel = generate(range(4), workers=2)
    assert np.array_equal(serial[0], parallel[0]) and np.array_equal(serial[1], parallel[1])


def test_every_class_appears_in_a_small_dataset():
    _squares, labels = generate(range(20))
    assert set(labels.tolist()) == set(range(NUM_CLASSES))


def test_save_and_load_round_trip(tmp_path):
    squares, labels = generate(range(2))
    path = tmp_path / "subfolder" / "squares.npz"
    save(path, squares, labels)
    loaded_squares, loaded_labels = load(path)
    assert np.array_equal(loaded_squares, squares) and np.array_equal(loaded_labels, labels)
