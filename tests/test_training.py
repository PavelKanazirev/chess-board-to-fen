import numpy as np
import pytest

pytest.importorskip("tensorflow")  # skip these tests if the "train" extra isn't installed

from board2fen.training import make_dataset  # noqa: E402


def small_dataset(n=50):
    rng = np.random.default_rng(0)
    squares = rng.integers(0, 256, (n, 32, 32), dtype=np.uint8)
    labels = np.arange(n, dtype=np.uint8) % 13
    return squares, labels


def test_batches_have_a_channel_axis_and_the_right_types():
    squares, labels = small_dataset()
    batches = list(make_dataset(squares, labels, batch_size=16))
    assert len(batches) == 4  # 16 + 16 + 16 + 2
    x, y = batches[0]
    assert x.shape == (16, 32, 32, 1) and x.dtype.name == "float32"
    assert y.dtype.name == "int32"


def test_validation_data_is_passed_through_unchanged():
    squares, labels = small_dataset()
    x, y = next(iter(make_dataset(squares, labels, batch_size=50)))
    assert np.array_equal(x.numpy()[..., 0], squares.astype(np.float32))
    assert np.array_equal(y.numpy(), labels)


def test_training_data_is_shuffled_and_augmented():
    squares, labels = small_dataset()
    x, y = next(iter(make_dataset(squares, labels, batch_size=50, training=True)))
    assert not np.array_equal(y.numpy(), labels)  # shuffled
    assert sorted(y.numpy().tolist()) == sorted(labels.tolist())  # but nothing lost
    assert x.shape == (50, 32, 32, 1)
    assert 0.0 <= float(x.numpy().min()) and float(x.numpy().max()) <= 255.0
