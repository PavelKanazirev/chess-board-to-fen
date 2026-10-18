import chess
import pytest

from board2fen.labels import CLASSES, NUM_CLASSES, square_of_index


def test_there_are_13_classes_and_empty_is_first():
    assert NUM_CLASSES == 13
    assert CLASSES[0] == "empty"


@pytest.mark.parametrize(
    ("index", "square"),
    [(0, chess.A8), (7, chess.H8), (56, chess.A1), (63, chess.H1), (12, chess.E7)],
)
def test_image_order(index, square):
    assert square_of_index(index) == square


def test_flipped_board_is_rotated_180_degrees():
    assert square_of_index(0, flipped=True) == chess.H1
    assert square_of_index(63, flipped=True) == chess.A8


@pytest.mark.parametrize("index", [-1, 64])
def test_index_out_of_range_is_rejected(index):
    with pytest.raises(ValueError):
        square_of_index(index)
