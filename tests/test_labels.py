import chess
import pytest

from board2fen.labels import (
    CLASSES,
    NUM_CLASSES,
    board_to_labels,
    labels_to_board,
    labels_to_fen,
    lichess_analysis_url,
    square_of_index,
)


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


def test_start_position_labels():
    labels = board_to_labels(chess.Board())
    assert [CLASSES[i] for i in labels[0:8]] == list("rnbqkbnr")
    assert [CLASSES[i] for i in labels[8:16]] == ["p"] * 8
    assert set(labels[16:48]) == {0}  # ranks 6 to 3 are empty
    assert [CLASSES[i] for i in labels[56:64]] == list("RNBQKBNR")


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    ],
)
def test_labels_round_trip(fen):
    board = chess.Board(fen)
    assert labels_to_board(board_to_labels(board)).board_fen() == board.board_fen()


def test_start_position_fen():
    assert labels_to_fen(board_to_labels(chess.Board())) == chess.STARTING_FEN


def test_flipped_labels_give_the_same_fen():
    labels = board_to_labels(chess.Board())
    seen_from_black = labels[::-1]
    assert labels_to_fen(seen_from_black, flipped=True) == labels_to_fen(labels)


def test_castling_rights_are_guessed_from_king_and_rooks():
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    assert labels_to_fen(board_to_labels(board), turn="b") == "4k3/8/8/8/8/8/8/4K2R b K - 0 1"


def test_wrong_number_of_labels_is_rejected():
    with pytest.raises(ValueError):
        labels_to_board([0] * 63)


def test_invalid_turn_is_rejected():
    with pytest.raises(ValueError):
        labels_to_fen([0] * 64, turn="white")


def test_lichess_url_has_no_spaces():
    url = lichess_analysis_url(chess.STARTING_FEN)
    assert url.startswith("https://lichess.org/analysis/")
    assert " " not in url
