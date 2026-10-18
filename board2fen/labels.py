"""Square labels and conversions between label lists, python-chess boards and FEN.

The 64 squares are always handled in *image order*: row by row from the top-left corner of a
board seen from White's side. Index 0 is a8, index 7 is h8, index 56 is a1 and index 63 is h1.
"""

from collections.abc import Sequence

import chess

CLASSES = ["empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def square_of_index(index: int, flipped: bool = False) -> chess.Square:
    """Map an image-order index (0..63) to a python-chess square.

    ``flipped`` means the picture was taken from Black's side (the board is rotated by 180°).
    """
    if not 0 <= index < 64:
        raise ValueError(f"index must be in 0..63, got {index}")
    if flipped:
        index = 63 - index
    row, col = divmod(index, 8)
    return chess.square(col, 7 - row)


def board_to_labels(board: chess.Board) -> list[int]:
    """The class index of every square, in image order (White at the bottom)."""
    labels = []
    for index in range(64):
        piece = board.piece_at(square_of_index(index))
        labels.append(CLASS_TO_INDEX[piece.symbol()] if piece else CLASS_TO_INDEX["empty"])
    return labels


def labels_to_board(labels: Sequence[int], flipped: bool = False) -> chess.Board:
    """Build a board from 64 class indices in image order."""
    if len(labels) != 64:
        raise ValueError(f"expected 64 labels, got {len(labels)}")
    board = chess.Board(None)  # None = an empty board
    for index, label in enumerate(labels):
        if label != CLASS_TO_INDEX["empty"]:
            piece = chess.Piece.from_symbol(CLASSES[label])
            board.set_piece_at(square_of_index(index, flipped), piece)
    return board


def labels_to_fen(labels: Sequence[int], turn: str = "w", flipped: bool = False) -> str:
    """A full FEN string for 64 class indices.

    An image shows only where the pieces stand. ``turn`` (side to move) must come from the user,
    castling rights are guessed from the king and rook positions, and en passant is unknown.
    """
    if turn not in ("w", "b"):
        raise ValueError(f"turn must be 'w' or 'b', got {turn!r}")
    board = labels_to_board(labels, flipped)
    board.turn = chess.WHITE if turn == "w" else chess.BLACK
    # Allow all four castles, then let python-chess drop those whose king or rook has moved.
    board.castling_rights = chess.BB_CORNERS
    board.castling_rights = board.clean_castling_rights()
    return board.fen()
