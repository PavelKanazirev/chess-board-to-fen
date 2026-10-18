"""Square labels and conversions between label lists, python-chess boards and FEN.

The 64 squares are always handled in *image order*: row by row from the top-left corner of a
board seen from White's side. Index 0 is a8, index 7 is h8, index 56 is a1 and index 63 is h1.
"""

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
