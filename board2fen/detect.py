"""Find the chessboard in a photo, straighten it, and cut it into 64 squares.

Images come in as RGB numpy arrays (see scene.py). Corners are always in the order
a8, h8, h1, a1: top-left, top-right, bottom-right, bottom-left of the board.
"""

import numpy as np


def order_corners(points) -> np.ndarray:
    """Sort 4 (x, y) points into the order top-left, top-right, bottom-right, bottom-left.

    In image coordinates y grows downwards. So the top-left point has the smallest x + y and the
    bottom-right the largest; the top-right has the smallest y - x and the bottom-left the largest.
    """
    pts = np.asarray(points, np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(total)], pts[np.argmin(diff)], pts[np.argmax(total)], pts[np.argmax(diff)]],
        np.float32,
    )
