"""Find the chessboard in a photo, straighten it, and cut it into 64 squares.

Images come in as RGB numpy arrays (see scene.py). Corners are always in the order
a8, h8, h1, a1: top-left, top-right, bottom-right, bottom-left of the board.
"""

import cv2
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


def to_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale version of an RGB image (unchanged if it already is grayscale)."""
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def warp_board(image: np.ndarray, corners: np.ndarray, img_size: int = 32, oversample: int = 2) -> np.ndarray:
    """Straighten the board inside ``corners`` into a grayscale square of 8 * img_size pixels.

    We warp at ``oversample`` times the final size first and then shrink with INTER_AREA, which
    averages pixels instead of skipping them. That keeps thin lines from flickering in and out.
    """
    n = 8 * img_size * oversample
    target = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    matrix = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), target)
    big = cv2.warpPerspective(
        to_gray(image), matrix, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return cv2.resize(big, (8 * img_size, 8 * img_size), interpolation=cv2.INTER_AREA)


def split_squares(board: np.ndarray, img_size: int) -> np.ndarray:
    """Cut a straightened (8 * img_size)² board into 64 squares, in image order (a8, b8, ..., h1).

    Returns an array of shape (64, img_size, img_size).
    """
    expected = (8 * img_size, 8 * img_size)
    if board.shape != expected:
        raise ValueError(f"expected a board of shape {expected}, got {board.shape}")
    # (rows of squares, pixel row, columns of squares, pixel column) -> squares first, then pixels
    return board.reshape(8, img_size, 8, img_size).transpose(0, 2, 1, 3).reshape(64, img_size, img_size)
