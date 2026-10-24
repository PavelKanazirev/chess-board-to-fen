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


def hull_to_quad(hull: np.ndarray) -> np.ndarray | None:
    """Simplify a convex hull to its 4 corners, or None if it isn't 4-sided."""
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    return order_corners(approx) if len(approx) == 4 else None


def largest_quad(edges: np.ndarray, min_area: float) -> np.ndarray | None:
    """The biggest roughly square quadrilateral outlined by the edge pixels, or None."""
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for contour in contours:
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        if area < min_area or area <= best_area:
            continue
        quad = hull_to_quad(hull)
        if quad is None:
            continue
        sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
        if sides.min() < 0.5 * sides.max():  # far too stretched to be a chessboard
            continue
        best, best_area = quad, area
    return best


def detect_board(image: np.ndarray, min_area_frac: float = 0.15) -> np.ndarray | None:
    """Find the board's 4 corners (a8, h8, h1, a1) in a photo, or None if no board is visible.

    The board is taken to be the largest roughly square outline in the picture's edges.
    ``min_area_frac`` ignores outlines covering less than that fraction of the picture.
    """
    gray = to_gray(image)
    scale = min(1.0, 800.0 / max(gray.shape))  # work on a copy at most 800 pixels wide
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blurred, 40, 120), np.ones((3, 3), np.uint8))
    quad = largest_quad(edges, min_area_frac * gray.size)
    return None if quad is None else quad / scale
