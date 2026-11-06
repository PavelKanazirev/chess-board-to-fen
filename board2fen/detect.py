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


def line_intersection(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray | None:
    """Where the line through p1, q1 crosses the line through p2, q2 (None if they're parallel)."""
    d1, d2 = q1 - p1, q2 - p2
    denom = d1[0] * d2[1] - d1[1] * d2[0]  # 2D cross product: 0 means parallel
    if abs(denom) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def hull_to_quad(hull: np.ndarray) -> np.ndarray | None:
    """Turn a convex hull into the board's 4 corners, or None if it isn't roughly 4-sided.

    Where a board corner has little contrast against the background, the hull "cuts the corner"
    and gets an extra short edge. So instead of trusting the hull's vertices, we take its 4 longest
    edges (the real board sides) and intersect neighbouring ones.
    """
    poly = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float64)
    if len(poly) < 4:
        return None
    edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    lengths = np.array([np.linalg.norm(end - start) for start, end in edges])
    longest = sorted(np.argsort(lengths)[-4:])  # the 4 longest edges, in their order around the hull
    if lengths[longest].sum() < 0.6 * lengths.sum():
        return None  # the outline is too ragged to be a quadrilateral
    sides = [edges[i] for i in longest]
    corners = [line_intersection(*sides[i - 1], *sides[i]) for i in range(4)]
    if any(corner is None for corner in corners):
        return None
    return order_corners(corners)


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


def fit_grid(profile: np.ndarray, min_fraction: float = 0.7, min_gain: float = 2.5) -> tuple[float, float]:
    """Where 9 evenly spaced lines (the borders of 8 squares) sit best on a 1-D edge profile.

    Tries every square size from ``min_fraction`` of the largest possible one up to the largest,
    at every start position, and keeps the lines with the strongest average edge. Returns the
    positions of the first and the last line.

    A smaller grid must be ``min_gain`` times as strong as the grid that fills the whole profile,
    or the whole profile is returned. Without a margin, the edges of pieces can line up by chance,
    but not that well: measured, at most 1.8 times on rendered boards, and at least 2.7 times on
    the photos of a mat with a margin.
    """
    size = len(profile)

    def strength(starts: np.ndarray, spacing: float) -> np.ndarray:
        lines = np.round(starts[:, None] + spacing * np.arange(9)).astype(int)  # one row per start
        return profile[np.minimum(lines, size - 1)].mean(axis=1)

    best_score, first, last = -1.0, 0.0, float(size)
    for spacing in np.arange(min_fraction * size / 8, size / 8 + 0.01, 0.25):
        starts = np.arange(int(size - 8 * spacing) + 1)
        scores = strength(starts, spacing)
        best = int(scores.argmax())
        if scores[best] > best_score:
            best_score, first, last = float(scores[best]), float(starts[best]), starts[best] + 8 * spacing
    if best_score < min_gain * strength(np.zeros(1), size / 8)[0]:
        return 0.0, float(size)
    return first, float(last)


def refine_to_grid(image: np.ndarray, corners: np.ndarray, size: int = 512) -> np.ndarray:
    """Shrink a board outline to its 8 x 8 grid of squares, e.g. inside a mat with a margin.

    Straightens the outline, then measures how strong the vertical edges are in every column of
    pixels (and the horizontal edges in every row). The borders of the squares make 9 evenly
    spaced peaks in these two profiles, which ``fit_grid`` finds. Without a margin the grid fills
    the whole outline, and the corners stay where they are.
    """
    target = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    matrix = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), target)
    board = cv2.warpPerspective(to_gray(image), matrix, (size, size), borderMode=cv2.BORDER_REPLICATE)
    board = cv2.GaussianBlur(board, (5, 5), 0)
    columns = np.abs(cv2.Sobel(board, cv2.CV_32F, 1, 0)).mean(axis=0)  # vertical edges per column
    rows = np.abs(cv2.Sobel(board, cv2.CV_32F, 0, 1)).mean(axis=1)  # horizontal edges per row
    left, right = fit_grid(columns)
    top, bottom = fit_grid(rows)
    grid = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    # Back from the straightened picture to the photo: the inverse of the same perspective transform.
    return cv2.perspectiveTransform(grid[None], np.linalg.inv(matrix))[0]


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
    # First with normal edges. If that finds no board, try again with more sensitive edges and
    # thicker lines, which join up the broken outline of a low-contrast board.
    for low, high, thickening in ((40, 120, 1), (15, 45, 3)):
        edges = cv2.dilate(cv2.Canny(blurred, low, high), np.ones((3, 3), np.uint8), iterations=thickening)
        quad = largest_quad(edges, min_area_frac * gray.size)
        if quad is not None:
            return quad / scale
    return None


def corner_error(found: np.ndarray, truth: np.ndarray) -> float:
    """The worst corner distance, as a fraction of the board's top edge (0.01 = 1%)."""
    side = float(np.linalg.norm(truth[1] - truth[0]))
    return float(np.linalg.norm(np.asarray(found) - truth, axis=1).max()) / side
