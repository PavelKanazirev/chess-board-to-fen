"""Photo-like scenes: a rendered board seen at an angle, on a background, with camera effects.

Images are numpy arrays in RGB order, shape (height, width, 3), dtype uint8, the same as Pillow.
OpenCV assumes BGR order when it encodes or saves images, so we convert only at those points.
"""

import math
import random

import cv2
import numpy as np


def numpy_rng(rng: random.Random) -> np.random.Generator:
    """A numpy generator seeded from ``rng``, so one seed still controls everything."""
    return np.random.default_rng(rng.randrange(2**32))


def random_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    """A smooth two-colour gradient with blurred noise, like a table or a wall."""
    color1 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    color2 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    angle = rng.uniform(0, 2 * math.pi)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    t = xx * math.cos(angle) + yy * math.sin(angle)
    t = (t - t.min()) / max(1e-6, float(t.max() - t.min()))  # 0..1 across the image
    image = color1 * (1 - t[..., None]) + color2 * t[..., None]
    image += numpy_rng(rng).normal(0, rng.uniform(0, 20), (height, width, 1))
    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(1, 5))
    return np.clip(image, 0, 255).astype(np.uint8)


def random_board_corners(width: int, height: int, rng: random.Random) -> np.ndarray:
    """Where the board's corners land in the picture, as a (4, 2) array of (x, y).

    The order is always a8, h8, h1, a1: clockwise from the top-left. The board is scaled, moved,
    rotated by up to 12 degrees and each corner nudged a little, like a camera that isn't
    exactly above the board.
    """
    for _ in range(100):
        size = rng.uniform(0.5, 0.75) * min(width, height)
        cx = width / 2 + rng.uniform(-0.1, 0.1) * width
        cy = height / 2 + rng.uniform(-0.1, 0.1) * height
        angle = math.radians(rng.uniform(-12, 12))
        cos, sin = math.cos(angle), math.sin(angle)
        corners = []
        for ux, uy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = ux * size / 2, uy * size / 2
            x, y = x * cos - y * sin, x * sin + y * cos  # rotate around the centre
            x += rng.uniform(-0.07, 0.07) * size  # nudge each corner: a slanted view
            y += rng.uniform(-0.07, 0.07) * size
            corners.append((cx + x, cy + y))
        corners = np.array(corners, np.float32)
        inside = (corners >= 0).all() and (corners[:, 0] < width).all() and (corners[:, 1] < height).all()
        if inside:
            return corners
    raise RuntimeError("could not place the board inside the image")


def paste_board(background: np.ndarray, board_image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the square, top-down ``board_image`` so its corners land on ``corners``."""
    height, width = background.shape[:2]
    n = board_image.shape[0]
    source = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    matrix = cv2.getPerspectiveTransform(source, corners)
    warped = cv2.warpPerspective(board_image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), matrix, (width, height))
    alpha = mask[..., None].astype(np.float32) / 255  # 1 = board, 0 = background
    return (warped * alpha + background * (1 - alpha)).astype(np.uint8)
