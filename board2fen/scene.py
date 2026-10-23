"""Photo-like scenes: a rendered board seen at an angle, on a background, with camera effects.

Images are numpy arrays in RGB order, shape (height, width, 3), dtype uint8, the same as Pillow.
OpenCV assumes BGR order when it encodes or saves images, so we convert only at those points.
"""

import math
import random

import chess
import cv2
import numpy as np

from board2fen.render import render_random_board


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


def photo_effects(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Camera imperfections: exposure, uneven light, blur, sensor noise and JPEG compression."""
    out = image.astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-40, 40)  # contrast and brightness
    if rng.random() < 0.5:  # a lamp off to one side: darker further away from it
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        lamp_x, lamp_y = rng.uniform(0, w), rng.uniform(0, h)
        distance = np.sqrt((xx - lamp_x) ** 2 + (yy - lamp_y) ** 2) / max(h, w)
        out *= (1 - rng.uniform(0, 0.5) * distance)[..., None]
    if rng.random() < 0.4:  # slightly out of focus
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.5))
    out += numpy_rng(rng).normal(0, rng.uniform(0, 8), out.shape)  # sensor noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    quality = rng.randint(35, 95)
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def render_scene(board: chess.Board, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """``board`` as a random "photo". Returns the RGB image and the board corners (a8, h8, h1, a1)."""
    board_image = np.asarray(render_random_board(board, rng))
    n = board_image.shape[0]
    height = rng.randint(int(n * 1.3), int(n * 2.0))
    width = height + rng.randint(0, height // 3)
    corners = random_board_corners(width, height, rng)
    scene = paste_board(random_background(width, height, rng), board_image, corners)
    return photo_effects(scene, rng), corners
