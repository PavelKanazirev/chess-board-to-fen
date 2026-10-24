"""Training data: rendered scenes cut into labelled 32x32 squares.

Each scene is straightened with its *true* corners plus a small random error, so the squares look
like the output of a slightly imperfect detector while every label stays correct.
"""

import random

import numpy as np

from board2fen.detect import split_squares, warp_board
from board2fen.labels import CLASS_TO_INDEX, board_to_labels
from board2fen.positions import random_position
from board2fen.scene import render_scene

EMPTY = CLASS_TO_INDEX["empty"]


def scene_squares(
    seed: int, img_size: int = 32, corner_noise: float = 0.015, keep_empty: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """The labelled squares of one random scene.

    Returns ``(squares, labels)``: uint8 arrays of shape (n, img_size, img_size) and (n,).
    ``corner_noise`` moves each corner by up to that fraction of the board size, like a detector
    that is slightly off. ``keep_empty`` keeps only that fraction of the empty squares, because
    most squares are empty and the classifier needs to see plenty of pieces.
    """
    rng = random.Random(seed)
    board = random_position(rng)
    image, corners = render_scene(board, rng)
    side = float(np.linalg.norm(corners[1] - corners[0]))
    noise = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)]) * corner_noise * side
    squares = split_squares(warp_board(image, corners + noise, img_size), img_size)
    labels = np.array(board_to_labels(board), np.uint8)
    keep = np.array([label != EMPTY or rng.random() < keep_empty for label in labels])
    return squares[keep], labels[keep]
