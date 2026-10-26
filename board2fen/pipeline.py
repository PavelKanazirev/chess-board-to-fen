"""The whole recognition: photo -> board detection -> 64 squares -> classifier -> FEN."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from board2fen.detect import detect_board, split_squares, warp_board
from board2fen.infer import SquareClassifier
from board2fen.labels import labels_to_fen


@dataclass
class Recognition:
    """Everything ``recognize`` found out about one picture."""

    fen: str
    labels: np.ndarray  # (64,) class index per square, in image order
    confidences: np.ndarray  # (64,) probability of the chosen class
    corners: np.ndarray  # (4, 2) board corners that were used
    detected: bool  # False: no board was found, the whole picture was used


def recognize(
    image: np.ndarray,
    classifier: SquareClassifier,
    detect: bool = True,
    flipped: bool = False,
    turn: str = "w",
) -> Recognition:
    """Recognise the position in an RGB picture.

    ``detect=False`` treats the whole picture as the board (a cropped screenshot). ``flipped``
    means the picture was taken from Black's side, and ``turn`` is the side to move, which a
    picture can't show.
    """
    height, width = image.shape[:2]
    corners = detect_board(image) if detect else None
    detected = corners is not None
    if corners is None:
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    squares = split_squares(warp_board(image, corners, classifier.img_size), classifier.img_size)
    probabilities = classifier.predict(squares)
    labels = probabilities.argmax(axis=1)
    return Recognition(
        fen=labels_to_fen(labels.tolist(), turn=turn, flipped=flipped),
        labels=labels,
        confidences=probabilities.max(axis=1),
        corners=corners,
        detected=detected,
    )


def load_image(path: str | Path) -> np.ndarray:
    """Read a JPG/PNG file as an RGB array."""
    data = np.fromfile(path, np.uint8)  # unlike cv2.imread, this also works with non-ASCII paths
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"not an image file: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR; we use RGB everywhere
