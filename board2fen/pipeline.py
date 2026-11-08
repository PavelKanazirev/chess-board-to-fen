"""The whole recognition: photo -> board detection -> 64 squares -> classifier -> FEN."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from board2fen.detect import detect_board, order_corners, split_squares, warp_board
from board2fen.infer import SquareClassifier
from board2fen.labels import labels_to_fen


@dataclass
class Recognition:
    """Everything ``recognize`` found out about one picture."""

    fen: str
    labels: np.ndarray  # (64,) class index per square, in image order
    confidences: np.ndarray  # (64,) probability of the chosen class
    corners: np.ndarray  # (4, 2) board corners that were used
    board: np.ndarray  # the straightened grayscale board, (8 * img_size, 8 * img_size)
    detected: bool  # False: no corners, so the whole picture was used


def recognize(
    image: np.ndarray,
    classifier: SquareClassifier,
    detect: bool = True,
    flipped: bool = False,
    turn: str = "w",
    corners: np.ndarray | None = None,
) -> Recognition:
    """Recognise the position in an RGB picture.

    ``detect=False`` treats the whole picture as the board (a cropped screenshot). ``flipped``
    means the picture was taken from Black's side, and ``turn`` is the side to move, which a
    picture can't show. Given ``corners`` (a8, h8, h1, a1), e.g. clicked by hand, are used
    instead of detecting the board.

    """
    height, width = image.shape[:2]
    if corners is None and detect:
        corners = detect_board(image)

    detected = corners is not None
    if corners is None:
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    board = warp_board(image, corners, classifier.img_size)
    squares = split_squares(board, classifier.img_size)
    probabilities = classifier.predict(squares)
    labels = probabilities.argmax(axis=1)
    return Recognition(
        fen=labels_to_fen(labels.tolist(), turn=turn, flipped=flipped),
        labels=labels,
        confidences=probabilities.max(axis=1),
        corners=corners,
        board=board,
        detected=detected,
    )


def load_image(path: str | Path) -> np.ndarray:
    """Read a JPG/PNG file as an RGB array."""
    data = np.fromfile(path, np.uint8)  # unlike cv2.imread, this also works with non-ASCII paths
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"not an image file: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR; we use RGB everywhere


def save_pgm(path: str | Path, gray: np.ndarray) -> Path:
    """Write a grayscale image as a binary PGM file: a short text header, then one byte per pixel.

    PGM is about the simplest image format there is, so the C++ demo can read it without an image
    library, as a program on a small device that gets its frames from a camera driver would.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 image, got {gray.dtype} of shape {gray.shape}")
    height, width = gray.shape
    path = Path(path)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + gray.tobytes())
    return path


def corners_file(photo: str | Path) -> Path:
    """Where the hand-clicked corners of a photo are kept: next to it, board.jpg -> board.corners.json."""
    return Path(photo).with_suffix(".corners.json")


def save_corners(photo: str | Path, corners: np.ndarray) -> Path:
    """Store a photo's board corners, sorted into the order a8, h8, h1, a1."""
    path = corners_file(photo)
    corners = np.round(order_corners(corners), 1)  # a tenth of a pixel is plenty
    path.write_text(json.dumps({"corners": corners.tolist()}, indent=2))
    return path


def load_corners(photo: str | Path) -> np.ndarray | None:
    """The hand-clicked corners of a photo, or None if it has none."""
    path = corners_file(photo)
    if not path.exists():
        return None
    return np.float32(json.loads(path.read_text())["corners"])
