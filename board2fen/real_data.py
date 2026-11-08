"""Real photos, labelled by their file names.

A photo's name is the FEN's piece placement with "-" instead of "/", because "/" can't be part of
a file name, for example ``1B2b3-Kp6-8-8-2k5-8-8-8.JPG``. A second photo of the same position may
add anything after an underscore: ``1B2b3-Kp6-8-8-2k5-8-8-8_2.jpg``.
"""

import shutil
import zipfile
from pathlib import Path

import chess
import numpy as np

from board2fen.detect import detect_board, split_squares, warp_board
from board2fen.labels import board_to_labels
from board2fen.pipeline import load_corners, load_image

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def labels_from_filename(path: str | Path) -> np.ndarray:
    """The class index of all 64 squares in image order; ValueError if the name isn't a placement."""
    board = chess.Board(None)
    board.set_board_fen(Path(path).stem.split("_")[0].replace("-", "/"))
    return np.array(board_to_labels(board))


def find_photos(folder: str | Path) -> list[Path]:
    """All photos in a folder, sorted by name, so that every run sees the same order."""
    return sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in PHOTO_SUFFIXES)


def extract_dataset(archive: Path, target: Path, keep: tuple[str, ...]) -> None:
    """Unpack the ``keep`` files of a GitHub zip archive, without its long top-level folder.

    Only a complete extraction gets the name ``target``, so a crash can't leave half a dataset.
    """
    partial = target.with_name(target.name + ".part")
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            _top_folder, _, name = member.filename.partition("/")
            if name.startswith(keep) and not member.is_dir():
                path = partial / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    partial.rename(target)


def group_positions(labels: np.ndarray, max_changes: int = 3) -> np.ndarray:
    """A group number for every photo; ``labels`` has 64 labels per photo.

    Two photos are close if their positions differ in at most ``max_changes`` squares: one move,
    a capture or en passant. Photos linked by a chain of close photos get the same group number,
    so in practice a group is one game, photographed move by move.
    """
    close = (labels[:, None, :] != labels[None, :, :]).sum(axis=2) <= max_changes
    group = np.arange(len(labels))
    while True:  # every photo takes the smallest group number among its close photos, until stable
        new = np.where(close, group[None, :], len(labels)).min(axis=1)
        if np.array_equal(new, group):
            return group
        group = new


def split_photos(photos: list[Path], test: float = 0.2, val: float = 0.1) -> dict[str, list[Path]]:
    """Split photos into "train", "val" and "test", keeping every group (game) in one part.

    A random split of single photos would put nearly identical neighbours from one game into train
    and test, and the test would measure memory instead of recognition. So whole groups are
    shared out, the largest first, each to the part that is furthest below its target size.
    """
    group = group_positions(np.stack([labels_from_filename(path) for path in photos]))
    targets = {"train": (1 - test - val) * len(photos), "val": val * len(photos), "test": test * len(photos)}
    parts = {name: [] for name in targets}
    ids, sizes = np.unique(group, return_counts=True)
    for g in ids[np.argsort(-sizes, kind="stable")]:  # largest first; ties in a fixed order
        name = max(parts, key=lambda name: targets[name] - len(parts[name]))
        parts[name] += [photo for photo, photo_group in zip(photos, group, strict=True) if photo_group == g]
    return {name: sorted(members) for name, members in parts.items()}


def photo_squares(path: Path, img_size: int = 32) -> tuple[np.ndarray, np.ndarray] | None:
    """The 64 squares of a photo and their labels, None without a board.

    Cut with the hand-clicked corners if the photo has them, otherwise by our own detector.
    """
    image = load_image(path)
    corners = load_corners(path)
    if corners is None:
        corners = detect_board(image)

    return split_squares(warp_board(image, corners, img_size), img_size), labels_from_filename(path)
