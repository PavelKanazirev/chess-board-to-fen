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

from board2fen.labels import board_to_labels

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
