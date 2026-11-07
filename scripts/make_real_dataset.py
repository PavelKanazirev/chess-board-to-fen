"""Cut the real photos into labelled squares: data/real/train.npz, val.npz and test.npz.

Usage:
    python scripts/make_real_dataset.py

The photos are split by game (see board2fen.real_data.split_photos), and every photo is cut by
our own detector, so the squares look exactly like what the pipeline feeds the classifier.
"""

import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from board2fen.dataset import EMPTY, save
from board2fen.real_data import find_photos, photo_squares, split_photos


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=Path("data/real/chess-dataset/labeled_originals"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--keep-empty", type=float, default=0.3, help="empty squares kept in train and val")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rng = random.Random(0)
    for name, photos in split_photos(find_photos(args.dir)).items():
        with Pool(args.workers) as pool:
            results = pool.map(photo_squares, photos)
        found = [result for result in results if result is not None]
        squares = np.concatenate([result[0] for result in found])
        labels = np.concatenate([result[1] for result in found]).astype(np.uint8)
        if name != "test":  # like the rendered data: fewer empty squares, more pieces to learn from
            keep = np.array([label != EMPTY or rng.random() < args.keep_empty for label in labels])
            squares, labels = squares[keep], labels[keep]
        save(args.out_dir / f"{name}.npz", squares, labels)
        missing = len(photos) - len(found)
        print(f"{name}: {len(photos)} photos ({missing} without a board), {len(labels)} squares")


if __name__ == "__main__":
    main()
