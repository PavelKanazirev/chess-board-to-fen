"""Generate the training and validation datasets from rendered scenes.

Usage:
    python scripts/generate_dataset.py                      # 4000 + 400 boards into data/
    python scripts/generate_dataset.py --boards 50 --val-boards 10 --out-dir data/small

Training and validation boards come from different seed ranges, so no board (and no square of
it) ever appears in both: the validation score measures how well the model handles boards it
has never seen.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from board2fen.dataset import generate, save
from board2fen.labels import CLASSES

VALIDATION_SEED_OFFSET = 1_000_000


def describe(name: str, squares: np.ndarray, labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=len(CLASSES))
    print(f"{name}: {len(labels)} squares, {squares.nbytes / 1e6:.0f} MB in memory")
    print("  " + "  ".join(f"{cls}:{count}" for cls, count in zip(CLASSES, counts, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--boards", type=int, default=4000, help="training boards")
    parser.add_argument("--val-boards", type=int, default=400, help="validation boards")
    parser.add_argument("--img-size", type=int, default=32, help="square size in pixels")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    splits = {
        "train": range(args.boards),
        "val": range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + args.val_boards),
    }
    for name, seeds in splits.items():
        start = time.perf_counter()
        squares, labels = generate(seeds, args.img_size, args.workers)
        path = args.out_dir / f"{name}.npz"
        save(path, squares, labels)
        describe(name, squares, labels)
        print(f"  {len(seeds)} boards in {time.perf_counter() - start:.0f} s -> {path}")


if __name__ == "__main__":
    main()
