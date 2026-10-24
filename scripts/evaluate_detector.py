"""Measure how well detect_board() finds boards in rendered scenes, where the truth is known.

Usage:
    python scripts/evaluate_detector.py --scenes 200
    python scripts/evaluate_detector.py --scenes 200 --save-failures data/detector_failures
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np

from board2fen.detect import corner_error, detect_board
from board2fen.positions import random_position
from board2fen.scene import render_scene


def save_failure(path: Path, image: np.ndarray, truth: np.ndarray, found: np.ndarray | None) -> None:
    """Save the scene with the true outline in green and the detected one in red."""
    path.parent.mkdir(parents=True, exist_ok=True)
    picture = image.copy()
    cv2.polylines(picture, [truth.astype(np.int32)], True, (0, 255, 0), 3)
    if found is not None:
        cv2.polylines(picture, [found.astype(np.int32)], True, (255, 0, 0), 3)
    cv2.imwrite(str(path), cv2.cvtColor(picture, cv2.COLOR_RGB2BGR))  # OpenCV saves BGR


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenes", type=int, default=200, help="how many scenes to test")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first scene")
    parser.add_argument("--tolerance", type=float, default=0.03, help="allowed corner error (0.03 = 3%%)")
    parser.add_argument("--save-failures", type=Path, help="folder for pictures of the failed scenes")
    args = parser.parse_args()

    errors, failures = [], []
    start = time.perf_counter()
    for seed in range(args.seed, args.seed + args.scenes):
        rng = random.Random(seed)
        image, truth = render_scene(random_position(rng), rng)
        found = detect_board(image)
        error = float("inf") if found is None else corner_error(found, truth)
        errors.append(error)
        if error > args.tolerance:
            failures.append(seed)
            if args.save_failures:
                save_failure(args.save_failures / f"scene_{seed}.jpg", image, truth, found)
    elapsed = time.perf_counter() - start

    errors = np.array(errors)
    found = np.isfinite(errors)
    print(f"scenes:        {len(errors)}  ({elapsed / len(errors) * 1000:.0f} ms each, including rendering)")
    print(f"board found:   {found.sum()}  ({found.mean():.1%})")
    print(f"within 1%:     {(errors < 0.01).sum()}  ({(errors < 0.01).mean():.1%})")
    good = errors < args.tolerance
    print(f"within {args.tolerance:.0%}:     {good.sum()}  ({good.mean():.1%})")
    if found.any():
        print(f"median error:  {np.median(errors[found]):.2%} of the board size")
    print(f"failed seeds:  {failures}")


if __name__ == "__main__":
    main()
