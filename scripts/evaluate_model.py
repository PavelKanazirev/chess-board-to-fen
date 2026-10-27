"""Evaluate a trained square classifier on data it has never seen.

Usage:
    python scripts/evaluate_model.py                       # squares from 300 unseen test boards
    python scripts/evaluate_model.py --scenes 200          # plus the whole pipeline on 200 photos
    python scripts/evaluate_model.py --npz data/val.npz    # squares from a saved dataset instead

Test boards use seeds from 2,000,000 and test photos from 3,000,000: never used for training
(seeds 0...) or validation (1,000,000...).
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from board2fen.dataset import generate, load
from board2fen.evaluation import (
    confusion_matrix,
    error_gallery,
    format_confusion,
    per_class_accuracy,
    top_confusions,
)
from board2fen.infer import SquareClassifier, load_classifier
from board2fen.labels import CLASSES, board_to_labels
from board2fen.pipeline import recognize
from board2fen.positions import random_position
from board2fen.scene import render_scene

TEST_SEED_OFFSET = 2_000_000
SCENE_SEED_OFFSET = 3_000_000


def predict(classifier: SquareClassifier, squares: np.ndarray) -> np.ndarray:
    """The predicted class of every square, computed in batches of 1024."""
    batches = [classifier.predict(squares[start : start + 1024]) for start in range(0, len(squares), 1024)]
    return np.concatenate(batches).argmax(axis=1)


def evaluate_squares(
    classifier: SquareClassifier, squares: np.ndarray, labels: np.ndarray, out_dir: Path
) -> None:
    predicted = predict(classifier, squares)
    matrix = confusion_matrix(labels, predicted)
    mistakes = int((predicted != labels).sum())
    print(f"squares: {len(labels)}   accuracy: {(predicted == labels).mean():.4%}   mistakes: {mistakes}")
    accuracy = per_class_accuracy(matrix)
    print("per class: " + "  ".join(f"{name} {acc:.2%}" for name, acc in zip(CLASSES, accuracy, strict=True)))
    print(format_confusion(matrix))
    print("most frequent mistakes (true -> predicted):", top_confusions(matrix) or "none")
    gallery = error_gallery(squares, labels, predicted)
    if gallery is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "errors.png"), gallery)
        print(f"pictures of the mistakes -> {out_dir / 'errors.png'}")


def evaluate_scenes(classifier: SquareClassifier, count: int) -> None:
    """The whole pipeline on rendered photos: detect the board, cut 64 squares, classify them."""
    found = right_squares = perfect_boards = 0
    for seed in range(SCENE_SEED_OFFSET, SCENE_SEED_OFFSET + count):
        rng = random.Random(seed)
        board = random_position(rng)
        image, _true_corners = render_scene(board, rng)
        result = recognize(image, classifier)
        if not result.detected:
            continue  # all 64 squares count as wrong
        found += 1
        right = int((result.labels == np.array(board_to_labels(board))).sum())
        right_squares += right
        perfect_boards += right == 64
    print(
        f"photos: {count}   board found: {found}   squares right: {right_squares / (64 * count):.2%}   "
        f"boards fully right: {perfect_boards} ({perfect_boards / count:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/square_classifier.keras"))
    parser.add_argument("--npz", type=Path, help="evaluate squares from this dataset file")
    parser.add_argument("--test-boards", type=int, default=300, help="unseen boards to generate")
    parser.add_argument("--scenes", type=int, default=0, help="photos for the end-to-end test")
    parser.add_argument("--out-dir", type=Path, default=Path("data/evaluation"))
    args = parser.parse_args()

    classifier = load_classifier(args.model)
    if args.npz:
        squares, labels = load(args.npz)
    else:
        seeds = range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.test_boards)
        squares, labels = generate(seeds, classifier.img_size, workers=4)
    evaluate_squares(classifier, squares, labels, args.out_dir)
    if args.scenes:
        evaluate_scenes(classifier, args.scenes)


if __name__ == "__main__":
    main()
