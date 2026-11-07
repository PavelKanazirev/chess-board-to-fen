"""Evaluate a square classifier on real photos labelled by their file names.

Usage:
    python scripts/evaluate_real.py
    python scripts/evaluate_real.py --limit 50
    python scripts/evaluate_real.py --dir data/real/chess-dataset/labeled_preprocessed --no-detect
    python scripts/evaluate_real.py --dir data/real/my_photos
    python scripts/evaluate_real.py --split test --model models/square_classifier.keras


Get the photos with scripts/download_data.py. Inference only: this doesn't need TensorFlow.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from board2fen.detect import split_squares
from board2fen.evaluation import (
    confusion_matrix,
    error_gallery,
    per_class_accuracy,
    square_metrics,
    top_confusions,
)
from board2fen.infer import load_classifier
from board2fen.labels import CLASSES
from board2fen.pipeline import load_image, recognize
from board2fen.real_data import find_photos, labels_from_filename, split_photos


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=Path("data/real/chess-dataset/labeled_originals"))
    parser.add_argument("--model", type=Path, default=Path("models/square_classifier_int8.onnx"))
    parser.add_argument("--no-detect", action="store_true", help="the pictures show only the board")
    parser.add_argument("--limit", type=int, help="use only this many photos, evenly spread")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="only one part of the photos, split by game as for training",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/evaluation_real"))
    args = parser.parse_args()

    photos = find_photos(args.dir)
    if args.split != "all":
        photos = split_photos(photos)[args.split]
    if args.limit and args.limit < len(photos):
        photos = [photos[i] for i in np.linspace(0, len(photos) - 1, args.limit, dtype=int)]
    classifier = load_classifier(args.model)

    true, predicted, squares = [], [], []
    detected = 0
    for path in photos:
        result = recognize(load_image(path), classifier, detect=not args.no_detect)
        true.append(labels_from_filename(path))
        predicted.append(result.labels)
        squares.append(split_squares(result.board, classifier.img_size))
        detected += result.detected
    true, predicted, squares = np.concatenate(true), np.concatenate(predicted), np.concatenate(squares)

    print(f"{len(photos)} photos from {args.dir}, model {args.model.name}")
    if not args.no_detect:
        print(f"a board was found in {detected}")
    for name, value in square_metrics(true, predicted).items():
        print(f"{name:>9}: {value:.2%}")
    matrix = confusion_matrix(true, predicted)
    accuracy = per_class_accuracy(matrix)
    print("per class: " + "  ".join(f"{name} {acc:.0%}" for name, acc in zip(CLASSES, accuracy, strict=True)))
    print("most frequent mistakes (true -> predicted):", top_confusions(matrix))
    gallery = error_gallery(squares, true, predicted)
    if gallery is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_dir / "errors.png"), gallery)
        print(f"pictures of the mistakes -> {args.out_dir / 'errors.png'}")


if __name__ == "__main__":
    main()
