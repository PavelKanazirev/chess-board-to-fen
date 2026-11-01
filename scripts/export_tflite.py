"""Export the trained model to TensorFlow Lite (fp32, fp16, int8) and compare the versions.

Usage:
    python scripts/export_tflite.py
    python scripts/export_tflite.py --model models/square_classifier.keras --out-dir models

For each format it prints the file size, the accuracy on unseen test squares, how often it
agrees with the Keras model, and the time for one board (64 squares) on one CPU thread.
"""

import argparse
import time
from pathlib import Path

import keras
import numpy as np

from board2fen.dataset import generate
from board2fen.export import QUANTIZATIONS, export_tflite
from board2fen.infer import KerasClassifier, SquareClassifier, TFLiteClassifier

TEST_SEED_OFFSET = 2_000_000


def ms_per_board(classifier: SquareClassifier, squares: np.ndarray, runs: int = 50) -> float:
    """Median time to classify one board (64 squares), after a few warm-up runs."""
    board = squares[:64]
    for _ in range(5):
        classifier.predict(board)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        classifier.predict(board)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/square_classifier.keras"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--calibration-boards", type=int, default=50, help="training boards for int8")
    parser.add_argument("--test-boards", type=int, default=50, help="unseen boards for the comparison")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    calibration, _labels = generate(range(args.calibration_boards), workers=4)  # training seeds
    test_squares, test_labels = generate(
        range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.test_boards), workers=4
    )

    reference = KerasClassifier(args.model)
    keras_classes = reference.predict(test_squares).argmax(axis=1)
    rows = [("keras", args.model, reference)]
    for quantization in QUANTIZATIONS:
        path = args.out_dir / f"{args.model.stem}_{quantization}.tflite"
        export_tflite(model, path, quantization, calibration)
        rows.append((f"tflite {quantization}", path, TFLiteClassifier(path, num_threads=1)))

    print(f"\n{len(test_labels)} test squares from {args.test_boards} unseen boards\n")
    print("| format | file | size (KB) | accuracy | agrees with Keras | ms per board |")
    print("|---|---|---:|---:|---:|---:|")
    for name, path, classifier in rows:
        classes = classifier.predict(test_squares).argmax(axis=1)
        print(
            f"| {name} | `{path.name}` | {path.stat().st_size / 1024:.0f} | "
            f"{(classes == test_labels).mean():.2%} | {(classes == keras_classes).mean():.2%} | "
            f"{ms_per_board(classifier, test_squares):.2f} |"
        )


if __name__ == "__main__":
    main()
