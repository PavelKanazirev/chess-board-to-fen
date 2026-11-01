"""Compare every exported model: file size, accuracy, agreement with Keras, and speed.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --test-boards 100 --threads 1

Run it on the target device (a Raspberry Pi, a Jetson, ...) to get real edge numbers.
"""

import argparse
import platform
from pathlib import Path

from board2fen.dataset import generate
from board2fen.export import ms_per_board
from board2fen.infer import KerasClassifier, OnnxClassifier, SquareClassifier, TFLiteClassifier

TEST_SEED_OFFSET = 2_000_000


def open_model(path: Path, threads: int) -> SquareClassifier:
    """Load a model; the edge runtimes are limited to ``threads`` CPU threads."""
    if path.suffix == ".tflite":
        return TFLiteClassifier(path, num_threads=threads)
    if path.suffix == ".onnx":
        return OnnxClassifier(path, num_threads=threads)
    return KerasClassifier(path)  # Keras always uses every core


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-boards", type=int, default=100, help="unseen boards to test on")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for TFLite and ONNX")
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.models_dir.iterdir()
        if path.suffix in (".keras", ".tflite", ".onnx") and not path.name.startswith("smoke")
    )
    squares, labels = generate(range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.test_boards), workers=4)

    results = []
    for path in paths:
        classifier = open_model(path, args.threads)
        classes = classifier.predict(squares).argmax(axis=1)
        results.append((path, classes, ms_per_board(classifier, squares)))
    keras_classes = next((classes for path, classes, _ in results if path.suffix == ".keras"), None)

    print(f"\nCPU: {platform.processor() or platform.machine()}")
    print(
        f"{len(labels)} test squares from {args.test_boards} unseen boards; "
        f"TFLite/ONNX on {args.threads} thread(s), Keras on all cores\n"
    )
    print("| model | size (KB) | accuracy | agrees with Keras | ms per board |")
    print("|---|---:|---:|---:|---:|")
    for path, classes, ms in results:
        agreement = "-" if keras_classes is None else f"{(classes == keras_classes).mean():.2%}"
        print(
            f"| `{path.name}` | {path.stat().st_size / 1024:.0f} | {(classes == labels).mean():.2%} | "
            f"{agreement} | {ms:.2f} |"
        )


if __name__ == "__main__":
    main()
