"""Train the square classifier on data/train.npz and check it on data/val.npz.

Usage:
    python scripts/train.py --epochs 10
    python scripts/train.py --epochs 1 --train data/smoke/train.npz --val data/smoke/val.npz

Saves the trained model (.keras) and the per-epoch history (.history.csv) next to each other.
"""

import argparse
from pathlib import Path

import keras

from board2fen.dataset import load
from board2fen.model import build_model, compile_model
from board2fen.training import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.npz"))
    parser.add_argument("--val", type=Path, default=Path("data/val.npz"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models/square_classifier.keras"))
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)  # Python, numpy and TensorFlow random numbers
    train_squares, train_labels = load(args.train)
    val_squares, val_labels = load(args.val)
    train_data = make_dataset(train_squares, train_labels, args.batch_size, training=True, seed=args.seed)
    val_data = make_dataset(val_squares, val_labels, args.batch_size)

    model = compile_model(build_model(train_squares.shape[1]), args.learning_rate)
    model.summary()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, verbose=1),
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(args.out.with_suffix(".history.csv")),
    ]
    model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks, verbose=2)

    model.save(args.out)
    _loss, accuracy = model.evaluate(val_data, verbose=0)
    print(f"validation accuracy: {accuracy:.4%}  ->  {args.out}")


if __name__ == "__main__":
    main()
