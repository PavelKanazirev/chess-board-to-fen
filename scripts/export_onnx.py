"""Export the trained model to ONNX: 32-bit float and int8.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/square_classifier.keras --out-dir models

Compare all exported formats afterwards with: python scripts/benchmark.py
"""

import argparse
from pathlib import Path

import keras

from board2fen.dataset import generate
from board2fen.export import export_onnx, quantize_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/square_classifier.keras"))
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--calibration-boards", type=int, default=50, help="training boards for int8")
    args = parser.parse_args()

    model = keras.models.load_model(args.model)
    fp32 = export_onnx(model, args.out_dir / f"{args.model.stem}.onnx")
    calibration, _labels = generate(range(args.calibration_boards), workers=4)  # training seeds
    int8 = quantize_onnx(fp32, args.out_dir / f"{args.model.stem}_int8.onnx", calibration)
    for path in (fp32, int8):
        print(f"{path}  {path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
