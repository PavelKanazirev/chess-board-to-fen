# chess-board-to-fen

[![CI](https://github.com/PavelKanazirev/chess-board-to-fen/actions/workflows/ci.yml/badge.svg)](https://github.com/PavelKanazirev/chess-board-to-fen/actions/workflows/ci.yml)

> **Status: work in progress.** This project is being built step by step, one pull request at a time.

**Goal:** take a photo or screenshot of a chessboard and output the position as a [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) string, using a small neural network deployed with **TensorFlow Lite** and **ONNX Runtime**, from Python and from C++.

## Problem definition

| | |
|---|---|
| **Input** | A photo or screenshot of a chessboard (JPG/PNG), the whole board visible, White at the bottom (or a `--flip` flag). |
| **Output** | The piece placement as FEN, e.g. `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`, plus a lichess analysis link. |
| **Core task** | Classify each of the 64 squares into 13 classes: empty, `P N B R Q K` (white), `p n b r q k` (black). |
| **Edge constraints** | Model < 300 KB (int8 < 100 KB); one board (64 squares) in < 50 ms on one CPU core; inference without TensorFlow (ONNX Runtime / LiteRT only). |
| **Success metrics** | Square accuracy, whole-board accuracy (all 64 correct), latency per board, model size per format. Measured on synthetic **and** real photos. |
| **Out of scope** | Side to move, castling rights, en passant and move counters: they can't be seen in an image. |

**Main risks**

- Training on synthetic 2D pieces may not transfer to photos of real 3D pieces. Mitigation: public real-photo datasets and my own photos for fine-tuning and testing.
- Board detection can fail on busy backgrounds or boards with wide frames.
- Dataset licenses: only openly licensed data, recorded in `DATA_SOURCES.md` and downloaded by a script, never committed.

## Tech stack

- **Python 3.13**, TensorFlow / Keras 3 (training), OpenCV (board detection), python-chess + Pillow (synthetic data)
- **TensorFlow Lite / LiteRT** and **ONNX / ONNX Runtime** (edge inference, int8 quantization)
- **C++17**, CMake, ONNX Runtime C++ API
- pytest, ruff, GitHub Actions


## Repository

`https://github.com/PavelKanazirev/chess-board-to-fen`. Clone it with:

```bash
git clone https://github.com/PavelKanazirev/chess-board-to-fen.git
```

## Development setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Git Bash; in PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```


## Results on real photos

500 real photos of 3D pieces taken from above ([data sources](DATA_SOURCES.md)) are split **by game**: positions that follow each other move by move always stay in the same part, so the 100 test photos come from games the model has never seen. The model is fine-tuned on 350 photos (plus the rendered data, so it doesn't forget it); 50 photos are for validation.

| Test photos (100, unseen games) | Trained on rendered boards only | Fine-tuned on real photos |
|---|---:|---:|
| Squares right | 77.03 % | 97.62 % |
| Occupied or empty right | 94.61 % | 99.22 % |
| Pieces right | 8.73 % | **90.24 %** |
| Whole boards right | 0 % | **32 %** |

The int8 ONNX model gives the same results (90.48 % of pieces, 35 whole boards), and the rendered test data stays at 99.96 % of squares. The most frequent mistakes are queens and kings, which look much alike from above.

Splitting single photos at random instead of by game makes the same short training look 15 points better on pieces (86 % instead of 71 %), because the test photos then have nearly identical neighbours in the training data. All photos show one board and one set of pieces, so results on other boards will be lower.

```bash
python scripts/download_data.py
python scripts/evaluate_real.py --split test
```


## C++ demo (ONNX Runtime)

[`cpp/`](cpp) is a small C++17 program that recognises a straightened board with the ONNX Runtime C++ API: no Python, TensorFlow or OpenCV, as on an embedded device whose camera pipeline already delivers a straightened grayscale frame. CMake downloads the prebuilt ONNX Runtime package once.

```bash
python scripts/predict.py photo.jpg --model models/square_classifier_int8.onnx --save-board board.pgm
cmake -S cpp -B build
cmake --build build --config Release
build/Release/board2fen_cpp models/square_classifier_int8.onnx board.pgm   # Linux: build/board2fen_cpp
```

On a 64-bit ARM Linux board (e.g. a Raspberry Pi) the same commands download the aarch64 package. When cross-compiling, point CMake at an extracted package instead: `-DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-aarch64-1.31.1`.


## License

[MIT](LICENSE)
