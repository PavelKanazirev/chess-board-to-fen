# chess-board-to-fen

[![CI](https://github.com/PavelKanazirev/chess-board-to-fen/actions/workflows/ci.yml/badge.svg)](https://github.com/PavelKanazirev/chess-board-to-fen/actions/workflows/ci.yml)

> **Status: work in progress.** This project is being built step by step, one pull request at a time. See the [roadmap](#roadmap).

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


## License

[MIT](LICENSE)
