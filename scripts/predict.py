"""Recognise the chess position in a photo or screenshot and print its FEN.

Usage:
    python scripts/predict.py photo.jpg
    python scripts/predict.py screenshot.png --no-detect      # the picture is just the board
    python scripts/predict.py photo.jpg --flip --black        # seen from Black's side, Black to move
    python scripts/predict.py --render 42                     # try it on a rendered test photo
"""

import argparse
import random
import time
from pathlib import Path

import chess
import cv2

from board2fen.infer import load_classifier
from board2fen.labels import lichess_analysis_url, square_of_index
from board2fen.pipeline import load_image, recognize
from board2fen.positions import random_position
from board2fen.scene import render_scene


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", type=Path, help="a JPG or PNG file")
    parser.add_argument("--render", type=int, metavar="SEED", help="use a rendered test photo instead")
    parser.add_argument("--model", type=Path, default=Path("models/square_classifier.keras"))
    parser.add_argument("--no-detect", action="store_true", help="the picture shows only the board")
    parser.add_argument("--flip", action="store_true", help="the picture is taken from Black's side")
    parser.add_argument("--black", action="store_true", help="Black to move")
    args = parser.parse_args()
    if (args.image is None) == (args.render is None):
        parser.error("give either an image file or --render SEED")

    expected = None
    if args.render is not None:
        rng = random.Random(args.render)
        board = random_position(rng)
        image, _corners = render_scene(board, rng)
        expected = board.board_fen()
        path = Path("data") / f"render_{args.render}.jpg"
        path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"rendered test photo -> {path}")
    else:
        image = load_image(args.image)

    classifier = load_classifier(args.model)
    start = time.perf_counter()
    turn = "b" if args.black else "w"
    result = recognize(image, classifier, detect=not args.no_detect, flipped=args.flip, turn=turn)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(result.fen)
    print(lichess_analysis_url(result.fen))
    found = "board found" if result.detected else "no board found, used the whole picture"
    print(f"{found}, {elapsed_ms:.0f} ms, lowest confidence {result.confidences.min():.2f}")
    unsure = [
        chess.square_name(square_of_index(index, args.flip))
        for index in range(64)
        if result.confidences[index] < 0.9
    ]
    if unsure:
        print("not sure about:", " ".join(unsure))
    if expected is not None:
        verdict = "correct" if result.fen.split()[0] == expected else "WRONG"
        print(f"expected {expected}: {verdict}")


if __name__ == "__main__":
    main()
