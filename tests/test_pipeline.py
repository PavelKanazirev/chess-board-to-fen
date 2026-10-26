import random
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest
from board2fen.pipeline import load_image, recognize

from board2fen.infer import load_classifier
from board2fen.positions import random_position
from board2fen.render import render_board
from board2fen.scene import render_scene

MODEL = Path(__file__).resolve().parent.parent / "models" / "square_classifier.keras"
POSITION = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"


@pytest.fixture(scope="module")
def classifier():
    if not MODEL.exists():
        pytest.skip("no trained model in models/")
    pytest.importorskip("keras")
    return load_classifier(MODEL)


def test_load_image_returns_rgb(tmp_path):
    image = np.zeros((10, 20, 3), np.uint8)
    image[:, :, 0] = 255  # pure red in RGB
    path = tmp_path / "red.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert np.array_equal(load_image(path), image)


def test_load_image_rejects_other_files(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not a picture")
    with pytest.raises(ValueError):
        load_image(path)


def test_recognize_photos(classifier):
    right = 0
    for seed in range(3_000_000, 3_000_005):  # test photos, never trained on
        rng = random.Random(seed)
        board = random_position(rng)
        image, _corners = render_scene(board, rng)
        result = recognize(image, classifier)
        assert result.detected
        right += result.fen.split()[0] == board.board_fen()
    assert right >= 4


def test_recognize_a_screenshot_without_detection(classifier):
    screenshot = np.asarray(render_board(chess.Board(POSITION), square_px=40))
    result = recognize(screenshot, classifier, detect=False)
    assert not result.detected
    assert result.fen.split()[0] == POSITION
    assert result.confidences.min() > 0.5


def test_recognize_a_board_seen_from_blacks_side(classifier):
    # From Black's side the squares are rotated by 180 degrees, but the pieces still stand upright.
    # So rotate the position, not the picture: flipping vertically and horizontally = 180 degrees.
    seen_from_black = chess.Board(POSITION).transform(chess.flip_vertical).transform(chess.flip_horizontal)
    screenshot = np.asarray(render_board(seen_from_black, square_px=40))
    result = recognize(screenshot, classifier, detect=False, flipped=True, turn="b")
    assert result.fen.split()[:2] == [POSITION, "b"]
