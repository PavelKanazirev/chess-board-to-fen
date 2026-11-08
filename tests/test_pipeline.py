import random
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest
from PIL import Image

from board2fen.detect import split_squares
from board2fen.infer import load_classifier
from board2fen.pipeline import load_corners, load_image, recognize, save_corners, save_pgm
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


def test_save_pgm_writes_a_header_and_raw_pixels(tmp_path):
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)  # 3 rows, 4 columns
    path = save_pgm(tmp_path / "small.pgm", gray)
    assert path.read_bytes() == b"P5\n4 3\n255\n" + bytes(range(12))
    assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), gray)  # OpenCV agrees


def test_save_pgm_rejects_color_images(tmp_path):
    with pytest.raises(ValueError):
        save_pgm(tmp_path / "color.pgm", np.zeros((4, 4, 3), np.uint8))


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


def test_recognize_returns_the_straightened_board(classifier):
    screenshot = np.asarray(render_board(chess.Board(POSITION), square_px=40))
    result = recognize(screenshot, classifier, detect=False)
    assert result.board.shape == (8 * classifier.img_size, 8 * classifier.img_size)
    assert result.board.dtype == np.uint8
    # The board is exactly what the classifier saw: classifying it again gives the same labels.
    squares = split_squares(result.board, classifier.img_size)
    assert np.array_equal(classifier.predict(squares).argmax(axis=1), result.labels)


def test_load_image_turns_phone_photos_upright(tmp_path):
    # Phones store the picture as the sensor saw it, plus an EXIF tag that says how to turn it.
    image = Image.new("RGB", (40, 20), (255, 0, 0))  # wide
    exif = Image.Exif()
    exif[0x0112] = 6  # "Orientation": show it turned by 90 degrees clockwise
    image.save(tmp_path / "phone.jpg", exif=exif)
    assert load_image(tmp_path / "phone.jpg").shape == (40, 20, 3)  # tall, as the phone showed it


def test_corners_round_trip_next_to_the_photo(tmp_path):
    photo = tmp_path / "8-8-8-8-8-8-8-K7.jpg"
    assert load_corners(photo) is None
    clicked = np.float32([[90, 390], [110, 70], [420, 400], [430, 90]])  # clicked in any order
    path = save_corners(photo, clicked)
    assert path.name == "8-8-8-8-8-8-8-K7.corners.json"
    assert load_corners(photo).tolist() == [[110, 70], [430, 90], [420, 400], [90, 390]]  # a8 h8 h1 a1


def test_recognize_uses_given_corners():
    class AlwaysEmpty:
        img_size = 32

        def predict(self, squares):
            probabilities = np.zeros((len(squares), 13), np.float32)
            probabilities[:, 0] = 1.0
            return probabilities

    image = np.full((300, 400, 3), 128, np.uint8)  # no board to detect
    corners = np.float32([[50, 20], [350, 30], [340, 280], [60, 270]])
    result = recognize(image, AlwaysEmpty(), corners=corners)
    assert result.detected
    assert np.array_equal(result.corners, corners)
    assert not recognize(image, AlwaysEmpty()).detected
