import itertools
import random

import chess
import cv2
import numpy as np
import pytest

from board2fen.detect import (
    corner_error,
    detect_board,
    line_intersection,
    order_corners,
    split_squares,
    to_gray,
    warp_board,
)
from board2fen.labels import board_to_labels
from board2fen.positions import random_position
from board2fen.render import render_board
from board2fen.scene import paste_board, random_background, random_board_corners, render_scene

SQUARE = [[10, 10], [90, 10], [90, 90], [10, 90]]  # top-left, top-right, bottom-right, bottom-left


@pytest.mark.parametrize("shuffled", list(itertools.permutations(SQUARE)))
def test_order_corners_sorts_any_order(shuffled):
    assert order_corners(shuffled).tolist() == SQUARE


@pytest.mark.parametrize("seed", range(5))
def test_order_corners_restores_scene_corners(seed):
    rng = random.Random(seed)
    _image, corners = render_scene(random_position(rng), rng)
    shuffled = corners[[2, 0, 3, 1]]
    assert np.array_equal(order_corners(shuffled), corners)


def test_warp_board_size_and_type():
    image = np.zeros((300, 400, 3), np.uint8)
    corners = np.float32([[50, 20], [350, 30], [340, 280], [60, 270]])
    board = warp_board(image, corners, img_size=32)
    assert board.shape == (256, 256)
    assert board.dtype == np.uint8


def test_warp_board_straightens_a_pasted_board():
    board_image = np.asarray(render_board(chess.Board(), square_px=32))  # 256 x 256, RGB
    rng = random.Random(4)
    corners = random_board_corners(500, 400, rng)
    scene = paste_board(random_background(500, 400, rng), board_image, corners)
    straight = warp_board(scene, corners, img_size=32)
    assert np.abs(straight.astype(int) - to_gray(board_image).astype(int)).mean() < 10


def test_split_squares_keeps_image_order():
    size = 4
    board = np.zeros((8 * size, 8 * size), np.uint8)
    for index in range(64):
        row, col = divmod(index, 8)
        board[row * size : (row + 1) * size, col * size : (col + 1) * size] = index
    squares = split_squares(board, size)
    assert squares.shape == (64, size, size)
    assert [int(square[0, 0]) for square in squares] == list(range(64))
    assert all((square == index).all() for index, square in enumerate(squares))


def test_split_squares_rejects_the_wrong_size():
    with pytest.raises(ValueError):
        split_squares(np.zeros((250, 256), np.uint8), 32)


def test_squares_from_a_scene_match_the_labels():
    rng = random.Random(11)
    board = chess.Board()
    image, corners = render_scene(board, rng)
    squares = split_squares(warp_board(image, corners, img_size=32), 32)
    labels = np.array(board_to_labels(board))
    contrast = squares.reshape(64, -1).std(axis=1)  # a piece adds dark and light pixels
    assert contrast[labels != 0].mean() > 2 * contrast[labels == 0].mean()


def clear_scene():
    """The start position, slightly tilted, on a plain dark background."""
    board_image = np.asarray(render_board(chess.Board(), square_px=40))  # 320 x 320
    corners = np.float32([[110, 70], [430, 90], [420, 400], [90, 390]])
    scene = paste_board(np.full((480, 520, 3), 60, np.uint8), board_image, corners)
    return scene, corners


def test_detect_board_finds_a_clear_board():
    scene, corners = clear_scene()
    found = detect_board(scene)
    assert found is not None
    assert np.abs(found - corners).max() < 0.03 * 320  # every corner within 3% of the board size


def test_detect_board_returns_none_without_a_board():
    assert detect_board(np.full((300, 400, 3), 128, np.uint8)) is None


def test_detect_board_ignores_small_squares():
    image = np.full((400, 400, 3), 200, np.uint8)
    cv2.rectangle(image, (10, 10), (60, 60), (0, 0, 0), thickness=-1)  # about 2% of the picture
    assert detect_board(image) is None


def test_corner_error_is_relative_to_the_board_size():
    truth = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert corner_error(truth, truth) == 0
    moved = truth.copy()
    moved[2] += [6, 8]  # one corner off by 10 pixels (a 6-8-10 triangle)
    assert corner_error(moved, truth) == pytest.approx(0.10)


def test_line_intersection():
    horizontal = (np.array([0.0, 0.0]), np.array([1.0, 0.0]))  # the line y = 0
    vertical = (np.array([5.0, -1.0]), np.array([5.0, 1.0]))  # the line x = 5
    assert line_intersection(*horizontal, *vertical).tolist() == [5.0, 0.0]


def test_parallel_lines_do_not_intersect():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert line_intersection(a, b, a + [0, 5], b + [0, 5]) is None


def test_detect_board_on_random_scenes():
    good = 0
    for seed in range(30):
        rng = random.Random(seed)
        image, truth = render_scene(random_position(rng), rng)
        found = detect_board(image)
        good += found is not None and corner_error(found, truth) < 0.03
    assert good >= 26
