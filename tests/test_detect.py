import itertools
import random

import chess
import numpy as np
import pytest

from board2fen.detect import order_corners, to_gray, warp_board
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
