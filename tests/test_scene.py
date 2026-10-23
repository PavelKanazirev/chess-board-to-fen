import random

import chess
import cv2
import numpy as np
import pytest

from board2fen.render import render_board
from board2fen.scene import (
    paste_board,
    photo_effects,
    random_background,
    random_board_corners,
    render_scene,
)


def test_background_shape_and_type():
    background = random_background(120, 80, random.Random(0))
    assert background.shape == (80, 120, 3)
    assert background.dtype == np.uint8


@pytest.mark.parametrize("seed", range(10))
def test_corners_are_inside_the_image_and_in_order(seed):
    corners = random_board_corners(400, 300, random.Random(seed))
    assert corners.shape == (4, 2)
    assert (corners >= 0).all() and (corners[:, 0] < 400).all() and (corners[:, 1] < 300).all()
    a8, h8, h1, a1 = corners
    assert h8[0] > a8[0] and h1[0] > a1[0]  # the h-file is to the right of the a-file
    assert a1[1] > a8[1] and h1[1] > h8[1]  # rank 1 is below rank 8


def test_paste_board_puts_the_board_where_the_corners_say():
    background = np.zeros((100, 100, 3), np.uint8)
    red_board = np.full((40, 40, 3), (255, 0, 0), np.uint8)
    corners = np.float32([[20, 20], [60, 20], [60, 60], [20, 60]])
    scene = paste_board(background, red_board, corners)
    assert (scene[40, 40] == (255, 0, 0)).all()  # inside the corners: board
    assert (scene[5, 5] == 0).all() and (scene[80, 80] == 0).all()  # outside: background


def test_warping_back_with_the_corners_recovers_the_board():
    board_image = np.asarray(render_board(chess.Board(), square_px=32))  # 256 x 256
    rng = random.Random(3)
    corners = random_board_corners(400, 350, rng)
    scene = paste_board(random_background(400, 350, rng), board_image, corners)
    top_down = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    back = cv2.warpPerspective(scene, cv2.getPerspectiveTransform(corners, top_down), (256, 256))
    assert np.abs(back.astype(int) - board_image.astype(int)).mean() < 10


def test_photo_effects_change_the_image_but_not_its_shape():
    image = np.full((60, 80, 3), 128, np.uint8)
    out = photo_effects(image, random.Random(0))
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.array_equal(out, image)
    assert np.array_equal(out, photo_effects(image, random.Random(0)))  # same seed, same result


def test_render_scene_is_reproducible_and_bigger_than_the_board():
    image, corners = render_scene(chess.Board(), random.Random(5))
    again, corners_again = render_scene(chess.Board(), random.Random(5))
    assert np.array_equal(image, again) and np.array_equal(corners, corners_again)
    height, width = image.shape[:2]
    assert (corners[:, 0] < width).all() and (corners[:, 1] < height).all()
    board_width = np.linalg.norm(corners[1] - corners[0])
    assert board_width < width
