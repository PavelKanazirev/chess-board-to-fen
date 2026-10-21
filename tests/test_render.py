import chess
import numpy as np
import pytest

from board2fen.render import DEFAULT_THEME, HIGHLIGHT, find_fonts, has_chess_glyphs, render_board


def test_at_least_one_font_with_chess_glyphs_is_found():
    fonts = find_fonts()
    assert fonts
    assert all(has_chess_glyphs(path) for path in fonts)


def test_a_missing_font_is_rejected(tmp_path):
    assert not has_chess_glyphs(tmp_path / "no_such_font.ttf")


def square_pixels(image, index, square_px):
    """The pixels of one square (image order index) as a numpy array of shape (H, W, 3)."""
    row, col = divmod(index, 8)
    pixels = np.asarray(image)
    return pixels[row * square_px : (row + 1) * square_px, col * square_px : (col + 1) * square_px]


@pytest.mark.parametrize("square_px", [32, 64])
def test_board_size(square_px):
    image = render_board(chess.Board(), square_px=square_px)
    assert image.size == (8 * square_px, 8 * square_px)
    assert image.mode == "RGB"


def test_empty_board_has_the_theme_colours():
    light, dark = DEFAULT_THEME
    image = render_board(chess.Board(None), square_px=32)
    assert (square_pixels(image, 0, 32) == light).all()  # a8 is a light square
    assert (square_pixels(image, 1, 32) == dark).all()  # b8 is dark
    assert (square_pixels(image, 63, 32) == light).all()  # h1 is light


def test_pieces_are_drawn_in_their_colour():
    image = render_board(chess.Board(), square_px=64)
    empty_dark = square_pixels(image, 17, 64).mean()  # b6, an empty dark square
    white_rook = square_pixels(image, 56, 64).mean()  # a1, also a dark square
    black_rook = square_pixels(image, 7, 64).mean()  # h8, also a dark square
    assert black_rook < empty_dark < white_rook


def test_highlighted_square_is_tinted():
    light, _dark = DEFAULT_THEME
    image = render_board(chess.Board(None), square_px=32, highlights=(0,))
    a8 = square_pixels(image, 0, 32)
    assert (a8 == [(a + b) // 2 for a, b in zip(light, HIGHLIGHT, strict=True)]).all()
    assert (square_pixels(image, 2, 32) == light).all()  # c8 is not highlighted


def test_coordinates_are_drawn_on_the_edge_squares():
    plain = render_board(chess.Board(None), square_px=64)
    labelled = render_board(chess.Board(None), square_px=64, coordinates=True)
    for index in (0, 56, 63):  # a8 has "8", a1 has "1" and "a", h1 has "h"
        assert (square_pixels(labelled, index, 64) != square_pixels(plain, index, 64)).any()
    assert (square_pixels(labelled, 27, 64) == square_pixels(plain, 27, 64)).all()  # d5: no label
