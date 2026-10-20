from board2fen.render import find_fonts, has_chess_glyphs


def test_at_least_one_font_with_chess_glyphs_is_found():
    fonts = find_fonts()
    assert fonts
    assert all(has_chess_glyphs(path) for path in fonts)


def test_a_missing_font_is_rejected(tmp_path):
    assert not has_chess_glyphs(tmp_path / "no_such_font.ttf")
