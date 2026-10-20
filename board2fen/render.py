"""Synthetic chessboard images: training data with perfect labels and no manual work.

Pieces are drawn with the chess symbols that Unicode defines (U+265A..U+265F), using any
installed font that contains them. We always draw the *filled* symbol and colour it ourselves:
white with a black outline for White, black for Black.
"""

import functools
import os
from pathlib import Path

import chess
from PIL import Image, ImageDraw, ImageFont

from board2fen.labels import square_of_index

GLYPHS = {"k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f"}

FONT_NAMES = (
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf",
    "seguisym.ttf",  # Segoe UI Symbol, part of Windows
    "FreeSerif.ttf",
    "NotoSansSymbols2-Regular.ttf",
)
FONT_DIRS = (
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
)

# (light square, dark square) colours as (red, green, blue).
DEFAULT_THEME = ((240, 217, 181), (181, 136, 99))


def has_chess_glyphs(font_path: str | Path) -> bool:
    """True if the font really draws chess pieces.

    A font without a symbol draws a placeholder box instead, and every missing symbol gets the
    same box. So we draw a knight and a bishop: both must leave ink, and they must differ.
    """
    try:
        font = ImageFont.truetype(str(font_path), 48)
    except OSError:
        return False
    images = []
    for glyph in (GLYPHS["n"], GLYPHS["b"]):
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).text((8, 0), glyph, font=font, fill=255)
        images.append(image)
    knight, bishop = images
    return knight.getbbox() is not None and bishop.getbbox() is not None and knight != bishop


@functools.cache
def find_fonts() -> tuple[str, ...]:
    """Paths of all installed fonts from FONT_NAMES that contain chess glyphs."""
    found = []
    for folder in FONT_DIRS:
        if not folder.is_dir():
            continue
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                path = os.path.join(root, name)
                if name in FONT_NAMES and has_chess_glyphs(path):
                    found.append(path)
    if not found:
        raise RuntimeError(
            "No font with chess glyphs found. On Linux run `sudo apt install fonts-dejavu-core`."
        )
    return tuple(found)


def draw_piece(
    draw: ImageDraw.ImageDraw, x0: int, y0: int, size: int, symbol: str, font: ImageFont.FreeTypeFont
) -> None:
    """Draw one piece (``symbol`` like "N" or "q") centred in the square at (x0, y0)."""
    glyph = GLYPHS[symbol.lower()]
    left, top, right, bottom = font.getbbox(glyph)  # the ink's box, relative to the text origin
    x = x0 + size / 2 - (left + right) / 2
    y = y0 + size / 2 - (top + bottom) / 2
    fill = (255, 255, 255) if symbol.isupper() else (0, 0, 0)
    draw.text((x, y), glyph, font=font, fill=fill, stroke_width=max(1, size // 32), stroke_fill=(0, 0, 0))


def render_board(
    board: chess.Board,
    square_px: int = 64,
    theme: tuple[tuple[int, int, int], tuple[int, int, int]] = DEFAULT_THEME,
    font_path: str | None = None,
) -> Image.Image:
    """A top-down picture of ``board`` with White at the bottom, 8 * square_px pixels wide."""
    light, dark = theme
    font = ImageFont.truetype(font_path or find_fonts()[0], int(square_px * 0.85))
    image = Image.new("RGB", (8 * square_px, 8 * square_px))
    draw = ImageDraw.Draw(image)
    for index in range(64):
        row, col = divmod(index, 8)
        x0, y0 = col * square_px, row * square_px
        color = light if (row + col) % 2 == 0 else dark
        draw.rectangle([x0, y0, x0 + square_px - 1, y0 + square_px - 1], fill=color)
        piece = board.piece_at(square_of_index(index))
        if piece:
            draw_piece(draw, x0, y0, square_px, piece.symbol(), font)
    return image
