"""Synthetic chessboard images: training data with perfect labels and no manual work.

Pieces are drawn with the chess symbols that Unicode defines (U+265A..U+265F), using any
installed font that contains them. We always draw the *filled* symbol and colour it ourselves:
white with a black outline for White, black for Black.
"""

import functools
import os
import random
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

THEMES = (
    DEFAULT_THEME,  # brown
    ((238, 238, 210), (118, 150, 86)),  # green
    ((222, 227, 230), (140, 162, 173)),  # blue-grey
    ((240, 240, 240), (170, 140, 200)),  # purple
    ((249, 229, 198), (209, 139, 71)),  # wood
    ((255, 255, 255), (150, 150, 150)),  # black-and-white print
)
HIGHLIGHT = (205, 210, 106)  # the yellow tint many sites use to mark the last move


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
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    size: int,
    symbol: str,
    font: ImageFont.FreeTypeFont,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    """Draw one piece (``symbol`` like "N" or "q") centred in the square at (x0, y0).

    ``dx`` and ``dy`` shift it away from the centre, in pixels.
    """
    glyph = GLYPHS[symbol.lower()]
    left, top, right, bottom = font.getbbox(glyph)  # the ink's box, relative to the text origin
    x = x0 + size / 2 - (left + right) / 2 + dx
    y = y0 + size / 2 - (top + bottom) / 2 + dy
    fill = (255, 255, 255) if symbol.isupper() else (0, 0, 0)
    draw.text((x, y), glyph, font=font, fill=fill, stroke_width=max(1, size // 32), stroke_fill=(0, 0, 0))


def draw_coordinates(
    draw: ImageDraw.ImageDraw, square_px: int, light: tuple, dark: tuple, font_path: str
) -> None:
    """Rank numbers down the a-file and file letters along rank 1, like most chess websites."""
    font = ImageFont.truetype(font_path, max(8, square_px // 5))
    for i in range(8):
        # Each label uses the colour of the *other* square type, so it stays readable.
        # In column 0 the square of row i is light when i is even; in row 7, column i is dark.
        rank_color = dark if i % 2 == 0 else light
        draw.text((2, i * square_px + 1), str(8 - i), font=font, fill=rank_color)
        file_color = light if i % 2 == 0 else dark
        position = ((i + 1) * square_px - 2, 8 * square_px - 2)
        draw.text(position, "abcdefgh"[i], font=font, fill=file_color, anchor="rd")


def render_board(
    board: chess.Board,
    square_px: int = 64,
    theme: tuple[tuple[int, int, int], tuple[int, int, int]] = DEFAULT_THEME,
    font_path: str | None = None,
    *,
    highlights: tuple[int, ...] = (),
    coordinates: bool = False,
    piece_scale: float = 0.85,
    piece_jitter: float = 0.0,
    rng: random.Random | None = None,
) -> Image.Image:
    """A top-down picture of ``board`` with White at the bottom, 8 * square_px pixels wide.

    ``highlights`` are image-order indices of squares to tint, like a "last move" marker.
    ``coordinates`` draws the 1-8 and a-h labels on the edge squares.
    ``piece_scale`` is the piece size as a fraction of the square. ``piece_jitter`` moves each
    piece by up to that fraction of the square in x and y, using ``rng`` for the random numbers.
    """
    light, dark = theme
    rng = rng or random.Random(0)
    font_path = font_path or find_fonts()[0]
    font = ImageFont.truetype(font_path, int(square_px * piece_scale))
    image = Image.new("RGB", (8 * square_px, 8 * square_px))
    draw = ImageDraw.Draw(image)
    for index in range(64):
        row, col = divmod(index, 8)
        x0, y0 = col * square_px, row * square_px
        color = light if (row + col) % 2 == 0 else dark
        if index in highlights:
            color = tuple((a + b) // 2 for a, b in zip(color, HIGHLIGHT, strict=True))
        draw.rectangle([x0, y0, x0 + square_px - 1, y0 + square_px - 1], fill=color)
    if coordinates:
        draw_coordinates(draw, square_px, light, dark, font_path)
    for index in range(64):
        piece = board.piece_at(square_of_index(index))
        if piece:
            row, col = divmod(index, 8)
            dx = rng.uniform(-piece_jitter, piece_jitter) * square_px
            dy = rng.uniform(-piece_jitter, piece_jitter) * square_px
            draw_piece(draw, col * square_px, row * square_px, square_px, piece.symbol(), font, dx, dy)

    return image
