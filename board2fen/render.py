"""Synthetic chessboard images: training data with perfect labels and no manual work.

Pieces are drawn with the chess symbols that Unicode defines (U+265A..U+265F), using any
installed font that contains them. We always draw the *filled* symbol and colour it ourselves:
white with a black outline for White, black for Black.
"""

import functools
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
