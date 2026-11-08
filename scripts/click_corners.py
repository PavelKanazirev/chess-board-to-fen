"""Click the 4 corners of the squares in a photo where the detector finds the wrong board.

Usage:
    python scripts/click_corners.py data/real/my_photos/<photo>.jpg [more photos ...]

Click the 4 outer corners of the 8 x 8 squares (not of the board's frame), in any order.
Backspace removes the last click, Enter saves and goes on to the next photo, Escape skips one.
The corners are saved next to the photo (<photo>.corners.json); predict.py, evaluate_real.py
and make_real_dataset.py then use them instead of the detector.
"""

import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from board2fen.detect import order_corners
from board2fen.pipeline import load_image, save_corners


class CornerPicker:
    """A window that shows a photo, scaled to fit the screen, and collects 4 clicks."""

    def __init__(self, root: tk.Tk, photo: Path, max_side: int = 900):
        self.root, self.photo = root, photo
        image = Image.fromarray(load_image(photo))
        self.scale = min(1.0, max_side / max(image.size))
        shown = image.resize((round(image.width * self.scale), round(image.height * self.scale)))
        self.tk_image = ImageTk.PhotoImage(shown)  # keep a reference, or Tk shows an empty window
        self.canvas = tk.Canvas(root, width=shown.width, height=shown.height, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.clicks: list[tuple[int, int]] = []
        self.saved = None
        root.title(f"{photo.name}: click the 4 corners of the squares, then press Enter")
        self.canvas.bind("<Button-1>", self.click)
        root.bind("<BackSpace>", self.undo)
        root.bind("<Return>", self.save)
        root.bind("<Escape>", lambda _event: root.destroy())

    def click(self, event: tk.Event) -> None:
        if len(self.clicks) < 4:
            self.clicks.append((event.x, event.y))
            self.redraw()

    def undo(self, _event: tk.Event | None = None) -> None:
        if self.clicks:
            self.clicks.pop()
            self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("mark")  # everything we drew has the tag "mark"; the photo hasn't
        for x, y in self.clicks:
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="red", width=2, tags="mark")
        if len(self.clicks) == 4:
            outline = order_corners(self.clicks).flatten().tolist()
            self.canvas.create_polygon(outline, outline="lime", fill="", width=2, tags="mark")

    def save(self, _event: tk.Event | None = None) -> None:
        if len(self.clicks) == 4:
            corners = np.float32(self.clicks) / self.scale  # back to the photo's own pixels
            self.saved = save_corners(self.photo, corners)
            print(f"corners -> {self.saved}")
            self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("photos", type=Path, nargs="+")
    args = parser.parse_args()
    for photo in args.photos:
        root = tk.Tk()
        CornerPicker(root, photo)
        root.mainloop()


if __name__ == "__main__":
    main()
