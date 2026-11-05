"""Download the public real-photo dataset into data/real. See DATA_SOURCES.md for its license.

Usage:
    python scripts/download_data.py

About 2.2 GB. The dataset is pinned to one commit, so everybody gets the same photos.
"""

import argparse
import shutil
import urllib.request
from pathlib import Path

from board2fen.real_data import extract_dataset, find_photos

COMMIT = "62a8f5b0a45a931c91fd11bf6e1fe0c9fa0d25d9"
URL = f"https://github.com/samryan18/chess-dataset/archive/{COMMIT}.zip"
KEEP = ("LICENSE.md", "labeled_originals/", "labeled_preprocessed/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/real"))
    args = parser.parse_args()

    target = args.out_dir / "chess-dataset"
    if target.exists():
        print(f"{target} already exists")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "chess-dataset.zip"
    if not archive.exists():
        print(f"downloading {URL} ...")
        partial = args.out_dir / "chess-dataset.zip.part"
        with urllib.request.urlopen(URL) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial.rename(archive)  # only a complete download gets the final name
    print("extracting ...")
    extract_dataset(archive, target, KEEP)
    archive.unlink()
    print(f"{len(find_photos(target / 'labeled_originals'))} photos -> {target}")


if __name__ == "__main__":
    main()
