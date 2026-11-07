import subprocess
import sys
import zipfile
from pathlib import Path

import chess
import numpy as np
import pytest

from board2fen.labels import CLASS_TO_INDEX, board_to_labels
from board2fen.real_data import (
    extract_dataset,
    find_photos,
    group_positions,
    labels_from_filename,
    split_photos,
)
from board2fen.render import render_board

ROOT = Path(__file__).resolve().parent.parent


def test_labels_from_filename_reads_the_placement():
    labels = labels_from_filename("data/1B2b3-Kp6-8-8-2k5-8-8-8.JPG")
    assert labels.shape == (64,)
    assert labels[1] == CLASS_TO_INDEX["B"]  # b8
    assert labels[4] == CLASS_TO_INDEX["b"]  # e8
    assert labels[8] == CLASS_TO_INDEX["K"]  # a7
    assert (labels != CLASS_TO_INDEX["empty"]).sum() == 5


def test_labels_from_filename_ignores_what_follows_an_underscore():
    first = labels_from_filename("8-8-8-8-8-8-8-K7.jpg")
    second = labels_from_filename("8-8-8-8-8-8-8-K7_2.jpg")
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", ["IMG_1234.jpg", "8-8-8.jpg", "9-8-8-8-8-8-8-8.jpg"])
def test_labels_from_filename_rejects_other_names(name):
    with pytest.raises(ValueError):
        labels_from_filename(name)


def test_find_photos_is_sorted_and_skips_other_files(tmp_path):
    for name in ("b.JPG", "a.png", "notes.txt"):
        (tmp_path / name).touch()
    assert [path.name for path in find_photos(tmp_path)] == ["a.png", "b.JPG"]


def test_extract_dataset_keeps_only_the_wanted_files(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("repo-abc123/LICENSE.md", "MIT")
        zip_file.writestr("repo-abc123/photos/a.jpg", "pixels")
        zip_file.writestr("repo-abc123/assets/logo.png", "not needed")
    target = tmp_path / "dataset"
    extract_dataset(archive, target, ("LICENSE.md", "photos/"))
    files = sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
    assert files == ["LICENSE.md", "photos/a.jpg"]


def test_evaluate_real_script_on_rendered_boards(tmp_path):
    pytest.importorskip("onnxruntime")
    model = ROOT / "models" / "square_classifier_int8.onnx"
    for placement in ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R", "8/8/8/4k3/8/8/4K3/8"):
        image = render_board(chess.Board(placement + " w - - 0 1"), square_px=40)
        image.save(tmp_path / (placement.replace("/", "-") + ".png"))
    command = [sys.executable, str(ROOT / "scripts" / "evaluate_real.py"), "--dir", str(tmp_path)]
    command += ["--no-detect", "--model", str(model), "--out-dir", str(tmp_path / "out")]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert "2 photos" in output
    assert "boards: 100.00%" in output


def game(start_fen: str, moves: int) -> list[chess.Board]:
    """The positions of a game played from ``start_fen``: always the first legal move."""
    board = chess.Board(start_fen)
    positions = [board.copy()]
    for _ in range(moves):
        board.push(next(iter(board.legal_moves)))
        positions.append(board.copy())
    return positions


def test_group_positions_links_a_game_move_by_move():
    positions = game(chess.STARTING_FEN, 3) + [chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")]
    labels = np.stack([np.array(board_to_labels(board)) for board in positions])
    group = group_positions(labels)
    assert len(set(group[:4])) == 1  # the 4 positions of the game belong together
    assert group[4] != group[0]  # the lone kings don't


def test_split_photos_keeps_every_game_in_one_part(tmp_path):
    games = [
        game(chess.STARTING_FEN, 5),  # 6 photos
        game("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 2),  # 3 photos
        game("7k/8/8/8/8/8/8/K6Q w - - 0 1", 0),  # 1 photo
    ]
    photo_game = {}
    for number, positions in enumerate(games):
        for board in positions:
            path = tmp_path / (board.board_fen().replace("/", "-") + ".jpg")
            path.touch()
            photo_game[path] = number
    parts = split_photos(find_photos(tmp_path))
    part_of = {path: name for name, part in parts.items() for path in part}
    assert sorted(part_of) == sorted(photo_game)  # every photo in exactly one part
    for number in range(len(games)):
        assert len({part_of[path] for path, g in photo_game.items() if g == number}) == 1
    # Largest game first, each to the part furthest below its target (7, 1 and 2 of 10 photos):
    # 6 photos -> train, 3 -> test, then 1 -> train (a tie with val; train comes first).
    assert [len(parts[name]) for name in ("train", "val", "test")] == [7, 0, 3]
