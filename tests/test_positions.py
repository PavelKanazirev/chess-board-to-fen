import random

import chess

from board2fen.positions import random_game_position, random_position, random_scattered_position


def test_the_same_seed_gives_the_same_position():
    assert random_position(random.Random(7)).fen() == random_position(random.Random(7)).fen()


def test_game_positions_are_legal():
    for seed in range(20):
        board = random_game_position(random.Random(seed))
        assert board.is_valid()


def test_scattered_positions_contain_every_piece_type():
    seen = set()
    for seed in range(50):
        board = random_scattered_position(random.Random(seed))
        seen.update(piece.symbol() for piece in board.piece_map().values())
    assert seen == set("PNBRQKpnbrqk")


def test_random_position_mixes_both_kinds():
    boards = [random_position(random.Random(seed)) for seed in range(40)]
    kinds = {board.is_valid() for board in boards}
    assert kinds == {True, False}
    assert all(isinstance(board, chess.Board) for board in boards)
