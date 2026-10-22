"""Random chess positions to render as training data."""

import random

import chess

PIECE_SYMBOLS = "PNBRQKpnbrqk"


def random_game_position(rng: random.Random, max_plies: int = 120) -> chess.Board:
    """The position after a random number of random legal moves from the start.

    Looks like a real game, but most squares hold pawns or nothing.
    """
    board = chess.Board()
    for _ in range(rng.randint(0, max_plies)):
        moves = list(board.legal_moves)
        if not moves:  # checkmate or stalemate
            break
        board.push(rng.choice(moves))
    return board


def random_scattered_position(rng: random.Random) -> chess.Board:
    """Random pieces on random squares. Not a legal position, but every piece type is common."""
    board = chess.Board(None)
    density = rng.uniform(0.1, 0.6)
    for square in chess.SQUARES:
        if rng.random() < density:
            board.set_piece_at(square, chess.Piece.from_symbol(rng.choice(PIECE_SYMBOLS)))
    return board


def random_position(rng: random.Random) -> chess.Board:
    """Half game-like positions, half scattered ones, so the classifier sees enough rare pieces."""
    if rng.random() < 0.5:
        return random_game_position(rng)
    return random_scattered_position(rng)
