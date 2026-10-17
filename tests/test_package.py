from importlib.metadata import version

import board2fen


def test_package_has_docstring():
    assert board2fen.__doc__


def test_version_matches_pyproject():
    # The version in pyproject.toml (package metadata) and in the code must agree.
    assert board2fen.__version__ == version("board2fen")
