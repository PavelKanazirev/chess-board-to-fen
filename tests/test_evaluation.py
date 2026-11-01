import numpy as np

from board2fen.evaluation import (
    confusion_matrix,
    error_gallery,
    format_confusion,
    ms_per_board,
    per_class_accuracy,
    top_confusions,
)
from board2fen.labels import CLASSES

TRUTH = np.array([0, 0, 1, 1, 1, 2])
PREDICTED = np.array([0, 0, 1, 2, 2, 2])  # two "P" squares taken for "N"


def test_confusion_matrix_counts_truth_against_prediction():
    matrix = confusion_matrix(TRUTH, PREDICTED, num_classes=3)
    assert matrix.tolist() == [[2, 0, 0], [0, 1, 2], [0, 0, 1]]
    assert matrix.sum() == len(TRUTH)
    assert np.trace(matrix) == 4  # the diagonal: correct answers


def test_per_class_accuracy():
    accuracy = per_class_accuracy(confusion_matrix(TRUTH, PREDICTED, num_classes=4))
    assert accuracy[:3].tolist() == [1.0, 1 / 3, 1.0]
    assert np.isnan(accuracy[3])  # class 3 never occurs


def test_top_confusions_lists_the_most_frequent_mistakes():
    matrix = confusion_matrix(TRUTH, PREDICTED)
    assert top_confusions(matrix) == [("P", "N", 2)]
    assert top_confusions(confusion_matrix(TRUTH, TRUTH)) == []


def test_format_confusion_has_a_row_per_class():
    text = format_confusion(confusion_matrix(TRUTH, PREDICTED))
    lines = text.splitlines()
    assert len(lines) == 1 + len(CLASSES)
    assert all(name in text for name in CLASSES)


def test_error_gallery_shows_only_the_mistakes():
    squares = np.zeros((len(TRUTH), 32, 32), np.uint8)
    gallery = error_gallery(squares, TRUTH, PREDICTED, columns=8)
    assert gallery.shape == (32 * 3 + 14, 8 * 32 * 3)  # 2 mistakes fit in one row
    assert error_gallery(squares, TRUTH, TRUTH) is None


def test_ms_per_board_times_one_board():
    class CountingClassifier:
        img_size = 32
        calls = 0

        def predict(self, squares):
            self.calls += 1
            assert len(squares) == 64  # always exactly one board
            return np.zeros((len(squares), 13))

    classifier = CountingClassifier()
    ms = ms_per_board(classifier, np.zeros((100, 32, 32), np.uint8), runs=10)
    assert ms >= 0
    assert classifier.calls == 5 + 10  # warm-up runs + timed runs
