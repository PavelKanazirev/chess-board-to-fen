"""Measuring a classifier: accuracy per class, the confusion matrix, and pictures of mistakes.

Plain numpy and OpenCV, no TensorFlow: these functions work with predictions from any runtime
(Keras today, TensorFlow Lite and ONNX Runtime later).
"""

import cv2
import numpy as np

from board2fen.labels import CLASSES


def confusion_matrix(truth: np.ndarray, predicted: np.ndarray, num_classes: int = len(CLASSES)) -> np.ndarray:
    """Count every (true class, predicted class) pair: rows are the truth, columns the prediction.

    The diagonal holds the correct answers; everything else is a mistake.
    """
    matrix = np.zeros((num_classes, num_classes), np.int64)
    np.add.at(matrix, (np.asarray(truth), np.asarray(predicted)), 1)
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    """For each true class, the fraction recognised correctly (NaN for a class that never occurs)."""
    totals = matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diag(matrix) / totals


def top_confusions(matrix: np.ndarray, k: int = 5) -> list[tuple[str, str, int]]:
    """The k most frequent mistakes as (true class, predicted class, count), most frequent first."""
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    result = []
    for flat_index in np.argsort(mistakes, axis=None)[::-1][:k]:
        true_class, predicted_class = divmod(int(flat_index), matrix.shape[1])
        count = int(mistakes[true_class, predicted_class])
        if count == 0:
            break
        result.append((CLASSES[true_class], CLASSES[predicted_class], count))
    return result


def format_confusion(matrix: np.ndarray) -> str:
    """The confusion matrix as a text table with class names."""
    header = "true\\pred " + " ".join(f"{name:>5}" for name in CLASSES)
    rows = [
        f"{name:>9} " + " ".join(f"{value:5d}" for value in row)
        for name, row in zip(CLASSES, matrix, strict=True)
    ]
    return "\n".join([header, *rows])


def error_gallery(
    squares: np.ndarray, truth: np.ndarray, predicted: np.ndarray, max_items: int = 48, columns: int = 8
) -> np.ndarray | None:
    """A grayscale image of misclassified squares, each labelled "true>predicted".

    Returns None when there are no mistakes.
    """
    wrong = np.flatnonzero(np.asarray(truth) != np.asarray(predicted))[:max_items]
    if len(wrong) == 0:
        return None
    zoom = squares.shape[1] * 3
    tile_height = zoom + 14  # room for the text under each square
    rows = (len(wrong) + columns - 1) // columns
    gallery = np.full((rows * tile_height, columns * zoom), 255, np.uint8)
    for n, index in enumerate(wrong):
        row, col = divmod(n, columns)
        y, x = row * tile_height, col * zoom
        gallery[y : y + zoom, x : x + zoom] = cv2.resize(
            squares[index], (zoom, zoom), interpolation=cv2.INTER_NEAREST
        )
        text = f"{CLASSES[truth[index]]}>{CLASSES[predicted[index]]}"
        cv2.putText(gallery, text, (x + 2, y + zoom + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1, cv2.LINE_AA)
    return gallery
