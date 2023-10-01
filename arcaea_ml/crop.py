from math import floor
from typing import Tuple

import numpy as np

from .types import Mat


def crop_xywh(mat: Mat, rect: Tuple[int, int, int, int]):
    x, y, w, h = rect
    return mat[y : y + h, x : x + w]


def is_black_edge(list_of_pixels: Mat, black_pixel: Mat, ratio: float = 0.5):
    pixels = list_of_pixels.reshape([-1, 3])
    return np.count_nonzero(np.all(pixels < black_pixel, axis=1)) > floor(  # type: ignore  # noqa: E501
        len(pixels) * ratio
    )


def crop_black_edges(img_bgr: Mat, black_threshold: int = 50):
    cropped = img_bgr.copy()
    black_pixel = np.array([black_threshold] * 3, img_bgr.dtype)
    height, width = img_bgr.shape[:2]
    left = 0
    right = width
    top = 0
    bottom = height

    for i in range(width):
        column = cropped[:, i]
        if not is_black_edge(column, black_pixel):
            break
        left += 1

    for i in sorted(range(width), reverse=True):
        column = cropped[:, i]
        if i <= left + 1 or not is_black_edge(column, black_pixel):
            break
        right -= 1

    for i in range(height):
        row = cropped[i]
        if not is_black_edge(row, black_pixel):
            break
        top += 1

    for i in sorted(range(height), reverse=True):
        row = cropped[i]
        if i <= top + 1 or not is_black_edge(row, black_pixel):
            break
        bottom -= 1

    return cropped[top:bottom, left:right]
