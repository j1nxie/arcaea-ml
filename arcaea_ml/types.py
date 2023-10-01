from typing import Any, NamedTuple, Self, Tuple, Union
from collections.abc import Iterable
import numpy as np

Mat = np.ndarray[Any, np.dtype[np.generic]]


class XYWHRect(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    def __add__(self, other: Union[Self, Tuple[int, int, int, int]]):
        if not isinstance(other, Iterable) or len(other) != 4:
            raise ValueError()

        return self.__class__(*[a + b for a, b in zip(self, other)])

    def __sub__(self, other: Union[Self, Tuple[int, int, int, int]]):
        if not isinstance(other, Iterable) or len(other) != 4:
            raise ValueError()

        return self.__class__(*[a - b for a, b in zip(self, other)])
