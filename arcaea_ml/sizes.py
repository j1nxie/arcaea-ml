from typing import Tuple, Union


def apply_factor(num: Union[int, float], factor: float):
    return num * factor


class Sizes:
    def __init__(self, factor: float):
        self.factor = factor

    def apply_factor(self, num: Union[int, float]):
        return apply_factor(num, self.factor)

    @property
    def TOP_BAR_HEIGHT(self):
        return self.apply_factor(50)

    @property
    def JACKET_RIGHT_FROM_HOR_MID(self):
        return self.apply_factor(-235)

    @property
    def JACKET_WIDTH(self):
        return self.apply_factor(375)

    @property
    def SCORE_PANEL(self) -> Tuple[int, int]:
        return tuple(self.apply_factor(num) for num in [447, 233])  # type: ignore
