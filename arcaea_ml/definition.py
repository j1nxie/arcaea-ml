from attrs import define, field
from typing import Iterable

from .types import XYWHRect


def iterable_to_xywh_field(__iter: Iterable) -> XYWHRect:
    return XYWHRect(*__iter)


@define(kw_only=True)
class Device:
    version = field(type=int)  # type: ignore
    uuid = field(type=int)  # type: ignore
    name = field(type=str)  # type: ignore
    crop_black_edges = field(type=bool)  # type: ignore
    factor = field(type=float)  # type: ignore
