from typing import Union

from .crop import crop_black_edges, crop_xywh
from .definition import Device
from .sizes import Sizes
from .types import Mat, XYWHRect


def to_int(num: Union[int, float]) -> int:
    return round(num)


class DeviceRois:
    def __init__(self, device: Device, img: Mat):
        self.device = device
        self.sizes = Sizes(self.device.factor)
        self.__img = img

    @staticmethod
    def construct_int_xywh_rect(x, y, w, h) -> XYWHRect:
        return XYWHRect(*[to_int(item) for item in [x, y, w, h]])

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img: Mat):
        self.__img = (
            crop_black_edges(img) if self.device.crop_black_edges else img.copy()
        )

    @property
    def h(self):
        return self.img.shape[0]

    @property
    def vmid(self):
        return self.h / 2

    @property
    def w(self):
        return self.img.shape[1]

    @property
    def hmid(self):
        return self.w / 2

    @property
    def h_without_top_bar(self):
        return self.h - self.sizes.TOP_BAR_HEIGHT

    @property
    def h_without_top_bar_mid(self):
        return self.sizes.TOP_BAR_HEIGHT + self.h_without_top_bar / 2

    @property
    def jacket_rect(self):
        return self.construct_int_xywh_rect(
            x=self.hmid
            + self.sizes.JACKET_RIGHT_FROM_HOR_MID
            - self.sizes.JACKET_WIDTH,
            y=self.h_without_top_bar_mid - self.sizes.SCORE_PANEL[1] / 2,
            w=self.sizes.JACKET_WIDTH,
            h=self.sizes.JACKET_WIDTH,
        )

    @property
    def jacket(self):
        return crop_xywh(self.img, self.jacket_rect)


class DeviceAutoRois(DeviceRois):
    @staticmethod
    def get_factor(width: int, height: int):
        ratio = width / height
        return ((width / 16) * 9) / 720 if ratio < (16 / 9) else height / 720

    def __init__(self, img: Mat):
        factor = self.get_factor(img.shape[1], img.shape[0])
        self.sizes = Sizes(factor)
        self.__img = None
        self.img = img

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, img: Mat):
        self.__img = crop_black_edges(img)
