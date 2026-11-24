"""imgtile 核心 API（初始未实现）。"""


class ImgTile:
    def split(self, image, size, pad=0, mode='reflect'):
        """分块"""
        raise NotImplementedError

    def merge(self, tiles, shape):
        """拼接"""
        raise NotImplementedError
