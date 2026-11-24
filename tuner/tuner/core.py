"""tuner 核心 API（初始未实现）。"""


class Tuner:
    def __init__(self, scale, tolerance_cents):
        """构造"""
        raise NotImplementedError

    def correct(self, samples, rate):
        """校正"""
        raise NotImplementedError

    def quantize(self, hz):
        """单音高量化"""
        raise NotImplementedError
