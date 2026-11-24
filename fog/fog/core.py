"""fog 核心 API（初始未实现）。"""


class Fog:
    def __init__(self, density, height_falloff=0.0):
        """构造"""
        raise NotImplementedError

    def factor(self, distance, height=0.0):
        """雾量"""
        raise NotImplementedError
