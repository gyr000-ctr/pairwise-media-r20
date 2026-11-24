"""bezier3d 核心 API（初始未实现）。"""


class Bezier3D:
    def point(self, c, t):
        """取点"""
        raise NotImplementedError

    def frame(self, c, t):
        """切线/法线/副法线"""
        raise NotImplementedError

    def length(self, c, n=64):
        """弧长"""
        raise NotImplementedError
