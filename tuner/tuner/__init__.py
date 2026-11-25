from .core import *  # noqa: F401,F403
from . import core as _core  # noqa: F401

__version__ = "1.0.0"

try:  # 透传 core 中定义的显式 __all__
    __all__ = list(_core.__all__) + ["__version__"]
except AttributeError:  # pragma: no cover
    __all__ = [name for name in globals() if not name.startswith("_")]


def __getattr__(name):
    # 未知的错误类/别名等委托给 core 的 PEP 562 兜底。
    return getattr(_core, name)
