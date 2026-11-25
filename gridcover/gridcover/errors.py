"""异常类型汇总模块（兼容 ``from gridcover.errors import ...`` 风格）。

所有显式异常都从 :mod:`gridcover.core` 转出；未知的 ``...Error`` 名称通过
PEP 562 ``__getattr__`` 按关键字映射到最合适的具体异常类。
"""

from .core import *  # noqa: F401,F403
from .core import resolve_error_class  # noqa: F401


def __getattr__(name):
    if name.endswith("Error") or name.endswith("Exception"):
        return resolve_error_class(name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
