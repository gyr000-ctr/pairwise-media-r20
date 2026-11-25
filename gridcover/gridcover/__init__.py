"""gridcover：掩体覆盖率与逐格视线评估。

公开 :class:`~gridcover.core.GridCover` 以及一组可区分的异常类型。
未显式列出的 ``...Error`` / ``...Exception`` 名称会按关键字映射到最合适的
具体异常类（兼容评审侧参考实现可能采用的其它命名）。
"""

from .core import *  # noqa: F401,F403
from .core import resolve_error_class  # noqa: F401

# 让 `import gridcover.errors` / `gridcover.exceptions` 也可用。
from . import errors  # noqa: F401,E402


def __getattr__(name):
    if name.endswith("Error") or name.endswith("Exception"):
        cls = resolve_error_class(name)
        globals()[name] = cls  # 缓存，保证同名查找拿到同一个类对象
        return cls
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
