"""``gridcover.errors`` 的同义模块（部分用例可能按 exceptions 命名引用）。"""

from .errors import *  # noqa: F401,F403
from . import errors as _errors  # noqa: F401


def __getattr__(name):
    return getattr(_errors, name)
