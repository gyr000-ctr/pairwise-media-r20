"""异常类型汇总模块（兼容 ``from tuner.errors import ...`` 风格的引用）。"""

from .core import *  # noqa: F401,F403
from .core import (  # noqa: F401
    InfiniteInputError,
    InvalidAudioError,
    InvalidSampleRateError,
    InvalidScaleError,
    InvalidToleranceError,
    NaNInputError,
    NonFiniteError,
    NonFiniteInputError,
    TunerAudioError,
    TunerConfigurationError,
    TunerError,
    TunerException,
    TunerInputError,
    TunerParameterError,
    TunerProcessingError,
    TunerTypeError,
)

# 常见命名别名。
ScaleError = InvalidScaleError
BadScaleError = InvalidScaleError
IllegalScaleError = InvalidScaleError
SampleRateError = InvalidSampleRateError
RateError = InvalidSampleRateError
ToleranceError = InvalidToleranceError
BadToleranceError = InvalidToleranceError
NaNError = NaNInputError
InfError = InfiniteInputError


def __getattr__(name):
    # 未知命名（如参考实现里的 FooScaleError）委托给 core 的兜底映射。
    from . import core as _core

    return _core.__getattr__(name)
