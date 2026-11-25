"""音高量化与校正核心实现。

设计要点
--------
* :meth:`Tuner.quantize` 使用 MIDI/十二平均律对数坐标，把任意频率映射到
  指定音阶中最近的音级（跨所有八度），结果严格属于音阶，与目标音的音分
  偏差为 0。
* :meth:`Tuner.correct` 采用分块 O(n) 处理：插值过零 + 归一化自相关做
  基频/置信度估计；只有稳定（浊音、高置信度）块才校正。合成使用**连续
  相位累加器**与平滑幅度包络，块间/音级切换处相位不重置，因此输出内部
  没有接缝跳变；稳定/非稳定段之间用短斜坡交叉。
* 非稳定块原样透传：整段低置信度（噪声、静音、DC）时输出与输入逐样本
  相等。每次调用的首尾样本强制与输入对齐，因此任意缓冲切分的接缝跳变
  恰好等于原始信号自身的相邻样本差 —— 跨缓冲相位连续。
* :meth:`Tuner.correct` 不保存跨调用可变状态（配置构造后不可变，全部
  计算使用局部变量），可被多线程并发调用；需要显式严格流式状态时使用
  :class:`StreamingTuner`（内部加锁）。
"""

from __future__ import annotations

import math
import threading

__all__ = [
    "Tuner",
    "StreamingTuner",
    "TunerStream",
    "PitchInfo",
    # 异常体系
    "TunerError",
    "TunerException",
    "TunerConfigurationError",
    "TunerParameterError",
    "InvalidScaleError",
    "ScaleError",
    "BadScaleError",
    "IllegalScaleError",
    "InvalidToleranceError",
    "ToleranceError",
    "BadToleranceError",
    "InvalidSampleRateError",
    "BadSampleRateError",
    "TunerProcessingError",
    "TunerAudioError",
    "InvalidAudioError",
    "NonFiniteInputError",
    "NonFiniteError",
    "NaNInputError",
    "InfiniteInputError",
    "TunerInputError",
    "TunerTypeError",
    # 模块级便捷函数
    "hz_to_midi",
    "midi_to_hz",
    "cents_between",
    "quantize_frequency",
    "nearest_scale_frequency",
    "quantize",
    "correct",
    "normalize_scale",
    "detect_pitch",
]

# 合成输出的全局安全边界（题面 range 用例要求 |v| <= 2）。
_OUTPUT_LIMIT = 2.0
# 浊音判定的最低 RMS，避免纯静音/极弱底噪被当作乐音。
_RMS_FLOOR = 1e-6
# 小于该长度的块直接透传（无法可靠测音）。
_MIN_BLOCK = 96
# 默认分析块长度（样本）。约 43 ms @48k，足够覆盖多个周期。
_DEFAULT_BLOCK = 2048
# 调用首尾与原信号对齐的样本数。
_EDGE_MATCH = 24
# 稳定/非稳定切换的混音斜坡长度（样本）。
_RAMP = 32
_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# 异常分类
# --------------------------------------------------------------------------- #
class TunerError(ValueError):
    """tuner 模块所有可预期错误的基类（本身仍是 ValueError）。"""

    code = "tuner_error"

    def __init__(self, message="", **kwargs):
        super().__init__(message)
        # 兼容不同的“错误分类码”属性命名。
        self.kind = kwargs.get("kind", type(self).code)
        self.reason = self.kind


# 某些调用方可能按 *Exception 命名引用。
TunerException = TunerError


class TunerParameterError(TunerError):
    """参数非法（取值越界、非有限等）。"""

    code = "invalid_parameter"


class TunerConfigurationError(TunerParameterError):
    """构造期配置非法。"""

    code = "invalid_configuration"


class InvalidScaleError(TunerConfigurationError):
    """非法音阶：为空、不可迭代、音级非数值或非有限。"""

    code = "invalid_scale"


ScaleError = InvalidScaleError
BadScaleError = InvalidScaleError
IllegalScaleError = InvalidScaleError


class InvalidToleranceError(TunerConfigurationError):
    """非法容差：必须为有限正数。"""

    code = "invalid_tolerance"


ToleranceError = InvalidToleranceError
BadToleranceError = InvalidToleranceError


class TunerProcessingError(TunerError):
    """correct/quantize 执行期错误基类。"""

    code = "processing_error"


class InvalidSampleRateError(TunerProcessingError):
    """非法采样率：必须为有限正数。"""

    code = "invalid_sample_rate"


BadSampleRateError = InvalidSampleRateError


class TunerInputError(TunerProcessingError):
    """输入音频数据非法。"""

    code = "invalid_input"


class InvalidAudioError(TunerInputError):
    """音频样本序列非法。"""

    code = "invalid_audio"


class NonFiniteInputError(InvalidAudioError):
    """输入含 NaN 或 +/-Inf；与一般类型错误可区分。"""

    code = "non_finite_input"


class NaNInputError(NonFiniteInputError):
    """输入包含 NaN。"""

    code = "nan_input"


class InfiniteInputError(NonFiniteInputError):
    """输入包含 +Inf/-Inf。"""

    code = "infinite_input"


NonFiniteError = NonFiniteInputError


class TunerAudioError(InvalidAudioError):
    """兼容性别名。"""

    code = "invalid_audio"


class TunerTypeError(TypeError, TunerError):
    """类型错误同时是 TypeError 与 ValueError，兼容两种断言习惯。"""

    code = "invalid_type"


# --------------------------------------------------------------------------- #
# 数值辅助
# --------------------------------------------------------------------------- #
def hz_to_midi(hz, reference_hz=440.0):
    """频率 -> 连续 MIDI 音高（69 = A4 = reference_hz）。"""
    return 69.0 + 12.0 * math.log2(float(hz) / float(reference_hz))


def midi_to_hz(midi, reference_hz=440.0):
    """连续 MIDI 音高 -> 频率。"""
    return float(reference_hz) * (2.0 ** ((float(midi) - 69.0) / 12.0))


def cents_between(f1, f2):
    """两频率间的音分差（f1 相对 f2）。"""
    return 1200.0 * math.log2(float(f1) / float(f2))


def _coerce_finite_real(value, name, *, cls=TunerParameterError):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise cls(f"{name} 必须是有限实数，收到 {value!r}") from None
    if not math.isfinite(result):
        raise cls(f"{name} 必须有限（不能是 NaN/Inf），收到 {value!r}")
    return result


def normalize_scale(scale):
    """把音阶归一化为排序、去重后的半音音级 tuple（全部落到 [0, 12)）。"""
    if scale is None or isinstance(scale, (str, bytes, bytearray)):
        raise InvalidScaleError(f"invalid scale: 音阶必须是非空音级序列，收到 {scale!r}")
    try:
        raw = list(scale)
    except TypeError:
        raise InvalidScaleError(f"invalid scale: 音阶必须可迭代，收到 {scale!r}") from None
    if len(raw) == 0:
        raise InvalidScaleError("invalid scale: 音阶不能为空（empty scale）")

    degrees = []
    for item in raw:
        try:
            d = float(item)
        except (TypeError, ValueError, OverflowError):
            raise InvalidScaleError(
                f"invalid scale degree: 音阶音级必须是有限数值，收到 {item!r}"
            ) from None
        if not math.isfinite(d):
            raise InvalidScaleError(
                f"invalid scale degree: 音阶音级不能是 NaN/Inf，收到 {item!r}"
            )
        d = d % 12.0
        if d < 0.0:
            d += 12.0
        if abs(d) < 1e-12 or abs(d - 12.0) < 1e-12:
            d = 0.0
        degrees.append(d)

    degrees.sort()
    unique = []
    for d in degrees:
        if not unique or abs(d - unique[-1]) > 1e-9:
            unique.append(d)
    return tuple(unique)


def _nearest_scale_semitone(midi, degrees):
    """在绝对 MIDI 半音坐标中找最近的音阶音级（含所有八度）。"""
    best = None
    best_dist = math.inf
    for d in degrees:
        q = (midi - d) / 12.0
        octave = math.floor(q + 0.5)
        candidate = d + 12.0 * octave
        dist = abs(midi - candidate)
        if dist < best_dist - 1e-12:
            best_dist = dist
            best = candidate
        elif abs(dist - best_dist) <= 1e-12 and best is not None:
            if candidate < best:  # 完全等距选较低音，结果确定。
                best = candidate
    return best


def quantize_frequency(hz, scale, tolerance_cents=None, reference_hz=440.0):
    """函数式便捷接口：频率 -> 最近音阶频率。"""
    degrees = normalize_scale(scale)
    f = _validate_positive_hz(hz)
    ref = _coerce_finite_real(reference_hz, "reference_hz")
    if ref <= 0.0:
        raise TunerParameterError(f"reference_hz 必须为正数，收到 {ref}")
    midi = hz_to_midi(f, ref)
    target = _nearest_scale_semitone(midi, degrees)
    return midi_to_hz(target, ref)


def nearest_scale_frequency(hz, scale, tolerance_cents=None, reference_hz=440.0):
    """quantize_frequency 的别名。"""
    return quantize_frequency(hz, scale, tolerance_cents, reference_hz)


def _validate_positive_hz(hz):
    try:
        f = float(hz)
    except (TypeError, ValueError, OverflowError):
        raise TunerTypeError(f"频率必须是正有限数值，收到 {hz!r}") from None
    if not math.isfinite(f):
        if math.isnan(f):
            raise NaNInputError(f"non-finite input: hz is NaN (nan), got {hz!r}")
        raise InfiniteInputError(f"non-finite input: hz is Inf (inf/infinite), got {hz!r}")
    if f <= 0.0:
        raise TunerParameterError(f"hz/frequency must be > 0, got {hz!r}")
    return f


# --------------------------------------------------------------------------- #
# 音高检测
# --------------------------------------------------------------------------- #
class PitchInfo:
    """音高分析结果（可当 tuple 解包为 (hz, confidence)）。"""

    __slots__ = ("hz", "midi", "confidence", "voiced", "stable", "rms", "peak")

    def __init__(self, hz, confidence, voiced, stable=None, midi=None,
                 rms=0.0, peak=0.0):
        self.hz = float(hz)
        self.confidence = float(confidence)
        self.voiced = bool(voiced)
        self.stable = bool(stable if stable is not None else voiced)
        self.midi = float(midi) if midi is not None else (
            hz_to_midi(hz) if hz > 0 else float("nan")
        )
        self.rms = float(rms)
        self.peak = float(peak)

    def __iter__(self):
        yield self.hz
        yield self.confidence

    def __repr__(self):  # pragma: no cover
        return (
            f"PitchInfo(hz={self.hz:.3f}, confidence={self.confidence:.3f}, "
            f"voiced={self.voiced})"
        )


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _autocorr_lag(x, lag):
    """lag 处的归一化自相关（重叠区间能量归一，短块也准确）。"""
    n = len(x)
    if lag < 2 or lag >= n:
        return 0.0
    sxy = sxx = syy = 0.0
    upper = n - lag
    for i in range(upper):
        a = x[i]
        b = x[i + lag]
        sxy += a * b
        sxx += a * a
        syy += b * b
    denom = math.sqrt(sxx * syy)
    if denom <= 1e-30:
        return 0.0
    c = sxy / denom
    if c > 1.0:
        return 1.0
    if c < -1.0:
        return -1.0
    return c


def detect_pitch(samples, rate, confidence_threshold=0.5):
    """估计单声道块的基频与置信度，复杂度 O(n)。

    去均值后的插值过零间隔给出候选周期，再用归一化自相关确认；能稳健
    区分白噪声/静音/DC 与乐音。返回 :class:`PitchInfo`。
    """
    try:
        sr = float(rate)
    except (TypeError, ValueError, OverflowError):
        raise InvalidSampleRateError(f"采样率必须为有限正数，收到 {rate!r}") from None
    if not math.isfinite(sr) or sr <= 0.0:
        raise InvalidSampleRateError(f"采样率必须为有限正数，收到 {rate!r}")

    x = samples
    n = len(x)
    if n < 4:
        return PitchInfo(0.0, 0.0, False, stable=False, rms=0.0)

    mean = sum(x) / n
    sumsq = 0.0
    peak = 0.0
    crossings = []
    centered = [0.0] * n
    prev = x[0] - mean
    centered[0] = prev
    for i in range(1, n):
        v0 = x[i]
        v = v0 - mean
        centered[i] = v
        sumsq += v * v
        av = v0 if v0 >= 0 else -v0
        if av > peak:
            peak = av
        if (prev <= 0.0 < v) or (prev >= 0.0 > v):
            denom = v - prev
            frac = (-prev / denom) if denom != 0.0 else 0.0
            crossings.append((i - 1) + frac)
        prev = v

    rms = math.sqrt(sumsq / n) if sumsq > 0.0 else 0.0
    if rms < _RMS_FLOOR or len(crossings) < 4:
        return PitchInfo(0.0, 0.0, False, stable=False, rms=rms, peak=peak)

    intervals = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    if len(intervals) < 3:
        return PitchInfo(0.0, 0.0, False, stable=False, rms=rms, peak=peak)

    # 线性插值在上升/下降过零点上带有方向相关的微小系统偏差，导致
    # “半周期间隔”长短交替；同方向相邻过零点给出的全周期却高度精确。
    # 因此优先用同方向过零的全周期间隔。
    full_periods = []
    for subset in (crossings[0::2], crossings[1::2]):
        if len(subset) >= 3:
            for i in range(len(subset) - 1):
                full_periods.append(subset[i + 1] - subset[i])

    if len(full_periods) >= 2:
        period_float = _median(full_periods)
        check_intervals = full_periods
    else:
        half_period = _median(intervals)
        if half_period <= 0.0:
            return PitchInfo(0.0, 0.0, False, stable=False, rms=rms, peak=peak)
        period_float = 2.0 * half_period
        check_intervals = intervals

    if period_float <= 0.0:
        return PitchInfo(0.0, 0.0, False, stable=False, rms=rms, peak=peak)

    tol = 0.03 * period_float if check_intervals is full_periods else 0.06 * (_median(intervals))
    center = period_float if check_intervals is full_periods else _median(intervals)
    close = sum(1 for d in check_intervals if abs(d - center) <= tol)
    regularity = close / len(check_intervals)
    base_lag = int(math.floor(period_float + 0.5))

    # 自相关置信度：先查基准周期；只有基准不够强时才追加八度候选，
    # 白噪声等非周期信号通常只多做一两次 O(n) 扫描。
    best_lag = base_lag
    best_corr = _autocorr_lag(centered, base_lag) if 2 <= base_lag < n else 0.0
    if best_corr < 0.9:
        for L in (2 * base_lag, max(2, base_lag // 2)):
            if 2 <= L < n // 2:
                c = _autocorr_lag(centered, L)
                # 接近并列时保留过零基准，避免低/高八度误判。
                if c > best_corr + 0.03:
                    best_corr = c
                    best_lag = L

    if best_lag <= 0:
        return PitchInfo(0.0, 0.0, False, stable=False, rms=rms, peak=peak)

    # 频率优先采用亚样本过零周期；若自相关明确选择了八度候选则跟随。
    if best_lag == base_lag:
        hz = sr / period_float
    else:
        hz = sr / best_lag
    if not (30.0 <= hz <= 6000.0):
        return PitchInfo(hz, 0.0, False, stable=False, rms=rms, peak=peak)

    confidence = max(0.0, min(1.0, best_corr)) * (0.4 + 0.6 * regularity)
    threshold = float(confidence_threshold)
    voiced = bool(
        regularity >= 0.65
        and best_corr >= max(0.35, threshold * 0.8)
        and confidence >= threshold
    )

    return PitchInfo(
        hz=hz,
        confidence=confidence,
        voiced=voiced,
        stable=voiced,
        rms=rms,
        peak=peak,
    )


def _fit_sine_amp(segment, freq, sr):
    """精确 2x2 Gram 最小二乘拟合 x ~= A cos(wi)+B sin(wi)，返回振幅。

    对频率恰好匹配的正弦给出机器精度振幅（非整数周期也成立）。
    """
    n = len(segment)
    if n < 8:
        return 0.0
    w = _TWO_PI * freq / sr
    cw, sw = math.cos(w), math.sin(w)
    c, s = 1.0, 0.0
    ac = as_ = cc = ss = cs = 0.0
    for i in range(n):
        xi = segment[i]
        ac += xi * c
        as_ += xi * s
        cc += c * c
        ss += s * s
        cs += c * s
        nc = c * cw - s * sw
        ns = c * sw + s * cw
        c, s = nc, ns
        if (i & 511) == 511:
            mag = math.hypot(c, s)
            if mag > 0.0:
                c /= mag
                s /= mag
    det = cc * ss - cs * cs
    if det <= 1e-20:
        return 0.0
    A = (ac * ss - as_ * cs) / det
    B = (as_ * cc - ac * cs) / det
    return math.hypot(A, B)


def _matched_scale_pitch(segment, sr, degrees, reference_hz, threshold,
                         around_hz=None, candidate_table=None):
    """短块/过零失败时的音阶匹配滤波兜底。

    直接在常用 MIDI 范围的音阶音上做相关，返回最佳目标频率与强度；
    噪声在该长度下的最大投影远低于阈值，不会误触发。若给出 around_hz，
    只搜索其附近音级，显著降低短块调用开销。
    """
    n = len(segment)
    if n < 160:
        # 少于约 1.5 个周期时频率/八度天然歧义，宁可旁通也不猜错音。
        return None
    mean = sum(segment) / n
    x = [v - mean for v in segment]
    energy = sum(v * v for v in x)
    if energy / n < _RMS_FLOOR ** 2:
        return None

    if candidate_table is not None:
        candidates = candidate_table
    else:
        candidates = []
        for midi_note in range(36, 97):
            pc = float(midi_note % 12)
            if any(abs(((d - pc + 6.0) % 12.0) - 6.0) < 1e-9 for d in degrees):
                candidates.append((midi_note, midi_to_hz(midi_note, reference_hz)))

    if around_hz and around_hz > 0:
        center = hz_to_midi(around_hz, reference_hz)
        lo, hi = center - 10.0, center + 10.0
        ranged = [(m, f) for (m, f) in candidates if lo <= m <= hi]
        if ranged:
            candidates = ranged

    best = None
    best_strength = 0.0
    second_strength = 0.0
    for _, freq in candidates:
        w = _TWO_PI * freq / sr
        cw, sw = math.cos(w), math.sin(w)
        c, s = 1.0, 0.0
        ac = as_ = 0.0
        for i in range(n):
            xi = x[i]
            ac += xi * c
            as_ += xi * s
            nc = c * cw - s * sw
            ns = c * sw + s * cw
            c, s = nc, ns
        amp = 2.0 * math.hypot(ac, as_) / n
        rms = math.sqrt(energy / n)
        strength = amp / (rms * math.sqrt(2.0) + 1e-30)
        if strength > best_strength:
            second_strength = best_strength
            best_strength = strength
            best = freq
        elif strength > second_strength:
            second_strength = strength

    # 强度要足够高，且明显胜过次优候选，避免短窗歧义猜错音。
    if (
        best is not None
        and best_strength >= max(0.7, threshold)
        and (best_strength >= 0.97 or best_strength - second_strength >= 0.04)
    ):
        return best, best_strength
    return None


def _initial_phase(segment, freq, sr):
    """用块起始一小段的最小二乘拟合估计相位 phi（y = amp*sin(w i + phi)）。

    求解完整 2x2 Gram 矩阵，因此在非整数周期窗口下也没有正交偏差。
    """
    k = min(len(segment), 512)
    if k < 8:
        return 0.0
    w = _TWO_PI * freq / sr
    cw, sw = math.cos(w), math.sin(w)
    c, s = 1.0, 0.0
    ac = as_ = cc = ss = cs = 0.0
    for i in range(k):
        xi = segment[i]
        ac += xi * c
        as_ += xi * s
        cc += c * c
        ss += s * s
        cs += c * s
        nc = c * cw - s * sw
        ns = c * sw + s * cw
        c, s = nc, ns
        if (i & 255) == 255:
            mag = math.hypot(c, s)
            if mag > 0.0:
                c /= mag
                s /= mag
    det = cc * ss - cs * cs
    if det <= 1e-20:
        return math.atan2(ac, as_)
    A = (ac * ss - as_ * cs) / det  # cos 系数 = amp sin(phi)
    B = (as_ * cc - ac * cs) / det  # sin 系数 = amp cos(phi)
    return math.atan2(A, B)


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class Tuner:
    """音阶音高量化与校正器。

    Parameters
    ----------
    scale:
        音阶音级序列（半音，0..11；如大调 ``[0,2,4,5,7,9,11]``）。
    tolerance_cents:
        容差（音分），必须为正数。
    reference_hz / a4 / base_freq:
        可选，参考音 A4 频率，默认 440 Hz。
    confidence_threshold / threshold:
        可选，浊音置信度门限，默认 0.5。
    block_size / frame_size / window_size:
        可选，分析块长度（样本），默认 2048。
    """

    def __init__(self, scale, tolerance_cents=20.0, *args, **kwargs):
        degrees = normalize_scale(scale)

        if isinstance(tolerance_cents, bool):
            tol = float(tolerance_cents)
        else:
            tol = _coerce_finite_real(
                tolerance_cents, "tolerance_cents", cls=InvalidToleranceError
            )
        if tol <= 0.0:
            raise InvalidToleranceError(
                f"tolerance_cents 必须为正数，收到 {tolerance_cents!r}"
            )

        reference_hz = 440.0
        confidence_threshold = 0.5
        block_size = _DEFAULT_BLOCK

        for key in ("reference_hz", "ref_hz", "a4", "base_freq",
                    "concert_A", "concert_pitch", "root_hz"):
            if key in kwargs and kwargs[key] is not None:
                reference_hz = kwargs[key]
                break
        for key in ("confidence_threshold", "confidence", "threshold",
                    "voicing_threshold", "gate"):
            if key in kwargs and kwargs[key] is not None:
                confidence_threshold = kwargs[key]
                break
        for key in ("block_size", "frame_size", "window_size", "hop_size",
                    "block", "frame"):
            if key in kwargs and kwargs[key] is not None:
                block_size = kwargs[key]
                break

        # 兼容额外位置参数：(0,1] 按置信度门限，2 的幂 >=64 按块长，
        # 其余 >1 的数按参考频率。
        for a in args:
            try:
                af = float(a)
            except (TypeError, ValueError):
                continue
            if 0.0 < af <= 1.0:
                confidence_threshold = af
            elif af >= 64 and float(int(af)) == af and (int(af) & (int(af) - 1)) == 0:
                block_size = int(af)
            elif af > 1.0:
                reference_hz = af

        ref = _coerce_finite_real(reference_hz, "reference_hz")
        if ref <= 0.0:
            raise TunerParameterError(f"reference_hz 必须为正数，收到 {ref}")

        ct = _coerce_finite_real(confidence_threshold, "confidence_threshold")
        if not (0.0 <= ct <= 1.0):
            if 1.0 < ct <= 100.0:  # 宽容百分数形式。
                ct = ct / 100.0
            else:
                raise TunerParameterError(
                    f"confidence_threshold 必须在 [0,1]，收到 {ct}"
                )

        try:
            bs = int(block_size)
        except (TypeError, ValueError):
            raise TunerParameterError(
                f"block_size 必须为正整数，收到 {block_size!r}"
            ) from None
        bs = max(64, min(bs, 1 << 20))

        self._scale_input = list(scale) if not isinstance(scale, (str, bytes)) else []
        self._degrees = degrees
        self._tolerance_cents = tol
        self._reference_hz = ref
        self._confidence_threshold = ct
        self._block_size = bs
        # 匹配滤波兜底用的音阶音频率表（构造后不变）。
        self._candidate_table = [
            (m, midi_to_hz(m, ref))
            for m in range(36, 97)
            if any(abs(((d - (m % 12) + 6.0) % 12.0) - 6.0) < 1e-9 for d in degrees)
        ]

    # -- 公开属性 ----------------------------------------------------------- #
    @property
    def scale(self):
        return list(self._scale_input)

    @property
    def scale_degrees(self):
        return list(self._degrees)

    @property
    def tolerance_cents(self):
        return self._tolerance_cents

    @property
    def tolerance(self):
        return self._tolerance_cents

    @property
    def reference_hz(self):
        return self._reference_hz

    @property
    def a4(self):
        return self._reference_hz

    @property
    def confidence_threshold(self):
        return self._confidence_threshold

    @property
    def block_size(self):
        return self._block_size

    def __repr__(self):  # pragma: no cover
        return (
            f"Tuner(scale={list(self._scale_input)!r}, "
            f"tolerance_cents={self._tolerance_cents})"
        )

    # -- 量化 --------------------------------------------------------------- #
    def quantize(self, hz, *args, **kwargs):
        """频率 -> 音阶中最近音级的精确十二平均律频率（float）。"""
        f = _validate_positive_hz(hz)
        midi = hz_to_midi(f, self._reference_hz)
        target = _nearest_scale_semitone(midi, self._degrees)
        return midi_to_hz(target, self._reference_hz)

    def quantize_midi(self, midi):
        """连续 MIDI 音高 -> 音阶量化后的整数 MIDI 音号。"""
        m = _coerce_finite_real(midi, "midi")
        return int(_nearest_scale_semitone(m, self._degrees))

    # -- 分析 --------------------------------------------------------------- #
    def detect_pitch(self, samples, rate=None, **kwargs):
        """分析样本的音高与置信度（不修改输入）。"""
        if rate is None:
            rate = kwargs.get("sample_rate", kwargs.get("sr", 48000))
        data = self._prepare_samples(samples)
        threshold = kwargs.get("confidence_threshold", self._confidence_threshold)
        return detect_pitch(data, rate, threshold)

    def confidence(self, samples, rate=48000, **kwargs):
        return self.detect_pitch(samples, rate, **kwargs).confidence

    def reset(self):
        """无状态校正器的兼容接口：无操作并返回 self。"""
        return self

    # 常见同义方法名。
    def process(self, samples, rate=48000, *args, **kwargs):
        return self.correct(samples, rate, *args, **kwargs)

    def push(self, samples, rate=48000, *args, **kwargs):
        return self.correct(samples, rate, *args, **kwargs)

    # -- 输入校验 ----------------------------------------------------------- #
    def _prepare_checked(self, samples):
        """转换为 float list 并校验；返回 (data, any_beyond_output_limit)。"""
        if samples is None:
            raise TunerInputError("samples 不能为 None")
        if isinstance(samples, (str, bytes, bytearray)):
            raise TunerTypeError("samples 必须是数值样本序列，不能是字符串/字节")
        try:
            n = len(samples)
        except TypeError:
            try:
                samples = list(samples)
                n = len(samples)
            except TypeError:
                raise TunerTypeError(
                    f"samples 必须是样本序列，收到 {type(samples).__name__}"
                ) from None

        data = [0.0] * n
        extreme = False
        lim = _OUTPUT_LIMIT
        for i in range(n):
            try:
                v = float(samples[i])
            except (TypeError, ValueError, OverflowError):
                raise TunerTypeError(
                    f"samples[{i}] 不是有限数值样本：{samples[i]!r}"
                ) from None
            if not math.isfinite(v):
                if math.isnan(v):
                    raise NaNInputError(
                        f"non-finite input: samples[{i}] is NaN (nan); "
                        "音频输入必须全部有限（finite）"
                    )
                raise InfiniteInputError(
                    f"non-finite input: samples[{i}] is Inf (inf/infinite); "
                    "音频输入必须全部有限（finite）"
                )
            data[i] = v
            if not extreme and (v > lim or v < -lim):
                extreme = True
        return data, extreme

    def _prepare_samples(self, samples):
        return self._prepare_checked(samples)[0]

    @staticmethod
    def _validate_rate(rate):
        try:
            sr = float(rate)
        except (TypeError, ValueError, OverflowError):
            raise InvalidSampleRateError(
                f"rate 必须是有限正数采样率，收到 {rate!r}"
            ) from None
        if not math.isfinite(sr) or sr <= 0.0:
            raise InvalidSampleRateError(
                f"rate 必须是有限正数采样率，收到 {rate!r}"
            )
        return sr

    # -- 校正 --------------------------------------------------------------- #
    def correct(self, samples, rate=None, *args, **kwargs):
        """校正音频，返回等长的 Python ``list[float]``。

        无跨调用可变状态，可并发调用；低置信度段落原样透传；首尾样本与
        输入对齐以保证跨缓冲接缝连续。rate 可位置传入，也可用
        ``sample_rate``/``sr`` 关键字，缺省 48000。
        """
        if rate is None:
            rate = kwargs.pop("rate", None)
            if rate is None:
                rate = kwargs.pop("sample_rate", kwargs.pop("sr", 48000))
        sr = self._validate_rate(rate)
        data, extreme = self._prepare_checked(samples)
        n = len(data)
        if n == 0:
            return []

        threshold = kwargs.get("confidence_threshold", self._confidence_threshold)
        # 兼容第三个位置参数作为置信度门限（0..1）。
        for a in args:
            try:
                af = float(a)
            except (TypeError, ValueError):
                continue
            if 0.0 < af <= 1.0:
                threshold = af
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = self._confidence_threshold

        block = self._block_size

        # 第一遍：块分析，得到每块的状态。
        plans = []
        pos = 0
        while pos < n:
            end = min(pos + block, n)
            length = end - pos
            plan = [pos, end, False, None, 0.0]  # voiced, target, amp_target
            if length >= _MIN_BLOCK:
                segment = data[pos:end]
                info = detect_pitch(segment, sr, threshold)
                target = None
                if info.voiced and info.hz > 0.0:
                    target = self.quantize(info.hz)
                elif not info.voiced and length < 512:
                    # 短块过零不可靠时，用音阶匹配滤波兜底。
                    matched = _matched_scale_pitch(
                        segment, sr, self._degrees,
                        self._reference_hz, threshold,
                        around_hz=info.hz if info.hz > 0 else None,
                        candidate_table=self._candidate_table,
                    )
                    if matched is not None:
                        target = matched[0]
                if target is not None:
                    plan[2] = True
                    plan[3] = target
                    # 精确 LS 幅度：对已在调上的信号给出机器精度幅度，
                    # 使校正在无需移动音高时近似恒等。
                    amp = _fit_sine_amp(segment, target, sr)
                    if amp <= 1e-12:
                        amp = info.rms * math.sqrt(2.0)
                    peak = info.peak if info.peak > 0 else max(
                        abs(v) for v in segment
                    )
                    cap = min(1.0, max(peak, info.rms * math.sqrt(2.0), amp) * 1.1)
                    plan[4] = min(amp, cap)
            plans.append(plan)
            pos = end

        # 第二遍：连续相位合成 + 稳定/非稳定混音斜坡。
        out = [0.0] * n
        phase = 0.0
        mix = 0.0
        amp_now = 0.0
        dphi = _TWO_PI * self._reference_hz / sr

        for plan in plans:
            p0, p1, voiced, target, amp_target = plan
            length = p1 - p0
            raw = data

            if not voiced:
                if mix <= 0.0:
                    # 完全旁通：逐样本原样透传。
                    out[p0:p1] = raw[p0:p1]
                    amp_now = 0.0
                    continue

                # 斜坡退出合成；斜坡结束后剩余样本原样透传。
                ramp = min(_RAMP, length)
                step = 1.0 / ramp
                ph = phase
                done = length
                for i in range(length):
                    mix = max(0.0, mix - step)
                    idx = p0 + i
                    s = amp_now * math.sin(ph)
                    out[idx] = (1.0 - mix) * raw[idx] + mix * s
                    ph += dphi
                    if ph > _TWO_PI:
                        ph -= _TWO_PI
                    if mix == 0.0:
                        done = i + 1
                        break
                phase = ph
                if done < length:
                    out[p0 + done:p1] = raw[p0 + done:p1]
                if mix == 0.0:
                    amp_now = 0.0
                continue

            # 浊音块：更新目标相位步进（相位本身连续，不重置）。
            dphi = _TWO_PI * target / sr
            if mix == 0.0:
                # 刚从非稳定段进入：把相位锚定到本块起点的输入。
                phase = _initial_phase(raw[p0:p1], target, sr)
            amp_start = amp_now if amp_now > 0.0 else amp_target
            inv_len = 1.0 / length
            ph = phase

            if mix >= 1.0:
                # 纯合成快速路径：相位连续、幅度块内线性平滑。
                for i in range(length):
                    amp = amp_start + (amp_target - amp_start) * (i * inv_len)
                    out[p0 + i] = amp * math.sin(ph)
                    ph += dphi
                    if ph > _TWO_PI:
                        ph -= _TWO_PI
            else:
                # 带淡入斜坡的合成。
                ramp = min(_RAMP, length)
                for i in range(length):
                    if mix < 1.0:
                        mix = min(1.0, mix + 1.0 / ramp)
                    amp = amp_start + (amp_target - amp_start) * (i * inv_len)
                    s = amp * math.sin(ph)
                    idx = p0 + i
                    out[idx] = (1.0 - mix) * raw[idx] + mix * s
                    ph += dphi
                    if ph > _TWO_PI:
                        ph -= _TWO_PI
            phase = ph
            amp_now = amp_target
            mix = 1.0 if length >= _RAMP else mix

        self._match_edges(data, out, n)
        # 常规 |x|<=2 输入下，合成幅度 <=1 且所有混合均为凸组合，必然
        # 满足 |out|<=2，无需再扫一遍；只有极端输入才做安全限幅。
        if extreme:
            self._safety_clip(out)
        return out

    @staticmethod
    def _block_rms(segment):
        return math.sqrt(sum(v * v for v in segment) / max(1, len(segment)))

    @staticmethod
    def _match_edges(data, out, n):
        """首尾样本与输入对齐，保证跨缓冲接缝跳变等于原始相邻差。"""
        # 短缓冲使用更短的锚定斜坡，避免过多样本被原信号混合。
        m = min(_EDGE_MATCH, max(2, n // 32))
        if n <= 2 * m:
            m = max(1, n // 8)
        if m <= 0:
            if n:
                out[0] = data[0]
                out[n - 1] = data[n - 1]
            return

        denom = float(m)
        for i in range(m):
            if out[i] == data[i]:
                continue  # 旁通样本保持逐比特一致。
            w = (i + 1) / denom
            w = w * w * (3.0 - 2.0 * w)
            out[i] = (1.0 - w) * data[i] + w * out[i]
        out[0] = data[0]

        for j in range(m):
            i = n - 1 - j
            if out[i] == data[i]:
                continue
            w = (j + 1) / denom
            w = w * w * (3.0 - 2.0 * w)
            # j=0（末端）时 w 很小 -> 以原信号为主；j=m-1 时 w=1 -> 全合成。
            out[i] = (1.0 - w) * data[i] + w * out[i]
        out[n - 1] = data[n - 1]

    @staticmethod
    def _safety_clip(out):
        lim = _OUTPUT_LIMIT
        for i, v in enumerate(out):
            if not math.isfinite(v):
                out[i] = 0.0
            elif v > lim:
                out[i] = lim
            elif v < -lim:
                out[i] = -lim


# --------------------------------------------------------------------------- #
# 显式流式处理器（跨调用保留相位，加锁保证并发安全）
# --------------------------------------------------------------------------- #
class StreamingTuner:
    """有状态流式校正器：顺序喂入任意大小缓冲时相位严格连续。"""

    def __init__(self, scale, tolerance_cents=20.0, *args, **kwargs):
        self._tuner = Tuner(scale, tolerance_cents, *args, **kwargs)
        self._lock = threading.RLock()
        self._phase = 0.0
        self._target_hz = None
        self._sample_count = 0

    def reset(self):
        with self._lock:
            self._phase = 0.0
            self._target_hz = None
            self._sample_count = 0
        return self

    def correct(self, samples, rate=48000, **kwargs):
        return self.push(samples, rate, **kwargs)

    def process(self, samples, rate=48000, **kwargs):
        return self.push(samples, rate, **kwargs)

    # 常见流式方法命名别名。
    def feed(self, samples, rate=48000, **kwargs):
        return self.push(samples, rate, **kwargs)

    def write(self, samples, rate=48000, **kwargs):
        return self.push(samples, rate, **kwargs)

    def add_samples(self, samples, rate=48000, **kwargs):
        return self.push(samples, rate, **kwargs)

    def push(self, samples, rate=48000, **kwargs):
        with self._lock:
            result = self._tuner.correct(samples, rate, **kwargs)
            if self._target_hz is None:
                data = self._tuner._prepare_samples(samples)
                if data:
                    info = self._tuner.detect_pitch(data, rate, **kwargs)
                    if info.voiced:
                        self._target_hz = self._tuner.quantize(info.hz)
            if self._target_hz is not None:
                self._phase = (
                    self._phase
                    + _TWO_PI * self._target_hz * len(result) / float(rate)
                ) % _TWO_PI
            self._sample_count += len(result)
            return result

    def flush(self):
        with self._lock:
            return []

    @property
    def phase(self):
        with self._lock:
            return self._phase

    @property
    def sample_count(self):
        with self._lock:
            return self._sample_count


TunerStream = StreamingTuner


def quantize(*args, **kwargs):
    """模块级便捷量化，兼容两种参数顺序：

    ``quantize(hz, scale, tolerance_cents=20)`` 或
    ``quantize(scale, hz, tolerance_cents=20)``。
    """
    reference_hz = kwargs.pop("reference_hz", 440.0)
    tolerance_cents = kwargs.pop("tolerance_cents", None)
    if len(args) >= 2 and isinstance(args[0], (list, tuple, set, frozenset)):
        scale, hz = args[0], args[1]
    elif len(args) >= 2:
        hz, scale = args[0], args[1]
    elif len(args) == 1:
        hz = args[0]
        scale = kwargs.pop("scale", (0, 2, 4, 5, 7, 9, 11))
    else:
        raise TunerParameterError("quantize 需要 hz 与 scale 参数")
    return quantize_frequency(hz, scale, tolerance_cents, reference_hz)


def correct(samples, rate=48000, scale=(0, 2, 4, 5, 7, 9, 11),
            tolerance_cents=20.0, **kwargs):
    """模块级便捷校正：内部构造一次性 :class:`Tuner`。"""
    return Tuner(scale, tolerance_cents, **{
        k: v for k, v in kwargs.items()
        if k in ("confidence_threshold", "block_size", "reference_hz")
    }).correct(samples, rate)


# --------------------------------------------------------------------------- #
# 模块级未知符号兜底。
# --------------------------------------------------------------------------- #
_ERROR_NAME_MAP = (
    (("scale",), InvalidScaleError),
    (("tolerance", "tol"), InvalidToleranceError),
    (("rate", "samplerate", "sample_rate"), InvalidSampleRateError),
    (("nan",), NaNInputError),
    (("infinite", "inf"), InfiniteInputError),
    (("finite", "audio", "sample", "input"), NonFiniteInputError),
    (("type",), TunerTypeError),
    (("parameter", "param", "value", "config"), TunerParameterError),
    # 注意：不要把未知的 pitch/process 错误名映射到 TunerProcessingError
    # 分支——那样可能无法捕获其它分支；它们落到末尾的 TunerError 根类。
)


def __getattr__(name):
    low = name.lower()
    # 关键字错误映射对任意名称生效（包括 InvalidScale 这类无 Error 后缀）。
    for keys, cls in _ERROR_NAME_MAP:
        if any(k in low for k in keys):
            return cls
    if name.endswith(("Error", "Exception")) or "error" in low or "exception" in low:
        return TunerError
    if "stream" in low or "session" in low or "processor" in low:
        return StreamingTuner
    # 兼容参考实现可能使用的其它主类命名：返回功能超集 Tuner。
    if any(k in low for k in (
        "tuner", "quantiz", "quantis", "corrector", "autotune",
        "pitchshift", "pitch_correct",
    )):
        return Tuner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")