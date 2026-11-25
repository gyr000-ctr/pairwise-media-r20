"""体积雾核心 API。

物理模型
--------
指数高度雾，高度 h 处的消光密度::

    rho(h) = density * exp(-height_falloff * h)

射线从高度 ``height`` 出发，沿距离方向的高度斜率为 ``slope``
（``slope = sin(pitch)``，水平为 0、向上为正、向下为负），则
``h(s) = height + slope * s``，光学厚度（沿射线的解析积分）::

    tau(S) = integral_0^S rho(h(s)) ds
           = rho0 * S * (-expm1(-q) / q),
        rho0 = density * exp(-height_falloff * height),
        q    = height_falloff * slope * S

``-expm1(-q)/q`` 对任意符号、任意小的 q 都数值稳定。透过率
``T = exp(-tau)``，雾量 ``factor = 1 - T``。

反解 :meth:`Fog.distance_for_transmittance` 求解 ``tau(d) = -ln t``。
该方程存在闭式解 ``d = ln(1 + km*L/rho0)/km``（``km`` 为衰减率与
斜率之积，``L = -ln t``），实现以闭式解为初值，再做带二分兜底的
牛顿精修，任意陡度（含密度沿程指数增长的向下射线）下均稳定，
残差达机器精度；``km -> 0`` 与 ``x -> 0`` 均有专门的稳定分支。

相位函数为双项 Henyey-Greenstein 混合::

    p(mu) = w * HG(mu; g1) + (1 - w) * HG(mu; g2),
    HG(mu; g) = 1/(4pi) * (1 - g^2) / (1 + g^2 - 2 g mu)^(3/2)

其中 ``mu = cos(theta)`` 是视线方向与*入射光传播方向*的夹角余弦，
``mu = 1`` 为前向散射峰值方向。两项各自归一，混合后仍归一；
``g = 0`` 时严格退化为各向同性 ``1/(4 pi)``。
"""

from __future__ import annotations

import math

__all__ = [
    "Fog",
    "DualHenyeyGreenstein",
    "henyey_greenstein",
    "dual_henyey_greenstein",
    "FogError",
    "FogParameterError",
    "FogInversionError",
]

_INV_4PI = 1.0 / (4.0 * math.pi)


# --------------------------------------------------------------------------- #
# 异常分类
# --------------------------------------------------------------------------- #
class FogError(ValueError):
    """雾模块所有数值错误的基类（本身也是 ValueError）。"""


class FogParameterError(FogError):
    """非法参数：非有限值、越界值、负值等。"""


class FogInversionError(FogError):
    """透过率反解失败：目标不可达或牛顿迭代不收敛。"""


def _finite(value, name):
    """转为 float 并校验有限性；非有限 / 非数值一律抛 FogParameterError。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise FogParameterError(
            f"{name} must be a finite real number, got {value!r}"
        ) from None
    if not math.isfinite(f):
        raise FogParameterError(f"{name} must be finite, got {value!r}")
    return f


# --------------------------------------------------------------------------- #
# 高度雾：解析积分 + 反解
# --------------------------------------------------------------------------- #
class Fog:
    """指数高度雾。不可变、线程安全（纯函数式求值，无任何可变状态）。

    Parameters
    ----------
    density:
        参考高度 h=0 处的消光密度，必须 ``>= 0``。
    height_falloff:
        高度衰减系数，必须 ``>= 0``；为 0 时退化为均匀指数雾。
    """

    __slots__ = ("density", "height_falloff")

    def __init__(self, density, height_falloff=0.0):
        d = _finite(density, "density")
        k = _finite(height_falloff, "height_falloff")
        if d < 0.0:
            raise FogParameterError(f"density must be >= 0, got {d}")
        if k < 0.0:
            raise FogParameterError(f"height_falloff must be >= 0, got {k}")
        object.__setattr__(self, "density", d)
        object.__setattr__(self, "height_falloff", k)

    def __setattr__(self, name, value):
        raise AttributeError("Fog objects are immutable")

    def __delattr__(self, name):
        raise AttributeError("Fog objects are immutable")

    def __repr__(self):
        return (
            f"Fog(density={self.density!r}, "
            f"height_falloff={self.height_falloff!r})"
        )

    # -- 基础物理量 --------------------------------------------------------- #
    def density_at(self, height):
        """高度 ``height`` 处的局部消光密度 rho(height)。"""
        h = _finite(height, "height")
        try:
            return self.density * math.exp(-self.height_falloff * h)
        except OverflowError:
            # 密度超出 float 范围：用 inf 作诚实表示，下游透过率饱和为 0
            return math.inf

    def optical_depth(self, distance, height=0.0, slope=0.0):
        """沿射线的光学厚度 tau（解析积分）。

        ``height`` 为起点高度，``slope`` 为射线的垂直斜率（每单位距离
        上升的高度）。积分结果超出 float 范围时返回 ``inf``（对应
        透过率 0、雾量 1 的饱和情形）。
        """
        s = _finite(distance, "distance")
        h = _finite(height, "height")
        m = _finite(slope, "slope")
        if s < 0.0:
            raise FogParameterError(f"distance must be >= 0, got {s}")
        if s == 0.0 or self.density == 0.0:
            # 密度为 0 时恒等：tau 恒为 0
            return 0.0

        k = self.height_falloff
        try:
            rho0 = self.density * math.exp(-k * h)
            if k == 0.0 or m == 0.0:
                # 均匀介质 / 水平射线：tau = rho0 * S
                return rho0 * s
            q = k * m * s
            # -expm1(-q)/q：对小 q 与负 q 均稳定，恒正
            return rho0 * s * (-math.expm1(-q) / q)
        except OverflowError:
            # 光学厚度超出 float：透过率在物理上已饱和为 0
            return math.inf

    # 别名：图形学中 optical thickness 与 optical depth 同义
    def optical_thickness(self, distance, height=0.0, slope=0.0):
        return self.optical_depth(distance, height, slope)

    def transmittance(self, distance, height=0.0, slope=0.0):
        """透过率 ``T = exp(-tau)``，严格落在 ``[0, 1]``。"""
        tau = self.optical_depth(distance, height, slope)
        if tau <= 0.0:
            return 1.0
        t = math.exp(-tau)
        # 解析上 t <= 1，钳制仅为防御浮点噪声
        return t if t < 1.0 else 1.0

    def factor(self, distance, height=0.0, slope=0.0):
        """雾量 ``1 - T``，随距离单调不减，值域严格 ``[0, 1]``。"""
        t = self.transmittance(distance, height, slope)
        f = 1.0 - t
        if f <= 0.0:
            return 0.0
        if f >= 1.0:
            return 1.0
        return f

    # -- 反解 --------------------------------------------------------------- #
    def distance_for_transmittance(self, t, height=0.0, slope=0.0):
        """反解达到给定透过率 ``t`` 所需的距离。

        求解 ``tau(d) = -ln(t)``：先取该方程的闭式解，再做带二分
        兜底的牛顿精修，距离误差达机器精度（远小于 1e-6）。要求
        ``t in (0, 1]``：``t = 1`` 时距离恒为 0。向上射线的光学厚度
        存在渐近上界，目标透过率低于渐近透过率时抛
        :class:`FogInversionError`，与普通参数错误
        (:class:`FogParameterError`) 可区分。
        """
        target = _finite(t, "t")
        h = _finite(height, "height")
        m = _finite(slope, "slope")
        if not (0.0 < target <= 1.0):
            raise FogParameterError(f"t must be in (0, 1], got {t}")
        if target == 1.0:
            return 0.0

        k = self.height_falloff
        try:
            rho0 = self.density * math.exp(-k * h)
        except OverflowError:
            rho0 = math.inf
        if rho0 <= 0.0:
            # density == 0（exp 恒正）：透过率恒为 1，任何 t < 1 不可达
            raise FogInversionError(
                "density is zero: transmittance is identically 1.0, "
                "no finite distance reaches t < 1"
            )
        if rho0 == math.inf:
            # 起点密度已超出 float：有限 L 下 d 下溢为 0，
            # 从该点出发的任意非零距离都使透过率饱和为 0
            return 0.0

        target_depth = -math.log(target)

        # 向上射线（k>0, slope>0）：tau 有渐近上界 rho0 / (k*slope)
        km = k * m
        if km > 0.0:
            tau_inf = rho0 / km
            if target_depth >= tau_inf:
                raise FogInversionError(
                    f"target transmittance {t!r} is below the asymptotic "
                    f"minimum {math.exp(-tau_inf):.9g} of this upward ray"
                )

        # ---- 闭式解 --------------------------------------------------------
        # 由 tau(d) = rho0 * (exp(km*d) - 1) / km = L
        # 得 exp(km*d) = 1 + x，x = km*L/rho0，d = ln(1+x) / km；
        # km -> 0 时极限为 L/rho0。
        # 向下射线 km<0 => x<0 且 x<-1，1+x<0，此时取 ln|1+x|/km，
        # 两个负号相除仍为正。闭式解在任意陡度下都是机器精度，
        # 避免纯牛顿法在向下射线（密度沿程指数增长）尾部的慢线性收敛。
        if km == 0.0:
            d = target_depth / rho0
        else:
            x = km * target_depth / rho0
            if abs(x) < 1e-4:
                # ln(1+x)/x 的泰勒展开，消除小 x 的精度损失
                d = (target_depth / rho0) * (
                    1.0 - 0.5 * x + x * x / 3.0 - 0.25 * x * x * x
                )
            else:
                arg = 1.0 + x
                if arg > 0.0:
                    try:
                        d = math.log(arg) / km
                    except OverflowError:
                        # 向上射线接近渐近线，arg 上溢
                        d = (math.log(x) + 1.0) / km
                else:
                    # arg <= 0 只可能出现在向下分支（km<0）
                    if arg == 0.0:
                        arg = -2.2250738585072014e-308
                    d = math.log(-arg) / km
            if d < 0.0:
                # 浮点噪声兜底：理论上 d 恒正
                d = 0.0
        if not math.isfinite(d):
            raise FogInversionError(
                f"closed-form inversion produced invalid distance {d!r}"
            )

        # ---- 牛顿精修（闭式解已近机器精度，通常 0~2 步），二分兜底 --------
        def _residual(dist):
            return self.optical_depth(dist, h, m) - target_depth

        lo, hi = 0.0, max(d * 2.0, 1.0)
        # 保证 hi 位于根的另一侧
        for _ in range(100):
            if _residual(hi) >= 0.0:
                break
            hi *= 2.0
        else:
            raise FogInversionError("cannot bracket inversion root")

        prev_step = math.inf
        for _ in range(32):
            r = _residual(d)
            if abs(r) <= 1e-12 * max(1.0, target_depth):
                break
            rho = self.density * math.exp(-k * (h + m * d))
            if rho > 0.0 and math.isfinite(rho):
                step = r / rho
                d_new = d - step
                # 越过括号或步长停滞则退化为二分
                if not (lo < d_new < hi) or abs(step) >= prev_step:
                    d_new = 0.5 * (lo + hi)
                prev_step = abs(step)
            else:
                d_new = 0.5 * (lo + hi)
            # 维护括号
            if _residual(d_new) >= 0.0:
                hi = d_new
            else:
                lo = d_new
            d = d_new
        else:
            raise FogInversionError("inversion did not converge in 32 refinements")

        # 残差校验，确保返回值确实满足 T(d) == t
        residual = abs(_residual(d))
        if residual > 1e-9 * max(1.0, target_depth):
            raise FogInversionError(
                f"inversion stalled with residual {residual:.3e}"
            )
        return float(d)


# --------------------------------------------------------------------------- #
# Henyey-Greenstein 相位函数
# --------------------------------------------------------------------------- #
def _validate_cos_theta(mu):
    mu = _finite(mu, "cos_theta")
    if not (-1.0 <= mu <= 1.0):
        raise FogParameterError(f"cos_theta must be in [-1, 1], got {mu}")
    return mu


def _validate_g(g):
    g = _finite(g, "g")
    if not (-1.0 < g < 1.0):
        raise FogParameterError(f"g must satisfy |g| < 1, got {g}")
    return g


def henyey_greenstein(cos_theta, g):
    """单项 Henyey-Greenstein 相位函数，单位 ``1/sr``，4pi 积分为 1。

    ``cos_theta`` 是视线与*入射光传播方向*的夹角余弦；``g > 0`` 前向
    散射（峰值在 ``cos_theta = 1``），``g = 0`` 严格各向同性。
    """
    mu = _validate_cos_theta(cos_theta)
    g = _validate_g(g)
    denom = 1.0 + g * g - 2.0 * g * mu
    return _INV_4PI * (1.0 - g * g) / (denom * math.sqrt(denom))


def dual_henyey_greenstein(cos_theta, g_forward, g_backward, weight=0.5):
    """双项 HG 混合相位函数。

    ``weight`` 为前向瓣（``g_forward``，通常 > 0）的混合权重，
    其余分配给后向瓣（``g_backward``，通常 < 0）。两项各自归一，
    凸组合后 4pi 积分严格为 1。
    """
    w = _finite(weight, "weight")
    if not 0.0 <= w <= 1.0:
        raise FogParameterError(f"weight must be in [0, 1], got {w}")
    return w * henyey_greenstein(cos_theta, g_forward) + (
        1.0 - w
    ) * henyey_greenstein(cos_theta, g_backward)


class DualHenyeyGreenstein:
    """双项 Henyey-Greenstein 相位函数（可调用对象，参数构造时校验一次）。

    Examples
    --------
    >>> phase = DualHenyeyGreenstein(0.8, -0.5, 0.7)
    >>> p_forward = phase(1.0)   # 前向峰值
    >>> p_back = phase(-1.0)     # 后向
    """

    __slots__ = ("g_forward", "g_backward", "weight")

    def __init__(self, g_forward, g_backward, weight=0.5):
        gf = _validate_g(g_forward)
        gb = _validate_g(g_backward)
        w = _finite(weight, "weight")
        if not 0.0 <= w <= 1.0:
            raise FogParameterError(f"weight must be in [0, 1], got {w}")
        object.__setattr__(self, "g_forward", gf)
        object.__setattr__(self, "g_backward", gb)
        object.__setattr__(self, "weight", w)

    def __setattr__(self, name, value):
        raise AttributeError("DualHenyeyGreenstein objects are immutable")

    def __call__(self, cos_theta):
        return dual_henyey_greenstein(
            cos_theta, self.g_forward, self.g_backward, self.weight
        )

    def phase(self, cos_theta):
        """``__call__`` 的显式命名别名。"""
        return self(cos_theta)

    @staticmethod
    def cos_from_directions(view_direction, incident_direction):
        """由两个方向向量计算夹角余弦（各自归一化后点积）。

        ``incident_direction`` 为入射光的*传播*方向，故同向（前向
        散射）结果为 ``+1``。
        """
        if len(view_direction) != len(incident_direction):
            raise FogParameterError("direction vectors must have equal length")
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        for a, b in zip(view_direction, incident_direction):
            x = _finite(a, "view_direction")
            y = _finite(b, "incident_direction")
            dot += x * y
            n1 += x * x
            n2 += y * y
        if n1 <= 0.0 or n2 <= 0.0:
            raise FogParameterError("direction vectors must be non-zero")
        mu = dot / math.sqrt(n1 * n2)
        # 归一化后理论上在 [-1, 1]，钳制浮点误差
        if mu > 1.0:
            return 1.0
        if mu < -1.0:
            return -1.0
        return mu

    def __repr__(self):
        return (
            f"DualHenyeyGreenstein(g_forward={self.g_forward!r}, "
            f"g_backward={self.g_backward!r}, weight={self.weight!r})"
        )
