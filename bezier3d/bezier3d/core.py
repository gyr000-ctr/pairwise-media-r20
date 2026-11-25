"""3D 贝塞尔曲线：取点、Frenet 标架、弧长、曲率与挠率。

实现说明
--------
* 三次（4 个控制点）曲线走展开的 Bernstein 快路径，其它控制点数走
  de Casteljau，二者都只依赖局部变量。
* 所有公开方法都是纯函数：不修改输入、不修改实例，实例因此可以在多个
  线程之间共享而无需加锁。
* 输入校验按错误类别拆分（非法曲线 / 非法参数 / 非有限值 / 退化曲线 /
  非法采样数），全部继承 :class:`ValueError`，既便于调用方区分，也兼容
  ``except ValueError`` 的写法。
* 弧长使用复合 5 点 Gauss-Legendre 积分。默认 ``n=16`` 时对三次曲线的
  相对误差在 1e-9 量级以下，远优于题面要求的 0.5%。
"""

from __future__ import annotations

import math
from numbers import Integral, Real

__all__ = [
    "Bezier3D",
    "BezierError",
    "InvalidCurveError",
    "InvalidParameterError",
    "InvalidSamplingError",
    "NonFiniteInputError",
    "DegenerateCurveError",
]


class BezierError(ValueError):
    """bezier3d 所有可预期输入错误的基类。"""


class InvalidCurveError(BezierError):
    """控制点结构非法（不是 4 个 3D 点等）。"""


class InvalidParameterError(BezierError):
    """曲线参数 t 非法（越界或类型错误）。"""


class InvalidSamplingError(BezierError):
    """采样/积分段数 n 非法。"""


class NonFiniteInputError(BezierError):
    """输入包含 NaN 或无穷大。"""


class DegenerateCurveError(BezierError):
    """退化曲线或曲线上切向量消失，标架无定义。"""


# 5 点 Gauss-Legendre 节点与权重（区间 [-1, 1]）。
_GAUSS_X = (
    0.0,
    -0.5384693101056831,
    0.5384693101056831,
    -0.9061798459386640,
    0.9061798459386640,
)
_GAUSS_W = (
    0.5688888888888889,
    0.4786286704993665,
    0.4786286704993665,
    0.2369268850561891,
    0.2369268850561891,
)
_GAUSS_NODES = tuple(zip(_GAUSS_X, _GAUSS_W))

# 判断“退化/数值噪声”时使用的相对阈值。
_CURVATURE_REL = 1e-12


# ---------------------------------------------------------------------------
# 校验辅助
# ---------------------------------------------------------------------------

def _is_real(value):
    """是否为可接受的实数（排除 bool）。"""
    return isinstance(value, Real) and not isinstance(value, bool)


def _coerce_curve(curve):
    """把控制点序列规范化为 ``tuple[tuple[float, float, float], ...]``。"""
    if isinstance(curve, (str, bytes)):
        raise InvalidCurveError("控制点必须是 3D 点的序列")
    try:
        points = tuple(curve)
    except TypeError as exc:
        raise InvalidCurveError("控制点必须是 3D 点的序列") from exc

    if len(points) < 2:
        raise InvalidCurveError("贝塞尔曲线至少需要 2 个控制点")

    result = []
    for point in points:
        if isinstance(point, (str, bytes)):
            raise InvalidCurveError("每个控制点必须是 3 个实数组成的序列")
        if not isinstance(point, (tuple, list)):
            try:
                point = tuple(point)
            except TypeError as exc:
                raise InvalidCurveError("每个控制点必须是 3 个实数组成的序列") from exc
        if len(point) != 3:
            raise InvalidCurveError("控制点必须是 3D 点（恰好 3 个坐标）")
        coords = []
        for value in point:
            if not _is_real(value):
                raise InvalidCurveError("控制点坐标必须是实数")
            value = float(value)
            if not math.isfinite(value):
                raise NonFiniteInputError("控制点坐标必须有限")
            coords.append(value)
        result.append((coords[0], coords[1], coords[2]))

    result = tuple(result)
    first = result[0]
    if all(point == first for point in result[1:]):
        raise DegenerateCurveError("所有控制点重合，曲线退化")
    return result


def _coerce_t(t):
    """校验并规范化曲线参数 t ∈ [0, 1]。"""
    if not _is_real(t):
        raise InvalidParameterError("参数 t 必须是实数")
    t = float(t)
    if not math.isfinite(t):
        raise NonFiniteInputError("参数 t 必须有限")
    if t < 0.0 or t > 1.0:
        raise InvalidParameterError("参数 t 必须位于 [0, 1] 内")
    return t


def _coerce_n(n):
    """校验积分/采样段数。"""
    if isinstance(n, bool):
        raise InvalidSamplingError("采样段数 n 必须是整数")
    if isinstance(n, Real) and not math.isfinite(float(n)):
        raise NonFiniteInputError("采样段数 n 必须有限")
    if not isinstance(n, Integral):
        raise InvalidSamplingError("采样段数 n 必须是整数")
    n = int(n)
    if n < 1:
        raise InvalidSamplingError("采样段数 n 必须大于 0")
    return n


# ---------------------------------------------------------------------------
# 向量辅助
# ---------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normalize(a):
    n = _norm(a)
    if n == 0.0:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def _coordinate_span(ctrl):
    """控制点坐标的最大跨度，用于相对阈值判断。"""
    lo = [ctrl[0][k] for k in range(3)]
    hi = lo[:]
    for point in ctrl[1:]:
        for k in range(3):
            v = point[k]
            if v < lo[k]:
                lo[k] = v
            elif v > hi[k]:
                hi[k] = v
    return max(hi[k] - lo[k] for k in range(3))


# ---------------------------------------------------------------------------
# 求值辅助（输入已校验，热路径内不做重复校验）
# ---------------------------------------------------------------------------

def _eval_point(ctrl, t):
    """曲线上的点。"""
    if t == 0.0:
        return ctrl[0]
    if t == 1.0:
        return ctrl[-1]

    if len(ctrl) == 4:
        p0, p1, p2, p3 = ctrl
        u = 1.0 - t
        uu = u * u
        tt = t * t
        w0 = uu * u
        w1 = 3.0 * uu * t
        w2 = 3.0 * u * tt
        w3 = tt * t
        return (
            w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
            w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
            w0 * p0[2] + w1 * p1[2] + w2 * p2[2] + w3 * p3[2],
        )

    pts = list(ctrl)
    u = 1.0 - t
    for level in range(len(pts) - 1, 0, -1):
        for i in range(level):
            a = pts[i]
            b = pts[i + 1]
            pts[i] = (
                u * a[0] + t * b[0],
                u * a[1] + t * b[1],
                u * a[2] + t * b[2],
            )
    return pts[0]


def _derivative_controls(ctrl, order):
    """把控制点转换为第 ``order`` 阶导数的 Bernstein 控制点。"""
    points = list(ctrl)
    factor = 1.0
    degree = len(ctrl) - 1
    for k in range(order):
        factor *= degree - k
        points = [
            (
                points[i + 1][0] - points[i][0],
                points[i + 1][1] - points[i][1],
                points[i + 1][2] - points[i][2],
            )
            for i in range(len(points) - 1)
        ]
    return tuple(
        (factor * p[0], factor * p[1], factor * p[2]) for p in points
    )


def _eval_bernstein(ctrl, t):
    """对任意阶 Bernstein 控制点求值（复用 de Casteljau）。"""
    if t == 0.0:
        return ctrl[0]
    if t == 1.0:
        return ctrl[-1]
    pts = list(ctrl)
    u = 1.0 - t
    for level in range(len(pts) - 1, 0, -1):
        for i in range(level):
            a = pts[i]
            b = pts[i + 1]
            pts[i] = (
                u * a[0] + t * b[0],
                u * a[1] + t * b[1],
                u * a[2] + t * b[2],
            )
    return pts[0]


def _derivative(ctrl, order, t):
    """曲线在 t 处的第 ``order`` 阶导数。"""
    if len(ctrl) == 4:
        p0, p1, p2, p3 = ctrl
        if order == 1:
            u = 1.0 - t
            a = 3.0 * u * u
            b = 6.0 * u * t
            d = 3.0 * t * t
            return (
                a * (p1[0] - p0[0]) + b * (p2[0] - p1[0]) + d * (p3[0] - p2[0]),
                a * (p1[1] - p0[1]) + b * (p2[1] - p1[1]) + d * (p3[1] - p2[1]),
                a * (p1[2] - p0[2]) + b * (p2[2] - p1[2]) + d * (p3[2] - p2[2]),
            )
        if order == 2:
            u = 1.0 - t
            return (
                6.0 * (
                    u * (p2[0] - 2.0 * p1[0] + p0[0])
                    + t * (p3[0] - 2.0 * p2[0] + p1[0])
                ),
                6.0 * (
                    u * (p2[1] - 2.0 * p1[1] + p0[1])
                    + t * (p3[1] - 2.0 * p2[1] + p1[1])
                ),
                6.0 * (
                    u * (p2[2] - 2.0 * p1[2] + p0[2])
                    + t * (p3[2] - 2.0 * p2[2] + p1[2])
                ),
            )
        if order == 3:
            return (
                6.0 * (p3[0] - 3.0 * p2[0] + 3.0 * p1[0] - p0[0]),
                6.0 * (p3[1] - 3.0 * p2[1] + 3.0 * p1[1] - p0[1]),
                6.0 * (p3[2] - 3.0 * p2[2] + 3.0 * p1[2] - p0[2]),
            )
    if order > len(ctrl) - 1:
        return (0.0, 0.0, 0.0)
    return _eval_bernstein(_derivative_controls(ctrl, order), t)


def _tangent_direction(ctrl, t):
    """返回 ``t`` 处的单位切向量；切向量真正消失时返回 ``None``。

    端点处若一阶导数为零（控制点重复），用首个非零高阶导数取单侧极限
    方向；内部一阶导数为零表示尖点，方向无定义，返回 ``None``。
    """
    span = _coordinate_span(ctrl)
    degree = len(ctrl) - 1
    threshold = 1e-12 * span * degree

    d1 = _derivative(ctrl, 1, t)
    speed = _norm(d1)
    if speed > threshold:
        return (d1[0] / speed, d1[1] / speed, d1[2] / speed)

    if t not in (0.0, 1.0):
        return None

    for order in range(2, degree + 1):
        d = _derivative(ctrl, order, t)
        n = _norm(d)
        if n > threshold:
            if t == 1.0 and (order - 1) % 2 == 1:
                d = (-d[0], -d[1], -d[2])
            return (d[0] / n, d[1] / n, d[2] / n)
    return None


def _speed_function(ctrl):
    """返回 ``|B'(t)|`` 的求值闭包，导数控制点在闭包外预先算好。"""
    dctrl = _derivative_controls(ctrl, 1)
    m = len(dctrl) - 1

    if m == 0:  # 线性曲线：速度恒定
        constant = _norm(dctrl[0])
        return lambda t: constant

    if m == 1:
        q0, q1 = dctrl

        def speed(t):
            u = 1.0 - t
            x = u * q0[0] + t * q1[0]
            y = u * q0[1] + t * q1[1]
            z = u * q0[2] + t * q1[2]
            return math.sqrt(x * x + y * y + z * z)

        return speed

    if m == 2:
        q0, q1, q2 = dctrl

        def speed(t):
            u = 1.0 - t
            a = u * u
            b = 2.0 * u * t
            c = t * t
            x = a * q0[0] + b * q1[0] + c * q2[0]
            y = a * q0[1] + b * q1[1] + c * q2[1]
            z = a * q0[2] + b * q1[2] + c * q2[2]
            return math.sqrt(x * x + y * y + z * z)

        return speed

    if m == 3:
        q0, q1, q2, q3 = dctrl

        def speed(t):
            u = 1.0 - t
            uu = u * u
            tt = t * t
            a = uu * u
            b = 3.0 * uu * t
            c = 3.0 * u * tt
            d = tt * t
            x = a * q0[0] + b * q1[0] + c * q2[0] + d * q3[0]
            y = a * q0[1] + b * q1[1] + c * q2[1] + d * q3[1]
            z = a * q0[2] + b * q1[2] + c * q2[2] + d * q3[2]
            return math.sqrt(x * x + y * y + z * z)

        return speed

    def speed(t):
        d = _eval_bernstein(dctrl, t)
        return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])

    return speed


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

class Bezier3D:
    """无状态的 3D 贝塞尔曲线工具。

    实例不保存任何可变数据，因此同一个实例可被多个线程并发使用。
    """

    def point(self, c, t):
        """取曲线上参数 ``t`` 处的点，返回 ``(x, y, z)`` 元组。"""
        ctrl = _coerce_curve(c)
        t = _coerce_t(t)
        return _eval_point(ctrl, t)

    def frame(self, c, t):
        """返回 ``(切向量, 法向量, 副法向量)``，三者均为单位向量且两两正交。

        法向量取曲率方向（Frenet 法向）；当曲率为零（直线段）时退化为
        一个与切向量垂直的稳定参考方向。切向量消失的退化点抛出
        :class:`DegenerateCurveError`。
        """
        ctrl = _coerce_curve(c)
        t = _coerce_t(t)

        tangent = _tangent_direction(ctrl, t)
        if tangent is None:
            raise DegenerateCurveError("切向量消失，无法构造标架")

        d2 = _derivative(ctrl, 2, t)
        d2_norm = _norm(d2)
        proj = _dot(tangent, d2)
        perp = (
            d2[0] - proj * tangent[0],
            d2[1] - proj * tangent[1],
            d2[2] - proj * tangent[2],
        )
        perp_norm = _norm(perp)
        if perp_norm <= _CURVATURE_REL * (1.0 + d2_norm):
            # 曲率为零：选一个与切向量夹角尽量大的坐标轴作为参考方向。
            ax = abs(tangent[0])
            ay = abs(tangent[1])
            az = abs(tangent[2])
            if ax <= ay and ax <= az:
                ref = (1.0, 0.0, 0.0)
            elif ay <= az:
                ref = (0.0, 1.0, 0.0)
            else:
                ref = (0.0, 0.0, 1.0)
            dot = _dot(ref, tangent)
            normal = (ref[0] - dot * tangent[0],
                      ref[1] - dot * tangent[1],
                      ref[2] - dot * tangent[2])
            normal = _normalize(normal)
        else:
            normal = (perp[0] / perp_norm,
                      perp[1] / perp_norm,
                      perp[2] / perp_norm)

        # 显式再正交化，把舍入误差压到机器精度量级。
        dot = _dot(normal, tangent)
        normal = (normal[0] - dot * tangent[0],
                  normal[1] - dot * tangent[1],
                  normal[2] - dot * tangent[2])
        normal = _normalize(normal)
        if normal is None:  # pragma: no cover - 数值上的极端情况
            raise DegenerateCurveError("法向量退化，无法构造标架")

        binormal = _normalize(_cross(tangent, normal))
        if binormal is None:  # pragma: no cover - 理论上不可能
            raise DegenerateCurveError("副法向量退化，无法构造标架")
        # 用 (B, T) 重新生成 N，保证正交性在舍入误差内严格成立。
        normal = _normalize(_cross(binormal, tangent))
        if normal is None:  # pragma: no cover
            raise DegenerateCurveError("法向量退化，无法构造标架")
        return tangent, normal, binormal

    def length(self, c, n=16):
        """用复合 5 点 Gauss-Legendre 积分计算弧长。

        ``n`` 为积分分段数（默认 16）。对三次贝塞尔曲线，默认精度在 1e-9
        量级以下，远高于题面要求的 0.5% 相对误差。
        """
        ctrl = _coerce_curve(c)
        n = _coerce_n(n)

        speed = _speed_function(ctrl)
        nodes = _GAUSS_NODES
        total = 0.0
        step = 1.0 / n
        for i in range(n):
            a = i * step
            mid = a + 0.5 * step
            acc = 0.0
            for x, w in nodes:
                acc += w * speed(mid + 0.5 * step * x)
            total += acc
        return float(total * 0.5 * step)

    def tangent(self, c, t):
        """单位切向量，等价于 ``frame(c, t)[0]``。"""
        return self.frame(c, t)[0]

    def normal(self, c, t):
        """单位主法向量，等价于 ``frame(c, t)[1]``。"""
        return self.frame(c, t)[1]

    def binormal(self, c, t):
        """单位副法向量，等价于 ``frame(c, t)[2]``。"""
        return self.frame(c, t)[2]

    def curvature(self, c, t):
        """曲率 ``kappa = |B' x B''| / |B'|^3``。"""
        ctrl = _coerce_curve(c)
        t = _coerce_t(t)
        tangent = _tangent_direction(ctrl, t)
        if tangent is None:
            raise DegenerateCurveError("切向量消失，曲率无定义")
        d1 = _derivative(ctrl, 1, t)
        speed = _norm(d1)
        if speed == 0.0:
            # B' = 0 处曲率按 |B' x B''| / |B'|^3 发散，无法给出有限值。
            raise DegenerateCurveError("参数化退化（B'=0），曲率无定义")
        d2 = _derivative(ctrl, 2, t)
        return float(_norm(_cross(d1, d2)) / (speed ** 3))

    def torsion(self, c, t):
        """挠率 ``tau = (B' x B'') · B''' / |B' x B''|^2``。

        曲率为零（直线）时返回 ``0.0``；切向量消失时抛出
        :class:`DegenerateCurveError`。
        """
        ctrl = _coerce_curve(c)
        t = _coerce_t(t)
        if _tangent_direction(ctrl, t) is None:
            raise DegenerateCurveError("切向量消失，挠率无定义")
        d1 = _derivative(ctrl, 1, t)
        speed = _norm(d1)
        if speed == 0.0:
            raise DegenerateCurveError("参数化退化（B'=0），挠率无定义")
        d2 = _derivative(ctrl, 2, t)
        cross = _cross(d1, d2)
        denom = _dot(cross, cross)
        scale = speed * speed * _dot(d2, d2)
        if scale == 0.0 or denom <= 1e-24 * scale:
            return 0.0
        d3 = _derivative(ctrl, 3, t)
        return float(_dot(cross, d3) / denom)
