"""题面完整约束的验证测试（解析积分、双项 HG、牛顿反解、性能等）。"""

import math
import threading
import time

import pytest

import fog
from fog import (
    DualHenyeyGreenstein,
    Fog,
    FogError,
    FogInversionError,
    FogParameterError,
    dual_henyey_greenstein,
    henyey_greenstein,
)

INV_4PI = 1.0 / (4.0 * math.pi)


# --------------------------------------------------------------------------- #
# 1. 解析积分 vs 1e4 步数值积分，透过率误差 <= 1e-6
# --------------------------------------------------------------------------- #
def _numerical_transmittance(f, distance, height, slope, steps=10_000):
    """中点法数值积分光学厚度，再取 exp。"""
    h = distance / steps
    total = 0.0
    for i in range(steps):
        s = (i + 0.5) * h
        total += f.density_at(height + slope * s)
    return math.exp(-total * h)


RAY_CONFIGS = [
    # density, falloff, distance, height, slope
    (0.5, 0.0, 10.0, 0.0, 0.0),      # 均匀雾（水平）
    (0.7, 0.35, 20.0, 2.0, 0.0),     # 水平高度雾
    (0.6, 0.4, 15.0, 1.0, 0.5),      # 向上
    (0.9, 0.25, 12.0, 5.0, -0.4),    # 向下（密度沿程增加）
    (0.2, 1.2, 8.0, 0.5, 0.9),       # 强衰减、陡仰角
    (1.5, 0.8, 3.0, 0.0, -0.7),      # 向下高密度
    (0.0, 0.5, 10.0, 3.0, 0.3),      # 零密度恒等
]


@pytest.mark.parametrize("d,k,s,h,m", RAY_CONFIGS)
def test_analytic_matches_numerical_transmittance(d, k, s, h, m):
    f = Fog(d, k)
    analytic = f.transmittance(s, h, m)
    numeric = _numerical_transmittance(f, s, h, m)
    assert abs(analytic - numeric) <= 1e-6


def test_analytic_matches_numerical_optical_depth():
    f = Fog(0.8, 0.45)
    for h, m in [(0.0, 0.3), (2.0, -0.6), (1.5, 0.0)]:
        s = 12.0
        numeric_tau = -math.log(_numerical_transmittance(f, s, h, m))
        assert abs(f.optical_depth(s, h, m) - numeric_tau) <= 1e-6


def test_analytic_formula_handles_tiny_q():
    # q -> 0 时必须稳定退化到 rho0 * S，不得出现 NaN
    f = Fog(0.5, 1e-12)
    tau = f.optical_depth(1.0, 0.0, 1e-12)
    assert math.isfinite(tau)
    assert abs(tau - 0.5) <= 1e-9


# --------------------------------------------------------------------------- #
# 2. 雾量单调性与值域
# --------------------------------------------------------------------------- #
def test_factor_monotone_all_ray_kinds():
    f = Fog(0.6, 0.4)
    for h, m in [(0.0, 0.0), (1.0, 0.5), (3.0, -0.5)]:
        prev = -1.0
        for i in range(40):
            v = f.factor(0.25 * i, h, m)
            assert 0.0 <= v <= 1.0
            assert v >= prev - 1e-15
            prev = v


def test_factor_zero_distance_and_saturation():
    f = Fog(0.5)
    assert f.factor(0.0) == 0.0
    assert f.factor(1000.0) <= 1.0
    assert isinstance(f.factor(1.0), float)


def test_height_fog_decays_above_reference():
    # 参考高度以上指数衰减：h=100, k=0.5 时密度 ~ e-50，雾量 <= 1e-9
    f = Fog(0.5, 0.5)
    assert f.factor(10.0, 100.0) <= 1e-9


def test_higher_height_less_fog():
    f = Fog(0.5, 0.5)
    assert f.factor(10.0, 5.0) < f.factor(10.0, 0.0)
    assert f.factor(10.0, 0.0, 0.3) < f.factor(10.0, 0.0, 0.0)


def test_density_increases_fog():
    assert Fog(1.0).factor(5.0) >= Fog(0.5).factor(5.0)


# --------------------------------------------------------------------------- #
# 3. 密度为 0 时恒等
# --------------------------------------------------------------------------- #
def test_zero_density_identity():
    f = Fog(0.0, 0.7)
    for s, h, m in [(0.0, 0.0, 0.0), (10.0, 3.0, 0.4), (10.0, 3.0, -0.4)]:
        assert f.optical_depth(s, h, m) == 0.0
        assert f.transmittance(s, h, m) == 1.0
        assert f.factor(s, h, m) == 0.0
        assert f.density_at(h) == 0.0


def test_zero_density_inverse_is_distinguishable_error():
    f = Fog(0.0, 0.5)
    with pytest.raises(FogInversionError):
        f.distance_for_transmittance(0.5)
    # 反解错误与参数错误是不同类型，但都是 FogError/ValueError
    assert issubclass(FogInversionError, FogError)
    assert issubclass(FogParameterError, FogError)


# --------------------------------------------------------------------------- #
# 4. 双项 Henyey-Greenstein：归一化、各向同性、前向峰值
# --------------------------------------------------------------------------- #
def _phase_integral(phase, samples=400_001):
    """2*pi * ∫_{-1}^{1} p(mu) dmu，辛普森法。"""
    n = samples - 1  # 偶数区间
    if n % 2:
        n += 1
    h = 2.0 / n
    total = phase(-1.0) + phase(1.0)
    for i in range(1, n):
        mu = -1.0 + i * h
        total += (4.0 if i % 2 else 2.0) * phase(mu)
    return 2.0 * math.pi * (h / 3.0) * total


def test_hg_normalized():
    for g in (-0.9, -0.3, 0.0, 0.3, 0.9):
        assert abs(_phase_integral(lambda mu, g=g: henyey_greenstein(mu, g)) - 1.0) <= 1e-7


def test_dual_hg_normalized():
    phase = DualHenyeyGreenstein(0.8, -0.5, 0.7)
    assert abs(_phase_integral(phase) - 1.0) <= 1e-7
    for w in (0.0, 0.25, 0.5, 1.0):
        p = DualHenyeyGreenstein(0.6, -0.4, w)
        assert abs(_phase_integral(p) - 1.0) <= 1e-7


def test_isotropic_when_g_zero():
    expected = INV_4PI
    assert henyey_greenstein(0.0, 0.0) == expected
    for mu in (-1.0, -0.37, 0.0, 0.51, 1.0):
        assert abs(henyey_greenstein(mu, 0.0) - expected) <= 1e-9
    # 双项且两个 g 都为 0
    p = DualHenyeyGreenstein(0.0, 0.0, 0.3)
    for mu in (-1.0, 0.0, 1.0):
        assert abs(p(mu) - expected) <= 1e-9


def test_forward_peak_aligns_with_incident_direction():
    phase = DualHenyeyGreenstein(0.8, -0.5, 0.7)
    mus = [-1.0 + 0.02 * i for i in range(101)]
    # 前向瓣占优时全局最大值在 mu = +1（入射传播方向）
    assert max(mus, key=phase) == 1.0
    assert phase(1.0) > phase(-1.0)
    # 方向向量：视线与入射光同向 -> mu = 1
    mu = DualHenyeyGreenstein.cos_from_directions((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    assert abs(mu - 1.0) <= 1e-12
    mu = DualHenyeyGreenstein.cos_from_directions((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert abs(mu + 1.0) <= 1e-12


def test_single_hg_forward_backward_lobes():
    assert henyey_greenstein(1.0, 0.7) > henyey_greenstein(-1.0, 0.7)
    assert henyey_greenstein(-1.0, -0.7) > henyey_greenstein(1.0, -0.7)


def test_dual_hg_mixture_weights():
    # w=1 退化为前向单项，w=0 退化为后向单项
    assert dual_henyey_greenstein(0.4, 0.6, -0.3, 1.0) == pytest.approx(
        henyey_greenstein(0.4, 0.6)
    )
    assert dual_henyey_greenstein(0.4, 0.6, -0.3, 0.0) == pytest.approx(
        henyey_greenstein(0.4, -0.3)
    )


# --------------------------------------------------------------------------- #
# 5. 牛顿反解 distance_for_transmittance
# --------------------------------------------------------------------------- #
INVERSE_CASES = [
    (0.5, 0.0, 0.0, 0.0),
    (0.7, 0.35, 2.0, 0.0),
    (0.6, 0.4, 1.0, 0.5),
    (0.9, 0.25, 5.0, -0.4),
    (0.2, 1.2, 0.5, 0.9),
]


@pytest.mark.parametrize("d,k,h,m", INVERSE_CASES)
def test_inverse_roundtrip_distance(d, k, h, m):
    f = Fog(d, k)
    checked = 0
    for d0 in (0.05, 0.5, 2.0, 10.0, 30.0):
        t = f.transmittance(d0, h, m)
        if t <= 0.0:
            # 透过率在 float 下已下溢为 0，超出反解定义域 t in (0,1]
            continue
        # 条件数过滤：t 的 1 ulp 舍入带来的距离误差 ~ eps*tau/rho(d0)，
        # 该量级超过 1e-7 时 1e-6 的距离精度在 float64 下不可表示
        # （仅出现在逼近向上射线渐近线的平坦区，那里改用透过率残差校验）
        rho_local = f.density_at(h + m * d0)
        if 2e-16 * (-math.log(t)) / max(rho_local, 1e-300) > 1e-7:
            continue
        d_solved = f.distance_for_transmittance(t, h, m)
        assert abs(d_solved - d0) <= 1e-6
        checked += 1
    assert checked >= 3


@pytest.mark.parametrize("d,k,h,m", INVERSE_CASES)
def test_inverse_roundtrip_transmittance(d, k, h, m):
    f = Fog(d, k)
    targets = [0.99, 0.9, 0.5, 0.1, 0.01]
    if k > 0.0 and m > 0.0:
        # 向上射线存在渐近最小透过率，只取可达目标并追加一个逼近渐近线的
        rho0 = d * math.exp(-k * h)
        t_min = math.exp(-rho0 / (k * m))
        targets = [t for t in targets if t > t_min * 1.001]
        targets.append(t_min * 1.05)
    assert targets
    for t in targets:
        d_solved = f.distance_for_transmittance(t, h, m)
        t_back = f.transmittance(d_solved, h, m)
        assert abs(t_back - t) <= 1e-6


def test_inverse_boundaries():
    f = Fog(0.5, 0.3)
    assert f.distance_for_transmittance(1.0) == 0.0
    with pytest.raises(FogParameterError):
        f.distance_for_transmittance(0.0)
    with pytest.raises(FogParameterError):
        f.distance_for_transmittance(-0.1)
    with pytest.raises(FogParameterError):
        f.distance_for_transmittance(1.01)
    with pytest.raises(FogParameterError):
        f.distance_for_transmittance(float("nan"))


def test_inverse_uniform_closed_form():
    # 均匀雾解析解 d = -ln(t) / density
    f = Fog(0.7)
    for t in (0.8, 0.5, 0.13, 0.001):
        expected = -math.log(t) / 0.7
        assert abs(f.distance_for_transmittance(t) - expected) <= 1e-10


def test_inverse_upward_asymptote_distinguishable():
    f = Fog(0.6, 0.4)
    h, m = 1.0, 0.5
    tau_inf = f.density * math.exp(-0.4 * h) / (0.4 * m)
    t_min = math.exp(-tau_inf)
    # 可达：略高于渐近最小值
    d_ok = f.distance_for_transmittance(t_min * 1.01, h, m)
    assert d_ok > 0.0
    # 不可达：低于渐近最小值 -> FogInversionError（区别于参数错误）
    with pytest.raises(FogInversionError):
        f.distance_for_transmittance(t_min * 0.5, h, m)


def test_inverse_monotone_in_t():
    f = Fog(0.6, 0.4)
    ts = [0.99, 0.8, 0.5, 0.2, 0.05]
    ds = [f.distance_for_transmittance(t, 1.0, -0.3) for t in ts]
    for a, b in zip(ds, ds[1:]):
        assert b > a  # 透过率越小，距离越远


# --------------------------------------------------------------------------- #
# 6. 非法参数分类报错
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn",
    [
        lambda: Fog(-0.1, 0.0),
        lambda: Fog(0.5, -1.0),
        lambda: Fog(float("nan"), 0.0),
        lambda: Fog(0.5, float("inf")),
        lambda: Fog(0.5).factor(-1.0),
        lambda: Fog(0.5).factor(float("nan")),
        lambda: Fog(0.5).factor(1.0, float("inf")),
        lambda: Fog(0.5).optical_depth(1.0, 0.0, float("nan")),
        lambda: DualHenyeyGreenstein(1.0, 0.0),
        lambda: DualHenyeyGreenstein(0.5, -2.0),
        lambda: DualHenyeyGreenstein(0.5, 0.5, 1.5),
        lambda: DualHenyeyGreenstein(0.5, 0.5, -0.1),
        lambda: henyey_greenstein(2.0, 0.5),
        lambda: henyey_greenstein(0.0, float("nan")),
    ],
)
def test_invalid_params_raise_parameter_error(fn):
    with pytest.raises(FogParameterError):
        fn()
    with pytest.raises(ValueError):  # 向后兼容：仍是 ValueError
        fn()


def test_error_types_distinguishable():
    with pytest.raises(FogParameterError):
        Fog(-1.0)
    f = Fog(0.3, 0.5)
    with pytest.raises(FogInversionError):
        # 向上射线达不到的透过率
        f.distance_for_transmittance(1e-12, 0.0, 1.0)
    # 两类错误互不混淆
    assert not issubclass(FogInversionError, FogParameterError)


# --------------------------------------------------------------------------- #
# 7. 性能：10 万次求值 <= 300 ms
# --------------------------------------------------------------------------- #
def test_performance_100k_factor():
    f = Fog(0.6, 0.4)
    n = 100_000
    start = time.perf_counter()
    acc = 0.0
    for i in range(n):
        acc += f.factor(0.0001 * i, 0.05 * (i % 7), 0.001 * (i % 3))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert 0.0 < acc <= n
    assert elapsed_ms <= 300.0, f"100k evaluations took {elapsed_ms:.1f} ms"


def test_performance_100k_phase():
    phase = DualHenyeyGreenstein(0.8, -0.5, 0.7)
    n = 100_000
    start = time.perf_counter()
    acc = 0.0
    for i in range(n):
        acc += phase(-1.0 + 2.0 * (i % 997) / 997.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert acc > 0.0
    assert elapsed_ms <= 300.0, f"100k phase evaluations took {elapsed_ms:.1f} ms"


# --------------------------------------------------------------------------- #
# 8. 并发安全与不可变
# --------------------------------------------------------------------------- #
def test_thread_safety():
    f = Fog(0.6, 0.4)
    phase = DualHenyeyGreenstein(0.8, -0.5, 0.7)
    errors = []
    ref_factor = f.factor(7.0, 1.0, 0.2)
    ref_inv = f.distance_for_transmittance(0.37, 1.0, 0.2)
    ref_phase = phase(0.33)

    def worker():
        try:
            for _ in range(2000):
                assert f.factor(7.0, 1.0, 0.2) == ref_factor
                assert f.distance_for_transmittance(0.37, 1.0, 0.2) == ref_inv
                assert phase(0.33) == ref_phase
        except Exception as exc:  # pragma: no cover - 仅并发失败时触发
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_immutability():
    f = Fog(0.5, 0.3)
    with pytest.raises(AttributeError):
        f.density = 0.9
    with pytest.raises(AttributeError):
        f.new_attr = 1.0
    p = DualHenyeyGreenstein(0.5, -0.5)
    with pytest.raises(AttributeError):
        p.g_forward = 0.1


def test_determinism():
    f = Fog(0.5, 0.3)
    assert f.factor(2.0, 1.0, 0.2) == f.factor(2.0, 1.0, 0.2)
    p = DualHenyeyGreenstein(0.5, -0.5, 0.6)
    assert p(0.4) == p(0.4)


# --------------------------------------------------------------------------- #
# 9. 极端值饱和与辅助 API
# --------------------------------------------------------------------------- #
def test_overflow_saturates_instead_of_raising():
    f = Fog(1.0, 1.0)
    assert f.optical_depth(1e308, -1e308, -1.0) == math.inf
    assert f.transmittance(1e308, -1e308, -1.0) == 0.0
    assert f.factor(1e308, -1e308, -1.0) == 1.0
    assert f.density_at(-2000.0) == math.inf
    # 起点密度溢出：任意非零距离透过率都饱和为 0，反解退化为 0
    assert f.distance_for_transmittance(0.5, -2000.0, 0.0) == 0.0


def test_aliases_and_helpers():
    f = Fog(0.6, 0.4)
    assert f.optical_thickness(7.0, 1.0, 0.2) == f.optical_depth(7.0, 1.0, 0.2)
    cos = DualHenyeyGreenstein.cos_from_directions
    assert abs(cos((0, 1, 0), (0, 1, 0)) - 1.0) <= 1e-12
    assert abs(cos((1, 0, 0), (0, 1, 0))) <= 1e-12
    assert abs(cos((1, 0, 0), (-1, 0, 0)) + 1.0) <= 1e-12
    # 非单位向量自动归一化
    assert abs(cos((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)) - 1.0) <= 1e-12
    phase = DualHenyeyGreenstein(0.7, -0.3, 0.6)
    assert phase.phase(0.5) == phase(0.5)
    with pytest.raises(FogParameterError):
        cos((1.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(FogParameterError):
        cos((0.0, 0.0), (1.0, 0.0))


def test_transmittance_monotone_decreasing():
    f = Fog(0.6, 0.4)
    prev = 1.0
    for i in range(30):
        t = f.transmittance(0.5 * i, 1.0, -0.3)
        assert 0.0 <= t <= 1.0
        assert t <= prev + 1e-15
        prev = t
