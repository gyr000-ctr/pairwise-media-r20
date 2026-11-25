import math
import threading
import time

import pytest

import bezier3d

C = ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _dense_length(curve, samples=20000):
    prev = bezier3d.Bezier3D().point(curve, 0.0)
    total = 0.0
    for i in range(1, samples + 1):
        cur = bezier3d.Bezier3D().point(curve, i / samples)
        total += math.dist(prev, cur)
        prev = cur
    return total


def test_passes_through_endpoints():
    b = bezier3d.Bezier3D()
    assert b.point(C, 0.0) == pytest.approx(C[0], abs=1e-12)
    assert b.point(C, 1.0) == pytest.approx(C[-1], abs=1e-12)


def test_tangent_matches_central_difference():
    b = bezier3d.Bezier3D()
    h = 1e-6
    for i in range(1, 40):
        t = i / 40.0
        fd = _normalize(
            tuple(
                (p - m) / (2.0 * h)
                for p, m in zip(b.point(C, t + h), b.point(C, t - h))
            )
        )
        assert max(abs(a - c) for a, c in zip(b.frame(C, t)[0], fd)) <= 1e-6


def test_length_matches_dense_sampling():
    b = bezier3d.Bezier3D()
    for curve in (C, ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, -1.0, 2.0), (5.0, 5.0, 5.0))):
        dense = _dense_length(curve)
        assert abs(b.length(curve) - dense) / dense <= 0.005


def test_frame_is_orthonormal():
    b = bezier3d.Bezier3D()
    for i in range(101):
        tangent, normal, binormal = b.frame(C, i / 100.0)
        for u, v in ((tangent, normal), (tangent, binormal), (normal, binormal)):
            assert abs(sum(a * c for a, c in zip(u, v))) <= 1e-9
        for v in (tangent, normal, binormal):
            assert abs(sum(a * a for a in v) - 1.0) <= 1e-9


def test_degenerate_curves_are_reported_distinctly():
    b = bezier3d.Bezier3D()
    with pytest.raises(bezier3d.DegenerateCurveError):
        b.point(((1.0, 2.0, 3.0),) * 4, 0.5)
    with pytest.raises(bezier3d.NonFiniteInputError):
        b.point(C, float("nan"))
    with pytest.raises(bezier3d.InvalidCurveError):
        b.point(((0.0, 0.0, 0.0), (1.0, 1.0)), 0.5)
    for exc in (
        bezier3d.DegenerateCurveError,
        bezier3d.NonFiniteInputError,
        bezier3d.InvalidCurveError,
    ):
        assert issubclass(exc, ValueError)
    assert len({bezier3d.DegenerateCurveError,
                bezier3d.NonFiniteInputError,
                bezier3d.InvalidCurveError}) == 3


def test_ten_thousand_samples_under_500ms():
    b = bezier3d.Bezier3D()
    start = time.perf_counter()
    for i in range(10000):
        b.point(C, (i % 1000) / 999.0)
    assert time.perf_counter() - start <= 0.5


def test_concurrent_use_is_safe():
    b = bezier3d.Bezier3D()
    failures = []
    barrier = threading.Barrier(8)

    def worker(seed):
        barrier.wait()
        try:
            for i in range(500):
                t = (seed * 37 + i) % 1000 / 999.0
                tangent, normal, _ = b.frame(C, t)
                assert abs(sum(x * y for x, y in zip(tangent, normal))) <= 1e-9
                assert b.point(C, 0.0) == C[0]
            assert b.length(C) > 1.0
        except BaseException as exc:  # pragma: no cover - 仅在失败时执行
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_inputs_are_not_mutated():
    b = bezier3d.Bezier3D()
    curve = [list(point) for point in C]
    snapshot = [point[:] for point in curve]
    b.point(curve, 0.5)
    b.frame(curve, 0.5)
    b.length(curve)
    assert curve == snapshot
