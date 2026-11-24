import math

import pytest

import bezier3d

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = ((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))

CASES = [
    ("start", lambda: abs(bezier3d.Bezier3D().point(C, 0.0)[0]) < 1e-9),
    ("end", lambda: abs(bezier3d.Bezier3D().point(C, 1.0)[0] - 1.0) < 1e-9),
    ("mid", lambda: 0.0 <= bezier3d.Bezier3D().point(C, 0.5)[0] <= 1.0),
    ("frame_orth", lambda: abs(sum(a * b for a, b in zip(bezier3d.Bezier3D().frame(C, 0.5)[0], bezier3d.Bezier3D().frame(C, 0.5)[1]))) < 1e-9),
    ("length", lambda: bezier3d.Bezier3D().length(C) > 1.0),
    ("bad_t", lambda: _raises(lambda: bezier3d.Bezier3D().point(C, 1.5))),
    ("bad_curve", lambda: _raises(lambda: bezier3d.Bezier3D().point(((0, 0, 0),) * 4, 0.5))),
    ("type", lambda: isinstance(bezier3d.Bezier3D().point(C, 0.5), tuple)),
    ("det", lambda: bezier3d.Bezier3D().point(C, 0.3) == bezier3d.Bezier3D().point(C, 0.3)),
    ("frame_len", lambda: len(bezier3d.Bezier3D().frame(C, 0.5)) == 3),
    ("unit_tangent", lambda: abs(math.sqrt(sum(c * c for c in bezier3d.Bezier3D().frame(C, 0.5)[0])) - 1.0) < 1e-6),
    ("length_type", lambda: isinstance(bezier3d.Bezier3D().length(C), float)),
    ("nan", lambda: _raises(lambda: bezier3d.Bezier3D().point(C, float('nan')))),
    ("no_mut", lambda: bezier3d.Bezier3D().point(C, 0.4) is not None),
    ("bad_len_n", lambda: _raises(lambda: bezier3d.Bezier3D().length(C, 0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
