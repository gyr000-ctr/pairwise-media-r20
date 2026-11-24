import math

import pytest

import fog

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f(d=0.5):
    return fog.Fog(d, 0.0)

CASES = [
    ("zero_dist", lambda: _f().factor(0.0) == 0.0),
    ("range", lambda: 0.0 <= _f().factor(10.0) <= 1.0),
    ("monotone", lambda: _f().factor(1.0) <= _f().factor(10.0)),
    ("zero_density", lambda: _f(0.0).factor(10.0) == 0.0),
    ("bad_density", lambda: _raises(lambda: fog.Fog(-0.1, 0.0))),
    ("bad_falloff", lambda: _raises(lambda: fog.Fog(0.5, -1.0))),
    ("height_falloff", lambda: fog.Fog(0.5, 0.5).factor(10.0, 100.0) <= fog.Fog(0.5, 0.5).factor(10.0, 0.0)),
    ("type", lambda: isinstance(_f().factor(1.0), float)),
    ("det", lambda: _f().factor(2.0) == _f().factor(2.0)),
    ("nan", lambda: _raises(lambda: _f().factor(float('nan')))),
    ("far_saturate", lambda: _f(10.0).factor(1000.0) <= 1.0),
    ("no_mut", lambda: _f().factor(3.0) is not None),
    ("density_effect", lambda: _f(1.0).factor(5.0) >= _f(0.5).factor(5.0)),
    ("negative_dist", lambda: _raises(lambda: _f().factor(-1.0))),
    ("height_only", lambda: isinstance(fog.Fog(0.0, 0.5).factor(5.0, 5.0), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
