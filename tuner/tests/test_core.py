import math

import pytest

import tuner

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

SCALE = [0, 2, 4, 5, 7, 9, 11]

CASES = [
    ("quantize_a440", lambda: abs(tuner.Tuner(SCALE, 20).quantize(440.0) - 440.0) < 440.0 * 0.05),
    ("quantize_off", lambda: tuner.Tuner(SCALE, 20).quantize(445.0) != 445.0),
    ("length", lambda: len(tuner.Tuner(SCALE, 20).correct([0.1] * 100, 48000)) == 100),
    ("bad_scale", lambda: _raises(lambda: tuner.Tuner([], 20))),
    ("bad_tolerance", lambda: _raises(lambda: tuner.Tuner(SCALE, 0))),
    ("bad_rate", lambda: _raises(lambda: tuner.Tuner(SCALE, 20).correct([0.1], 0))),
    ("empty", lambda: tuner.Tuner(SCALE, 20).correct([], 48000) == []),
    ("type", lambda: isinstance(tuner.Tuner(SCALE, 20).correct([0.1], 48000), list)),
    ("det", lambda: tuner.Tuner(SCALE, 20).correct([0.1] * 10, 48000) == tuner.Tuner(SCALE, 20).correct([0.1] * 10, 48000)),
    ("range", lambda: all(abs(v) <= 2.0 for v in tuner.Tuner(SCALE, 20).correct([0.5] * 100, 48000))),
    ("nan", lambda: _raises(lambda: tuner.Tuner(SCALE, 20).correct([float('nan')], 48000))),
    ("quantize_type", lambda: isinstance(tuner.Tuner(SCALE, 20).quantize(440.0), float)),
    ("no_mut", lambda: tuner.Tuner(SCALE, 20).correct([0.2] * 10, 48000) is not None),
    ("octave", lambda: abs(tuner.Tuner(SCALE, 20).quantize(880.0) - 880.0) < 880.0 * 0.05),
    ("bad_hz", lambda: _raises(lambda: tuner.Tuner(SCALE, 20).quantize(0.0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
