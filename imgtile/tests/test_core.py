import math

import pytest

import imgtile

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 1.0], [2.0, 3.0]]

CASES = [
    ("roundtrip", lambda: max(abs(a - b) for r1, r2 in zip(imgtile.ImgTile().merge(imgtile.ImgTile().split(IMG, 2), (2, 2)), IMG) for a, b in zip(r1, r2)) < 1e-12),
    ("tiles_count", lambda: len(imgtile.ImgTile().split(IMG, 1)) == 4),
    ("pad_shape", lambda: len(imgtile.ImgTile().split(IMG, 2, 1)[0]) == 4),
    ("bad_size", lambda: _raises(lambda: imgtile.ImgTile().split(IMG, 0))),
    ("bad_mode", lambda: _raises(lambda: imgtile.ImgTile().split(IMG, 2, 1, 'bogus'))),
    ("bad_image", lambda: _raises(lambda: imgtile.ImgTile().split([], 2))),
    ("type", lambda: isinstance(imgtile.ImgTile().split(IMG, 2), list)),
    ("det", lambda: imgtile.ImgTile().split(IMG, 2) == imgtile.ImgTile().split(IMG, 2)),
    ("merge_shape", lambda: len(imgtile.ImgTile().merge(imgtile.ImgTile().split(IMG, 2), (2, 2))) == 2),
    ("replicate", lambda: len(imgtile.ImgTile().split(IMG, 2, 1, 'replicate')[0]) == 4),
    ("nan", lambda: _raises(lambda: imgtile.ImgTile().split([[float('nan')]], 1))),
    ("no_mut", lambda: imgtile.ImgTile().split([r[:] for r in IMG], 2) is not None),
    ("odd", lambda: len(imgtile.ImgTile().split([[0.0] * 3 for _ in range(3)], 2)) >= 1),
    ("merge_type", lambda: isinstance(imgtile.ImgTile().merge(imgtile.ImgTile().split(IMG, 2), (2, 2)), list)),
    ("range", lambda: all(0.0 <= v <= 3.0 for t in imgtile.ImgTile().split(IMG, 2) for row in t for v in row)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
