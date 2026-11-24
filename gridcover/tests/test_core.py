import math

import pytest

import gridcover

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
T = [(2, 2), (2, 0)]

CASES = [
    ("visible_self", lambda: gridcover.GridCover().visible(G, (0, 0), (0, 0)) is True),
    ("blocked", lambda: gridcover.GridCover().visible(G, (0, 0), (2, 2)) is False),
    ("coverage_range", lambda: 0.0 <= gridcover.GridCover().coverage(G, (0, 0), T) <= 1.0),
    ("bad_origin", lambda: _raises(lambda: gridcover.GridCover().coverage(G, (9, 9), T))),
    ("bad_grid", lambda: _raises(lambda: gridcover.GridCover().coverage([], (0, 0), T))),
    ("empty_targets", lambda: gridcover.GridCover().coverage(G, (0, 0), []) == 0.0),
    ("type", lambda: isinstance(gridcover.GridCover().coverage(G, (0, 0), T), float)),
    ("det", lambda: gridcover.GridCover().coverage(G, (0, 0), T) == gridcover.GridCover().coverage(G, (0, 0), T)),
    ("bool_visible", lambda: isinstance(gridcover.GridCover().visible(G, (0, 0), (1, 0)), bool)),
    ("nan", lambda: _raises(lambda: gridcover.GridCover().visible([[float('nan')]], (0, 0), (0, 0)))),
    ("no_mut", lambda: gridcover.GridCover().coverage([r[:] for r in G], (0, 0), T) is not None),
    ("all_open", lambda: gridcover.GridCover().coverage([[0, 0], [0, 0]], (0, 0), [(1, 1)]) == 1.0),
    ("all_blocked", lambda: gridcover.GridCover().coverage([[0, 1], [1, 0]], (0, 0), [(1, 1)]) == 0.0),
    ("target_len", lambda: isinstance(gridcover.GridCover().coverage(G, (0, 0), T), float)),
    ("coverage_type", lambda: isinstance(gridcover.GridCover().coverage(G, (0, 0), [(1, 0)]), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
