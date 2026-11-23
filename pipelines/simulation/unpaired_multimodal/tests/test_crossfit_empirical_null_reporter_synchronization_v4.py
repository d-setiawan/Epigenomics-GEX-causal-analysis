from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import crossfit_empirical_null_reporter_synchronization_v4 as calibration  # noqa: E402


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def test_connected_pair_components_preserve_disjoint_hypotheses() -> None:
    components = calibration._connected_pair_components(
        {
            _pair("a", "b"),
            _pair("b", "c"),
            _pair("d", "e"),
        }
    )

    assert components == [
        {_pair("a", "b"), _pair("b", "c")},
        {_pair("d", "e")},
    ]


def test_pair_components_merge_transitively_without_pair_expansion() -> None:
    components = calibration._connected_pair_components(
        {
            _pair("a", "b"),
            _pair("c", "d"),
            _pair("b", "c"),
        }
    )

    assert len(components) == 1
    assert components[0] == {
        _pair("a", "b"),
        _pair("b", "c"),
        _pair("c", "d"),
    }
    assert _pair("a", "d") not in components[0]
