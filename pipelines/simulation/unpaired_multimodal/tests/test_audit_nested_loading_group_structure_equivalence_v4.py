from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_nested_loading_group_structure_equivalence_v4 as audit  # noqa: E402


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def test_maximum_matching_uses_distinct_truth_pairs() -> None:
    truth = {_pair("a", "b"), _pair("c", "d")}
    candidates = [
        {_pair("a", "b"), _pair("c", "d")},
        {_pair("a", "b")},
    ]

    matching = audit.maximum_truth_matching(candidates, truth)

    assert len(matching) == 2
    assert set(matching.values()) == truth


def test_maximum_matching_does_not_count_duplicate_coverage_twice() -> None:
    truth = {_pair("a", "b")}
    candidates = [{_pair("a", "b")}, {_pair("a", "b")}]

    matching = audit.maximum_truth_matching(candidates, truth)

    assert len(matching) == 1


def test_maximum_matching_ignores_nontruth_alternatives() -> None:
    truth = {_pair("a", "b")}
    candidates = [{_pair("x", "y"), _pair("a", "b")}]

    assert audit.maximum_truth_matching(candidates, truth) == {
        0: _pair("a", "b")
    }
