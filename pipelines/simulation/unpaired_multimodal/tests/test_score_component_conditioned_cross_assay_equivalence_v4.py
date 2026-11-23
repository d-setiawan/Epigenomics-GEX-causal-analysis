from __future__ import annotations

from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_component_conditioned_cross_assay_equivalence_v4 as scorer  # noqa: E402


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((left, right))


def test_maximum_matching_does_not_double_count_one_truth() -> None:
    candidates = [
        {_pair("a", "b")},
        {_pair("a", "b"), _pair("c", "d")},
    ]
    truths = [{_pair("a", "b")}, {_pair("c", "d")}]

    matching = scorer.maximum_truth_matching(candidates, truths)

    assert len(matching) == 2
    assert set(matching.values()) == {0, 1}


def test_equivalence_matching_accepts_any_member_cross_product() -> None:
    candidates = [{_pair("left_s", "right_s")}]
    truth_equivalence = [
        {
            _pair("left_p", "right_p"),
            _pair("left_p", "right_s"),
            _pair("left_s", "right_p"),
            _pair("left_s", "right_s"),
        }
    ]

    matching = scorer.maximum_truth_matching(candidates, truth_equivalence)

    assert matching == {0: 0}
