from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from structure_aware_collider_rescue import (  # noqa: E402
    aggregate_vote_evidence,
    select_anchored_collider_rescues,
)


def _votes(pair_metrics: dict[tuple[str, str], tuple[float, float, int]]) -> pd.DataFrame:
    rows = []
    for (left, right), (p_value, log_odds, selected_count) in pair_metrics.items():
        for vote in range(6):
            rows.append(
                {
                    "source_assay": "assay_a",
                    "node1": left,
                    "node2": right,
                    "selected": vote < selected_count,
                    "empirical_p_positive": p_value,
                    "pooled_log_odds": log_odds,
                    "vote": vote,
                }
            )
    return pd.DataFrame(rows)


def test_completes_only_the_anchored_empirically_separated_wedge() -> None:
    base = pd.DataFrame(
        [{"source_assay": "assay_a", "node1": "peak_a", "node2": "peak_c"}]
    )
    votes = _votes(
        {
            ("peak_b", "peak_c"): (0.010, 1.25, 6),
            ("peak_a", "peak_b"): (0.40, 0.20, 0),
            ("peak_d", "peak_c"): (0.010, 0.75, 6),
            ("peak_a", "peak_d"): (0.50, 0.10, 0),
        }
    )
    result = select_anchored_collider_rescues(base, votes)
    assert len(result.rescued_edges) == 1
    edge = result.rescued_edges.iloc[0]
    assert {edge["node1"], edge["node2"]} == {"peak_b", "peak_c"}
    assert edge["collider_center"] == "peak_c"
    assert edge["anchor_outer"] == "peak_a"
    center_endpoint = edge["endpoint1"] if edge["node1"] == "peak_c" else edge["endpoint2"]
    assert center_endpoint == "ARROW"


@pytest.mark.parametrize(
    "candidate,outer",
    [
        ((0.021, 1.20, 6), (0.40, 0.20, 0)),  # candidate empirical p too large
        ((0.010, 0.99, 6), (0.40, 0.20, 0)),  # effect below practical threshold
        ((0.010, 1.20, 4), (0.40, 0.20, 0)),  # unstable across votes
        ((0.010, 1.20, 6), (0.10, 0.80, 3)),  # outer pair not separated
    ],
)
def test_rejects_weak_unstable_or_shielded_candidates(candidate, outer) -> None:
    base = pd.DataFrame(
        [{"source_assay": "assay_a", "node1": "a", "node2": "c"}]
    )
    result = select_anchored_collider_rescues(
        base,
        _votes({("b", "c"): candidate, ("a", "b"): outer}),
    )
    assert result.rescued_edges.empty


def test_abstains_without_an_anchor_or_when_center_already_has_two_neighbors() -> None:
    votes = _votes({("b", "c"): (0.001, 2.0, 6), ("a", "b"): (0.5, 0.0, 0)})
    no_anchor = pd.DataFrame(columns=["source_assay", "node1", "node2"])
    assert select_anchored_collider_rescues(no_anchor, votes).rescued_edges.empty

    degree_two = pd.DataFrame(
        [
            {"source_assay": "assay_a", "node1": "a", "node2": "c"},
            {"source_assay": "assay_a", "node1": "d", "node2": "c"},
        ]
    )
    assert select_anchored_collider_rescues(degree_two, votes).rescued_edges.empty


def test_aggregator_canonicalizes_pairs_and_rejects_privileged_columns() -> None:
    votes = _votes({("z", "a"): (0.01, 1.2, 5)})
    aggregated = aggregate_vote_evidence(votes)
    assert aggregated.loc[0, ["node1", "node2"]].tolist() == ["a", "z"]
    assert aggregated.loc[0, "selected_vote_fraction"] == pytest.approx(5 / 6)

    leaked = votes.assign(true_role="hidden")
    with pytest.raises(ValueError, match="privileged"):
        aggregate_vote_evidence(leaked)
