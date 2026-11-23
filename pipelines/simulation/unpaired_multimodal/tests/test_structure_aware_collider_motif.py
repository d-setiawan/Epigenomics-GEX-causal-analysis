from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from structure_aware_collider_motif import select_unique_collider_motifs  # noqa: E402


def _base(edges: list[tuple[str, str]], assay: str = "assay_a") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source_assay": assay, "node1": left, "node2": right, "selected": True}
            for left, right in edges
        ],
        columns=["source_assay", "node1", "node2", "selected"],
    )


def _votes(
    metrics: dict[tuple[str, str], tuple[float, float, int]],
    assay: str = "assay_a",
) -> pd.DataFrame:
    rows = []
    for (left, right), (p_value, log_odds, selected_count) in metrics.items():
        for vote in range(6):
            rows.append(
                {
                    "source_assay": assay,
                    "node1": left,
                    "node2": right,
                    "selected": vote < selected_count,
                    "empirical_p_positive": p_value,
                    "pooled_log_odds": log_odds,
                    "vote": vote,
                }
            )
    return pd.DataFrame(rows)


def _pairs(frame: pd.DataFrame) -> set[frozenset[str]]:
    return {
        frozenset((str(row.node1), str(row.node2))) for row in frame.itertuples()
    }


def test_selects_complete_wedge_and_ignores_isolated_association() -> None:
    result = select_unique_collider_motifs(
        _base([("a", "c")]),
        _votes(
            {
                ("a", "c"): (0.001, 1.8, 6),
                ("b", "c"): (0.002, 1.6, 6),
                ("a", "b"): (0.40, 0.3, 0),
                ("d", "e"): (0.001, 1.4, 6),
            }
        ),
    )
    assert _pairs(result.direct_candidates) == {
        frozenset(("a", "c")),
        frozenset(("b", "c")),
    }
    assert set(result.direct_candidates["collider_center"]) == {"c"}
    isolated = result.demoted_or_unresolved.loc[
        result.demoted_or_unresolved["node1"].isin(["d", "e"])
    ]
    assert len(isolated) == 1


def test_keeps_stronger_collider_and_demotes_an_excess_spoke() -> None:
    result = select_unique_collider_motifs(
        _base([("a", "c"), ("b", "c"), ("c", "noise")]),
        _votes(
            {
                ("a", "c"): (0.001, 2.0, 6),
                ("b", "c"): (0.001, 1.8, 6),
                ("c", "noise"): (0.005, 1.2, 6),
                ("a", "b"): (0.40, 0.2, 0),
                ("a", "noise"): (0.50, 0.1, 0),
                ("b", "noise"): (0.60, 0.1, 0),
            }
        ),
    )
    assert _pairs(result.direct_candidates) == {
        frozenset(("a", "c")),
        frozenset(("b", "c")),
    }
    demoted = result.demoted_or_unresolved
    assert frozenset(("c", "noise")) in _pairs(demoted)
    assert demoted.loc[
        demoted.apply(lambda row: {row["node1"], row["node2"]} == {"c", "noise"}, axis=1),
        "reason",
    ].iloc[0] == "not_in_unique_best_wedge"


def test_resolves_triangle_only_when_weak_side_has_clear_margin() -> None:
    result = select_unique_collider_motifs(
        _base([("a", "c"), ("a", "b")]),
        _votes(
            {
                ("a", "c"): (0.001, 2.2, 6),
                ("b", "c"): (0.002, 2.0, 6),
                ("a", "b"): (0.05, 1.2, 4),
            }
        ),
    )
    assert _pairs(result.direct_candidates) == {
        frozenset(("a", "c")),
        frozenset(("b", "c")),
    }
    assert set(result.direct_candidates["collider_center"]) == {"c"}


def test_abstains_on_statistically_symmetric_triangle() -> None:
    result = select_unique_collider_motifs(
        _base([]),
        _votes(
            {
                ("a", "b"): (0.002, 2.20, 6),
                ("a", "c"): (0.002, 2.25, 6),
                ("b", "c"): (0.002, 2.22, 6),
            }
        ),
    )
    assert result.direct_candidates.empty
    assert result.component_diagnostics.iloc[0]["status"] == "unresolved_competing_wedges"
    assert len(result.demoted_or_unresolved) == 3


def test_selects_multiple_disconnected_motifs_in_one_assay() -> None:
    result = select_unique_collider_motifs(
        _base([("a", "c"), ("x", "z")]),
        _votes(
            {
                ("a", "c"): (0.001, 1.8, 6),
                ("b", "c"): (0.001, 1.7, 6),
                ("a", "b"): (0.5, 0.1, 0),
                ("x", "z"): (0.001, 1.9, 6),
                ("y", "z"): (0.001, 1.6, 6),
                ("x", "y"): (0.5, 0.1, 0),
            }
        ),
    )
    assert len(result.direct_candidates) == 4
    assert set(result.direct_candidates["collider_center"]) == {"c", "z"}
