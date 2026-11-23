from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from consensus_ca_boss_v4 import (  # noqa: E402
    attach_group_projection_once,
    consensus_typed_graph,
    direct_collider_children,
    replace_pairwise_state_layer,
)


def _graph(edge_type: str, endpoint2: str = "CIRCLE", include: bool = True) -> pd.DataFrame:
    if not include:
        return pd.DataFrame(columns=["node1", "endpoint1", "node2", "endpoint2", "edge_type"])
    return pd.DataFrame(
        [
            {
                "node1": "A",
                "endpoint1": "CIRCLE",
                "node2": "B",
                "endpoint2": endpoint2,
                "edge_type": edge_type,
                "orientation_source": "test",
                "state_score": 0.5,
            }
        ]
    )


def test_four_of_six_typed_consensus_and_endpoint_abstention() -> None:
    graphs = [
        _graph("cross_assay_shared_state", "ARROW"),
        _graph("cross_assay_shared_state", "ARROW"),
        _graph("cross_assay_shared_state", "CIRCLE"),
        _graph("cross_assay_shared_state", "CIRCLE"),
        _graph("cross_assay_shared_state", include=False),
        _graph("cross_assay_shared_state", include=False),
    ]

    consensus = consensus_typed_graph(graphs, minimum_count=4)

    assert len(consensus) == 1
    row = consensus.iloc[0]
    assert row["selection_count"] == 4
    assert row["edge_type"] == "cross_assay_shared_state"
    assert row["endpoint1"] == "CIRCLE"
    assert row["endpoint2"] == "CIRCLE"
    assert row["orientation_source"] == "consensus_abstained"


def test_type_disagreement_becomes_unresolved() -> None:
    graphs = [
        _graph("within_assay_direct_candidate"),
        _graph("within_assay_direct_candidate"),
        _graph("within_assay_shared_state"),
        _graph("within_assay_shared_state"),
        _graph("within_assay_direct_candidate", include=False),
        _graph("within_assay_direct_candidate", include=False),
    ]

    consensus = consensus_typed_graph(graphs, minimum_count=4)

    assert consensus.iloc[0]["edge_type"] == "unresolved"


def test_endpoint_marks_vote_independently() -> None:
    graphs = [
        _graph("within_assay_direct_candidate", "ARROW"),
        _graph("within_assay_direct_candidate", "ARROW"),
        pd.DataFrame(
            [{"node1": "A", "endpoint1": "TAIL", "node2": "B", "endpoint2": "CIRCLE", "edge_type": "within_assay_direct_candidate"}]
        ),
        pd.DataFrame(
            [{"node1": "A", "endpoint1": "TAIL", "node2": "B", "endpoint2": "ARROW", "edge_type": "within_assay_direct_candidate"}]
        ),
        _graph("within_assay_direct_candidate", include=False),
        _graph("within_assay_direct_candidate", include=False),
    ]

    consensus = consensus_typed_graph(graphs, minimum_count=4)

    row = consensus.iloc[0]
    # Endpoint 1 has no 2/3 majority and abstains; endpoint 2 has 3/4 arrows.
    assert row["endpoint1"] == "CIRCLE"
    assert row["endpoint2"] == "ARROW"
    assert row["endpoint1_vote_count"] == 2
    assert row["endpoint2_vote_count"] == 3


def test_empty_consensus_preserves_readable_schema() -> None:
    graphs = [_graph("cross_assay_shared_state", include=False) for _ in range(6)]
    consensus = consensus_typed_graph(graphs, minimum_count=4)

    assert consensus.empty
    assert {"node1", "endpoint1", "node2", "endpoint2", "edge_type"}.issubset(
        consensus.columns
    )


def test_group_projection_attaches_once_and_preserves_direct_edges() -> None:
    direct = pd.DataFrame(
        [
            {
                "node1": "peak_a",
                "endpoint1": "TAIL",
                "node2": "peak_c",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "orientation_source": "collider",
                "selection_count": 5,
                "selection_frequency": 5 / 6,
                "n_nested_runs": 6,
            }
        ]
    )
    projection = pd.DataFrame(
        [
            {
                "node1": "peak_a",
                "node2": "peak_c",
                "edge_type": "within_assay_shared_state",
                "selection_status": "selected",
                "projection_source": "group",
                "group1": "g1",
                "group2": "",
                "confidence": 0.9,
            },
            {
                "node1": "peak_b",
                "node2": "expr__gene",
                "edge_type": "prior_constrained_peak_gene",
                "selection_status": "selected",
                "projection_source": "group",
                "group1": "g2",
                "group2": "g3",
                "confidence": 0.8,
            },
            {
                "node1": "peak_x",
                "node2": "peak_y",
                "edge_type": "unresolved",
                "selection_status": "unresolved",
                "projection_source": "group",
                "group1": "g4",
                "group2": "g5",
                "confidence": 0.7,
            },
        ]
    )

    attached = attach_group_projection_once(direct, projection)

    assert len(attached) == 2
    preserved = attached.loc[
        attached["edge_type"].eq("within_assay_direct_candidate")
    ].iloc[0]
    assert preserved["endpoint1"] == "TAIL"
    assert preserved["endpoint2"] == "ARROW"
    assert bool(preserved["group_projection_selected"])
    gene = attached.loc[attached["edge_type"].eq("prior_constrained_peak_gene")].iloc[0]
    gene_endpoint = gene["endpoint1"] if gene["node1"].startswith("expr__") else gene["endpoint2"]
    peak_endpoint = gene["endpoint2"] if gene["node1"].startswith("expr__") else gene["endpoint1"]
    assert gene_endpoint == "ARROW"
    assert peak_endpoint == "TAIL"
    assert gene["selection_count"] == 0
    assert not attached["node1"].eq("peak_x").any()


def test_group_projection_replaces_preliminary_state_pairs_and_detects_colliders() -> None:
    consensus = pd.DataFrame(
        [
            {
                "node1": "parent_a",
                "endpoint1": "TAIL",
                "node2": "child",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            },
            {
                "node1": "parent_b",
                "endpoint1": "TAIL",
                "node2": "child",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            },
            {
                "node1": "wrong_state_a",
                "endpoint1": "CIRCLE",
                "node2": "wrong_state_b",
                "endpoint2": "CIRCLE",
                "edge_type": "cross_assay_shared_state",
            },
        ]
    )
    projection = pd.DataFrame(
        [
            {
                "node1": "state_a",
                "node2": "state_b",
                "edge_type": "cross_assay_shared_state",
                "selection_status": "selected",
                "projection_source": "group",
                "group1": "g1",
                "group2": "g2",
                "confidence": 0.9,
            }
        ]
    )

    assert direct_collider_children(consensus) == ("child",)
    replaced = replace_pairwise_state_layer(consensus, projection)
    assert len(replaced) == 3
    assert not replaced["node1"].eq("wrong_state_a").any()
    assert (
        replaced["edge_type"].eq("within_assay_direct_candidate").sum() == 2
    )
    assert replaced["node1"].eq("state_a").any()
