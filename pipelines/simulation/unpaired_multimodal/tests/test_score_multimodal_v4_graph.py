from __future__ import annotations

from pathlib import Path
import sys
from itertools import combinations

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from score_multimodal_v4_graph import score_v4  # noqa: E402


def _edge(
    left: str,
    right: str,
    endpoint_left: str = "CIRCLE",
    endpoint_right: str = "CIRCLE",
    edge_type: str = "within_assay_direct_candidate",
) -> dict[str, str]:
    return {
        "node1": left,
        "endpoint1": endpoint_left,
        "node2": right,
        "endpoint2": endpoint_right,
        "edge_type": edge_type,
    }


def test_perfect_typed_contract_passes_without_crediting_gene_prior() -> None:
    truth = pd.DataFrame(
        [
            {"node1": "p1__M1", "node2": "c1__M1", "truth_type": "within_assay_direct", "label": "p1_c", "mark": "M1", "parent": "p1__M1", "child": "c1__M1", "orientation_credit": True},
            {"node1": "q1__M1", "node2": "c1__M1", "truth_type": "within_assay_direct", "label": "q1_c", "mark": "M1", "parent": "q1__M1", "child": "c1__M1", "orientation_credit": True},
            {"node1": "p2__M2", "node2": "c2__M2", "truth_type": "within_assay_direct", "label": "p2_c", "mark": "M2", "parent": "p2__M2", "child": "c2__M2", "orientation_credit": True},
            {"node1": "q2__M2", "node2": "c2__M2", "truth_type": "within_assay_direct", "label": "q2_c", "mark": "M2", "parent": "q2__M2", "child": "c2__M2", "orientation_credit": True},
            {"node1": "p1__M1", "node2": "s1__M1", "truth_type": "within_assay_shared_state", "label": "state1", "mark": "M1"},
            {"node1": "s1__M1", "node2": "s2__M2", "node_set1": "p1__M1;s1__M1", "node_set2": "p2__M2;s2__M2", "truth_type": "cross_assay_shared_state", "label": "cross_state", "mark": "M1-M2"},
            {"node1": "p1__M1", "node2": "expr__G", "node_set1": "p1__M1;s1__M1", "node_set2": "expr__G", "truth_type": "slow_peak_gene", "label": "slow", "mark": "M1", "parent": "p1__M1", "child": "expr__G", "orientation_credit": False},
            {"node1": "p2__M2", "node2": "expr__G", "truth_type": "fast_peak_gene_challenge", "label": "fast", "mark": "M2", "parent": "p2__M2", "child": "expr__G", "orientation_credit": False},
            {"node1": "n1__M1", "node2": "expr__G", "truth_type": "sentinel_null", "label": "null1", "mark": "M1"},
            {"node1": "n2__M2", "node2": "expr__G", "truth_type": "sentinel_null", "label": "null2", "mark": "M2"},
        ]
    )
    edges = pd.DataFrame(
        [
            _edge("p1__M1", "c1__M1", "CIRCLE", "ARROW"),
            _edge("q1__M1", "c1__M1", "CIRCLE", "ARROW"),
            _edge("p2__M2", "c2__M2", "CIRCLE", "ARROW"),
            _edge("q2__M2", "c2__M2", "CIRCLE", "ARROW"),
            _edge("p1__M1", "s1__M1", edge_type="within_assay_shared_state"),
            _edge("p1__M1", "p2__M2", edge_type="cross_assay_shared_state"),
            _edge(
                "s1__M1",
                "expr__G",
                "CIRCLE",
                "ARROW",
                "prior_constrained_peak_gene",
            ),
        ]
    )

    metrics, pairs = score_v4(edges, truth)

    assert metrics["world_success"] == 1
    assert metrics["direct_raw_adjacency_macro_recall"] == 1.0
    assert metrics["direct_typed_candidate_macro_recall"] == 1.0
    assert metrics["direct_adjacency_macro_recall"] == 1.0
    assert metrics["collider_arrowhead_recall"] == 1.0
    assert metrics["matched_null_false_positive_rate"] == 0.0
    assert metrics["gene_sink_orientations_credited"] is False
    slow = pairs.loc[pairs["truth_type"] == "slow_peak_gene"].iloc[0]
    assert slow["credited_child_arrowhead"] == 0

    nodes = sorted(set(truth["node1"]) | set(truth["node2"]))
    dense = pd.DataFrame(
        [_edge(left, right) for left, right in combinations(nodes, 2)]
    )
    dense_metrics, _ = score_v4(dense, truth)
    assert dense_metrics["n_estimated_focal_edges"] > 30
    assert dense_metrics["density_class_success"] == 0
    assert dense_metrics["world_success"] == 0


def test_shared_one_way_call_and_null_rate_fail_gate() -> None:
    truth = pd.DataFrame(
        [
            {"node1": "p__M", "node2": "c__M", "truth_type": "within_assay_direct", "label": "pc", "mark": "M", "parent": "p__M", "child": "c__M", "orientation_credit": True},
            {"node1": "q__M", "node2": "c__M", "truth_type": "within_assay_direct", "label": "qc", "mark": "M", "parent": "q__M", "child": "c__M", "orientation_credit": True},
            {"node1": "p__M", "node2": "s__M", "truth_type": "within_assay_shared_state", "label": "shared", "mark": "M"},
            {"node1": "p__M", "node2": "expr__G", "truth_type": "slow_peak_gene", "label": "slow", "mark": "M"},
            {"node1": "f__M", "node2": "expr__G", "truth_type": "fast_peak_gene_challenge", "label": "fast", "mark": "M"},
            {"node1": "n__M", "node2": "expr__G", "truth_type": "sentinel_null", "label": "null", "mark": "M"},
        ]
    )
    edges = pd.DataFrame(
        [
            _edge("p__M", "c__M", "CIRCLE", "ARROW"),
            _edge("q__M", "c__M", "CIRCLE", "ARROW"),
            _edge("p__M", "s__M", "TAIL", "ARROW", "within_assay_shared_state"),
            _edge("p__M", "expr__G", edge_type="prior_constrained_peak_gene"),
            _edge("n__M", "expr__G", edge_type="prior_constrained_peak_gene"),
        ]
    )

    metrics, _ = score_v4(edges, truth)

    assert metrics["shared_state_false_one_way_count"] == 1
    assert metrics["matched_null_false_positive_rate"] == 1.0
    assert metrics["world_success"] == 0


def test_unlabelled_focal_edges_reduce_precision_and_fail_gate() -> None:
    truth = pd.DataFrame(
        [
            {"node1": "p__M", "node2": "c__M", "truth_type": "within_assay_direct", "label": "pc", "mark": "M", "parent": "p__M", "child": "c__M", "orientation_credit": True},
            {"node1": "q__M", "node2": "c__M", "truth_type": "within_assay_direct", "label": "qc", "mark": "M", "parent": "q__M", "child": "c__M", "orientation_credit": True},
            {"node1": "p__M", "node2": "s__M", "truth_type": "within_assay_shared_state", "label": "shared", "mark": "M"},
            {"node1": "p__M", "node2": "expr__G", "truth_type": "slow_peak_gene", "label": "slow", "mark": "M"},
            {"node1": "f__M", "node2": "expr__G", "truth_type": "fast_peak_gene_challenge", "label": "fast", "mark": "M"},
            {"node1": "n__M", "node2": "expr__G", "truth_type": "sentinel_null", "label": "null", "mark": "M"},
        ]
    )
    edges = pd.DataFrame(
        [
            _edge("p__M", "c__M", "CIRCLE", "ARROW"),
            _edge("q__M", "c__M", "CIRCLE", "ARROW"),
            _edge("p__M", "s__M", edge_type="within_assay_shared_state"),
            _edge("p__M", "expr__G", edge_type="prior_constrained_peak_gene"),
            _edge("q__M", "s__M", edge_type="unresolved"),
            _edge("c__M", "s__M", edge_type="unresolved"),
        ]
    )

    metrics, _ = score_v4(edges, truth)

    assert metrics["n_unsupported_focal_edges"] == 2
    assert metrics["n_false_typed_focal_edges"] == 2
    assert metrics["primary_skeleton_precision"] == 4 / 6
    assert metrics["precision_class_success"] == 0
    assert metrics["world_success"] == 0


def test_wrong_layer_adjacency_does_not_receive_typed_recall() -> None:
    truth = pd.DataFrame(
        [
            {"node1": "p__M", "node2": "c__M", "truth_type": "within_assay_direct", "label": "pc", "mark": "M", "parent": "p__M", "child": "c__M", "orientation_credit": True},
            {"node1": "q__M", "node2": "c__M", "truth_type": "within_assay_direct", "label": "qc", "mark": "M", "parent": "q__M", "child": "c__M", "orientation_credit": True},
            {"node1": "p__M", "node2": "s__M", "truth_type": "within_assay_shared_state", "label": "shared", "mark": "M"},
            {"node1": "p__M", "node2": "expr__G", "truth_type": "slow_peak_gene", "label": "slow", "mark": "M"},
            {"node1": "f__M", "node2": "expr__G", "truth_type": "fast_peak_gene_challenge", "label": "fast", "mark": "M"},
            {"node1": "n__M", "node2": "expr__G", "truth_type": "sentinel_null", "label": "null", "mark": "M"},
        ]
    )
    edges = pd.DataFrame(
        [
            _edge("p__M", "c__M", edge_type="within_assay_shared_state"),
            _edge("q__M", "c__M", edge_type="within_assay_shared_state"),
            _edge("p__M", "s__M", edge_type="within_assay_direct_candidate"),
            _edge("p__M", "expr__G", edge_type="cross_assay_shared_state"),
        ]
    )

    metrics, pairs = score_v4(edges, truth)

    assert metrics["direct_raw_adjacency_macro_recall"] == 1.0
    assert metrics["direct_typed_candidate_macro_recall"] == 0.0
    assert metrics["direct_adjacency_macro_recall"] == 0.0
    assert metrics["direct_adjacency_gate_requires_direct_candidate_type"] is True
    assert metrics["shared_state_adjacency_recall"] == 0.0
    assert metrics["slow_peak_gene_present"] == 0
    assert metrics["n_wrong_layer_primary_edges"] == 4
    assert pairs.loc[pairs["label"] == "pc", "adjacency_present"].iloc[0] == 1
