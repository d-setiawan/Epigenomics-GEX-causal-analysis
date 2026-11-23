from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_latent_loading_groups_v4 as scorer  # noqa: E402


MARKS = ("A", "B", "C", "D", "E", "F")
PRIMARY_DIMENSIONS = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2}
Q_DIMENSIONS = {"A": 1, "B": 2, "C": 0, "D": 2, "E": 0, "F": 1}


def _metadata() -> tuple[dict[str, object], pd.DataFrame]:
    parameters: dict[str, object] = {
        "marks": list(MARKS),
        "state_dimension": PRIMARY_DIMENSIONS,
        "q_dimension": Q_DIMENSIONS,
        "state_sign": {mark: -1.0 if index % 2 else 1.0 for index, mark in enumerate(MARKS)},
        "q_sign": {mark: 1.0 if index % 2 else -1.0 for index, mark in enumerate(MARKS)},
        "slow_mark": "A",
        "gene": "expr__gene",
    }
    rows = []
    for mark in MARKS:
        for role in ("P", "Q", "C", "S", "N1", "N2"):
            if role in {"P", "S"}:
                dimension: int | str = PRIMARY_DIMENSIONS[mark]
            elif role == "Q":
                dimension = Q_DIMENSIONS[mark]
            else:
                dimension = ""
            rows.append(
                {
                    "node": f"{mark}_{role}",
                    "mark": mark,
                    "role": role,
                    "state_dimension": dimension,
                }
            )
    return parameters, pd.DataFrame(rows)


def _hyperedges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hyperedge_id": "within_A",
                "edge_type": scorer.WITHIN_GROUP_TYPE,
                "selection_status": "selected",
                "group1": "gA",
                "group2": "",
                "members": "A_P;A_S",
            },
            {
                "hyperedge_id": "within_B",
                "edge_type": scorer.WITHIN_GROUP_TYPE,
                "selection_status": "selected",
                "group1": "gB",
                "group2": "",
                "members": "B_P;B_S",
            },
            {
                "hyperedge_id": "within_C",
                "edge_type": scorer.WITHIN_GROUP_TYPE,
                "selection_status": "selected",
                "group1": "gC",
                "group2": "",
                "members": "C_P;C_S",
            },
            {
                "hyperedge_id": "gene_group",
                "edge_type": scorer.WITHIN_GROUP_TYPE,
                "selection_status": "selected",
                "group1": "gGene",
                "group2": "",
                "members": "expr__gene",
            },
            {
                "hyperedge_id": "cross_axis0",
                "edge_type": scorer.CROSS_TYPE,
                "selection_status": "selected",
                "group1": "gA",
                "group2": "gB",
                "members": "A_P;A_S;B_P;B_S",
            },
            {
                "hyperedge_id": "cross_mismatch",
                "edge_type": scorer.CROSS_TYPE,
                "selection_status": "selected",
                "group1": "gA",
                "group2": "gC",
                "members": "A_P;A_S;C_P;C_S",
            },
            {
                "hyperedge_id": "gene_wrong_layer",
                "edge_type": scorer.CROSS_TYPE,
                "selection_status": "selected",
                "group1": "gA",
                "group2": "gGene",
                "members": "A_P;A_S;expr__gene",
            },
            {
                "hyperedge_id": "unresolved_valid_axis",
                "edge_type": "unresolved",
                "selection_status": "unresolved",
                "group1": "gA",
                "group2": "gB",
                "members": "A_P;A_S;B_P;B_S",
            },
        ]
    )


def _edges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Correct primary-axis cross-assay and local claims.
            {"node1": "A_S", "node2": "B_P", "edge_type": scorer.CROSS_TYPE},
            {"node1": "C_P", "node2": "C_S", "edge_type": scorer.WITHIN_EDGE_TYPE},
            # Q is credited according to q_dimension, not its mark's primary dimension.
            {"node1": "A_P", "node2": "C_Q", "edge_type": scorer.CROSS_TYPE},
            {"node1": "C_Q", "node2": "expr__gene", "edge_type": scorer.GENE_TYPE},
            # Same axis, deliberately placed in the wrong layer.
            {"node1": "A_P", "node2": "B_S", "edge_type": scorer.WITHIN_EDGE_TYPE},
            # Different axes and non-pure C must never receive state credit.
            {"node1": "A_Q", "node2": "A_P", "edge_type": scorer.WITHIN_EDGE_TYPE},
            {"node1": "A_C", "node2": "B_P", "edge_type": scorer.CROSS_TYPE},
            # A direct edge is outside this scorer rather than a shared-state false positive.
            {
                "node1": "A_P",
                "node2": "A_C",
                "edge_type": "within_assay_direct_candidate",
            },
        ]
    )


def _axis_hyperedges() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hyperedge_id": "axis_dim0",
                "axis_id": "latent_axis_001",
                "edge_type": scorer.CHROMATIN_AXIS_TYPE,
                "selection_status": "selected",
                "features": "A_P;A_S;B_P;B_S",
            },
            {
                "hyperedge_id": "axis_dim1_with_q",
                "axis_id": "latent_axis_002",
                "edge_type": scorer.CHROMATIN_AXIS_TYPE,
                "selection_status": "selected",
                "features": "A_Q;C_P;C_S",
            },
            {
                "hyperedge_id": "axis_slow_rna",
                "axis_id": "latent_axis_003",
                "edge_type": scorer.RNA_AXIS_TYPE,
                "selection_status": "selected",
                "features": "A_P;A_S;expr__gene",
            },
            {
                "hyperedge_id": "axis_mixed_dimensions",
                "axis_id": "latent_axis_004",
                "edge_type": scorer.CHROMATIN_AXIS_TYPE,
                "selection_status": "selected",
                "features": "A_P;C_P",
            },
            {
                "hyperedge_id": "axis_nonpure",
                "axis_id": "latent_axis_005",
                "edge_type": scorer.CHROMATIN_AXIS_TYPE,
                "selection_status": "selected",
                "features": "A_C;B_P",
            },
            {
                "hyperedge_id": "axis_wrong_layer",
                "axis_id": "latent_axis_006",
                "edge_type": scorer.CROSS_TYPE,
                "selection_status": "selected",
                "features": "A_P;B_P",
            },
            {
                "hyperedge_id": "axis_valid_but_unresolved",
                "axis_id": "latent_axis_007",
                "edge_type": "unresolved",
                "selection_status": "unresolved",
                "features": "E_P;F_P",
            },
        ]
    )


def test_truth_contract_is_sign_invariant_and_uses_q_dimension() -> None:
    parameters, roles = _metadata()
    truth = scorer.build_latent_loading_truth(parameters, roles)
    indexed = truth.features.set_index("feature")

    assert indexed.loc["A_P", "state_dimension"] == 0
    assert indexed.loc["A_S", "state_dimension"] == 0
    assert indexed.loc["A_Q", "state_dimension"] == 1
    assert indexed.loc["C_Q", "state_dimension"] == 0
    assert not indexed.loc["A_C", "pure_state_feature"]
    assert pd.isna(indexed.loc["A_C", "state_dimension"])
    assert indexed.loc["expr__gene", "state_dimension"] == 0
    assert truth.eligible_cross_dimensions == (0, 1, 2)
    assert len(truth.expected_assay_groups) == 6


def test_group_and_sparse_scoring_separate_axis_from_layer_correctness() -> None:
    parameters, roles = _metadata()
    result = scorer.score_latent_loading_groups(
        world_parameters=parameters,
        role_assignments=roles,
        typed_hyperedges=_hyperedges(),
        sparse_edges=_edges(),
    )

    hyper = result.hyperedge_scores.set_index("hyperedge_id")
    assert hyper.loc["cross_axis0", "typed_valid"] == 1
    assert hyper.loc["cross_mismatch", "axis_compatible"] == 0
    assert hyper.loc["gene_wrong_layer", "axis_compatible"] == 1
    assert hyper.loc["gene_wrong_layer", "type_correct"] == 0
    assert hyper.loc["unresolved_valid_axis", "typed_valid"] == 0

    edges = result.sparse_edge_scores.set_index(["node1", "node2"])
    assert edges.loc[("A_P", "C_Q"), "typed_valid"] == 1
    assert edges.loc[("C_Q", "expr__gene"), "typed_valid"] == 1
    assert edges.loc[("A_P", "B_S"), "axis_compatible"] == 1
    assert edges.loc[("A_P", "B_S"), "type_correct"] == 0
    assert edges.loc[("A_Q", "A_P"), "axis_compatible"] == 0
    assert edges.loc[("A_C", "B_P"), "axis_compatible"] == 0
    assert edges.loc[("A_P", "A_C"), "in_loading_scope"] == 0

    metrics = result.metrics
    assert metrics["n_selected_group_definitions"] == 4
    assert metrics["selected_group_definition_axis_precision"] == 1.0
    assert metrics["combined_within_assay_ps_group_recall"] == pytest.approx(0.5)
    assert metrics["combined_cross_assay_latent_axis_coverage"] == pytest.approx(1 / 3)
    assert metrics["combined_slow_gene_axis_support"] == 1
    assert metrics["n_selected_axis_mismatch_claims"] == 3
    assert metrics["n_selected_wrong_layer_claims"] == 2
    assert metrics["n_out_of_scope_sparse_edges"] == 1
    assert not metrics["direct_causal_direction_credited"]
    assert not metrics["cell_correspondence_used"]


def test_native_nary_axis_scoring_counts_each_hyperedge_once() -> None:
    parameters, roles = _metadata()
    result = scorer.score_latent_loading_groups(
        world_parameters=parameters,
        role_assignments=roles,
        typed_axis_hyperedges=_axis_hyperedges(),
    )

    axes = result.axis_hyperedge_scores.set_index("hyperedge_id")
    assert axes.loc["axis_dim0", "typed_valid"] == 1
    assert axes.loc["axis_dim1_with_q", "typed_valid"] == 1
    assert axes.loc["axis_slow_rna", "typed_valid"] == 1
    assert axes.loc["axis_slow_rna", "expected_edge_type"] == scorer.RNA_AXIS_TYPE
    assert axes.loc["axis_mixed_dimensions", "pure_member_set"] == 1
    assert axes.loc["axis_mixed_dimensions", "axis_compatible"] == 0
    assert axes.loc["axis_mixed_dimensions", "failure_reason"] == "mixed_state_dimensions"
    assert axes.loc["axis_nonpure", "pure_member_set"] == 0
    assert axes.loc["axis_nonpure", "failure_reason"] == "nonpure_peak_member"
    assert axes.loc["axis_wrong_layer", "axis_compatible"] == 1
    assert axes.loc["axis_wrong_layer", "type_correct"] == 0
    assert axes.loc["axis_valid_but_unresolved", "axis_compatible"] == 1
    assert axes.loc["axis_valid_but_unresolved", "typed_valid"] == 0
    assert axes.loc["axis_valid_but_unresolved", "failure_reason"] == "wrong_axis_layer"

    metrics = result.metrics
    assert metrics["n_selected_latent_axis_hyperedges"] == 6
    assert metrics["n_pure_selected_latent_axis_hyperedges"] == 5
    assert metrics["n_axis_compatible_selected_latent_axis_hyperedges"] == 4
    assert metrics["n_typed_valid_selected_latent_axis_hyperedges"] == 3
    assert metrics["selected_latent_axis_purity_precision"] == pytest.approx(5 / 6)
    assert metrics["selected_latent_axis_axis_precision"] == pytest.approx(4 / 6)
    assert metrics["selected_latent_axis_typed_precision"] == pytest.approx(3 / 6)
    assert metrics["native_recovered_latent_axis_dimensions"] == [0, 1]
    assert metrics["native_latent_axis_dimension_coverage"] == pytest.approx(2 / 3)
    assert metrics["native_slow_rna_axis_support"] == 1
    assert metrics["combined_cross_assay_latent_axis_coverage"] == pytest.approx(2 / 3)
    assert metrics["combined_slow_gene_axis_support"] == 1
    assert metrics["n_selected_latent_axis_mismatch_claims"] == 2
    assert metrics["n_selected_latent_axis_wrong_layer_claims"] == 1
    assert metrics["n_selected_axis_mismatch_claims"] == 2
    assert metrics["n_selected_wrong_layer_claims"] == 1


def test_native_axis_scorer_rejects_duplicate_ids_and_unknown_nodes() -> None:
    parameters, roles = _metadata()
    duplicate = _axis_hyperedges().iloc[:2].copy()
    duplicate.loc[:, "axis_id"] = "same_axis"
    with pytest.raises(ValueError, match="axis identifiers"):
        scorer.score_latent_loading_groups(
            world_parameters=parameters,
            role_assignments=roles,
            typed_axis_hyperedges=duplicate,
        )

    unknown = _axis_hyperedges().iloc[:1].copy()
    unknown.loc[:, "features"] = "A_P;not_in_scorer_bundle"
    with pytest.raises(ValueError, match="unknown scorer nodes"):
        scorer.score_latent_loading_groups(
            world_parameters=parameters,
            role_assignments=roles,
            typed_axis_hyperedges=unknown,
        )


def test_stale_dimensions_and_unknown_discovery_nodes_fail_closed() -> None:
    parameters, roles = _metadata()
    stale = roles.copy()
    stale.loc[stale["node"] == "A_Q", "state_dimension"] = 2
    with pytest.raises(ValueError, match="State dimension mismatch"):
        scorer.build_latent_loading_truth(parameters, stale)

    unknown = _edges().iloc[:1].copy()
    unknown.loc[:, "node2"] = "not_in_scorer_bundle"
    with pytest.raises(ValueError, match="unknown scorer nodes"):
        scorer.score_latent_loading_groups(
            world_parameters=parameters,
            role_assignments=roles,
            sparse_edges=unknown,
        )
