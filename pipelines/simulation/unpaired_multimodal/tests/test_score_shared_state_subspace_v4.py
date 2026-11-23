from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import score_shared_state_subspace_v4 as scorer  # noqa: E402
import shared_state_subspace_v4 as subspace  # noqa: E402


MARKS = tuple(f"M{index}" for index in range(6))
ROLES = ("P", "Q", "C", "S", "N1", "N2")
GENE = "expr__TEST"
ROUNDTRIP_BOUNDARY_FEATURE = "peak__M0__P"


def _parameters(seed: int = 49_001) -> dict[str, object]:
    # Exact axis maps are deliberately absent: they are outside this scorer.
    return {"seed": seed, "marks": list(MARKS), "gene": GENE}


def _roles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node": f"peak__{mark}__{role}", "mark": mark, "role": role}
            for mark in MARKS
            for role in ROLES
        ]
    )


ASSAY_PARTICIPATION_COLUMNS = [
    "assay_participation_partial_conjunction_p",
    "assay_participation_holm_p",
    "assay_participation_single_feature_order_statistic_p",
    "assay_participation_single_feature_p",
    "assay_participation_single_feature_witness_view_count",
    "assay_participation_single_feature_multiview_replication_gate",
    "assay_participation_partial_conjunction_multiview_replication_gate",
    "assay_participation_omnibus_route_count",
    "assay_participation_omnibus_p",
    "assay_participation_omnibus_holm_p",
    "assay_participation_winning_route",
    "assay_participation_selection_route",
    "assay_participation_selected",
    "assay_participation_order",
    "assay_participation_family_size",
    "assay_participation_contributing_feature_count",
    "assay_participation_minimum_witness_view_count",
    "assay_participation_multiview_replication_gate",
    "assay_participation_reporter",
]


def _refresh_assay_participation(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute the v9 complete-null omnibus contract from observable evidence."""

    output = frame.copy()
    participation = subspace._select_assay_state_participation(
        output,
        subspace.SharedStateSubspaceConfig(),
    ).set_index("modality")
    modalities = output["modality"].astype(str)
    repeated_columns = {
        "assay_participation_partial_conjunction_p": (
            "partial_conjunction_empirical_p"
        ),
        "assay_participation_holm_p": "assay_participation_holm_p",
        "assay_participation_single_feature_order_statistic_p": (
            "single_feature_order_statistic_p"
        ),
        "assay_participation_single_feature_p": "single_feature_bonferroni_p",
        "assay_participation_single_feature_witness_view_count": (
            "single_feature_witness_view_count"
        ),
        "assay_participation_single_feature_multiview_replication_gate": (
            "single_feature_multiview_replication_gate"
        ),
        "assay_participation_partial_conjunction_multiview_replication_gate": (
            "partial_conjunction_multiview_replication_gate"
        ),
        "assay_participation_omnibus_route_count": (
            "assay_participation_omnibus_route_count"
        ),
        "assay_participation_omnibus_p": "assay_participation_omnibus_p",
        "assay_participation_omnibus_holm_p": (
            "assay_participation_omnibus_holm_p"
        ),
        "assay_participation_winning_route": (
            "assay_participation_winning_route"
        ),
        "assay_participation_selection_route": (
            "assay_participation_selection_route"
        ),
        "assay_participation_order": "partial_conjunction_order",
        "assay_participation_family_size": "n_formally_eligible_features",
        "assay_participation_contributing_feature_count": (
            "contributing_feature_count"
        ),
        "assay_participation_minimum_witness_view_count": (
            "minimum_witness_view_count"
        ),
        "assay_participation_multiview_replication_gate": (
            "multiview_replication_gate"
        ),
    }
    for output_column, participation_column in repeated_columns.items():
        output[output_column] = modalities.map(participation[participation_column])
    output["assay_participation_selected"] = modalities.map(
        participation["selection_status"].eq("selected")
    )
    selected_reporter = modalities.map(participation["selected_reporter"])
    output["assay_participation_reporter"] = (
        output["assay_participation_selected"]
        & output["feature"].astype(str).eq(selected_reporter)
    )
    return output


def _assay_participation_provenance(
    membership: pd.DataFrame,
) -> dict[str, object]:
    participation = subspace._select_assay_state_participation(
        membership,
        subspace.SharedStateSubspaceConfig(),
    )
    selected_modalities = sorted(
        participation.loc[
            participation["selection_status"].eq("selected"), "modality"
        ].astype(str)
    )
    return {
        "edge_type": "assay_shared_state_participation",
        "feature_empirical_p_calibration": (
            "existing_state_depth_orthogonal_synchronized_conditional_"
            "wild_null"
        ),
        "within_assay_single_feature_test": (
            "bonferroni_minP_valid_under_arbitrary_feature_dependence"
        ),
        "within_assay_partial_conjunction_test": (
            "bonferroni_partial_conjunction_valid_under_arbitrary_"
            "feature_dependence"
        ),
        "assay_participation_null": "no_state_responsive_features_in_the_assay",
        "partial_conjunction_null": (
            "fewer_than_r_state_responsive_features_in_the_assay"
        ),
        "within_assay_omnibus_combination": (
            "bonferroni_over_predeclared_replication_gated_routes"
        ),
        "omnibus_route_count_policy": (
            "one_for_structural_singleton_otherwise_two"
        ),
        "failed_replication_route_p_value": 1.0,
        "across_assay_error_control": (
            "holm_strong_FWER_without_cross_assay_dependence_assumption"
        ),
        "assay_multiplicity_family": "all_observed_assays",
        "selection_rule": (
            "assay_participation_omnibus_holm_p_at_most_configured_"
            "error_probability"
        ),
        "complete_null_order_statistic_minP_used": True,
        "minimum_witness_views": 6,
        "affects_feature_membership": False,
        "affects_existing_typed_subspace_hyperedge": False,
        "reporter_or_witness_implies_feature_membership": False,
        "membership_annotation_columns": ASSAY_PARTICIPATION_COLUMNS,
        "records": participation.to_dict(orient="records"),
        "n_selected_assays": len(selected_modalities),
        "selected_modalities": sorted(selected_modalities),
    }


def _membership(selected: set[str] | None = None) -> pd.DataFrame:
    roles = _roles()
    eligible = set(
        roles.loc[roles["role"].isin(["P", "Q", "S", "C"]), "node"].astype(str)
    ) | {GENE}
    selected = eligible if selected is None else selected
    rows: list[dict[str, object]] = []
    for record in roles.to_dict(orient="records") + [
        {"node": GENE, "mark": "", "role": "GENE"}
    ]:
        feature = str(record["node"])
        mark = str(record["mark"])
        is_selected = feature in selected
        view_fit = 0.40 if is_selected else 0.01
        reference_fit = 0.43 if is_selected else 0.02
        null_upper = 0.05
        row = {
            "feature": feature,
            "modality": "rna" if feature == GENE else f"chrom_{mark}",
            "channel": "RNA" if feature == GENE else mark,
            "n_views": 6,
            "n_cells": 1000,
            "n_events": -1 if feature == GENE else 150,
            "n_nonevents": -1 if feature == GENE else 850,
            "binary_event_gate": True,
            "formal_test_eligible": True,
            "formal_test_abstention_reason": "",
            # Deliberately strong for selected and unresolved features alike:
            # this decomposition is descriptive and never a membership gate.
            "oof_marginal_state_fit": 0.90,
            "oof_depth_only_fit": 0.10,
            "oof_joint_fit": 0.80,
            "oof_partial_state_fit": 7.0 / 9.0,
            "median_conditional_wald": view_fit,
            "lower_quartile_conditional_wald": view_fit,
            "median_conditional_score_norm": 0.20 if is_selected else 0.01,
            "minimum_conditional_score_norm": 0.18 if is_selected else 0.005,
            "median_conditional_score_direction_stability": (
                0.95 if is_selected else 0.20
            ),
            "minimum_conditional_score_direction_stability": (
                0.90 if is_selected else 0.10
            ),
            **{
                f"conditional_wald__model_seed{view}": view_fit
                for view in range(6)
            },
            "empirical_null_p": 0.001 if is_selected else 0.8,
            "reference_conditional_wald": reference_fit,
            "empirical_null_fdr": 1.0,
            "empirical_null_family_size": 1 if feature == GENE else 6,
            "conditional_wald_null_median": 0.01,
            "conditional_wald_null_95pct": null_upper,
            "conditional_wald_above_null_95pct": reference_fit - null_upper,
            "conditional_wald_view_support": 1.0 if is_selected else 0.0,
            "feature_global_maxT_p": 0.02 if is_selected else 0.8,
            "feature_global_maxT_confirmation_gate": is_selected,
            "assay_global_maxT_p": 1.0,
            "assay_global_maxT_admission_gate": False,
            "direction_stability_diagnostic": is_selected,
            "feature_maxT_unconditional_confirmation_gate": is_selected,
            "feature_direction_stability_confirmation_gate": is_selected,
            "pre_redundancy_selection_status": (
                "selected" if is_selected else "unresolved"
            ),
            "pre_redundancy_abstention_reason": "",
            "selection_status": "selected" if is_selected else "unresolved",
            "abstention_reason": "",
            "redundancy_direction_stability_gate": is_selected,
            "redundancy_fallback_selected": False,
            "redundancy_demoted": False,
            "within_assay_strong_core_member": False,
            "within_assay_strong_core_count": 0,
            "within_assay_weak_maxT_tail_demoted": False,
            "direct_layer_overlap": feature == "peak__M0__C",
            "direct_layer_annotation_available": True,
            "subspace_id": scorer.SUBSPACE_ID if is_selected else "",
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    view_columns = [f"conditional_wald__model_seed{view}" for view in range(6)]
    for _, indices in frame.groupby("modality", sort=True).groups.items():
        group_indices = list(indices)
        eligible_indices = frame.loc[
            group_indices
        ].index[frame.loc[group_indices, "formal_test_eligible"]].tolist()
        frame.loc[group_indices, "empirical_null_family_size"] = len(
            eligible_indices
        )
        frame.loc[group_indices, "empirical_null_fdr"] = 1.0
        if eligible_indices:
            frame.loc[eligible_indices, "empirical_null_fdr"] = (
                scorer._benjamini_hochberg(
                    frame.loc[eligible_indices, "empirical_null_p"].to_numpy(
                        dtype=float
                    )
                )
            )
        assay_p = (
            float(frame.loc[eligible_indices, "feature_global_maxT_p"].min())
            if eligible_indices
            else 1.0
        )
        frame.loc[group_indices, "assay_global_maxT_p"] = assay_p
        frame.loc[
            group_indices, "assay_global_maxT_admission_gate"
        ] = assay_p <= 0.10
    reasons: list[str] = []
    for record in frame.to_dict(orient="records"):
        if not bool(record["formal_test_eligible"]):
            reasons.append(str(record["formal_test_abstention_reason"]))
            continue
        failures: list[str] = []
        if float(record["empirical_null_fdr"]) > 0.10:
            failures.append("empirical_null_fdr")
        if not bool(record["assay_global_maxT_admission_gate"]):
            failures.append("assay_global_maxT_admission")
        if not bool(record["feature_global_maxT_confirmation_gate"]):
            failures.append("feature_global_maxT_confirmation")
        if not bool(record["feature_direction_stability_confirmation_gate"]):
            failures.append("feature_direction_stability_confirmation")
        view_count = sum(float(record[column]) >= 0.05 for column in view_columns)
        if view_count < 4:
            failures.append("conditional_wald_view_support")
        reasons.append(";".join(failures))
    frame["pre_redundancy_abstention_reason"] = reasons
    frame["abstention_reason"] = reasons
    frame = _refresh_assay_participation(frame)
    return frame.loc[:, list(scorer.MEMBERSHIP_COLUMNS)]


def _hyperedge(
    membership: pd.DataFrame,
    *,
    direction: bool = False,
    exact_axis: bool = False,
    direct: bool = False,
) -> pd.DataFrame:
    selected = membership.loc[membership["selection_status"] == "selected"]
    members = sorted(selected["feature"].astype(str))
    modalities = sorted(set(selected["modality"].astype(str)))
    hyperedge_selected = len(modalities) >= 2
    return pd.DataFrame(
        [
            {
                "subspace_id": scorer.SUBSPACE_ID,
                "edge_type": scorer.SUBSPACE_EDGE_TYPE,
                "modalities": ";".join(modalities),
                "member_features": ";".join(members),
                "n_modalities": len(modalities),
                "n_features": len(members),
                "selection_status": "selected" if hyperedge_selected else "unresolved",
                "empirical_multimodality_p": 0.005,
                "error_control_method": scorer.EXPECTED_ERROR_CONTROL_METHOD,
                "conditional_on_common_state_model": True,
                "causal_interpretation": "hypothesis_only",
                "direction_claimed": direction,
                "exact_axis_pairing_claimed": exact_axis,
                "direct_causal_claimed": direct,
                "abstention_reason": "" if hyperedge_selected else "insufficient_modalities",
            }
        ]
    )


def _local_truth() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node1": "peak__M0__P",
                "node2": "peak__M0__C",
                "truth_type": "within_assay_direct",
                "label": "pc",
                "mark": "M0",
                "parent": "peak__M0__P",
                "child": "peak__M0__C",
                "orientation_credit": True,
            },
            {
                "node1": "peak__M0__Q",
                "node2": "peak__M0__C",
                "truth_type": "within_assay_direct",
                "label": "qc",
                "mark": "M0",
                "parent": "peak__M0__Q",
                "child": "peak__M0__C",
                "orientation_credit": True,
            },
            {
                "node1": "peak__M0__P",
                "node2": "peak__M0__S",
                "truth_type": "within_assay_shared_state",
                "label": "shared",
                "mark": "M0",
            },
            {
                "node1": "peak__M0__P",
                "node2": GENE,
                "truth_type": "slow_peak_gene",
                "label": "slow",
                "mark": "M0",
            },
            {
                "node1": "peak__M1__P",
                "node2": GENE,
                "truth_type": "fast_peak_gene_challenge",
                "label": "fast",
                "mark": "M1",
            },
            {
                "node1": "peak__M0__N1",
                "node2": GENE,
                "truth_type": "sentinel_null",
                "label": "null",
                "mark": "M0",
            },
        ]
    )


def _local_graph() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node1": "peak__M0__P",
                "endpoint1": "CIRCLE",
                "node2": "peak__M0__C",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            },
            {
                "node1": "peak__M0__Q",
                "endpoint1": "CIRCLE",
                "node2": "peak__M0__C",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
            },
        ]
    )


def _perfect_score(seed: int = 49_001) -> scorer.SharedStateSubspaceScore:
    membership = _membership()
    return scorer.score_shared_state_subspace(
        world_parameters=_parameters(seed),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
        typed_graph=_local_graph(),
        truth_typed_edges=_local_truth(),
    )


def test_perfect_dimension_agnostic_subspace_and_local_structure_pass() -> None:
    result = _perfect_score()
    metrics = result.metrics

    assert metrics["n_truth_state_responsive_chromatin_features"] == 24
    assert metrics["n_truth_state_responsive_features"] == 25
    assert metrics["subspace_member_precision"] == 1.0
    assert metrics["chromatin_subspace_member_precision"] == 1.0
    assert metrics["subspace_member_recall"] == 1.0
    assert metrics["P_member_recall"] == 1.0
    assert metrics["Q_member_recall"] == 1.0
    assert metrics["S_member_recall"] == 1.0
    assert metrics["sentinel_null_inclusion_rate"] == 0.0
    assert metrics["C_state_responsive_member_recall"] == 1.0
    assert metrics["rna_gene_subspace_support"] == 1
    assert metrics["assay_coverage"] == 1.0
    assert metrics["hyperedge_schema_consistent"] == 1
    assert metrics["n_unsupported_claim_flags"] == 0
    assert metrics["subspace_world_success"] == 1
    assert metrics["local_structure_success"] == 1
    assert metrics["joint_world_success"] == 1
    assert metrics["direct_layer_overlap_annotation_exact"] == 1
    assert metrics["exact_latent_axis_identity_scored"] is False
    assert metrics["exact_cross_assay_feature_pairs_scored"] is False
    assert metrics["cross_assay_causal_direction_scored"] is False
    assert metrics["rna_gene_exact_peak_link_scored"] is False
    assert len(result.hyperedge_scores) == 1
    assert "node1" not in result.hyperedge_scores.columns


def test_exact_axis_and_direct_claims_are_separately_penalized() -> None:
    membership = _membership()
    hyperedge = _hyperedge(
        membership, direction=True, exact_axis=True, direct=True
    )
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=hyperedge,
    )

    assert result.metrics["n_unsupported_claim_flags"] == 3
    assert result.metrics["n_direction_claims"] == 1
    assert result.metrics["n_exact_axis_pairing_claims"] == 1
    assert result.metrics["n_direct_causal_claims"] == 1
    assert result.metrics["hyperedge_schema_consistent"] == 0
    assert result.metrics["hyperedge_class_success"] == 0
    assert result.metrics["subspace_world_success"] == 0


def test_selected_hyperedge_empty_reason_survives_tsv_roundtrip(
    tmp_path: Path,
) -> None:
    membership = _membership()
    path = tmp_path / "typed_subspace_hyperedges.tsv"
    _hyperedge(membership).to_csv(path, sep="\t", index=False)
    roundtripped = pd.read_csv(path, sep="\t")

    # This is pandas' normal representation of a terminal empty TSV field.
    assert pd.isna(roundtripped.loc[0, "abstention_reason"])
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=roundtripped,
    )

    assert result.metrics["abstention_reason_coherent"] == 1
    assert result.metrics["hyperedge_schema_consistent"] == 1
    assert result.metrics["hyperedge_class_success"] == 1


def test_shared_subspace_label_must_be_conditional_and_hypothesis_only() -> None:
    membership = _membership()
    hyperedge = _hyperedge(membership)
    hyperedge.loc[0, "conditional_on_common_state_model"] = False
    hyperedge.loc[0, "causal_interpretation"] = "identified"
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=hyperedge,
    )

    assert result.metrics["conditional_common_state_interpretation"] == 0
    assert result.metrics["hypothesis_only_interpretation"] == 0
    assert result.metrics["hyperedge_schema_consistent"] == 0
    assert result.metrics["subspace_world_success"] == 0


def test_synchronized_maxT_error_control_is_part_of_hyperedge_contract() -> None:
    assert scorer.EXPECTED_ERROR_CONTROL_METHOD == (
        "synchronized_conditional_score_wild_feature_maxT_confirmation_"
        "borderline_direction_stability_"
        "within_assay_redundancy_confirmation_"
        "within_assay_strong_maxT_core_weak_tail_confirmation_"
        "assay_maxT_admission_"
        "within_assay_BH_conservative_order_2_partial_conjunction"
    )
    membership = _membership()
    hyperedge = _hyperedge(membership)
    hyperedge.loc[0, "empirical_multimodality_p"] = 0.20
    hyperedge.loc[0, "selection_status"] = "unresolved"
    hyperedge.loc[0, "abstention_reason"] = "multimodality_maxT"
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=hyperedge,
    )
    assert result.metrics["selection_status_coherent"] == 1
    assert result.metrics["empirical_multimodality_gate"] == 0
    assert result.metrics["hyperedge_schema_consistent"] == 1
    assert result.metrics["hyperedge_class_success"] == 0

    wrong_method = _hyperedge(membership)
    wrong_method.loc[0, "error_control_method"] = "unadjusted"
    wrong = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=wrong_method,
    )
    assert wrong.metrics["error_control_method_correct"] == 0
    assert wrong.metrics["hyperedge_schema_consistent"] == 0

    broken_membership = membership.copy()
    mask = broken_membership["modality"] == "chrom_M0"
    broken_membership.loc[mask, "feature_global_maxT_p"] = 0.20
    broken_membership.loc[mask, "feature_global_maxT_confirmation_gate"] = False
    broken_membership.loc[
        mask, "feature_maxT_unconditional_confirmation_gate"
    ] = False
    broken_membership.loc[mask, "assay_global_maxT_p"] = 0.20
    broken_membership.loc[mask, "assay_global_maxT_admission_gate"] = False
    with pytest.raises(ValueError, match="selection_status disagrees"):
        scorer.score_shared_state_subspace(
            world_parameters=_parameters(),
            role_assignments=_roles(),
            feature_membership=broken_membership,
            typed_subspace_hyperedges=_hyperedge(broken_membership),
        )


def test_feature_maxT_and_borderline_direction_stability_gate_membership() -> None:
    membership = _membership()
    target = membership["feature"].eq("peak__M0__Q")
    membership.loc[target, "feature_global_maxT_p"] = 0.20
    membership.loc[target, "feature_global_maxT_confirmation_gate"] = False
    membership.loc[target, "median_conditional_score_direction_stability"] = 0.20
    membership.loc[target, "minimum_conditional_score_direction_stability"] = 0.10
    membership.loc[target, "direction_stability_diagnostic"] = False
    membership.loc[target, "feature_maxT_unconditional_confirmation_gate"] = False
    membership.loc[target, "feature_direction_stability_confirmation_gate"] = False
    membership.loc[target, "pre_redundancy_selection_status"] = "unresolved"
    membership.loc[target, "pre_redundancy_abstention_reason"] = (
        "feature_global_maxT_confirmation;feature_direction_stability_confirmation"
    )
    membership.loc[target, "selection_status"] = "unresolved"
    membership.loc[target, "abstention_reason"] = (
        "feature_global_maxT_confirmation;feature_direction_stability_confirmation"
    )
    membership.loc[target, "redundancy_direction_stability_gate"] = False
    membership.loc[target, "subspace_id"] = ""

    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )
    assert result.metrics["subspace_member_precision"] == 1.0
    assert result.metrics["Q_member_recall"] == pytest.approx(5 / 6)


def test_assay_participation_uses_complete_null_omnibus_and_is_not_membership() -> None:
    membership = _membership()
    chrom = membership.loc[membership["modality"].eq("chrom_M0")]
    rna = membership.loc[membership["modality"].eq("rna")]

    # Chromatin uses r=2 of m=6: (6 - 2 + 1) * 0.001 = 0.005.
    assert chrom["assay_participation_order"].eq(2).all()
    assert chrom["assay_participation_family_size"].eq(6).all()
    assert chrom["assay_participation_partial_conjunction_p"].eq(0.005).all()
    assert chrom["assay_participation_single_feature_p"].eq(0.006).all()
    assert chrom["assay_participation_omnibus_route_count"].eq(2).all()
    assert chrom["assay_participation_omnibus_p"].eq(0.01).all()
    assert chrom["assay_participation_omnibus_holm_p"].eq(0.06).all()
    assert chrom["assay_participation_winning_route"].eq(
        "partial_conjunction"
    ).all()
    assert chrom["assay_participation_selection_route"].eq(
        "partial_conjunction"
    ).all()
    # RNA is structurally singleton, so it uses the predeclared r=1 exception.
    assert rna["assay_participation_order"].eq(1).all()
    assert rna["assay_participation_partial_conjunction_p"].eq(0.001).all()
    assert rna["assay_participation_omnibus_route_count"].eq(1).all()
    assert rna["assay_participation_omnibus_p"].eq(0.001).all()
    assert rna["assay_participation_omnibus_holm_p"].eq(0.007).all()
    assert rna["assay_participation_winning_route"].eq("single_feature").all()
    assert rna["assay_participation_selection_route"].eq("single_feature").all()
    assert membership["assay_participation_holm_p"].max() == pytest.approx(0.03)
    assert membership["assay_participation_selected"].all()
    assert membership[
        "assay_participation_minimum_witness_view_count"
    ].eq(6).all()

    # Make the leading M0 reporter fail strict feature membership without
    # changing its state/depth-calibrated empirical p-value.  The assay call
    # and reporter remain evidence annotations; neither promotes the feature.
    target = membership["feature"].eq("peak__M0__C")
    membership.loc[target, "feature_global_maxT_p"] = 0.20
    membership.loc[target, "feature_global_maxT_confirmation_gate"] = False
    membership.loc[target, "median_conditional_score_direction_stability"] = 0.20
    membership.loc[target, "minimum_conditional_score_direction_stability"] = 0.10
    membership.loc[target, "direction_stability_diagnostic"] = False
    membership.loc[target, "feature_maxT_unconditional_confirmation_gate"] = False
    membership.loc[target, "feature_direction_stability_confirmation_gate"] = False
    reason = (
        "feature_global_maxT_confirmation;feature_direction_stability_confirmation"
    )
    membership.loc[target, "pre_redundancy_selection_status"] = "unresolved"
    membership.loc[target, "pre_redundancy_abstention_reason"] = reason
    membership.loc[target, "selection_status"] = "unresolved"
    membership.loc[target, "abstention_reason"] = reason
    membership.loc[target, "redundancy_direction_stability_gate"] = False
    membership.loc[target, "subspace_id"] = ""
    membership = _refresh_assay_participation(membership)

    target_row = membership.loc[target].iloc[0]
    assert target_row["assay_participation_reporter"]
    assert target_row["assay_participation_selected"]
    assert target_row["selection_status"] == "unresolved"
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )
    assert result.metrics["C_state_responsive_member_recall"] == pytest.approx(5 / 6)


def test_one_strong_all_views_feature_uses_single_feature_omnibus_route() -> None:
    membership = _membership()
    m0 = membership["modality"].eq("chrom_M0")
    strong = membership["feature"].eq("peak__M0__P")
    membership.loc[m0, "empirical_null_p"] = 0.8
    membership.loc[m0, "reference_conditional_wald"] = 0.02
    membership.loc[m0, "conditional_wald_view_support"] = 0.0
    membership.loc[strong, "empirical_null_p"] = 0.001
    membership.loc[strong, "reference_conditional_wald"] = 0.43
    membership.loc[strong, "conditional_wald_view_support"] = 1.0
    membership = _refresh_assay_participation(membership)

    assert membership.loc[m0, "assay_participation_single_feature_p"].eq(
        0.006
    ).all()
    assert membership.loc[m0, "assay_participation_partial_conjunction_p"].eq(
        1.0
    ).all()
    assert membership.loc[
        m0, "assay_participation_single_feature_multiview_replication_gate"
    ].all()
    assert not membership.loc[
        m0,
        "assay_participation_partial_conjunction_multiview_replication_gate",
    ].any()
    assert membership.loc[m0, "assay_participation_omnibus_p"].eq(0.012).all()
    assert membership.loc[m0, "assay_participation_omnibus_holm_p"].eq(
        0.06
    ).all()
    assert membership.loc[m0, "assay_participation_winning_route"].eq(
        "single_feature"
    ).all()
    assert membership.loc[m0, "assay_participation_selected"].all()


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        (
            "assay_participation_partial_conjunction_p",
            0.006,
            "partial-conjunction p-value is inconsistent",
        ),
        (
            "assay_participation_holm_p",
            0.04,
            "Holm p-value is inconsistent",
        ),
        (
            "assay_participation_omnibus_p",
            0.02,
            "omnibus evidence is inconsistent",
        ),
        (
            "assay_participation_omnibus_holm_p",
            0.08,
            "omnibus Holm p-value is inconsistent",
        ),
    ],
)
def test_assay_participation_adjusted_p_values_are_recomputed_not_trusted(
    column: str, replacement: float, message: str
) -> None:
    membership = _membership()
    membership.loc[membership["modality"].eq("chrom_M0"), column] = replacement
    with pytest.raises(ValueError, match=message):
        scorer.score_shared_state_subspace(
            world_parameters=_parameters(),
            role_assignments=_roles(),
            feature_membership=membership,
            typed_subspace_hyperedges=_hyperedge(membership),
        )


def test_assay_participation_provenance_forbids_feature_promotion() -> None:
    membership = _membership()
    provenance = {
        "assay_state_participation": _assay_participation_provenance(membership)
    }
    audit = scorer._validate_assay_participation_provenance(
        membership, provenance
    )
    assert audit["n_selected_assays"] == 7

    malformed = json.loads(json.dumps(provenance))
    malformed["assay_state_participation"][
        "reporter_or_witness_implies_feature_membership"
    ] = True
    with pytest.raises(ValueError, match="provenance contract mismatch"):
        scorer._validate_assay_participation_provenance(membership, malformed)

    extra_record_field = json.loads(json.dumps(provenance))
    extra_record_field["assay_state_participation"]["records"][0][
        "unregistered_route_note"
    ] = "not frozen"
    with pytest.raises(ValueError, match="record fields are missing or extra"):
        scorer._validate_assay_participation_provenance(
            membership, extra_record_field
        )


def test_membership_rejects_unregistered_omnibus_columns() -> None:
    membership = _membership()
    membership["unregistered_omnibus_debug"] = 0.0
    with pytest.raises(ValueError, match="unregistered columns"):
        scorer.score_shared_state_subspace(
            world_parameters=_parameters(),
            role_assignments=_roles(),
            feature_membership=membership,
            typed_subspace_hyperedges=_hyperedge(membership),
        )


def test_single_route_survives_when_partial_conjunction_branch_fails_six_views() -> None:
    membership = _membership()
    row = membership.index[membership["feature"].eq("peak__M0__P")][0]
    view_columns = [f"conditional_wald__model_seed{view}" for view in range(6)]
    four_pass = [0.01, 0.01, 0.40, 0.40, 0.40, 0.40]
    membership.loc[row, view_columns] = four_pass
    membership.loc[row, "median_conditional_wald"] = pd.Series(four_pass).median()
    membership.loc[row, "lower_quartile_conditional_wald"] = pd.Series(
        four_pass
    ).quantile(0.25)
    membership.loc[row, "conditional_wald_view_support"] = 4.0 / 6.0
    membership = _refresh_assay_participation(membership)
    m0 = membership["modality"].eq("chrom_M0")
    assert membership.loc[row, "selection_status"] == "selected"
    assert membership.loc[m0, "assay_participation_selected"].all()
    assert membership.loc[
        m0, "assay_participation_minimum_witness_view_count"
    ].eq(4).all()
    assert membership.loc[
        m0,
        "assay_participation_single_feature_multiview_replication_gate",
    ].all()
    assert not membership.loc[
        m0,
        "assay_participation_partial_conjunction_multiview_replication_gate",
    ].any()
    assert membership.loc[m0, "assay_participation_winning_route"].eq(
        "single_feature"
    ).all()
    assert membership.loc[m0, "assay_participation_selection_route"].eq(
        "single_feature"
    ).all()
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )
    assert result.metrics["P_member_recall"] == 1.0

    three_pass = [0.01, 0.01, 0.01, 0.40, 0.40, 0.40]
    membership.loc[row, view_columns] = three_pass
    membership.loc[row, "median_conditional_wald"] = pd.Series(three_pass).median()
    membership.loc[row, "lower_quartile_conditional_wald"] = pd.Series(
        three_pass
    ).quantile(0.25)
    membership.loc[row, "conditional_wald_view_support"] = 3.0 / 6.0
    membership = _refresh_assay_participation(membership)
    with pytest.raises(ValueError, match="selection_status disagrees"):
        scorer.score_shared_state_subspace(
            world_parameters=_parameters(),
            role_assignments=_roles(),
            feature_membership=membership,
            typed_subspace_hyperedges=_hyperedge(membership),
        )


def test_C_is_positive_overlap_and_only_sentinels_are_negative_controls() -> None:
    base = _membership()
    selected = set(
        base.loc[base["selection_status"] == "selected", "feature"].astype(str)
    )
    selected |= {"peak__M1__N1"}
    membership = _membership(selected)
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )

    assert result.metrics["n_selected_ineligible_features"] == 1
    assert result.metrics["C_state_responsive_member_recall"] == 1.0
    assert result.metrics["sentinel_null_inclusion_rate"] == pytest.approx(1 / 12)
    assert result.metrics["membership_class_success"] == 0


def test_missing_C_state_responsiveness_fails_its_role_guardrail() -> None:
    base = _membership()
    selected = set(
        base.loc[
            (base["selection_status"] == "selected")
            & ~base["feature"].str.endswith("__C"),
            "feature",
        ].astype(str)
    )
    membership = _membership(selected)
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )

    assert result.metrics["C_state_responsive_member_recall"] == 0.0
    assert result.metrics["n_selected_ineligible_features"] == 0
    assert result.metrics["membership_class_success"] == 0


def test_hyperedge_member_set_must_equal_selected_membership() -> None:
    membership = _membership()
    hyperedge = _hyperedge(membership)
    members = hyperedge.loc[0, "member_features"].split(";")
    hyperedge.loc[0, "member_features"] = ";".join(members[:-1])
    hyperedge.loc[0, "n_features"] = len(members) - 1

    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=hyperedge,
    )

    assert result.metrics["member_set_exact"] == 0
    assert result.metrics["feature_count_exact"] == 0
    assert result.metrics["hyperedge_schema_consistent"] == 0


def test_null_delta_uses_reference_wald_not_descriptive_multiview_median() -> None:
    membership = _membership()
    assert not (
        membership["reference_conditional_wald"]
        == membership["median_conditional_wald"]
    ).any()
    scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )

    malformed = membership.copy()
    malformed["conditional_wald_above_null_95pct"] = (
        malformed["median_conditional_wald"]
        - malformed["conditional_wald_null_95pct"]
    )
    with pytest.raises(ValueError, match="inconsistent"):
        scorer.score_shared_state_subspace(
            world_parameters=_parameters(),
            role_assignments=_roles(),
            feature_membership=malformed,
            typed_subspace_hyperedges=_hyperedge(malformed),
        )


def test_oof_state_depth_decomposition_is_audited_but_never_selects_members() -> None:
    membership = _membership()
    sentinel = membership["feature"].eq("peak__M0__N1")
    assert membership.loc[sentinel, "oof_marginal_state_fit"].iloc[0] == 0.90
    assert membership.loc[sentinel, "selection_status"].iloc[0] == "unresolved"

    selected = membership["selection_status"].eq("selected")
    membership.loc[selected, "oof_marginal_state_fit"] = -0.80
    membership.loc[selected, "oof_depth_only_fit"] = -0.20
    membership.loc[selected, "oof_joint_fit"] = -0.50
    membership.loc[selected, "oof_partial_state_fit"] = -0.25
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )
    assert result.metrics["subspace_world_success"] == 1

    malformed = membership.copy()
    malformed.loc[selected, "oof_partial_state_fit"] = 0.75
    with pytest.raises(ValueError, match="decomposition is inconsistent"):
        scorer.score_shared_state_subspace(
            world_parameters=_parameters(),
            role_assignments=_roles(),
            feature_membership=malformed,
            typed_subspace_hyperedges=_hyperedge(malformed),
        )


def test_local_graph_is_required_only_for_joint_certification() -> None:
    membership = _membership()
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
    )

    assert result.metrics["subspace_world_success"] == 1
    assert result.metrics["local_structure_scored"] is False
    assert result.metrics["joint_world_success"] == 0


def test_false_local_edges_fail_precision_even_when_direct_collider_recall_passes() -> None:
    membership = _membership()
    graph = _local_graph()
    graph.loc[len(graph)] = {
        "node1": "peak__M0__Q",
        "endpoint1": "CIRCLE",
        "node2": "peak__M0__S",
        "endpoint2": "CIRCLE",
        "edge_type": "within_assay_direct_candidate",
    }
    result = scorer.score_shared_state_subspace(
        world_parameters=_parameters(),
        role_assignments=_roles(),
        feature_membership=membership,
        typed_subspace_hyperedges=_hyperedge(membership),
        typed_graph=graph,
        truth_typed_edges=_local_truth(),
    )

    assert result.metrics["within_assay_direct_class_success"] == 1
    assert result.metrics["within_assay_collider_class_success"] == 1
    assert result.metrics["local_graph_n_unsupported_focal_edges"] == 1
    assert result.metrics["local_graph_primary_skeleton_precision"] == pytest.approx(2 / 3)
    assert result.metrics["local_graph_precision_class_success"] == 0
    assert result.metrics["local_structure_success"] == 0
    assert result.metrics["joint_world_success"] == 0


def test_truth_requires_six_complete_assay_role_sets() -> None:
    roles = _roles().iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly one feature"):
        scorer.build_subspace_truth(_parameters(), roles)


def test_protocol_predeclares_independent_nonadaptive_fresh_blocks() -> None:
    protocol = scorer.validation_protocol()

    # The v10 subspace producer remains frozen. The v14 formal scorer binds the
    # v4.32 set-valued loading-structure representation into the joint source
    # closure, while the validation protocol is versioned at v16.
    assert scorer.CANDIDATE_MANIFEST_SCHEMA_VERSION == (
        "shared_state_subspace_v4.frozen_discovery.v10"
    )
    assert scorer.PROVENANCE_SCHEMA_VERSION == "shared_state_subspace_v4.v10"
    assert scorer.SCORER_SCHEMA_VERSION == "shared_state_subspace_v4.scorer.v14"
    assert scorer.GLOBAL_FREEZE_SCHEMA_VERSION.endswith(".v14")
    assert scorer.BLOCK_VALIDATION_SCHEMA_VERSION.endswith(".v14")
    assert protocol["schema_version"] == (
        "shared_state_subspace_v4.validation_protocol.v16"
    )
    assert protocol["fresh_seed_blocks"]["TRAIN"] == list(range(49_001, 49_025))
    assert protocol["fresh_seed_blocks"]["DEVELOPMENT"] == list(
        range(50_001, 50_017)
    )
    assert protocol["fresh_seed_blocks"]["LOCKED"] == list(range(51_001, 51_041))
    assert protocol["estimand"]["exact_latent_axis_identity_scored"] is False
    assert protocol["barrier"][
        "TRAIN_49001_49024_is_one_nonadaptive_confirmation_block"
    ] is True
    assert protocol["history_boundary"]["seeds_33001_33024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_36001_36024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_37001_37024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_38001_38024"] == (
        "diagnostic_and_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_39001_39024"] == (
        "aborted_pretruth_infrastructure_schema_mismatch_no_scorer_opened"
    )
    assert protocol["history_boundary"]["seeds_40001_40024"] == (
        "failed_scientific_confirmation_then_algorithm_development_only"
    )
    assert protocol["history_boundary"]["seeds_41001_41024"] == (
        "passed_confirmation_then_algorithm_changed_and_development_only"
    )
    assert protocol["history_boundary"]["seeds_34001_34016"] == (
        "opened_development_failed_v43_then_posthoc_passed_v44_development_only"
    )
    assert protocol["history_boundary"]["seeds_45001_45040"] == (
        "unopened_reserved_superseded_before_v44_fresh_validation"
    )
    assert protocol["history_boundary"]["seeds_46001_46024"] == (
        "passed_confirmation_then_DEV470_informed_v45_and_development_only"
    )
    assert protocol["history_boundary"]["seeds_47001_47016"] == (
        "opened_development_failed_v44_then_posthoc_passed_v45_development_only"
    )
    assert protocol["history_boundary"]["seeds_48001_48040"] == (
        "unopened_reserved_superseded_before_v45_fresh_validation"
    )
    assert set(protocol["fresh_seed_blocks"]["TRAIN"]).isdisjoint(
        range(33_001, 33_025)
    )
    assert set(protocol["fresh_seed_blocks"]["TRAIN"]).isdisjoint(
        range(36_001, 36_025)
    )
    assert set(protocol["fresh_seed_blocks"]["TRAIN"]).isdisjoint(
        range(37_001, 37_025)
    )
    assert set(protocol["fresh_seed_blocks"]["TRAIN"]).isdisjoint(
        range(38_001, 38_025)
    )
    assert set(protocol["fresh_seed_blocks"]["TRAIN"]).isdisjoint(
        range(39_001, 39_025)
    )
    assert set(protocol["fresh_seed_blocks"]["TRAIN"]).isdisjoint(
        range(40_001, 40_025)
    )
    payload = dict(protocol)
    digest = payload.pop("protocol_payload_sha256")
    assert digest == scorer.canonical_sha256(payload)


def test_joint_source_collector_hashes_every_required_dependency() -> None:
    hashes = scorer.collect_joint_executable_source_sha256(SCRIPTS)
    assert set(hashes) == set(scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES)
    assert hashes["shared_state_subspace_v4.py"] == scorer.file_sha256(
        SCRIPTS / "shared_state_subspace_v4.py"
    )
    assert (
        hashes["shared_state_subspace_v4.py"]
        == scorer.FROZEN_DISCOVERY_SOURCE_SHA256
    )


def test_scorer_schema_contract_matches_actual_discovery_producer() -> None:
    tree = ast.parse((SCRIPTS / "shared_state_subspace_v4.py").read_text())
    produced_schema_versions: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "schema_version"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                produced_schema_versions.add(value.value)
    assert scorer.CANDIDATE_MANIFEST_SCHEMA_VERSION in produced_schema_versions
    assert scorer.PROVENANCE_SCHEMA_VERSION in produced_schema_versions


def test_frozen_local_config_and_reachable_source_closure_are_explicit() -> None:
    assert (
        scorer.FROZEN_LOCAL_CONFIG["group_projection_min_relative_margin"]
        == 0.10
    )
    newly_reachable = {
        "hierarchical_sparse_edges.py",
        "crossfit_hierarchical_collider_v4.py",
        "score_latent_loading_groups_v4.py",
        "structure_aware_collider_rescue.py",
        "observed_collider_structure.py",
        "structure_anchored_loading_groups_v4.py",
        "loading_group_state_guard_v4.py",
        "loading_group_structure_uncertainty_guard_v4.py",
        "loading_group_competing_parent_observability_guard_v4.py",
        "loading_group_structure_equivalence_v4.py",
    }
    assert newly_reachable.issubset(scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES)
    assert all((SCRIPTS / source).is_file() for source in newly_reachable)

    # Recompute the source-only local import closure from the formal runner.
    # This prevents a newly imported production module from escaping the
    # content-addressed source map, while keeping diagnostics out of it.
    reachable: set[str] = set()
    pending = ["run_shared_state_subspace_block_v4.py"]
    while pending:
        filename = pending.pop()
        if filename in reachable:
            continue
        reachable.add(filename)
        tree = ast.parse((SCRIPTS / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                modules = []
            for module in modules:
                local_name = f"{module.split('.')[0]}.py"
                if (SCRIPTS / local_name).is_file() and local_name not in reachable:
                    pending.append(local_name)
    assert reachable == set(scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES)

    excluded_nonproduction = {
        "structure_anchored_state_reporter_v4.py",
        "diagnose_formal_train_direct_failures_v4.py",
        "diagnose_loading_metric_train_v4.py",
        "diagnose_rescue_only_completion_compatibility_v4.py",
        "freeze_formal_loading_recall_candidates_train_v4.py",
        "score_formal_loading_recall_diagnostic_train_v4.py",
        "crosscheck_empirical_null_rescue_old_train_v4.py",
    }
    assert excluded_nonproduction.isdisjoint(
        scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES
    )


def _attempts(
    *,
    failures: set[int] | None = None,
    candidate_hash: str = "a" * 64,
    embedding_manifest_hash: str = "2" * 64,
    embedding_hash: str = "3" * 64,
    local_manifest_hash: str = "e" * 64,
    local_edges_hash: str = "f" * 64,
) -> list[dict[str, object]]:
    failures = failures or set()
    return [
        {
            "world_seed": seed,
            "status": "timeout" if seed in failures else "ok",
            **(
                {}
                if seed in failures
                else {
                    "candidate_manifest_sha256": candidate_hash,
                    "subspace_embedding_manifest_sha256": (
                        embedding_manifest_hash
                    ),
                    "subspace_embedding_sha256": embedding_hash,
                    "local_pretruth_manifest_sha256": local_manifest_hash,
                    "local_consensus_edges_sha256": local_edges_hash,
                }
            ),
        }
        for seed in scorer.FRESH_TRAIN_SEEDS
    ]


CONFIG = dict(scorer.FROZEN_DISCOVERY_CONFIG)
CONFIG_HASH = scorer.canonical_sha256(CONFIG)
SOURCE_HASH = scorer.FROZEN_DISCOVERY_SOURCE_SHA256
SOURCE_MAP = {
    name: "1" * 64 for name in scorer.REQUIRED_JOINT_EXECUTABLE_SOURCES
}
SOURCE_MAP["shared_state_subspace_v4.py"] = SOURCE_HASH
SOURCE_MAP["cell_state_loading_groups.py"] = "c" * 64
SOURCE_MAP["channel_aware_boss_v4.py"] = "d" * 64


def test_global_pretruth_freeze_is_exact_and_self_hashed() -> None:
    manifest = scorer.make_global_pretruth_freeze(
        split="TRAIN",
        discovery_attempts=_attempts(),
        algorithm_source_sha256=SOURCE_HASH,
        algorithm_config_sha256=CONFIG_HASH,
        joint_executable_source_sha256=SOURCE_MAP,
    )
    audit = scorer.validate_global_pretruth_freeze(manifest, split="TRAIN")

    assert set(audit["record_map"]) == set(scorer.FRESH_TRAIN_SEEDS)
    assert manifest["created_before_any_scorer_read"] is True
    assert manifest["scorer_domain_opened"] is False

    tampered = dict(manifest)
    tampered["scorer_domain_opened"] = True
    with pytest.raises(ValueError, match="self-hash"):
        scorer.validate_global_pretruth_freeze(tampered)

    with pytest.raises(ValueError, match="exactly cover"):
        scorer.make_global_pretruth_freeze(
            split="TRAIN",
            discovery_attempts=_attempts()[:-1],
            algorithm_source_sha256=SOURCE_HASH,
            algorithm_config_sha256=CONFIG_HASH,
            joint_executable_source_sha256=SOURCE_MAP,
        )


def _write_candidate_bundle(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, str]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    membership_path = candidate / "subspace_feature_membership.tsv"
    hyperedge_path = candidate / "typed_subspace_hyperedges.tsv"
    provenance_path = candidate / "subspace_provenance.json"
    manifest_path = candidate / "frozen_discovery_manifest.json"
    membership = _membership()
    # Exercise the serialization boundary: default pandas parsing rounds this
    # value onto 0.1, which can invert a frozen threshold decision.
    boundary_index = membership.index[
        membership["feature"] == ROUNDTRIP_BOUNDARY_FEATURE
    ][0]
    membership.loc[boundary_index, "empirical_null_fdr"] = float(
        np.nextafter(0.1, float("inf"))
    )
    membership.loc[boundary_index, "selection_status"] = "unresolved"
    membership.loc[boundary_index, "abstention_reason"] = "empirical_null_fdr"
    membership.loc[boundary_index, "subspace_id"] = ""
    membership.to_csv(membership_path, sep="\t", index=False)
    _hyperedge(membership).to_csv(hyperedge_path, sep="\t", index=False)
    information_boundary = {"truth_or_scorer_used": False}
    embedding_root = tmp_path / "assay_local_state_embedding_v4"
    embedding_root.mkdir()
    embedding_path = embedding_root / scorer.assay_local_embedding.EMBEDDING_NAME
    embedding_path.write_text(
        "modality\tcell\tintegration__dim0\tintegration__dim1\t"
        "integration__dim2\tintegration_uncertainty\n"
    )
    embedding_provenance_path = (
        embedding_root / scorer.assay_local_embedding.PROVENANCE_NAME
    )
    embedding_provenance_path.write_text("{}\n")
    embedding_boundary = {
        "consumed": [
            "observed_background_matrix_per_assay",
            "observed_leave_focal_out_depth_per_assay",
            "assay_key",
        ],
        "not_consumed": [
            "observed_focal_values",
            "cross_assay_cell_identity",
            "cross_assay_feature_identity",
            "scorer",
            "truth_graph",
            "simulator_roles",
            "true_latent_state",
            "external_data",
        ],
        "truth_or_scorer_used": False,
        "cell_rows_compared_across_assays": False,
        "feature_columns_compared_across_assays": False,
        "same_assay_cells_used_for_fold_procrustes": True,
    }
    embedding_manifest_path = (
        embedding_root / scorer.assay_local_embedding.MANIFEST_NAME
    )
    embedding_manifest = {
        "schema_version": (
            scorer.assay_local_embedding.FROZEN_EVIDENCE_SCHEMA_VERSION
        ),
        "frozen_before_scorer_read": True,
        "algorithm_source_sha256": SOURCE_MAP[
            "assay_local_state_embedding_v4.py"
        ],
        "config": scorer.FROZEN_SUBSPACE_EMBEDDING_CONFIG,
        "formal_state_dimension": 3,
        "rank_policy": (
            "predeclared_simulator_rank_three_with_parallel_analysis_abstention"
        ),
        "zero_padding_used": False,
        "cross_assay_alignment_used": False,
        "all_assays_parallel_rank_support_formal_dimension": True,
        "accepted_assays": sorted(
            scorer.assay_local_embedding.EXPECTED_MODALITIES
        ),
        "input_file_sha256": {
            f"background_matrix__{modality}.npz": "4" * 64
            for modality in scorer.assay_local_embedding.EXPECTED_MODALITIES
        },
        "output_file_sha256": {
            embedding_path.name: scorer.file_sha256(embedding_path),
            embedding_provenance_path.name: scorer.file_sha256(
                embedding_provenance_path
            ),
        },
        "information_boundary": embedding_boundary,
    }
    embedding_manifest_path.write_text(
        json.dumps(embedding_manifest, sort_keys=True) + "\n"
    )
    embedding_sidecar_path = embedding_root / "frozen_evidence_manifest.sha256"
    embedding_sidecar_path.write_text(
        scorer.file_sha256(embedding_manifest_path)
        + f"  {embedding_manifest_path.name}\n"
    )
    safe_embedding_evidence = {
        "layout": "assay_local_state_embedding_v4",
        "files": {
            embedding_path.name: scorer.file_sha256(embedding_path),
            embedding_manifest_path.name: scorer.file_sha256(
                embedding_manifest_path
            ),
            embedding_sidecar_path.name: scorer.file_sha256(
                embedding_sidecar_path
            ),
            embedding_provenance_path.name: scorer.file_sha256(
                embedding_provenance_path
            ),
        },
        "upstream_algorithm_source_sha256": {
            "assay_local_state_embedding_v4.py": SOURCE_MAP[
                "assay_local_state_embedding_v4.py"
            ]
        },
        "formal_state_dimension": 3,
        "rank_policy": (
            "predeclared_simulator_rank_three_with_parallel_analysis_abstention"
        ),
    }
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": scorer.PROVENANCE_SCHEMA_VERSION,
                "config": CONFIG,
                "information_boundary": information_boundary,
                "formal_state_embedding": safe_embedding_evidence,
                "descriptive_state_depth_decomposition": (
                    scorer.EXPECTED_DESCRIPTIVE_DECOMPOSITION
                ),
                "assay_state_participation": (
                    _assay_participation_provenance(membership)
                ),
                "error_control": {
                    "binary_eligibility": (
                        scorer.EXPECTED_BINARY_ELIGIBILITY_POLICY
                    ),
                    "ineligible_placeholder_policy": (
                        scorer.EXPECTED_INELIGIBLE_PLACEHOLDER_POLICY
                    ),
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    source_hash = SOURCE_HASH
    manifest = {
        "schema_version": scorer.CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "algorithm_source_sha256": source_hash,
        "dependency_source_sha256": {
            "cell_state_loading_groups.py": "c" * 64,
            "assay_local_state_embedding_v4.py": SOURCE_MAP[
                "assay_local_state_embedding_v4.py"
            ],
        },
        "config": CONFIG,
        "output_sha256": {
            membership_path.name: scorer.file_sha256(membership_path),
            hyperedge_path.name: scorer.file_sha256(hyperedge_path),
            provenance_path.name: scorer.file_sha256(provenance_path),
        },
        "frozen_before_scorer_read": True,
        "information_boundary": information_boundary,
        "safe_embedding_freeze_evidence": safe_embedding_evidence,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    local_root = tmp_path / "local"
    local_edges_path = (
        local_root / scorer.FROZEN_LOCAL_CANDIDATE_NAME / "consensus_edges.tsv"
    )
    local_edges_path.parent.mkdir(parents=True)
    _local_graph().to_csv(local_edges_path, sep="\t", index=False)
    local_manifest_path = local_root / "pretruth_consensus_manifest.json"
    local_manifest_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "all_consensuses_built_before_truth_open": True,
                "consensus_sha256": {
                    scorer.FROZEN_LOCAL_CANDIDATE_NAME: "0" * 64
                },
                "consensus_file_sha256": {
                    scorer.FROZEN_LOCAL_CANDIDATE_NAME: scorer.file_sha256(
                        local_edges_path
                    )
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return (
        membership_path,
        hyperedge_path,
        manifest_path,
        embedding_manifest_path,
        embedding_path,
        local_manifest_path,
        local_edges_path,
        source_hash,
    )


def test_frozen_candidate_is_bound_to_all_world_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        membership_path,
        hyperedge_path,
        candidate_path,
        embedding_manifest_path,
        embedding_path,
        local_manifest_path,
        local_edges_path,
        source_hash,
    ) = (
        _write_candidate_bundle(tmp_path)
    )
    candidate_hash = scorer.file_sha256(candidate_path)
    embedding_manifest_hash = scorer.file_sha256(embedding_manifest_path)
    embedding_hash = scorer.file_sha256(embedding_path)
    local_manifest_hash = scorer.file_sha256(local_manifest_path)
    local_edges_hash = scorer.file_sha256(local_edges_path)
    manifest = scorer.make_global_pretruth_freeze(
        split="TRAIN",
        discovery_attempts=_attempts(
            candidate_hash=candidate_hash,
            embedding_manifest_hash=embedding_manifest_hash,
            embedding_hash=embedding_hash,
            local_manifest_hash=local_manifest_hash,
            local_edges_hash=local_edges_hash,
        ),
        algorithm_source_sha256=source_hash,
        algorithm_config_sha256=CONFIG_HASH,
        joint_executable_source_sha256=SOURCE_MAP,
    )
    global_path = tmp_path / "global.json"
    global_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    membership, hyperedges, local_edges, audit = scorer.load_frozen_subspace_candidate(
        feature_membership_path=membership_path,
        typed_subspace_hyperedges_path=hyperedge_path,
        candidate_manifest_path=candidate_path,
        subspace_embedding_manifest_path=embedding_manifest_path,
        subspace_embedding_path=embedding_path,
        local_pretruth_manifest_path=local_manifest_path,
        local_consensus_edges_path=local_edges_path,
        global_pretruth_manifest_path=global_path,
        split="TRAIN",
        world_seed=49_001,
    )
    assert len(membership) == 37
    assert (
        membership.loc[
            membership["feature"] == ROUNDTRIP_BOUNDARY_FEATURE,
            "empirical_null_fdr",
        ].iloc[0]
        == float(np.nextafter(0.1, float("inf")))
    )
    assert len(hyperedges) == 1
    assert len(local_edges) == 2
    assert audit["candidate_bytes_loaded_before_scorer_read"] is True
    assert audit["local_graph_bytes_loaded_before_scorer_read"] is True

    extra_embedding_file = embedding_path.parent / "unbound_extra.tsv"
    extra_embedding_file.write_text("unexpected\n")
    with pytest.raises(ValueError, match="missing or extra files"):
        scorer.load_frozen_subspace_candidate(
            feature_membership_path=membership_path,
            typed_subspace_hyperedges_path=hyperedge_path,
            candidate_manifest_path=candidate_path,
            subspace_embedding_manifest_path=embedding_manifest_path,
            subspace_embedding_path=embedding_path,
            local_pretruth_manifest_path=local_manifest_path,
            local_consensus_edges_path=local_edges_path,
            global_pretruth_manifest_path=global_path,
            split="TRAIN",
            world_seed=49_001,
        )
    extra_embedding_file.unlink()

    original_membership_bytes = membership_path.read_bytes()
    original_read_csv = scorer.pd.read_csv
    parse_started = False

    def mutate_disk_when_parse_starts(source: object, *args: object, **kwargs: object):
        nonlocal parse_started
        if not parse_started:
            parse_started = True
            membership_path.write_text("mutated_during_parse\n")
        assert not isinstance(source, (str, Path))
        return original_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(scorer.pd, "read_csv", mutate_disk_when_parse_starts)
    held_membership, _, _, _ = scorer.load_frozen_subspace_candidate(
        feature_membership_path=membership_path,
        typed_subspace_hyperedges_path=hyperedge_path,
        candidate_manifest_path=candidate_path,
        subspace_embedding_manifest_path=embedding_manifest_path,
        subspace_embedding_path=embedding_path,
        local_pretruth_manifest_path=local_manifest_path,
        local_consensus_edges_path=local_edges_path,
        global_pretruth_manifest_path=global_path,
        split="TRAIN",
        world_seed=49_001,
    )
    assert len(held_membership) == 37
    assert membership_path.read_text() == "mutated_during_parse\n"
    membership_path.write_bytes(original_membership_bytes)
    monkeypatch.setattr(scorer.pd, "read_csv", original_read_csv)

    original_local_edges = local_edges_path.read_text()
    local_edges_path.write_text(original_local_edges + "\n")
    with pytest.raises(ValueError, match="Local consensus edges differ"):
        scorer.load_frozen_subspace_candidate(
            feature_membership_path=membership_path,
            typed_subspace_hyperedges_path=hyperedge_path,
            candidate_manifest_path=candidate_path,
            subspace_embedding_manifest_path=embedding_manifest_path,
            subspace_embedding_path=embedding_path,
            local_pretruth_manifest_path=local_manifest_path,
            local_consensus_edges_path=local_edges_path,
            global_pretruth_manifest_path=global_path,
            split="TRAIN",
            world_seed=49_001,
        )
    local_edges_path.write_text(original_local_edges)

    membership_path.write_text(membership_path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        scorer.load_frozen_subspace_candidate(
            feature_membership_path=membership_path,
            typed_subspace_hyperedges_path=hyperedge_path,
            candidate_manifest_path=candidate_path,
            subspace_embedding_manifest_path=embedding_manifest_path,
            subspace_embedding_path=embedding_path,
            local_pretruth_manifest_path=local_manifest_path,
            local_consensus_edges_path=local_edges_path,
            global_pretruth_manifest_path=global_path,
            split="TRAIN",
            world_seed=49_001,
        )


def _block_records(
    manifest: dict[str, object], *, failures: set[int] | None = None
) -> list[dict[str, object]]:
    failures = failures or set()
    audit = scorer.validate_global_pretruth_freeze(manifest)
    rows = []
    for seed in scorer.FRESH_TRAIN_SEEDS:
        failed = seed in failures
        rows.append(
            {
                "world_seed": seed,
                "discovery_status": "timeout" if failed else "ok",
                "scoring_status": "not_run" if failed else "ok",
                "algorithm_source_sha256": SOURCE_HASH,
                "algorithm_config_sha256": CONFIG_HASH,
                "joint_candidate_identity_sha256": audit[
                    "joint_candidate_identity_sha256"
                ],
                "global_manifest_payload_sha256": audit[
                    "manifest_payload_sha256"
                ],
                **(
                    {}
                    if failed
                    else {
                        "candidate_manifest_sha256": "a" * 64,
                        "subspace_embedding_manifest_sha256": "2" * 64,
                        "subspace_embedding_sha256": "3" * 64,
                        "local_pretruth_manifest_sha256": "e" * 64,
                        "local_consensus_edges_sha256": "f" * 64,
                        "metrics": _perfect_score(seed).metrics,
                    }
                ),
            }
        )
    return rows


def test_block_gate_keeps_failures_in_denominator() -> None:
    failures = {49_001, 49_002}
    manifest = scorer.make_global_pretruth_freeze(
        split="TRAIN",
        discovery_attempts=_attempts(failures=failures),
        algorithm_source_sha256=SOURCE_HASH,
        algorithm_config_sha256=CONFIG_HASH,
        joint_executable_source_sha256=SOURCE_MAP,
    )
    result = scorer.validate_block_results(
        split="TRAIN",
        result_records=_block_records(manifest, failures=failures),
        global_pretruth_manifest=manifest,
    )

    assert result["summaries"]["n_registered_worlds"] == 24
    assert result["summaries"]["n_scored_worlds"] == 22
    assert result["summaries"]["n_joint_successful_worlds"] == 22
    assert result["summaries"]["joint_world_success_rate"] == pytest.approx(22 / 24)
    assert result["summaries"]["sentinel_null_feature_denominator"] == 22 * 12
    assert result["summaries"]["mean_P_member_recall"] == pytest.approx(22 / 24)
    assert result["summaries"][
        "sentinel_null_inclusion_rate_one_sided_95pct_Wilson_UCB"
    ] < 0.02
    assert result["summaries"]["mean_C_state_responsive_member_recall"] == pytest.approx(
        22 / 24
    )
    assert result["gate_checks"]["every_half_fraction_design_cell"] is True
    assert result["gate_checks"]["every_DOE_factor_level"] is True
    assert result["block_pass"] is True

    failures = {49_001, 49_002, 49_003}
    failed_manifest = scorer.make_global_pretruth_freeze(
        split="TRAIN",
        discovery_attempts=_attempts(failures=failures),
        algorithm_source_sha256=SOURCE_HASH,
        algorithm_config_sha256=CONFIG_HASH,
        joint_executable_source_sha256=SOURCE_MAP,
    )
    failed = scorer.validate_block_results(
        split="TRAIN",
        result_records=_block_records(failed_manifest, failures=failures),
        global_pretruth_manifest=failed_manifest,
    )
    assert failed["summaries"]["n_joint_successful_worlds"] == 21
    assert failed["gate_checks"]["minimum_successful_worlds"] is False
    assert failed["block_pass"] is False
