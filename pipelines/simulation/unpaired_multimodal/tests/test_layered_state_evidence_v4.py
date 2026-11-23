from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import layered_state_evidence_v4 as layered  # noqa: E402
import score_shared_state_subspace_v4 as subspace_scorer  # noqa: E402


def _empirical_null_calibrator(
    *, threshold: float = 2.0
) -> layered.empirical_null_state.FrozenEmpiricalNullStateCalibrator:
    return layered.empirical_null_state.FrozenEmpiricalNullStateCalibrator(
        schema_version=layered.empirical_null_state.SCHEMA,
        conditional_weight=1.5,
        marginal_null_median=0.0,
        marginal_null_iqr=1.0,
        conditional_excess_null_median=0.0,
        conditional_excess_null_iqr=1.0,
        maximum_null_score=1.0,
        second_largest_block_maximum_null_score=0.5,
        block_maximum_extrapolation=0.5,
        selection_threshold=threshold,
        training_source_blocks=("training_a", "training_b"),
        training_candidate_rows=20,
        training_null_rows=4,
    )


def _multievidence_calibrator(
    *, threshold: float = 2.0
) -> layered.multievidence_state.FrozenMultievidenceStateCalibrator:
    return layered.multievidence_state.FrozenMultievidenceStateCalibrator(
        schema_version=layered.multievidence_state.SCHEMA,
        component_columns=layered.multievidence_state.SCORE_COMPONENT_COLUMNS,
        component_null_medians=(0.0, 0.0, 0.0, 0.0),
        component_null_iqrs=(1.0, 1.0, 1.0, 1.0),
        route_names=layered.multievidence_state.ROUTE_NAMES,
        route_weights=layered.multievidence_state.ROUTE_WEIGHTS,
        route_null_medians=(0.0, 0.0, 0.0),
        route_null_iqrs=(1.0, 1.0, 1.0),
        maximum_null_family_score=1.0,
        second_largest_block_maximum_null_family_score=0.5,
        block_maximum_extrapolation=0.5,
        selection_threshold=threshold,
        training_source_blocks=("training_a", "training_b"),
        training_candidate_rows=20,
        training_null_rows=4,
    )


def _topology_consensus_calibrator(
    *, certified: bool = True
) -> layered.topology_consensus_state.FrozenTopologyConsensusStateCalibrator:
    return (
        layered.topology_consensus_state.FrozenTopologyConsensusStateCalibrator(
            schema_version=layered.topology_consensus_state.SCHEMA,
            minimum_direct_candidate_degree=1,
            state_specific_gate=(
                layered.topology_consensus_state.STATE_SPECIFIC_GATE
            ),
            null_control_method=(
                layered.topology_consensus_state.NULL_CONTROL_METHOD
            ),
            training_source_blocks=("training_a", "training_b"),
            training_observable_rows=20,
            training_null_rows=4,
            training_rule_eligible_null_rows=0 if certified else 1,
            sentinel_certified=certified,
        )
    )


def _membership() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "feature": "peak_A",
                "modality": "chrom_A",
                "formal_test_eligible": True,
                "feature_global_maxT_p": 0.02,
                "selection_status": "selected",
            },
            {
                "feature": "peak_B",
                "modality": "chrom_A",
                "formal_test_eligible": True,
                "feature_global_maxT_p": 0.95,
                "selection_status": "unresolved",
            },
            {
                "feature": "peak_C",
                "modality": "chrom_A",
                "formal_test_eligible": True,
                "feature_global_maxT_p": 0.04,
                "selection_status": "selected",
            },
            {
                "feature": "peak_Z",
                "modality": "chrom_A",
                "formal_test_eligible": False,
                "feature_global_maxT_p": 1.00,
                "selection_status": "unresolved",
            },
            {
                "feature": "peak_N",
                "modality": "chrom_B",
                "formal_test_eligible": True,
                "feature_global_maxT_p": 0.60,
                "selection_status": "selected",
            },
            {
                "feature": "gene_G",
                "modality": "rna",
                "formal_test_eligible": True,
                "feature_global_maxT_p": 0.01,
                "selection_status": "selected",
            },
        ]
    )
    frame["n_views"] = 6
    per_feature = {
        "peak_A": (0.001, 0.40, 1.0),
        "peak_B": (0.800, 0.02, 0.0),
        "peak_C": (0.002, 0.35, 1.0),
        "peak_Z": (1.000, 0.00, 0.0),
        "peak_N": (0.900, 0.01, 0.0),
        "gene_G": (0.001, 0.45, 1.0),
    }
    frame["empirical_null_p"] = frame["feature"].map(
        lambda feature: per_feature[str(feature)][0]
    )
    frame["reference_conditional_wald"] = frame["feature"].map(
        lambda feature: per_feature[str(feature)][1]
    )
    frame["conditional_wald_view_support"] = frame["feature"].map(
        lambda feature: per_feature[str(feature)][2]
    )

    # Build the repeated annotations from the same observable-only routine used
    # by the layered producer.  This keeps the fixture bound to both omnibus
    # routes, their branch-specific 6/6 gates, and Holm across assays.
    participation = layered.state_subspace._select_assay_state_participation(
        frame,
        layered.state_subspace.SharedStateSubspaceConfig(),
    ).set_index("modality")
    modalities = frame["modality"].astype(str)
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
        frame[output_column] = modalities.map(participation[participation_column])
    frame["assay_participation_selected"] = modalities.map(
        participation["selection_status"].eq("selected")
    )
    selected_reporter = modalities.map(participation["selected_reporter"])
    frame["assay_participation_reporter"] = (
        frame["assay_participation_selected"]
        & frame["feature"].astype(str).eq(selected_reporter)
    )

    # The layered v4 freeze carries the complete upstream membership table,
    # not a projection of just the columns consumed by this builder.
    boolean_defaults = {
        "binary_event_gate": True,
        "feature_global_maxT_confirmation_gate": False,
        "assay_global_maxT_admission_gate": False,
        "direction_stability_diagnostic": False,
        "feature_maxT_unconditional_confirmation_gate": False,
        "feature_direction_stability_confirmation_gate": False,
        "redundancy_direction_stability_gate": False,
        "redundancy_fallback_selected": False,
        "redundancy_demoted": False,
        "direct_layer_overlap": False,
        "direct_layer_annotation_available": True,
    }
    string_defaults = {
        "channel": "",
        "formal_test_abstention_reason": "",
        "pre_redundancy_selection_status": "unresolved",
        "pre_redundancy_abstention_reason": "",
        "abstention_reason": "",
        "subspace_id": "",
    }
    integer_defaults = {
        "n_cells": 1000,
        "n_events": 100,
        "n_nonevents": 900,
        "empirical_null_family_size": 1,
    }
    for column in subspace_scorer.MEMBERSHIP_COLUMNS:
        if column in frame:
            continue
        if column in boolean_defaults:
            frame[column] = boolean_defaults[column]
        elif column in string_defaults:
            frame[column] = string_defaults[column]
        elif column in integer_defaults:
            frame[column] = integer_defaults[column]
        else:
            frame[column] = 0.0
    frame.loc[frame["modality"].eq("rna"), ["n_events", "n_nonevents"]] = -1
    frame["channel"] = frame["modality"].map(
        lambda modality: "RNA" if modality == "rna" else str(modality)[6:]
    )
    frame["pre_redundancy_selection_status"] = frame["selection_status"]
    frame.loc[
        frame["feature"].eq("peak_A"),
        ["oof_marginal_state_fit", "oof_partial_state_fit"],
    ] = [0.1, 0.1]
    frame.loc[
        frame["feature"].eq("peak_B"),
        [
            "pre_redundancy_selection_status",
            "redundancy_demoted",
            "feature_global_maxT_p",
        ],
    ] = ["selected", True, 0.005]
    frame["subspace_id"] = frame["selection_status"].map(
        lambda status: subspace_scorer.SUBSPACE_ID if status == "selected" else ""
    )
    return frame.loc[:, list(subspace_scorer.MEMBERSHIP_COLUMNS)]


def _subspace() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subspace_id": "source_subspace",
                "edge_type": "cross_assay_shared_state_subspace",
                "selection_status": "selected",
                "empirical_multimodality_p": 0.01,
            }
        ]
    )


def _consensus() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "node1": "peak_A",
                "endpoint1": "TAIL",
                "node2": "peak_C",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "collider_motif_center": "peak_C",
                "orientation_source": "unique_positive_residual_wedge",
            },
            {
                "node1": "peak_A",
                "endpoint1": "CIRCLE",
                "node2": "peak_B",
                "endpoint2": "CIRCLE",
                "edge_type": "within_assay_shared_state",
                "collider_motif_center": "",
                "orientation_source": "group_projection",
            },
        ]
    )


def _groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.DataFrame(
        [
            {
                "loading_group": "group_1",
                "modality": "chrom_A",
                "members": "peak_B;peak_C",
                "group_confidence": 0.97,
                "minimum_member_centroid_cosine": 0.97,
                "selection_status": "selected",
            },
            {
                "loading_group": "singleton_1",
                "modality": "chrom_B",
                "members": "peak_N",
                "group_confidence": 0.91,
                "minimum_member_centroid_cosine": 1.0,
                "selection_status": "singleton_candidate",
            },
        ]
    )
    membership = pd.DataFrame(
        [
            {
                "loading_group": "group_1",
                "feature": "peak_B",
                "modality": "chrom_A",
            },
            {
                "loading_group": "group_1",
                "feature": "peak_C",
                "modality": "chrom_A",
            },
            {
                "loading_group": "singleton_1",
                "feature": "peak_N",
                "modality": "chrom_B",
            },
        ]
    )
    return groups, membership


def _structure_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_key": "peak_B|||peak_C",
                "opposite_parent_pair_key": "peak_A|||peak_C",
                "modality": "chrom_A",
                "parent_candidate": "peak_B",
                "noncollider_sibling_candidate": "peak_C",
                "collider_center": "peak_Z",
                "other_parent_candidate": "peak_A",
                "selected": True,
                "selection_stage": "formal_baseline",
            }
        ]
    )


def test_layers_are_kept_distinct_and_maxT_confirmation_is_conservative() -> None:
    groups, group_membership = _groups()
    result = layered.build_layered_state_evidence(
        _membership(),
        _subspace(),
        _consensus(),
        groups,
        group_membership,
        _structure_evidence(),
    )
    evidence = result.feature_evidence.set_index("feature")

    assert evidence.loc[
        ["peak_A", "peak_C", "gene_G"],
        "confirmed_depth_orthogonal_state_response",
    ].all()
    assert not evidence.loc[
        "peak_N", "confirmed_depth_orthogonal_state_response"
    ]
    assert evidence.loc["peak_N", "state_response_status"] == (
        "conditional_state_response_candidate_not_featurewise_confirmed"
    )
    assert evidence.loc["peak_B", "state_response_status"] == "unresolved"
    assert evidence.loc["peak_B", "state_equivalence_only_member"]
    assert evidence.loc["peak_B", "state_supported_member"]
    assert evidence.loc["peak_B", "state_membership_tier"] == (
        "within_assay_state_equivalence_member"
    )
    assert not evidence.loc["peak_B", "strict_state_member"]
    assert evidence.loc["peak_A", "strict_state_member"]
    assert not evidence.loc["peak_N", "state_supported_member"]

    assert evidence.loc["peak_B", "assay_local_loading_group_support"]
    assert evidence.loc["peak_C", "assay_local_loading_group_support"]
    assert evidence.loc["peak_N", "assay_local_singleton_candidate"]
    assert evidence.loc["peak_A", "within_assay_direct_role"] == "parent_candidate"
    assert evidence.loc["peak_C", "within_assay_direct_role"] == (
        "collider_center_candidate"
    )
    assert not evidence["layer_union_is_state_membership"].any()
    assert not evidence["cross_assay_pair_claimed"].any()
    assert not evidence["direct_cross_assay_effect_claimed"].any()

    subspace = result.typed_hyperedges.loc[
        result.typed_hyperedges["edge_type"].eq(
            "cross_assay_shared_state_subspace"
        )
    ].iloc[0]
    assert set(subspace["members"].split(";")) == {"peak_A", "peak_C", "gene_G"}
    assert set(subspace["modalities"].split(";")) == {"chrom_A", "rna"}
    assert subspace["hypothesis_only"]
    assert not subspace["direct_causal_claimed"]
    local = result.typed_hyperedges.loc[
        result.typed_hyperedges["edge_type"].eq(
            layered.loading_group_structure_equivalence.STATE_EQUIVALENCE_TYPE
        )
    ].iloc[0]
    assert set(local["members"].split(";")) == {
        "peak_A",
        "peak_B",
        "peak_C",
        "peak_Z",
    }
    assert set(local["nominal_members"].split(";")) == {"peak_B", "peak_C"}
    assert local["candidate_pair_count"] == 4
    assert json.loads(local["candidate_pairs_json"]) == [
        ["peak_A", "peak_C"],
        ["peak_A", "peak_Z"],
        ["peak_B", "peak_C"],
        ["peak_B", "peak_Z"],
    ]
    assert local["structure_equivalence_selected"]
    assert not local["exact_loading_group_selected"]
    assert not local["exact_pair_identified"]
    assert not local["candidate_pairs_may_be_expanded_as_edges"]
    assert local["n_modalities"] == 1
    assert local["loading_group_v430_retained"]
    assert local["complete_two_parent_comparison_available"]
    assert local["selected_parent_candidate"] == "peak_B"
    assert local["competing_parent_candidate"] == "peak_A"
    assert local["competing_parent_fixed_rule_eligible"]
    assert local["loading_group_guard_status"] == (
        layered.loading_group_competitor_guard.STATE_RETAINED_STATUS
    )
    equivalence = result.typed_hyperedges.loc[
        result.typed_hyperedges["edge_type"].eq(
            "within_assay_shared_state_equivalence_class"
        )
        & result.typed_hyperedges["modalities"].eq("chrom_A")
    ].iloc[0]
    assert set(equivalence["members"].split(";")) == {
        "peak_A",
        "peak_B",
        "peak_C",
    }
    assert set(equivalence["strict_members"].split(";")) == {
        "peak_A",
        "peak_C",
    }
    assert equivalence["equivalence_only_members"] == "peak_B"
    assert not equivalence["pairwise_expansion_permitted"]
    assert not equivalence["direct_causal_claimed"]

    direct = result.within_assay_direct_candidates.iloc[0]
    assert direct["layer"] == "within_assay_direct_candidate"
    assert direct["source_assay"] == "chrom_A"
    assert not direct["cross_assay_claim"]
    competitor_guard = result.provenance[
        "loading_group_competing_parent_observability_guard"
    ]
    assert competitor_guard["active"]
    assert competitor_guard["schema_version"] == (
        layered.loading_group_competitor_guard.SCHEMA
    )
    assert not competitor_guard["application_requires_truth"]
    structure_equivalence = result.provenance[
        "loading_group_structure_equivalence"
    ]
    assert structure_equivalence["active"]
    assert structure_equivalence["schema_version"] == (
        layered.loading_group_structure_equivalence.SCHEMA
    )
    assert not structure_equivalence[
        "candidate_pairs_may_be_expanded_as_edges"
    ]


def test_empirical_null_calibrator_adds_only_a_typed_membership_layer() -> None:
    membership = _membership()
    membership.loc[
        membership["feature"].eq("peak_N"),
        "oof_marginal_state_fit",
    ] = 3.0
    groups, group_membership = _groups()
    result = layered.build_layered_state_evidence(
        membership,
        _subspace(),
        _consensus(),
        groups,
        group_membership,
        _structure_evidence(),
        empirical_null_calibrator=_empirical_null_calibrator(),
    )
    evidence = result.feature_evidence.set_index("feature")

    assert evidence["empirical_null_calibration_active"].all()
    assert evidence.loc["peak_N", "empirical_null_state_member"]
    assert evidence.loc["peak_N", "state_calibrated_supported_member"]
    assert not evidence.loc["peak_N", "state_supported_member"]
    assert evidence.loc["peak_N", "state_calibrated_membership_tier"] == (
        layered.empirical_null_state.TIER
    )
    assert "within_assay_empirical_null_state_member" in evidence.loc[
        "peak_N", "support_layers"
    ]
    assert len(result.within_assay_direct_candidates) == 1
    assert not result.feature_evidence["cross_assay_pair_claimed"].any()
    assert not result.feature_evidence["direct_cross_assay_effect_claimed"].any()
    assert result.provenance["counts"]["n_empirical_null_state_members"] == 1
    assert not any(
        result.typed_hyperedges["edge_type"].astype(str).str.contains(
            "empirical_null"
        )
    )


def test_multievidence_calibrator_adds_only_a_typed_membership_layer() -> None:
    membership = _membership()
    target = membership["feature"].eq("peak_N")
    membership.loc[
        target,
        [
            "oof_marginal_state_fit",
            "oof_partial_state_fit",
            "oof_joint_fit",
            "conditional_wald_above_null_95pct",
            "median_conditional_score_norm",
        ],
    ] = [3.0, 2.0, 3.0, 3.0, 3.0]
    groups, group_membership = _groups()
    result = layered.build_layered_state_evidence(
        membership,
        _subspace(),
        _consensus(),
        groups,
        group_membership,
        _structure_evidence(),
        multievidence_calibrator=_multievidence_calibrator(),
    )
    evidence = result.feature_evidence.set_index("feature")

    assert evidence["multievidence_calibration_active"].all()
    assert evidence.loc[
        "peak_N", "multievidence_empirical_null_state_member"
    ]
    assert evidence.loc["peak_N", "state_v426_supported_member"]
    assert not evidence.loc["peak_N", "state_calibrated_supported_member"]
    assert evidence.loc["peak_N", "state_v426_membership_tier"] == (
        layered.multievidence_state.TIER
    )
    assert "within_assay_multievidence_empirical_null_state_member" in (
        evidence.loc["peak_N", "support_layers"]
    )
    assert len(result.within_assay_direct_candidates) == 1
    assert not result.feature_evidence["cross_assay_pair_claimed"].any()
    assert not result.feature_evidence["direct_cross_assay_effect_claimed"].any()
    assert result.provenance["counts"]["n_multievidence_state_members"] == 1
    assert not any(
        result.typed_hyperedges["edge_type"].astype(str).str.contains(
            "multievidence"
        )
    )


def test_cross_assay_edge_cannot_enter_direct_layer() -> None:
    groups, group_membership = _groups()
    consensus = _consensus()
    consensus.loc[0, "node2"] = "gene_G"
    with pytest.raises(ValueError, match="different modalities"):
        layered.build_layered_state_evidence(
            _membership(),
            _subspace(),
            consensus,
            groups,
            group_membership,
            _structure_evidence(),
        )


def test_topology_consensus_requires_state_gate_and_direct_support() -> None:
    features = pd.DataFrame(
        {
            "feature": ["weak_joint", "direct_only", "state_only", "upstream"],
            "modality": ["chrom_A"] * 4,
            "formal_test_eligible": [True] * 4,
            "state_v426_supported_member": [False, False, False, True],
            "state_v426_membership_tier": [
                "unresolved",
                "unresolved",
                "unresolved",
                "strict_state_member",
            ],
            "oof_marginal_state_fit": [0.01, -0.01, 0.01, 0.02],
            "oof_partial_state_fit": [0.02, -0.01, 0.02, 0.02],
            "within_assay_direct_candidate_degree": [1, 1, 0, 1],
        }
    )
    annotated = layered._annotate_topology_consensus_state_layer(
        features,
        _topology_consensus_calibrator(),
    ).set_index("feature")
    assert annotated.loc["weak_joint", "topology_consensus_state_member"]
    assert annotated.loc["weak_joint", "state_v427_supported_member"]
    assert not annotated.loc["direct_only", "state_v427_supported_member"]
    assert not annotated.loc["state_only", "state_v427_supported_member"]
    assert annotated.loc["upstream", "state_v427_supported_member"]
    assert not annotated.loc["upstream", "topology_consensus_state_member"]
    assert annotated.loc["weak_joint", "state_v427_membership_tier"] == (
        layered.topology_consensus_state.TIER
    )


def test_v430_downgrades_loading_group_that_overlaps_complete_collider() -> None:
    groups, group_membership = _groups()
    consensus = pd.concat(
        [
            _consensus(),
            pd.DataFrame(
                [
                    {
                        "node1": "peak_B",
                        "endpoint1": "TAIL",
                        "node2": "peak_C",
                        "endpoint2": "ARROW",
                        "edge_type": "within_assay_direct_candidate",
                        "collider_motif_center": "peak_C",
                        "orientation_source": "unique_positive_residual_wedge",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = layered.build_layered_state_evidence(
        _membership(),
        _subspace(),
        consensus,
        groups,
        group_membership,
        _structure_evidence(),
    )
    row = result.typed_hyperedges.loc[
        result.typed_hyperedges["relation_id"].eq("assay_local__group_1")
    ].iloc[0]

    assert row["selection_status"] == "unresolved"
    assert row["loading_group_v429_retained"]
    assert row["loading_group_guard_route"] == "unresolved"
    assert row["complete_direct_collider_overlap"]
    assert row["direct_collider_child_overlap"] == "peak_C"
    assert row["loading_group_guard_status"] == (
        layered.loading_group_competitor_guard.UPSTREAM_REJECTED_STATUS
    )
    assert row["loading_group_v430_guard_status"] == (
        layered.loading_group_structure_guard.DIRECT_OVERLAP_STATUS
    )
    assert len(result.within_assay_direct_candidates) == 2


def test_run_freezes_content_addressed_outputs_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups, group_membership = _groups()
    inputs = {
        "membership": _membership(),
        "subspace": _subspace(),
        "consensus": _consensus(),
        "groups": groups,
        "group_membership": group_membership,
        "structure": _structure_evidence(),
    }
    boundary_value = float(np.nextafter(0.1, float("inf")))
    inputs["membership"].loc[
        inputs["membership"].index[0], "reference_conditional_wald"
    ] = boundary_value
    paths: dict[str, Path] = {}
    for name, frame in inputs.items():
        path = tmp_path / f"{name}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[name] = path
    original_membership_bytes = paths["membership"].read_bytes()
    original_snapshot = layered._read_regular_bytes_once
    raced = False

    def mutate_after_snapshot(path: Path, label: str) -> bytes:
        nonlocal raced
        payload = original_snapshot(path, label)
        if label == "membership" and not raced:
            raced = True
            path.write_text("mutated_after_snapshot\n")
        return payload

    monkeypatch.setattr(layered, "_read_regular_bytes_once", mutate_after_snapshot)
    out = tmp_path / "candidate"
    summary = layered.run_layered_state_evidence(
        paths["membership"],
        paths["subspace"],
        paths["consensus"],
        paths["groups"],
        paths["group_membership"],
        paths["structure"],
        out,
    )

    manifest_path = out / "frozen_layered_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["candidate_frozen"]
    assert not manifest["truth_or_scorer_consumed_by_algorithm"]
    assert summary["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert manifest["input_sha256"]["membership"] == hashlib.sha256(
        original_membership_bytes
    ).hexdigest()
    assert set(manifest["dependency_source_sha256"]) == {
        "empirical_null_state_calibration_v4.py",
        "loading_group_state_guard_v4.py",
        "loading_group_structure_uncertainty_guard_v4.py",
        "loading_group_competing_parent_observability_guard_v4.py",
        "loading_group_structure_equivalence_v4.py",
        "multievidence_empirical_null_state_calibration_v4.py",
        "shared_state_subspace_v4.py",
        "state_equivalence_membership_v4.py",
        "topology_consensus_state_calibration_v4.py",
    }
    assert paths["membership"].read_text() == "mutated_after_snapshot\n"
    assert set(
        pd.read_csv(out / "layered_feature_evidence.tsv", sep="\t")["feature"]
    ) == set(_membership()["feature"])
    round_tripped = pd.read_csv(
        out / "layered_feature_evidence.tsv",
        sep="\t",
        float_precision="round_trip",
    )
    assert round_tripped.loc[
        round_tripped["feature"].eq(inputs["membership"].iloc[0]["feature"]),
        "reference_conditional_wald",
    ].iloc[0] == boundary_value
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        layered.run_layered_state_evidence(
            paths["membership"],
            paths["subspace"],
            paths["consensus"],
            paths["groups"],
            paths["group_membership"],
            paths["structure"],
            out,
        )


def test_scorer_domain_paths_are_rejected(tmp_path: Path) -> None:
    scorer_dir = tmp_path / "scorer"
    scorer_dir.mkdir()
    path = scorer_dir / "membership.tsv"
    _membership().to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="forbidden scorer domain"):
        layered._safe_observable_path(path, "membership")


def test_empirical_null_artifact_is_self_hashed_and_fail_closed() -> None:
    payload = {
        **_empirical_null_calibrator().to_payload(),
        "artifact_role": "future_application_only",
        "training_label_input_sha256": "1" * 64,
        "phase_a_manifest_sha256": "2" * 64,
        "training_rows_are_not_validation_rows": True,
        "application_requires_truth": False,
        "new_worlds_or_datasets_created": False,
    }
    payload["artifact_payload_sha256"] = layered._canonical_sha256(payload)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    calibrator, metadata = layered.load_empirical_null_state_calibrator(encoded)

    assert calibrator.selection_threshold == 2.0
    assert metadata["active"]
    assert not metadata["application_requires_truth"]

    payload["selection_threshold"] = 1.5
    with pytest.raises(ValueError, match="payload hash changed"):
        layered.load_empirical_null_state_calibrator(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        )


def test_multievidence_artifact_is_self_hashed_and_fail_closed() -> None:
    payload = {
        **_multievidence_calibrator().to_payload(),
        "artifact_role": "future_application_only",
        "training_label_input_sha256": "1" * 64,
        "phase_a_manifest_sha256": "2" * 64,
        "fixed_route_family_selected_without_application_labels": True,
        "retrospective_design_informed_by_cached_development_results": True,
        "training_rows_are_not_validation_rows": True,
        "application_requires_truth": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    payload["artifact_payload_sha256"] = layered._canonical_sha256(payload)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    calibrator, metadata = layered.load_multievidence_state_calibrator(encoded)

    assert calibrator.selection_threshold == 2.0
    assert metadata["active"]
    assert metadata["fixed_route_family"] == list(
        layered.multievidence_state.ROUTE_NAMES
    )
    assert not metadata["application_requires_truth"]

    payload["selection_threshold"] = 1.5
    with pytest.raises(ValueError, match="payload hash changed"):
        layered.load_multievidence_state_calibrator(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        )


def test_topology_consensus_artifact_is_self_hashed_and_fail_closed() -> None:
    payload = {
        **_topology_consensus_calibrator().to_payload(),
        "artifact_role": "future_application_only",
        "training_label_input_sha256": "1" * 64,
        "phase_a_manifest_sha256": "2" * 64,
        "fixed_state_topology_conjunction_selected_without_application_labels": True,
        "retrospective_design_informed_by_cached_development_results": True,
        "training_rows_are_not_validation_rows": True,
        "application_requires_truth": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    payload["artifact_payload_sha256"] = layered._canonical_sha256(payload)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    calibrator, metadata = (
        layered.load_topology_consensus_state_calibrator(encoded)
    )
    assert calibrator.sentinel_certified
    assert metadata["active"]
    assert metadata["fixed_state_topology_conjunction"]
    assert not metadata["application_requires_truth"]

    payload["minimum_direct_candidate_degree"] = 2
    with pytest.raises(ValueError, match="payload hash changed"):
        layered.load_topology_consensus_state_calibrator(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        )
