from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cell_state_loading_groups as loading_groups  # noqa: E402
import shared_state_subspace_v4 as subspace  # noqa: E402


def _observed_row(
    feature: str,
    modality: str,
    fit: float,
    stability: float,
) -> dict[str, object]:
    return {
        "feature": feature,
        "modality": modality,
        "channel": "RNA" if modality == "rna" else modality.removeprefix("chrom_"),
        "n_views": 6,
        "n_cells": 1000,
        "n_events": -1 if modality == "rna" else 150,
        "n_nonevents": -1 if modality == "rna" else 850,
        "binary_event_gate": True,
        "formal_test_eligible": True,
        "formal_test_abstention_reason": "",
        "oof_marginal_state_fit": 0.25,
        "oof_depth_only_fit": 0.10,
        "oof_joint_fit": 0.30,
        "oof_partial_state_fit": 2.0 / 9.0,
        "median_conditional_wald": fit,
        "lower_quartile_conditional_wald": fit * 0.95,
        "median_conditional_score_norm": max(fit / 4.0, 0.002),
        "minimum_conditional_score_norm": max(fit / 5.0, 0.001),
        "median_conditional_score_direction_stability": stability,
        "minimum_conditional_score_direction_stability": max(stability - 0.05, 0.0),
        **{
            f"conditional_wald__model_seed{view}": fit * (0.96 + 0.01 * view)
            for view in range(6)
        },
    }


def test_v3_defaults_encode_exact_hierarchical_membership_contract() -> None:
    config = subspace.SharedStateSubspaceConfig()

    assert config.null_repetitions == 999
    assert config.observed_model_views == 6
    assert config.minimum_conditional_wald_views == 4
    assert config.minimum_median_score_direction_stability == 0.80
    assert config.within_assay_strong_core_maxT_p == 0.01
    assert config.within_assay_strong_core_minimum_members == 4
    assert config.minimum_binary_events == 20
    assert config.minimum_binary_nonevents == 20
    assert config.assay_state_participation_order == 2
    assert config.maximum_assay_state_participation_error_probability == 0.10
    assert config.minimum_assay_state_participation_views == 6
    np.testing.assert_allclose(
        subspace._holm(np.asarray([0.01, 0.04, 0.03, 1.0])),
        np.asarray([0.04, 0.09, 0.09, 1.0]),
    )


def test_underfloor_chromatin_abstains_before_every_fit_and_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1701)
    n_cells = 120
    assay = loading_groups.AssayData(
        modality="chrom_X",
        cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
        coordinates=rng.normal(size=(n_cells, 3)),
        coordinate_uncertainty=np.zeros(n_cells),
        log_depth=rng.normal(size=n_cells),
        focal=pd.DataFrame(
            {
                "constant_zero": np.zeros(n_cells, dtype=int),
                "nineteen_events": np.r_[
                    np.ones(19, dtype=int), np.zeros(n_cells - 19, dtype=int)
                ],
                "nineteen_nonevents": np.r_[
                    np.zeros(19, dtype=int), np.ones(n_cells - 19, dtype=int)
                ],
            }
        ),
    )
    inputs = _loading_inputs(assay)
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("An under-floor feature entered a fitted model")

    monkeypatch.setattr(subspace, "_fit_assay_score_nuisance", forbidden)
    monkeypatch.setattr(subspace, "_orthogonal_score_feature", forbidden)
    monkeypatch.setattr(loading_groups, "_estimate_feature", forbidden)

    observed = subspace._fit_observed_evidence(inputs, config)
    reference, null = subspace.estimate_conditional_wild_null(inputs, config=config)
    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership = result.feature_membership.set_index("feature")

    assert reference == {}
    assert null == {}
    assert not membership["binary_event_gate"].any()
    assert not membership["formal_test_eligible"].any()
    assert set(membership["formal_test_abstention_reason"]) == {
        "insufficient_binary_events"
    }
    assert set(membership["selection_status"]) == {"unresolved"}
    assert set(membership["abstention_reason"]) == {"insufficient_binary_events"}
    assert (membership["empirical_null_family_size"] == 0).all()
    assert (membership["empirical_null_p"] == 1.0).all()
    assert (membership["empirical_null_fdr"] == 1.0).all()
    assert (membership["feature_global_maxT_p"] == 1.0).all()
    assert (membership["assay_global_maxT_p"] == 1.0).all()
    placeholder_columns = [
        *subspace._DESCRIPTIVE_DECOMPOSITION_FIELDS,
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "median_conditional_score_norm",
        "minimum_conditional_score_norm",
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
        *[f"conditional_wald__model_seed{view}" for view in range(6)],
        "reference_conditional_wald",
        "conditional_wald_null_median",
        "conditional_wald_null_95pct",
        "conditional_wald_above_null_95pct",
        "conditional_wald_view_support",
    ]
    assert (membership[placeholder_columns] == 0.0).all().all()
    edge = result.typed_subspace_hyperedges.iloc[0]
    assert edge["selection_status"] == "unresolved"
    assert edge["n_modalities"] == 0
    assert edge["n_features"] == 0
    assert edge["empirical_multimodality_p"] == 1.0


def test_single_class_fixed_training_fold_fails_closed_before_feature_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(1702)
    n_cells = 100
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    response = np.zeros(n_cells, dtype=int)
    assay = loading_groups.AssayData(
        modality="chrom_X",
        cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
        coordinates=rng.normal(size=(n_cells, 3)),
        coordinate_uncertainty=np.zeros(n_cells),
        log_depth=rng.normal(size=n_cells),
        focal=pd.DataFrame({"fold_bad": response}),
    )
    first_seed = subspace._stable_seed(
        config.random_seed, "orthogonal_score_view_folds", "0", "chrom_X"
    )
    first_test = subspace._response_independent_assay_folds(
        assay, requested=config.folds, seed=first_seed
    )[0][1]
    assert len(first_test) == 20
    assay.focal.loc[first_test, "fold_bad"] = 1

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("A fold-ineligible feature entered a fitted model")

    monkeypatch.setattr(subspace, "_fit_assay_score_nuisance", forbidden)
    monkeypatch.setattr(subspace, "_orthogonal_score_feature", forbidden)
    monkeypatch.setattr(loading_groups, "_estimate_feature", forbidden)
    inputs = _loading_inputs(assay)
    observed = subspace._fit_observed_evidence(inputs, config)
    reference, null = subspace.estimate_conditional_wild_null(inputs, config=config)
    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    row = result.feature_membership.iloc[0]

    assert row["n_events"] == 20
    assert row["n_nonevents"] == 80
    assert bool(row["binary_event_gate"])
    assert not bool(row["formal_test_eligible"])
    assert row["formal_test_abstention_reason"] == "single_class_training_fold"
    assert row["selection_status"] == "unresolved"
    assert row["abstention_reason"] == "single_class_training_fold"
    assert row["empirical_null_family_size"] == 0
    assert result.typed_subspace_hyperedges.iloc[0]["selection_status"] == (
        "unresolved"
    )


def test_prefit_abstention_cannot_admit_an_assay_or_hyperedge() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    sparse = _observed_row("sparse_chrom", "chrom_X", 1000.0, 1.0)
    sparse.update(
        {
            "n_events": 19,
            "n_nonevents": 981,
            "binary_event_gate": False,
            "formal_test_eligible": False,
            "formal_test_abstention_reason": "insufficient_binary_events",
        }
    )
    observed = pd.DataFrame(
        [sparse, _observed_row("gene", "rna", 0.20, 0.99)]
    )
    reference = {"gene": 0.20}
    null = {"gene": np.linspace(0.0, 0.04, 199)}

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership = result.feature_membership.set_index("feature")
    edge = result.typed_subspace_hyperedges.iloc[0]

    assert membership.loc["sparse_chrom", "empirical_null_family_size"] == 0
    assert membership.loc["sparse_chrom", "empirical_null_p"] == 1.0
    assert membership.loc["sparse_chrom", "feature_global_maxT_p"] == 1.0
    assert membership.loc["sparse_chrom", "assay_global_maxT_p"] == 1.0
    assert not membership.loc["sparse_chrom", "assay_global_maxT_admission_gate"]
    assert membership.loc["sparse_chrom", "selection_status"] == "unresolved"
    assert set(
        membership.index[membership["selection_status"] == "selected"]
    ) == {"gene"}
    assert edge["selection_status"] == "unresolved"
    assert edge["modalities"] == "rna"
    assert edge["abstention_reason"] == "insufficient_modalities"


def test_empirical_null_selects_members_without_pair_or_direct_claims() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("peak_a", "chrom_A", 0.20, 0.98),
            _observed_row("peak_b", "chrom_B", 0.18, 0.96),
            _observed_row("gene", "rna", 0.22, 0.99),
            _observed_row("collider_child", "chrom_C", 0.21, 0.99),
            _observed_row("sentinel_a", "chrom_A", 0.005, 0.35),
            _observed_row("sentinel_b", "chrom_B", -0.002, 0.20),
        ]
    )
    null = {
        feature: (
            np.linspace(-0.03, 0.04, 199)
            if not feature.startswith("sentinel")
            else np.linspace(-0.02, 0.03, 199)
        )
        for feature in observed["feature"]
    }

    result = subspace.select_shared_subspace_membership(
        observed,
        dict(zip(observed["feature"], observed["median_conditional_wald"], strict=True)),
        null,
        direct_layer_features=["collider_child"],
        config=config,
    )

    membership = result.feature_membership.set_index("feature")
    participation = result.assay_state_participation.set_index("modality")
    assert set(
        membership.index[membership["selection_status"] == "selected"]
    ) == {"peak_a", "peak_b", "collider_child", "gene"}
    assert membership.loc["collider_child", "direct_layer_overlap"]
    assert membership.loc["collider_child", "selection_status"] == "selected"
    assert membership.loc["sentinel_a", "selection_status"] == "unresolved"
    assert participation.loc[
        "rna", "assay_participation_omnibus_route_count"
    ] == 1
    assert participation.loc[
        "rna", "assay_participation_winning_route"
    ] == "single_feature"
    assert participation.loc[
        "rna", "assay_participation_selection_route"
    ] == "single_feature"
    edge = result.typed_subspace_hyperedges.iloc[0]
    assert edge["edge_type"] == "cross_assay_shared_state_subspace"
    assert edge["selection_status"] == "selected"
    assert set(str(edge["modalities"]).split(";")) == {
        "chrom_A", "chrom_B", "chrom_C", "rna"
    }
    assert bool(edge["conditional_on_common_state_model"])
    assert edge["causal_interpretation"] == "hypothesis_only"
    assert not bool(edge["direction_claimed"])
    assert not bool(edge["exact_axis_pairing_claimed"])
    assert not bool(edge["direct_causal_claimed"])
    assert edge["error_control_method"] == (
        "synchronized_conditional_score_wild_feature_maxT_confirmation_"
        "borderline_direction_stability_"
        "within_assay_redundancy_confirmation_"
        "within_assay_strong_maxT_core_weak_tail_confirmation_"
        "assay_maxT_admission_"
        "within_assay_BH_conservative_order_2_partial_conjunction"
    )
    assert result.provenance["schema_version"] == "shared_state_subspace_v4.v10"
    assert result.provenance["descriptive_state_depth_decomposition"] == {
        "method": "cross_fitted_state_plus_orthogonal_depth_predictive_fit",
        "fields": [
            "oof_marginal_state_fit",
            "oof_depth_only_fit",
            "oof_joint_fit",
            "oof_partial_state_fit",
        ],
        "affects_selection": False,
        "shared_state_depth_allocation": "state",
        "calibrated_primary_estimand": "depth_orthogonal_state_response",
    }


def test_assay_state_participation_is_first_class_but_cannot_mutate_membership() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("target_a", "chrom_A", 0.20, 0.99),
            _observed_row("target_b", "chrom_A", 0.18, 0.99),
            _observed_row("null_barrier_a", "chrom_B", 0.01, 0.20),
            _observed_row("null_barrier_b", "chrom_B", 0.01, 0.20),
        ]
    )
    reference = dict(
        zip(observed["feature"], observed["median_conditional_wald"], strict=True)
    )
    null = {
        "target_a": np.linspace(0.0, 0.04, 199),
        "target_b": np.linspace(0.0, 0.04, 199),
        # These null features create a world maxT barrier.  The target assay
        # therefore has replicated local evidence but no strict member.
        "null_barrier_a": np.full(199, 0.25),
        "null_barrier_b": np.full(199, 0.25),
    }

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership = result.feature_membership.set_index("feature")
    participation = result.assay_state_participation.set_index("modality")
    edge = result.typed_subspace_hyperedges.iloc[0]

    assert set(membership["selection_status"]) == {"unresolved"}
    assert edge["modalities"] == ""
    assert edge["n_features"] == 0
    assert edge["selection_status"] == "unresolved"
    assert participation.loc["chrom_A", "selection_status"] == "selected"
    assert participation.loc[
        "chrom_A", "partial_conjunction_empirical_p"
    ] == pytest.approx(0.005)
    assert participation.loc["chrom_A", "assay_participation_holm_p"] == (
        pytest.approx(0.01)
    )
    assert participation.loc["chrom_A", "selected_reporter"] == "target_a"
    assert participation.loc[
        "chrom_A", "single_feature_order_statistic_p"
    ] == pytest.approx(0.005)
    assert participation.loc[
        "chrom_A", "single_feature_bonferroni_p"
    ] == pytest.approx(0.01)
    assert participation.loc[
        "chrom_A", "single_feature_witness_view_count"
    ] == 6
    assert bool(
        participation.loc[
            "chrom_A", "single_feature_multiview_replication_gate"
        ]
    )
    assert bool(
        participation.loc[
            "chrom_A", "partial_conjunction_multiview_replication_gate"
        ]
    )
    assert participation.loc[
        "chrom_A", "assay_participation_omnibus_route_count"
    ] == 2
    assert participation.loc[
        "chrom_A", "assay_participation_omnibus_p"
    ] == pytest.approx(0.01)
    assert participation.loc[
        "chrom_A", "assay_participation_omnibus_holm_p"
    ] == pytest.approx(0.02)
    assert participation.loc[
        "chrom_A", "assay_participation_winning_route"
    ] == "partial_conjunction"
    assert participation.loc[
        "chrom_A", "assay_participation_selection_route"
    ] == "partial_conjunction"
    assert participation.loc[
        "chrom_A", "partial_conjunction_witness_features"
    ] == "target_a;target_b"
    assert participation.loc["chrom_A", "contributing_feature_count"] == 2
    evidence = json.loads(
        participation.loc["chrom_A", "all_eligible_feature_evidence"]
    )
    assert [record["feature"] for record in evidence] == ["target_a", "target_b"]
    assert all(record["partial_conjunction_witness"] for record in evidence)
    assert not bool(participation.loc["chrom_A", "feature_membership_claimed"])
    assert not bool(participation.loc["chrom_A", "direct_causal_claimed"])
    assert membership.loc[
        ["target_a", "target_b"], "assay_participation_selected"
    ].all()
    assert not membership.loc[
        ["null_barrier_a", "null_barrier_b"],
        "assay_participation_selected",
    ].any()
    assert membership.loc["target_a", "assay_participation_reporter"]
    assert not membership.loc["target_b", "assay_participation_reporter"]
    assert set(
        membership.loc[
            ["target_a", "target_b"], "assay_participation_order"
        ]
    ) == {2}
    assert set(
        membership.loc[
            ["target_a", "target_b"], "assay_participation_family_size"
        ]
    ) == {2}
    assert set(
        membership.loc[
            ["target_a", "target_b"],
            "assay_participation_contributing_feature_count",
        ]
    ) == {2}
    assert set(
        membership.loc[
            ["target_a", "target_b"],
            "assay_participation_minimum_witness_view_count",
        ]
    ) == {6}
    assert membership.loc[
        ["target_a", "target_b"],
        "assay_participation_multiview_replication_gate",
    ].all()
    assert np.allclose(
        membership.loc[
            ["target_a", "target_b"],
            "assay_participation_partial_conjunction_p",
        ],
        0.005,
    )
    assert np.allclose(
        membership.loc[
            ["target_a", "target_b"], "assay_participation_holm_p"
        ],
        0.01,
    )
    assert np.allclose(
        membership.loc[
            ["target_a", "target_b"], "assay_participation_omnibus_holm_p"
        ],
        0.02,
    )
    assert set(
        membership.loc[
            ["target_a", "target_b"], "assay_participation_selection_route"
        ]
    ) == {"partial_conjunction"}
    provenance = result.provenance["assay_state_participation"]
    assert provenance["selected_modalities"] == ["chrom_A"]
    assert provenance["affects_feature_membership"] is False
    assert provenance["affects_existing_typed_subspace_hyperedge"] is False
    assert provenance["across_assay_error_control"] == (
        "holm_strong_FWER_without_cross_assay_dependence_assumption"
    )


def test_assay_omnibus_selects_one_strong_replicated_signal_but_not_pure_null() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=999)
    observed_rows = [_observed_row("strong", "chrom_one", 0.20, 0.99)]
    observed_rows.extend(
        _observed_row(f"sentinel_{index}", "chrom_one", 0.01, 0.20)
        for index in range(5)
    )
    observed_rows.extend(
        _observed_row(f"pure_null_{index}", "chrom_null", 0.01, 0.20)
        for index in range(6)
    )
    observed = pd.DataFrame(observed_rows)
    reference = dict(
        zip(observed["feature"], observed["median_conditional_wald"], strict=True)
    )
    null = {
        feature: (
            np.linspace(0.0, 0.04, 999)
            if feature == "strong"
            else np.full(999, 0.08)
        )
        for feature in observed["feature"]
    }

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    participation = result.assay_state_participation.set_index("modality")
    membership = result.feature_membership.set_index("feature")

    assert participation.loc["chrom_one", "selection_status"] == "selected"
    assert participation.loc["chrom_null", "selection_status"] == "unresolved"
    assert participation.loc[
        "chrom_one", "partial_conjunction_empirical_p"
    ] == pytest.approx(1.0)
    assert participation.loc[
        "chrom_null", "partial_conjunction_empirical_p"
    ] == pytest.approx(1.0)
    assert participation.loc["chrom_one", "leading_reporter"] == "strong"
    assert participation.loc["chrom_one", "selected_reporter"] == "strong"
    assert participation.loc[
        "chrom_one", "single_feature_bonferroni_p"
    ] == pytest.approx(0.006)
    assert participation.loc[
        "chrom_one", "assay_participation_omnibus_p"
    ] == pytest.approx(0.012)
    assert participation.loc[
        "chrom_one", "assay_participation_omnibus_holm_p"
    ] == pytest.approx(0.024)
    assert participation.loc[
        "chrom_one", "assay_participation_winning_route"
    ] == "single_feature"
    assert participation.loc[
        "chrom_one", "assay_participation_selection_route"
    ] == "single_feature"
    assert bool(
        participation.loc[
            "chrom_one", "single_feature_multiview_replication_gate"
        ]
    )
    assert not bool(
        participation.loc[
            "chrom_one", "partial_conjunction_multiview_replication_gate"
        ]
    )
    assert membership.loc["strong", "selection_status"] == "selected"
    assert membership.loc[
        membership["modality"].eq("chrom_one"),
        "assay_participation_selected",
    ].all()
    assert membership.loc["strong", "assay_participation_reporter"]
    assert not membership.loc[
        "strong", "assay_participation_multiview_replication_gate"
    ]
    assert membership.loc[
        "strong", "assay_participation_minimum_witness_view_count"
    ] == 0
    assert set(
        membership.loc[
            membership.index.str.startswith(("sentinel_", "pure_null_")),
            "selection_status",
        ]
    ) == {"unresolved"}
    assert result.provenance["assay_state_participation"][
        "n_selected_assays"
    ] == 1


def test_conditional_wald_is_invariant_to_orthogonal_basis_rotation() -> None:
    rng = np.random.default_rng(7)
    n_cells = 420
    state = rng.normal(size=(n_cells, 4))
    depth = 0.5 * state[:, 0] + rng.normal(size=n_cells)
    response = 1.2 * state[:, 1] - 0.7 * state[:, 3] + 0.3 * depth
    response += rng.normal(scale=0.3, size=n_cells)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assay = loading_groups.AssayData(
        modality="rna",
        cell_ids=np.asarray([f"cell_{index}" for index in range(n_cells)]),
        coordinates=state,
        coordinate_uncertainty=np.zeros(n_cells),
        log_depth=depth,
        focal=pd.DataFrame({"expr__gene": response}),
    )
    rotated = loading_groups.AssayData(
        modality=assay.modality,
        cell_ids=assay.cell_ids,
        coordinates=state @ rotation,
        coordinate_uncertainty=assay.coordinate_uncertainty,
        log_depth=assay.log_depth,
        focal=assay.focal,
    )
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    nuisance = subspace._fit_assay_score_nuisance(assay, config, 11)
    rotated_nuisance = subspace._fit_assay_score_nuisance(rotated, config, 11)
    original_fit = subspace._orthogonal_score_feature(
        "rna", "expr__gene", assay, nuisance, config, 13
    )
    rotated_fit = subspace._orthogonal_score_feature(
        "rna", "expr__gene", rotated, rotated_nuisance, config, 13
    )

    assert original_fit.statistic == pytest.approx(rotated_fit.statistic, abs=1e-9)
    assert original_fit.score_norm == pytest.approx(rotated_fit.score_norm, abs=1e-9)


def test_loader_axis_scaling_is_undone_before_rotation_invariant_score() -> None:
    rng = np.random.default_rng(19)
    n_cells = 500
    raw = rng.normal(size=(n_cells, 3)) @ np.diag([2.5, 0.7, 1.4])
    depth = 0.4 * raw[:, 0] + rng.normal(size=n_cells)
    response = raw[:, 1] - 0.6 * raw[:, 2] + rng.normal(scale=0.4, size=n_cells)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    def loader_scaled(coordinates: np.ndarray) -> loading_groups.LoadingInputs:
        center = coordinates.mean(axis=0)
        scale = coordinates.std(axis=0, ddof=0)
        assay = loading_groups.AssayData(
            modality="rna",
            cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
            coordinates=(coordinates - center[None, :]) / scale[None, :],
            coordinate_uncertainty=np.zeros(n_cells),
            log_depth=depth,
            focal=pd.DataFrame({"expr__gene": response}),
        )
        return loading_groups.LoadingInputs(
            focal_root=Path("/tmp/focal"),
            embedding_path=Path("/tmp/analysis_embeddings.tsv"),
            coordinate_columns=tuple(f"integration__dim{i}" for i in range(3)),
            coordinate_center=center,
            coordinate_scale=scale,
            assays={"rna": assay},
            input_hashes={},
        )

    original, _ = subspace._rotation_invariant_local_inputs(loader_scaled(raw))
    rotated, _ = subspace._rotation_invariant_local_inputs(
        loader_scaled(raw @ rotation)
    )
    np.testing.assert_allclose(
        original.assays["rna"].coordinates @ original.assays["rna"].coordinates.T,
        rotated.assays["rna"].coordinates @ rotated.assays["rna"].coordinates.T,
        atol=1e-9,
    )
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    nuisance = subspace._fit_assay_score_nuisance(
        original.assays["rna"], config, 23
    )
    rotated_nuisance = subspace._fit_assay_score_nuisance(
        rotated.assays["rna"], config, 23
    )
    fit = subspace._orthogonal_score_feature(
        "rna", "expr__gene", original.assays["rna"], nuisance, config, 29
    )
    rotated_fit = subspace._orthogonal_score_feature(
        "rna", "expr__gene", rotated.assays["rna"], rotated_nuisance, config, 29
    )
    assert fit.statistic == pytest.approx(rotated_fit.statistic, abs=1e-8)


def test_descriptive_state_depth_decomposition_is_rotation_invariant_and_algebraic() -> None:
    rng = np.random.default_rng(29)
    n_cells = 420
    state = rng.normal(size=(n_cells, 3))
    depth = 0.7 * state[:, 0] + rng.normal(size=n_cells)
    response = 0.8 * state[:, 1] + 0.6 * depth + rng.normal(scale=0.4, size=n_cells)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    def inputs(coordinates: np.ndarray) -> loading_groups.LoadingInputs:
        assay = loading_groups.AssayData(
            modality="rna",
            cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
            coordinates=coordinates,
            coordinate_uncertainty=np.zeros(n_cells),
            log_depth=depth,
            focal=pd.DataFrame({"expr__gene": response}),
        )
        raw = loading_groups.LoadingInputs(
            focal_root=Path("/tmp/focal"),
            embedding_path=Path("/tmp/analysis_embeddings.tsv"),
            coordinate_columns=(
                "integration__dim0",
                "integration__dim1",
                "integration__dim2",
            ),
            coordinate_center=np.zeros(3),
            coordinate_scale=np.ones(3),
            assays={"rna": assay},
            input_hashes={},
        )
        return subspace._rotation_invariant_local_inputs(raw)[0]

    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    original = subspace._fit_descriptive_state_depth_decomposition(
        inputs(state), config
    ).set_index("feature")
    rotated = subspace._fit_descriptive_state_depth_decomposition(
        inputs(state @ rotation), config
    ).set_index("feature")

    fields = list(subspace._DESCRIPTIVE_DECOMPOSITION_FIELDS)
    np.testing.assert_allclose(original[fields], rotated[fields], atol=1e-9)
    assert np.isfinite(original[fields].to_numpy()).all()
    assert (original[fields].to_numpy() <= 1.0).all()
    row = original.loc["expr__gene"]
    expected_partial = (
        row["oof_joint_fit"] - row["oof_depth_only_fit"]
    ) / (1.0 - row["oof_depth_only_fit"])
    assert row["oof_partial_state_fit"] == pytest.approx(expected_partial)


def test_family_bh_has_power_for_one_strong_feature_out_of_six() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row(
                f"peak_{index}",
                "chrom_A",
                0.20 if index == 0 else 0.0,
                0.99 if index == 0 else 0.20,
            )
            for index in range(6)
        ]
    )
    null = {
        f"peak_{index}": np.linspace(-0.03, 0.04, 199)
        for index in range(6)
    }

    result = subspace.select_shared_subspace_membership(
        observed,
        dict(zip(observed["feature"], observed["median_conditional_wald"], strict=True)),
        null,
        config=config,
    )

    membership = result.feature_membership.set_index("feature")
    assert membership.loc["peak_0", "empirical_null_p"] == pytest.approx(0.005)
    assert membership.loc["peak_0", "empirical_null_fdr"] == pytest.approx(0.03)
    assert membership.loc["peak_0", "selection_status"] == "selected"
    assert set(membership["empirical_null_family_size"]) == {6}


def test_empirical_p_uses_fixed_reference_not_correlated_view_median() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [_observed_row(f"peak_{index}", "chrom_A", 0.20, 0.99) for index in range(6)]
    )
    # The six-view median looks strong, but the fixed-fold reference statistic
    # is null-like.  It must therefore fail the multiplier-bootstrap test.
    reference = {f"peak_{index}": 0.0 for index in range(6)}
    null = {
        f"peak_{index}": np.linspace(-0.03, 0.04, 199)
        for index in range(6)
    }

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )

    membership = result.feature_membership.set_index("feature")
    assert (membership["reference_conditional_wald"] == 0.0).all()
    assert (membership["empirical_null_p"] > 0.4).all()
    assert set(membership["selection_status"]) == {"unresolved"}


def test_exact_four_of_six_views_selects_but_three_does_not() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("peak_a", "chrom_A", 0.20, 0.10),
            _observed_row("peak_b", "chrom_B", 0.20, 0.99),
        ]
    )
    view_columns = [f"conditional_wald__model_seed{index}" for index in range(6)]
    observed.loc[observed["feature"] == "peak_a", view_columns] = [
        0.20,
        0.20,
        0.20,
        0.20,
        0.00,
        0.00,
    ]
    null = {feature: np.linspace(0.0, 0.04, 199) for feature in observed["feature"]}
    reference = {feature: 0.20 for feature in observed["feature"]}

    four = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")
    observed.loc[observed["feature"] == "peak_a", view_columns[3]] = 0.00
    three = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")

    assert four.loc["peak_a", "conditional_wald_view_support"] == pytest.approx(4 / 6)
    assert four.loc["peak_a", "selection_status"] == "selected"
    assert not four.loc["peak_a", "direction_stability_diagnostic"]
    assert three.loc["peak_a", "conditional_wald_view_support"] == pytest.approx(3 / 6)
    assert three.loc["peak_a", "selection_status"] == "unresolved"
    assert three.loc["peak_a", "abstention_reason"] == "conditional_wald_view_support"


def test_redundancy_confirmation_demotes_unstable_member_when_stable_peer_exists() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("stable", "chrom_A", 0.30, 0.90),
            _observed_row("unstable", "chrom_A", 0.32, 0.60),
            _observed_row("gene", "rna", 0.35, 0.90),
        ]
    )
    null = {
        feature: np.linspace(0.0, 0.04, 199)
        for feature in observed["feature"]
    }
    reference = {feature: 0.30 for feature in observed["feature"]}

    membership = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")

    assert membership.loc["unstable", "pre_redundancy_selection_status"] == (
        "selected"
    )
    assert membership.loc["stable", "selection_status"] == "selected"
    assert membership.loc["unstable", "selection_status"] == "unresolved"
    assert membership.loc["unstable", "redundancy_demoted"]
    assert membership.loc["unstable", "abstention_reason"] == (
        "within_assay_redundancy_direction_instability"
    )
    assert not membership.loc["stable", "redundancy_fallback_selected"]


def test_redundancy_confirmation_retains_one_preliminary_member_as_fallback() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("weaker", "chrom_A", 0.20, 0.60),
            _observed_row("stronger", "chrom_A", 0.30, 0.70),
            _observed_row("gene", "rna", 0.35, 0.90),
        ]
    )
    null = {
        feature: np.linspace(0.0, 0.04, 199)
        for feature in observed["feature"]
    }
    reference = {"weaker": 0.20, "stronger": 0.30, "gene": 0.35}

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership = result.feature_membership.set_index("feature")

    assert membership.loc["stronger", "selection_status"] == "selected"
    assert membership.loc["stronger", "redundancy_fallback_selected"]
    assert membership.loc["weaker", "selection_status"] == "unresolved"
    assert membership.loc["weaker", "redundancy_demoted"]
    assert set(
        result.typed_subspace_hyperedges.iloc[0]["modalities"].split(";")
    ) == {"chrom_A", "rna"}


def test_strong_core_confirmation_demotes_only_the_weak_maxT_tail() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    chromatic = [
        ("core_1", 1.10),
        ("core_2", 1.10),
        ("core_3", 1.10),
        # With 199 null draws ending at 1.0, this is exactly p = 0.01
        # and therefore belongs to the predeclared inclusive strong core.
        ("core_boundary", 1.00),
        ("weak_tail", 0.992),
    ]
    observed = pd.DataFrame(
        [
            *[
                _observed_row(feature, "chrom_A", reference, 0.95)
                for feature, reference in chromatic
            ],
            _observed_row("gene", "rna", 1.10, 0.95),
        ]
    )
    reference = {**dict(chromatic), "gene": 1.10}
    null = {
        feature: np.linspace(0.0, 1.0, 199)
        for feature in observed["feature"]
    }

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership = result.feature_membership.set_index("feature")

    assert membership.loc["core_boundary", "feature_global_maxT_p"] == pytest.approx(
        0.01
    )
    assert membership.loc["core_boundary", "within_assay_strong_core_member"]
    assert membership.loc["weak_tail", "feature_global_maxT_p"] == pytest.approx(
        0.015
    )
    assert membership.loc["weak_tail", "pre_redundancy_selection_status"] == (
        "selected"
    )
    assert not membership.loc["weak_tail", "redundancy_demoted"]
    assert membership.loc["weak_tail", "within_assay_strong_core_count"] == 4
    assert membership.loc["weak_tail", "within_assay_weak_maxT_tail_demoted"]
    assert membership.loc["weak_tail", "selection_status"] == "unresolved"
    assert membership.loc["weak_tail", "abstention_reason"] == (
        "within_assay_strong_maxT_core_weak_tail"
    )
    assert set(
        membership.loc[
            membership["modality"].eq("chrom_A")
            & membership["selection_status"].eq("selected")
        ].index
    ) == {"core_1", "core_2", "core_3", "core_boundary"}
    guard = result.provenance["error_control"][
        "strong_core_weak_tail_confirmation"
    ]
    assert guard["n_demoted"] == 1
    assert guard["may_add_members"] is False
    assert guard["may_remove_last_member_from_an_assay"] is False


def test_strong_core_confirmation_requires_four_strong_members() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    chromatic = [
        ("core_1", 1.10),
        ("core_2", 1.10),
        ("core_boundary", 1.00),
        ("weak_tail", 0.992),
    ]
    observed = pd.DataFrame(
        [
            *[
                _observed_row(feature, "chrom_A", reference, 0.95)
                for feature, reference in chromatic
            ],
            _observed_row("gene", "rna", 1.10, 0.95),
        ]
    )
    reference = {**dict(chromatic), "gene": 1.10}
    null = {
        feature: np.linspace(0.0, 1.0, 199)
        for feature in observed["feature"]
    }

    membership = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")

    assert membership.loc["weak_tail", "within_assay_strong_core_count"] == 3
    assert not membership.loc["weak_tail", "within_assay_weak_maxT_tail_demoted"]
    assert membership.loc["weak_tail", "selection_status"] == "selected"


def test_borderline_feature_maxT_requires_stable_score_direction() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("stable", "chrom_A", 1.10, 0.90),
            _observed_row("unstable", "chrom_B", 1.10, 0.60),
        ]
    )
    reference = {"stable": 0.93, "unstable": 0.93}
    null = {
        "stable": np.linspace(0.0, 1.0, 199),
        "unstable": np.linspace(0.0, 1.0, 199),
    }

    membership = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")

    assert 0.05 < membership.loc["stable", "feature_global_maxT_p"] <= 0.10
    assert membership.loc["stable", "selection_status"] == "selected"
    assert membership.loc["stable", "direction_stability_diagnostic"]
    assert not membership.loc[
        "unstable", "feature_maxT_unconditional_confirmation_gate"
    ]
    assert not membership.loc[
        "unstable", "feature_direction_stability_confirmation_gate"
    ]
    assert membership.loc["unstable", "selection_status"] == "unresolved"
    assert membership.loc["unstable", "abstention_reason"] == (
        "feature_direction_stability_confirmation"
    )


def test_assay_maxT_is_admission_gate_but_descriptive_fits_are_not() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("candidate", "chrom_A", 0.06, 0.99),
            _observed_row("null_barrier", "chrom_B", 0.00, 0.99),
        ]
    )
    null = {
        "candidate": np.linspace(0.0, 0.01, 199),
        "null_barrier": np.full(199, 0.08),
    }
    reference = {"candidate": 0.06, "null_barrier": 0.00}

    baseline = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")
    observed.loc[:, list(subspace._DESCRIPTIVE_DECOMPOSITION_FIELDS)] = -0.75
    perturbed = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    ).feature_membership.set_index("feature")

    assert baseline.loc["candidate", "empirical_null_fdr"] <= 0.10
    assert not baseline.loc["candidate", "assay_global_maxT_admission_gate"]
    assert baseline.loc["candidate", "selection_status"] == "unresolved"
    assert baseline.loc["candidate", "abstention_reason"] == (
        "assay_global_maxT_admission;feature_global_maxT_confirmation"
    )
    assert baseline["selection_status"].equals(perturbed["selection_status"])
    assert baseline["abstention_reason"].equals(perturbed["abstention_reason"])


def test_observable_collider_centers_are_post_selection_overlap(tmp_path: Path) -> None:
    path = tmp_path / "motif_provenance.json"
    path.write_text(
        """{
          "selected_collider_centers": ["peak_b", "peak_a"],
          "truth_or_scorer_used": false
        }\n"""
    )

    features, digest = subspace.load_observable_direct_layer_overlap(path)

    assert features == ("peak_a", "peak_b")
    assert len(digest) == 64


def test_formal_hard_embedding_provenance_is_bound_to_focal_inputs(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    embedding = prepared / "analysis_embeddings.tsv"
    embedding.write_text("modality\tcell\tintegration__dim0\n")
    focal_hashes = {
        "observed_focal__chrom_X.tsv": "a" * 64,
        "observed_focal__rna.tsv": "b" * 64,
    }
    (prepared / "alignment_provenance.json").write_text(
        """{
          "schema_version": "prepare_ca_boss_v4_inputs.alignment.v1",
          "method": "recomputed_independent_two_feature_fold_consensus",
          "input_policy": {
            "scorer_or_truth_read": false,
            "external_data_read": false,
            "supplied_analysis_embeddings_read": false,
            "supplied_alignment_provenance_read": false,
            "supplied_metacells_or_membership_read": false
          },
          "information_boundary": {
            "cell_rows_compared_across_assays": false,
            "feature_columns_compared_across_assays": false,
            "focal_values_used_for_alignment": false,
            "feature_folds_are_within_assay": true
          },
          "input_file_sha256": {
            "observed_focal__chrom_X.tsv": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "observed_focal__rna.tsv": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
          }
        }\n"""
    )
    (prepared / "preparation_manifest.json").write_text(
        """{
          "schema_version": "prepare_ca_boss_v4_inputs.run.v1",
          "alignment_method": "recomputed_two_feature_fold_background_consensus",
          "files_written": ["analysis_embeddings.tsv", "alignment_provenance.json"],
          "n_modalities": 2,
          "derange": false
        }\n"""
    )
    assay = loading_groups.AssayData(
        modality="rna",
        cell_ids=np.asarray(["c0", "c1"]),
        coordinates=np.asarray([[0.0], [1.0]]),
        coordinate_uncertainty=np.zeros(2),
        log_depth=np.asarray([0.0, 1.0]),
        focal=pd.DataFrame({"expr__gene": [0.0, 1.0]}),
    )
    inputs = loading_groups.LoadingInputs(
        focal_root=tmp_path,
        embedding_path=embedding,
        coordinate_columns=("integration__dim0",),
        coordinate_center=np.zeros(1),
        coordinate_scale=np.ones(1),
        assays={"rna": assay, "chrom_X": assay},
        input_hashes={
            embedding.name: subspace._sha256(embedding),
            **focal_hashes,
        },
    )

    evidence = subspace._validate_safe_embedding_freeze(inputs)

    assert evidence["layout"] == "prepare_ca_boss_v4_inputs"
    assert set(evidence["files"]) == {
        "analysis_embeddings.tsv",
        "alignment_provenance.json",
        "preparation_manifest.json",
    }


def _loading_inputs(*assays: loading_groups.AssayData) -> loading_groups.LoadingInputs:
    dimension = assays[0].coordinates.shape[1]
    return loading_groups.LoadingInputs(
        focal_root=Path("/tmp/focal"),
        embedding_path=Path("/tmp/analysis_embeddings.tsv"),
        coordinate_columns=tuple(f"integration__dim{i}" for i in range(dimension)),
        coordinate_center=np.zeros(dimension),
        coordinate_scale=np.ones(dimension),
        assays={assay.modality: assay for assay in assays},
        input_hashes={},
    )


def _empirical_p(reference: float, null: np.ndarray) -> float:
    return (1.0 + float(np.sum(null >= reference))) / (len(null) + 1.0)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_conditional_wild_null_controls_pure_depth_response(nonlinear: bool) -> None:
    rejections = 0
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        n_cells = 500
        depth = rng.normal(size=n_cells)
        if nonlinear:
            coordinates = np.column_stack(
                [
                    np.sin(1.4 * depth) + rng.normal(scale=0.30, size=n_cells),
                    depth**2 + rng.normal(scale=0.35, size=n_cells),
                    np.cos(0.8 * depth) + rng.normal(scale=0.30, size=n_cells),
                ]
            )
            response = np.sin(1.7 * depth) + 0.4 * depth**2
        else:
            coordinates = np.column_stack(
                [
                    depth + rng.normal(scale=0.25, size=n_cells),
                    -0.7 * depth + rng.normal(scale=0.35, size=n_cells),
                    rng.normal(size=n_cells),
                ]
            )
            response = depth
        response = response + rng.normal(scale=0.25, size=n_cells)
        assay = loading_groups.AssayData(
            modality="rna",
            cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
            coordinates=coordinates,
            coordinate_uncertainty=np.zeros(n_cells),
            log_depth=depth,
            focal=pd.DataFrame({"expr__gene": response}),
        )
        config = subspace.SharedStateSubspaceConfig(
            null_repetitions=199, random_seed=seed
        )
        reference, null = subspace.estimate_conditional_wild_null(
            _loading_inputs(assay), config=config
        )
        rejections += _empirical_p(reference["expr__gene"], null["expr__gene"]) <= 0.10

    assert rejections <= 1


def test_conditional_wild_score_has_depth_orthogonal_power() -> None:
    rng = np.random.default_rng(81)
    n_cells = 700
    depth = rng.normal(size=n_cells)
    innovation = rng.normal(size=n_cells)
    coordinates = np.column_stack(
        [depth + innovation, -0.4 * depth + rng.normal(size=n_cells), rng.normal(size=n_cells)]
    )
    response = depth + 0.9 * innovation + rng.normal(scale=0.4, size=n_cells)
    assay = loading_groups.AssayData(
        modality="rna",
        cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
        coordinates=coordinates,
        coordinate_uncertainty=np.zeros(n_cells),
        log_depth=depth,
        focal=pd.DataFrame({"expr__gene": response}),
    )
    reference, null = subspace.estimate_conditional_wild_null(
        _loading_inputs(assay),
        config=subspace.SharedStateSubspaceConfig(null_repetitions=199),
    )

    assert _empirical_p(reference["expr__gene"], null["expr__gene"]) == pytest.approx(
        0.005
    )


def test_sparse_binary_conditional_wild_null_and_power() -> None:
    null_p_values = []
    for seed in range(5):
        rng = np.random.default_rng(900 + seed)
        n_cells = 1500
        depth = rng.normal(size=n_cells)
        coordinates = np.column_stack(
            [
                np.sin(1.2 * depth) + rng.normal(scale=0.4, size=n_cells),
                0.7 * depth + rng.normal(scale=0.5, size=n_cells),
                rng.normal(size=n_cells),
            ]
        )
        response = rng.binomial(1, expit(-2.8 + 1.1 * depth))
        assay = loading_groups.AssayData(
            modality="chrom_X",
            cell_ids=np.asarray([f"c{index}" for index in range(n_cells)]),
            coordinates=coordinates,
            coordinate_uncertainty=np.zeros(n_cells),
            log_depth=depth,
            focal=pd.DataFrame({"chr1:1-2__X": response}),
        )
        reference, null = subspace.estimate_conditional_wild_null(
            _loading_inputs(assay),
            config=subspace.SharedStateSubspaceConfig(
                null_repetitions=199, random_seed=seed
            ),
        )
        null_p_values.append(
            _empirical_p(reference["chr1:1-2__X"], null["chr1:1-2__X"])
        )
    assert sum(value <= 0.10 for value in null_p_values) <= 1

    rng = np.random.default_rng(42)
    n_cells = 1500
    depth = rng.normal(size=n_cells)
    innovation = rng.normal(size=n_cells)
    coordinates = np.column_stack(
        [
            depth + innovation,
            -0.4 * depth + rng.normal(size=n_cells),
            rng.normal(size=n_cells),
        ]
    )
    response = rng.binomial(1, expit(-2.8 + 1.1 * depth + 0.5 * innovation))
    powered = loading_groups.AssayData(
        modality="chrom_X",
        cell_ids=np.asarray([f"p{index}" for index in range(n_cells)]),
        coordinates=coordinates,
        coordinate_uncertainty=np.zeros(n_cells),
        log_depth=depth,
        focal=pd.DataFrame({"chr1:1-2__X": response}),
    )
    reference, null = subspace.estimate_conditional_wild_null(
        _loading_inputs(powered),
        config=subspace.SharedStateSubspaceConfig(null_repetitions=199),
    )
    assert _empirical_p(reference["chr1:1-2__X"], null["chr1:1-2__X"]) == (
        pytest.approx(0.005)
    )


def test_feature_maxT_confirms_membership_inside_an_admitted_assay() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("strong_a", "chrom_A", 0.20, 0.99),
            _observed_row("sentinel_a", "chrom_A", 0.06, 0.99),
            _observed_row("strong_b", "chrom_B", 0.18, 0.99),
            _observed_row("null_barrier", "chrom_C", 0.00, 0.20),
        ]
    )
    null = {
        "strong_a": np.linspace(0.0, 0.04, 199),
        "sentinel_a": np.linspace(0.0, 0.01, 199),
        "strong_b": np.linspace(0.0, 0.04, 199),
        "null_barrier": np.full(199, 0.08),
    }
    reference = dict(
        zip(observed["feature"], observed["median_conditional_wald"], strict=True)
    )

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    membership = result.feature_membership.set_index("feature")

    assert membership.loc["sentinel_a", "empirical_null_fdr"] <= 0.10
    assert membership.loc["sentinel_a", "feature_global_maxT_p"] > 0.10
    assert not membership.loc[
        "sentinel_a", "feature_global_maxT_confirmation_gate"
    ]
    assert membership.loc["sentinel_a", "assay_global_maxT_admission_gate"]
    assert membership.loc["sentinel_a", "selection_status"] == "unresolved"
    assert membership.loc["sentinel_a", "abstention_reason"] == (
        "feature_global_maxT_confirmation"
    )
    assert set(
        membership.index[membership["selection_status"] == "selected"]
    ) == {"strong_a", "strong_b"}


def test_one_active_assay_cannot_select_multi_assay_hypothesis() -> None:
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = pd.DataFrame(
        [
            _observed_row("strong_a", "chrom_A", 0.20, 0.99),
            _observed_row("null_b", "chrom_B", 0.01, 0.99),
        ]
    )
    null = {
        "strong_a": np.linspace(0.0, 0.04, 199),
        "null_b": np.full(199, 0.08),
    }
    reference = dict(
        zip(observed["feature"], observed["median_conditional_wald"], strict=True)
    )

    result = subspace.select_shared_subspace_membership(
        observed, reference, null, config=config
    )
    edge = result.typed_subspace_hyperedges.iloc[0]

    assert edge["selection_status"] == "unresolved"
    assert edge["empirical_multimodality_p"] > 0.10
    assert edge["abstention_reason"] == "insufficient_modalities"


def test_mixed_axis_collider_is_selected_in_state_and_direct_layers() -> None:
    rng = np.random.default_rng(123)
    n_chrom = 2500
    state = rng.normal(size=(n_chrom, 3))
    depth = 0.35 * state[:, 0] + rng.normal(size=n_chrom)
    parent_p = rng.binomial(1, expit(-1.7 + 1.3 * state[:, 0]))
    parent_q = rng.binomial(1, expit(-1.7 + 1.3 * state[:, 1]))
    collider_c = rng.binomial(1, expit(-2.5 + 2.0 * parent_p + 2.0 * parent_q))
    chrom = loading_groups.AssayData(
        modality="chrom_X",
        cell_ids=np.asarray([f"ch{index}" for index in range(n_chrom)]),
        coordinates=state,
        coordinate_uncertainty=np.zeros(n_chrom),
        log_depth=depth,
        focal=pd.DataFrame(
            {
                "P": parent_p,
                "Q": parent_q,
                "C": collider_c,
                "N1": rng.binomial(1, 0.12, size=n_chrom),
                "N2": rng.binomial(1, 0.12, size=n_chrom),
            }
        ),
    )
    n_rna = 1800
    rna_state = rng.normal(size=(n_rna, 3))
    rna = loading_groups.AssayData(
        modality="rna",
        cell_ids=np.asarray([f"r{index}" for index in range(n_rna)]),
        coordinates=rna_state,
        coordinate_uncertainty=np.zeros(n_rna),
        log_depth=0.3 * rna_state[:, 0] + rng.normal(size=n_rna),
        focal=pd.DataFrame(
            {"gene": 1.2 * rna_state[:, 0] + rng.normal(scale=0.5, size=n_rna)}
        ),
    )
    inputs = _loading_inputs(chrom, rna)
    config = subspace.SharedStateSubspaceConfig(null_repetitions=199)
    observed = subspace._fit_observed_evidence(inputs, config)
    reference, null = subspace.estimate_conditional_wild_null(inputs, config=config)

    result = subspace.select_shared_subspace_membership(
        observed,
        reference,
        null,
        direct_layer_features=["C"],
        config=config,
    )
    membership = result.feature_membership.set_index("feature")

    assert set(
        membership.index[membership["selection_status"] == "selected"]
    ) == {"P", "Q", "C", "gene"}
    assert membership.loc["C", "direct_layer_overlap"]
    assert not membership.loc["N1", "direct_layer_overlap"]
    assert result.typed_subspace_hyperedges.iloc[0]["selection_status"] == "selected"
