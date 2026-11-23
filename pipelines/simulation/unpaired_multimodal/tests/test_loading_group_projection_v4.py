from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loading_group_projection_v4 as projection  # noqa: E402


def test_loader_hashes_and_parses_the_same_round_trip_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = float(np.nextafter(0.1, float("inf")))
    source = tmp_path / "view"
    source.mkdir()
    filenames = (
        "cell_loading_feature_models.tsv",
        "cell_loading_loadings.tsv",
        "cell_loading_pair_evidence.tsv",
    )
    original: dict[str, bytes] = {}
    for filename in filenames:
        path = source / filename
        path.write_text(f"value\n{boundary!r}\n", encoding="utf-8")
        original[filename] = path.read_bytes()

    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    original_snapshot = projection._read_regular_bytes_once
    mutated = False

    def mutate_after_snapshot(path: Path, label: str) -> bytes:
        nonlocal mutated
        payload = original_snapshot(path, label)
        if not mutated:
            mutated = True
            path.write_text("value\n0.9\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(projection, "make_evidence_view", capture)
    monkeypatch.setattr(
        projection, "_read_regular_bytes_once", mutate_after_snapshot
    )

    projection.load_evidence_view(source, view_id="round_trip")

    for key in ("feature_models", "loadings", "pair_evidence"):
        frame = captured[key]
        assert isinstance(frame, pd.DataFrame)
        assert frame.loc[0, "value"] == boundary
    assert captured["input_hashes"] == {
        filename: hashlib.sha256(original[filename]).hexdigest()
        for filename in filenames
    }


def _unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    return vector / np.linalg.norm(vector)


def _make_view(
    view_id: str,
    vectors: dict[str, np.ndarray],
    modalities: dict[str, str],
    fits: dict[str, float],
    *,
    perturbation: float = 0.0,
) -> projection.LoadingEvidenceView:
    seed = int.from_bytes(view_id.encode("utf-8"), byteorder="little") % (2**32)
    rng = np.random.default_rng(seed)
    adjusted: dict[str, np.ndarray] = {}
    for feature, vector in vectors.items():
        draw = vector + perturbation * rng.normal(size=len(vector))
        adjusted[feature] = draw / np.linalg.norm(draw)
    model_rows = []
    loading_rows = []
    for feature in sorted(adjusted):
        modality = modalities[feature]
        channel = "RNA" if modality == "rna" else modality.removeprefix("chrom_")
        model_rows.append(
            {
                "feature": feature,
                "modality": modality,
                "channel": channel,
                "oof_partial_state_fit": fits[feature],
                "conditional_loading_norm": 0.60,
                "loading_direction_stability": 0.95,
            }
        )
        for index, coefficient in enumerate(adjusted[feature]):
            loading_rows.append(
                {
                    "feature": feature,
                    "modality": modality,
                    "channel": channel,
                    "state_coordinate": f"integration__dim{index}",
                    "conditional_coefficient": coefficient,
                }
            )
    pair_rows = []
    features = sorted(adjusted)
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            cosine = float(np.dot(adjusted[left], adjusted[right]))
            stability = 0.95
            pair_rows.append(
                {
                    "pair_key": f"{left}|||{right}",
                    "node1": left,
                    "node2": right,
                    "modality1": modalities[left],
                    "modality2": modalities[right],
                    "same_assay": modalities[left] == modalities[right],
                    "absolute_loading_cosine": abs(cosine),
                    "node1_oof_partial_state_fit": fits[left],
                    "node2_oof_partial_state_fit": fits[right],
                    "node1_loading_stability": stability,
                    "node2_loading_stability": stability,
                    "pair_state_score": abs(cosine)
                    * stability
                    * np.sqrt(max(fits[left], 0.0) * max(fits[right], 0.0)),
                }
            )
    return projection.make_evidence_view(
        view_id=view_id,
        feature_models=pd.DataFrame(model_rows),
        loadings=pd.DataFrame(loading_rows),
        pair_evidence=pd.DataFrame(pair_rows),
    )


def _coherent_views() -> list[projection.LoadingEvidenceView]:
    vectors = {
        "a1": _unit([1.0, 0.04, 0.0]),
        "a2": _unit([0.98, -0.02, 0.0]),
        "a3": _unit([0.0, 1.0, 0.0]),
        "a4": _unit([0.45, 0.85, 0.0]),
        "b1": _unit([1.0, 0.02, 0.0]),
        "b2": _unit([0.97, -0.05, 0.0]),
        "b3": _unit([0.0, 1.0, 0.0]),
        "b4": _unit([0.35, 0.88, 0.0]),
        "expr__g": _unit([0.0, 0.0, 1.0]),
    }
    modalities = {
        **{feature: "chrom_A" for feature in ("a1", "a2", "a3", "a4")},
        **{feature: "chrom_B" for feature in ("b1", "b2", "b3", "b4")},
        "expr__g": "rna",
    }
    fits = {
        "a1": 0.13,
        "a2": 0.08,
        "a3": 0.03,
        "a4": 0.025,
        "b1": 0.12,
        "b2": 0.07,
        "b3": 0.03,
        "b4": 0.025,
        "expr__g": 0.10,
    }
    return [
        _make_view("view_0", vectors, modalities, fits, perturbation=0.004),
        _make_view("view_1", vectors, modalities, fits, perturbation=0.006),
        _make_view("view_2", vectors, modalities, fits, perturbation=0.005),
    ]


def _asymmetric_gene_views() -> list[projection.LoadingEvidenceView]:
    vectors = {
        "a1": _unit([1.0, 0.0, 0.0]),
        "a2": _unit([0.999, -0.005, 0.0]),
        "b1": _unit([1.0, 0.05, 0.0]),
        "b2": _unit([0.998, 0.055, 0.0]),
        # The gene is closest to chrom_B, while chrom_B is still closer to
        # chrom_A.  This is the case a mutual peak--gene rule wrongly rejects.
        "expr__g": _unit([0.96, 0.28, 0.0]),
    }
    modalities = {
        "a1": "chrom_A",
        "a2": "chrom_A",
        "b1": "chrom_B",
        "b2": "chrom_B",
        "expr__g": "rna",
    }
    fits = {
        "a1": 0.13,
        "a2": 0.09,
        "b1": 0.12,
        "b2": 0.08,
        "expr__g": 0.11,
    }
    return [
        _make_view("gene_0", vectors, modalities, fits),
        _make_view("gene_1", vectors, modalities, fits),
        _make_view("gene_2", vectors, modalities, fits),
    ]


def test_coherent_within_assay_groups_precede_typed_cross_group_projection() -> None:
    result = projection.estimate_loading_group_projection(_coherent_views())

    selected_members = set(result.groups["members"])
    assert "a1;a2" in selected_members
    assert "b1;b2" in selected_members
    assert "expr__g" in selected_members
    cross = result.group_match_evidence.loc[
        (result.group_match_evidence["modality1"] == "chrom_A")
        & (result.group_match_evidence["modality2"] == "chrom_B")
    ].iloc[0]
    assert cross["selection_status"] == "selected"
    assert cross["edge_type"] == "cross_assay_shared_state"

    cross_hyperedges = result.typed_hyperedges.loc[
        result.typed_hyperedges["edge_type"] == "cross_assay_shared_state"
    ]
    assert len(cross_hyperedges) == 1
    cross_projection = result.sparse_pair_projection.loc[
        result.sparse_pair_projection["edge_type"] == "cross_assay_shared_state"
    ]
    assert len(cross_projection) == 1
    assert {cross_projection.iloc[0]["node1"], cross_projection.iloc[0]["node2"]} == {
        "a1",
        "b1",
    }
    assert not result.provenance["input_policy"]["truth_scorer_roles_or_dag_read"]


def test_gene_uses_asymmetric_global_best_peak_and_prior_constrained_type() -> None:
    config = projection.LoadingProjectionConfig(cross_projection_relative_margin=0.01)
    result = projection.estimate_loading_group_projection(
        _asymmetric_gene_views(), config
    )

    peak_gene = result.group_match_evidence.loc[
        (result.group_match_evidence["modality1"] == "chrom_B")
        & (result.group_match_evidence["modality2"] == "rna")
    ].iloc[0]
    chromatin_match = result.group_match_evidence.loc[
        (result.group_match_evidence["modality1"] == "chrom_A")
        & (result.group_match_evidence["modality2"] == "chrom_B")
    ].iloc[0]
    assert peak_gene["median_absolute_centroid_cosine"] < chromatin_match[
        "median_absolute_centroid_cosine"
    ]
    assert peak_gene["gene_global_best"]
    assert peak_gene["gene_peak_neighbor_rank"] == 1
    assert not peak_gene["mutual_best"]
    assert peak_gene["selection_rule"] == "asymmetric_global_gene_best_peak"
    assert peak_gene["selection_status"] == "selected"
    assert peak_gene["edge_type"] == "prior_constrained_peak_gene"

    prior_hyperedges = result.typed_hyperedges.loc[
        result.typed_hyperedges["edge_type"] == "prior_constrained_peak_gene"
    ]
    prior_projections = result.sparse_pair_projection.loc[
        result.sparse_pair_projection["edge_type"] == "prior_constrained_peak_gene"
    ]
    assert len(prior_hyperedges) == 1
    assert len(prior_projections) == 1
    assert prior_hyperedges["hyperedge_id"].is_unique
    assert prior_projections["pair_key"].is_unique
    assert prior_projections.iloc[0]["projection_source"] == (
        "asymmetric_global_gene_best_peak_group"
    )
    assert result.provenance["identifiability"][
        "selected_group_match_attached_post_consensus_once"
    ]


def test_ambiguous_gene_best_peak_remains_unresolved_and_out_of_projection() -> None:
    result = projection.estimate_loading_group_projection(_asymmetric_gene_views())

    gene_matches = result.group_match_evidence.loc[
        (result.group_match_evidence["modality1"] == "rna")
        | (result.group_match_evidence["modality2"] == "rna")
    ]
    assert (gene_matches["selection_status"] == "unresolved").any()
    unresolved = gene_matches.loc[
        gene_matches["selection_status"] == "unresolved"
    ].iloc[0]
    assert unresolved["edge_type"] == "unresolved"
    assert not (
        result.sparse_pair_projection["edge_type"]
        == "prior_constrained_peak_gene"
    ).any()


def test_ambiguous_assay_local_group_abstains_from_pair_projection() -> None:
    vectors = {
        "a1": _unit([1.0, 0.0, 0.0]),
        "a2": _unit([0.99, 0.01, 0.0]),
        "a3": _unit([0.99, -0.01, 0.0]),
    }
    modalities = {feature: "chrom_A" for feature in vectors}
    fits = {feature: 0.08 for feature in vectors}
    views = [
        _make_view("ambiguous_0", vectors, modalities, fits),
        _make_view("ambiguous_1", vectors, modalities, fits),
    ]

    result = projection.estimate_loading_group_projection(views)

    assert result.groups.empty
    assert result.sparse_pair_projection.empty
    assert result.feature_consensus["locally_ambiguous_group_excluded"].all()
    assert not result.feature_consensus["singleton_bridge_eligible"].any()
    top = result.within_group_candidates.loc[
        result.within_group_candidates["aggregate_rank"] == 1
    ].iloc[0]
    assert top["selection_status"] == "unresolved"
    assert "margin" in top["selection_reason"]


def test_observed_direct_collider_child_is_excluded_before_group_ranking() -> None:
    vectors = {
        "a_child": _unit([1.0, 0.0, 0.0]),
        "a_s": _unit([0.995, 0.01, 0.0]),
        "a_p": _unit([0.96, 0.08, 0.0]),
        "a_other": _unit([0.0, 1.0, 0.0]),
    }
    modalities = {feature: "chrom_A" for feature in vectors}
    fits = {"a_child": 0.15, "a_s": 0.12, "a_p": 0.10, "a_other": 0.03}
    views = [
        _make_view("collider_0", vectors, modalities, fits),
        _make_view("collider_1", vectors, modalities, fits),
        _make_view("collider_2", vectors, modalities, fits),
    ]

    result = projection.estimate_loading_group_projection(
        views,
        projection.LoadingProjectionConfig(min_group_relative_margin=0.01),
        excluded_features=("a_child",),
    )

    selected_ids = set(
        result.groups.loc[
            result.groups["selection_status"] == "selected", "loading_group"
        ]
    )
    selected_membership = result.group_membership.loc[
        result.group_membership["loading_group"].isin(selected_ids)
    ]
    assert set(selected_membership["feature"]) == {"a_p", "a_s"}
    assert result.groups.loc[
        result.groups["selection_status"] == "selected", "members"
    ].iloc[0] == "a_p;a_s"
    assert "a_child" not in set(result.group_membership["feature"])
    assert result.feature_consensus.set_index("feature").loc[
        "a_child", "direct_collider_excluded"
    ]
    assert result.provenance["excluded_direct_collider_features"] == ["a_child"]
    assert result.provenance["excluded_observable_structure_features"] == [
        "a_child"
    ]


def test_observed_direct_parent_pair_cannot_be_relabelled_as_shared_state() -> None:
    vectors = {
        "a_parent1": _unit([1.0, 0.0, 0.0]),
        "a_parent2": _unit([0.999, 0.04, 0.0]),
        "a_state_sibling": _unit([0.94, -0.34, 0.0]),
        "a_other": _unit([0.0, 1.0, 0.0]),
    }
    modalities = {feature: "chrom_A" for feature in vectors}
    fits = {
        "a_parent1": 0.12,
        "a_parent2": 0.12,
        "a_state_sibling": 0.12,
        "a_other": 0.03,
    }
    views = [
        _make_view("parents_0", vectors, modalities, fits),
        _make_view("parents_1", vectors, modalities, fits),
        _make_view("parents_2", vectors, modalities, fits),
    ]

    unconstrained = projection.estimate_loading_group_projection(
        views,
        projection.LoadingProjectionConfig(min_group_relative_margin=0.01),
    )
    assert unconstrained.groups.loc[
        unconstrained.groups["selection_status"] == "selected", "members"
    ].iloc[0] == "a_parent1;a_parent2"

    constrained = projection.estimate_loading_group_projection(
        views,
        projection.LoadingProjectionConfig(min_group_relative_margin=0.01),
        forbidden_within_pairs=(("a_parent2", "a_parent1"),),
    )
    selected = constrained.groups.loc[
        constrained.groups["selection_status"] == "selected", "members"
    ].iloc[0]
    assert selected == "a_parent1;a_state_sibling"
    parent_pair = constrained.within_group_candidates.set_index("pair_key").loc[
        "a_parent1|||a_parent2"
    ]
    assert parent_pair["observable_structure_pair_excluded"]
    assert not parent_pair["selected_group"]
    assert constrained.provenance[
        "forbidden_observable_direct_parent_pairs"
    ] == [["a_parent1", "a_parent2"]]
    assert constrained.provenance["input_policy"][
        "direct_collider_parent_pairs_excluded_from_state_groups"
    ]


def test_only_layered_structure_anchored_pairs_can_form_chromatin_groups() -> None:
    vectors = {
        "a_parent1": _unit([1.0, 0.0, 0.0]),
        "a_parent2": _unit([0.999, 0.04, 0.0]),
        "a_state_sibling": _unit([0.94, -0.34, 0.0]),
        "a_other": _unit([0.92, -0.39, 0.0]),
    }
    modalities = {feature: "chrom_A" for feature in vectors}
    fits = {feature: 0.12 for feature in vectors}
    views = [
        _make_view("anchored_0", vectors, modalities, fits),
        _make_view("anchored_1", vectors, modalities, fits),
        _make_view("anchored_2", vectors, modalities, fits),
    ]
    result = projection.estimate_loading_group_projection(
        views,
        projection.LoadingProjectionConfig(min_group_relative_margin=0.01),
        forbidden_within_pairs=(("a_parent1", "a_parent2"),),
        permitted_within_pairs=(("a_parent1", "a_state_sibling"),),
    )
    selected = result.groups.loc[
        result.groups["selection_status"] == "selected", "members"
    ].tolist()
    assert selected == ["a_parent1;a_state_sibling"]
    candidates = result.within_group_candidates.set_index("pair_key")
    assert candidates.loc[
        "a_parent1|||a_state_sibling", "structure_anchor_pair_permitted"
    ]
    assert candidates.loc[
        "a_parent1|||a_state_sibling", "selection_reason"
    ] == "layered_state_sibling_structure_anchor"
    assert not candidates.loc[
        "a_other|||a_parent1", "structure_anchor_pair_permitted"
    ]
    assert candidates.loc[
        "a_other|||a_parent1", "selection_reason"
    ] == "not_layered_structure_anchor"
    assert result.provenance["permitted_structure_anchored_within_pairs"] == [
        ["a_parent1", "a_state_sibling"]
    ]
    assert result.provenance["input_policy"][
        "layered_state_sibling_structure_gate_applied"
    ]
    assert result.provenance["schema_version"] == "loading_group_projection_v4.v7"
    assert result.provenance["structure_anchor_semantics"] == {
        "selection_basis": "layered_state_sibling_structure_anchor",
        "formal_cross_assay_or_assay_local_state_tier_allowed": True,
        "assay_local_tier_does_not_claim_cross_assay_membership": True,
    }

    abstained = projection.estimate_loading_group_projection(
        views,
        permitted_within_pairs=(),
    )
    assert not abstained.groups["selection_status"].eq("selected").any()


def test_strict_singleton_candidates_are_diagnostic_not_local_group_claims() -> None:
    result = projection.estimate_loading_group_projection(_coherent_views())

    candidates = result.groups.loc[
        result.groups["selection_status"] == "singleton_candidate"
    ]
    assert not candidates.empty
    assert (candidates["n_members"] == 1).all()
    candidate_ids = set(candidates["loading_group"])
    assert not result.group_match_evidence["group1"].isin(candidate_ids).any()
    assert not result.group_match_evidence["group2"].isin(candidate_ids).any()
    assert not result.typed_hyperedges["group1"].isin(candidate_ids).any()
    assert not result.typed_hyperedges["group2"].isin(candidate_ids).any()
    assert result.provenance["identifiability"][
        "singleton_candidate_cannot_seed_an_axis"
    ]


def test_cross_group_similarity_is_typed_unresolved_without_forced_pair_edge() -> None:
    vectors = {
        "a1": _unit([1.0, 0.01, 0.0]),
        "a2": _unit([0.99, -0.01, 0.0]),
        "a3": _unit([0.0, 1.0, 0.0]),
        "b1": _unit([0.80, 0.60, 0.0]),
        "b2": _unit([0.82, 0.58, 0.0]),
        "b3": _unit([-0.60, 0.80, 0.0]),
        "expr__g": _unit([0.0, 0.0, 1.0]),
    }
    modalities = {
        **{feature: "chrom_A" for feature in ("a1", "a2", "a3")},
        **{feature: "chrom_B" for feature in ("b1", "b2", "b3")},
        "expr__g": "rna",
    }
    fits = {
        "a1": 0.12,
        "a2": 0.09,
        "a3": 0.02,
        "b1": 0.11,
        "b2": 0.08,
        "b3": 0.02,
        "expr__g": 0.10,
    }
    views = [
        _make_view("unresolved_0", vectors, modalities, fits, perturbation=0.002),
        _make_view("unresolved_1", vectors, modalities, fits, perturbation=0.003),
        _make_view("unresolved_2", vectors, modalities, fits, perturbation=0.002),
    ]

    result = projection.estimate_loading_group_projection(views)

    match = result.group_match_evidence.loc[
        (result.group_match_evidence["modality1"] == "chrom_A")
        & (result.group_match_evidence["modality2"] == "chrom_B")
    ].iloc[0]
    assert 0.75 < match["median_absolute_centroid_cosine"] < 0.90
    assert match["selection_status"] == "unresolved"
    unresolved = result.typed_hyperedges.loc[
        (result.typed_hyperedges["group1"] == match["group1"])
        & (result.typed_hyperedges["group2"] == match["group2"])
    ].iloc[0]
    assert unresolved["edge_type"] == "unresolved"
    assert unresolved["selection_status"] == "unresolved"
    assert not (
        result.sparse_pair_projection["edge_type"] == "cross_assay_shared_state"
    ).any()


def test_complete_pair_contract_and_forbidden_columns_are_enforced() -> None:
    view = _coherent_views()[0]
    incomplete = view.pair_evidence.iloc[:-1].copy()
    with pytest.raises(ValueError, match="complete"):
        projection.make_evidence_view(
            view_id="incomplete",
            feature_models=view.feature_models,
            loadings=view.loadings,
            pair_evidence=incomplete,
        )

    forbidden = view.feature_models.assign(truth_role="unknown")
    with pytest.raises(ValueError, match="forbidden"):
        projection.make_evidence_view(
            view_id="forbidden",
            feature_models=forbidden,
            loadings=view.loadings,
            pair_evidence=view.pair_evidence,
        )


def test_views_must_share_feature_and_coordinate_contracts() -> None:
    views = _coherent_views()
    changed = views[1].loadings.copy()
    changed.loc[changed["state_coordinate"] == "integration__dim2", "state_coordinate"] = (
        "integration__dim3"
    )
    with pytest.raises(ValueError, match="sequential"):
        projection.make_evidence_view(
            view_id="bad_coordinates",
            feature_models=views[1].feature_models,
            loadings=changed,
            pair_evidence=views[1].pair_evidence,
        )


def test_projection_labels_correlated_refits_as_stability_not_sample_size() -> None:
    result = projection.estimate_loading_group_projection(_coherent_views())
    semantics = result.provenance["view_semantics"]

    assert semantics["same_observed_cells_refit"] is True
    assert semantics["views_are_algorithmic_stability_repeats"] is True
    assert semantics["views_are_independent_worlds"] is False
    assert semantics["view_support_not_world_sample_size"] is True
