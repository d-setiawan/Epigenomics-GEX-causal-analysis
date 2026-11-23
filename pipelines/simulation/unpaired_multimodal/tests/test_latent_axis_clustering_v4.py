from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import latent_axis_clustering_v4 as axis_clustering  # noqa: E402
import loading_group_projection_v4 as loading_projection  # noqa: E402


def _unit(values: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    return vector / np.linalg.norm(vector)


def _make_view(
    view_id: str,
    vectors: dict[str, np.ndarray],
    modalities: dict[str, str],
    *,
    signs: dict[str, int] | None = None,
    perturbation: float = 0.0,
) -> loading_projection.LoadingEvidenceView:
    seed = int.from_bytes(view_id.encode("utf-8"), byteorder="little") % (2**32)
    rng = np.random.default_rng(seed)
    oriented: dict[str, np.ndarray] = {}
    for feature, vector in vectors.items():
        sign = int((signs or {}).get(feature, 1))
        oriented[feature] = _unit(sign * vector + perturbation * rng.normal(size=len(vector)))
    models = []
    loadings = []
    for feature in sorted(oriented):
        modality = modalities[feature]
        channel = "RNA" if modality == "rna" else modality.removeprefix("chrom_")
        models.append(
            {
                "feature": feature,
                "modality": modality,
                "channel": channel,
                "oof_partial_state_fit": 0.12,
                "conditional_loading_norm": 0.60,
                "loading_direction_stability": 0.95,
            }
        )
        for index, coefficient in enumerate(oriented[feature]):
            loadings.append(
                {
                    "feature": feature,
                    "modality": modality,
                    "channel": channel,
                    "state_coordinate": f"integration__dim{index}",
                    "conditional_coefficient": coefficient,
                }
            )
    pairs = []
    features = sorted(oriented)
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            cosine = abs(float(np.dot(oriented[left], oriented[right])))
            pairs.append(
                {
                    "pair_key": f"{left}|||{right}",
                    "node1": left,
                    "node2": right,
                    "modality1": modalities[left],
                    "modality2": modalities[right],
                    "same_assay": modalities[left] == modalities[right],
                    "absolute_loading_cosine": cosine,
                    "node1_oof_partial_state_fit": 0.12,
                    "node2_oof_partial_state_fit": 0.12,
                    "node1_loading_stability": 0.95,
                    "node2_loading_stability": 0.95,
                    "pair_state_score": 0.12 * cosine,
                }
            )
    return loading_projection.make_evidence_view(
        view_id=view_id,
        feature_models=pd.DataFrame(models),
        loadings=pd.DataFrame(loadings),
        pair_evidence=pd.DataFrame(pairs),
    )


def _groups(
    specs: dict[str, tuple[str, list[str], float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    membership_rows = []
    for group_id, (modality, features, confidence) in specs.items():
        group_rows.append(
            {
                "loading_group": group_id,
                "modality": modality,
                "members": ";".join(sorted(features)),
                "group_confidence": confidence,
                "selection_status": "selected",
            }
        )
        for feature in features:
            membership_rows.append(
                {
                    "loading_group": group_id,
                    "feature": feature,
                    "modality": modality,
                }
            )
    return pd.DataFrame(group_rows), pd.DataFrame(membership_rows)


def _with_singleton_candidates(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    specs: dict[str, tuple[str, str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows = []
    membership_rows = []
    for group_id, (modality, feature, confidence) in specs.items():
        group_rows.append(
            {
                "loading_group": group_id,
                "modality": modality,
                "members": feature,
                "group_confidence": confidence,
                "selection_status": "singleton_candidate",
            }
        )
        membership_rows.append(
            {
                "loading_group": group_id,
                "feature": feature,
                "modality": modality,
            }
        )
    return (
        pd.concat([groups, pd.DataFrame(group_rows)], ignore_index=True),
        pd.concat([membership, pd.DataFrame(membership_rows)], ignore_index=True),
    )


def _six_views(
    vectors: dict[str, np.ndarray],
    modalities: dict[str, str],
    sign_groups: list[list[str]] | None = None,
) -> list[loading_projection.LoadingEvidenceView]:
    views = []
    sign_groups = sign_groups or []
    for index in range(6):
        signs = {feature: 1 for feature in vectors}
        # Deliberately flip whole assay-local groups in alternating views.  A
        # projective axis must be invariant to these arbitrary signs.
        if index % 2:
            for feature_group in sign_groups:
                for feature in feature_group:
                    signs[feature] *= -1
        views.append(
            _make_view(
                f"view_{index}",
                vectors,
                modalities,
                signs=signs,
                perturbation=0.002,
            )
        )
    return views


def test_complete_link_recovers_multi_assay_axes_and_keeps_full_evidence() -> None:
    vectors = {
        "a1": _unit([1.0, 0.01, 0.0]),
        "a2": _unit([0.99, -0.01, 0.0]),
        "b1": _unit([-1.0, 0.02, 0.0]),
        "b2": _unit([-0.99, -0.02, 0.0]),
        "c1": _unit([1.0, 0.03, 0.0]),
        "c2": _unit([0.99, -0.03, 0.0]),
        "d1": _unit([0.01, 1.0, 0.0]),
        "d2": _unit([-0.01, 0.99, 0.0]),
        "e1": _unit([0.02, -1.0, 0.0]),
        "e2": _unit([-0.02, -0.99, 0.0]),
        "expr__g": _unit([-1.0, 0.0, 0.0]),
    }
    modalities = {
        **{feature: "chrom_A" for feature in ("a1", "a2")},
        **{feature: "chrom_B" for feature in ("b1", "b2")},
        **{feature: "chrom_C" for feature in ("c1", "c2")},
        **{feature: "chrom_D" for feature in ("d1", "d2")},
        **{feature: "chrom_E" for feature in ("e1", "e2")},
        "expr__g": "rna",
    }
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a1", "a2"], 0.93),
            "group_B": ("chrom_B", ["b1", "b2"], 0.92),
            "group_C": ("chrom_C", ["c1", "c2"], 0.91),
            "group_D": ("chrom_D", ["d1", "d2"], 0.90),
            "group_E": ("chrom_E", ["e1", "e2"], 0.89),
            "group_R": ("rna", ["expr__g"], 0.88),
        }
    )
    views = _six_views(
        vectors,
        modalities,
        sign_groups=[["b1", "b2"], ["e1", "e2"], ["expr__g"]],
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, views
    )

    selected = result.axis_candidates.loc[
        result.axis_candidates["selection_status"] == "selected"
    ]
    assert len(selected) == 2
    selected_group_sets = {
        frozenset(value.split(";")) for value in selected["loading_groups"]
    }
    assert frozenset({"group_A", "group_B", "group_C", "group_R"}) in selected_group_sets
    assert frozenset({"group_D", "group_E"}) in selected_group_sets
    rna_axis = selected.loc[selected["contains_rna"]].iloc[0]
    assert rna_axis["edge_type"] == "cross_assay_shared_state_axis_with_rna"
    assert result.group_pair_view_evidence.shape[0] == 15 * 6
    assert len(result.group_pair_evidence) == 15
    assert len(result.group_axis_evidence) == 6 * 2
    assert len(result.group_axis_view_evidence) == 6 * 2 * 6
    assert result.group_pair_view_evidence[
        "absolute_centroid_cosine"
    ].between(0.0, 1.0).all()
    assert not result.provenance["input_policy"][
        "truth_scorer_roles_or_dag_read"
    ]
    assert result.provenance["clustering_rule"][
        "thresholds_recoverable_from_long_evidence"
    ]


def test_strict_singleton_joins_only_a_coherent_unresolved_anchor_axis() -> None:
    vectors = {
        "a1": _unit([1.0, 0.01, 0.0]),
        "a2": _unit([0.99, -0.01, 0.0]),
        "c1": _unit([0.0, 1.0, 0.01]),
        "c2": _unit([0.0, 0.99, -0.01]),
        "b": _unit([-0.999, 0.02, 0.0]),
    }
    modalities = {
        "a1": "chrom_A",
        "a2": "chrom_A",
        "b": "chrom_B",
        "c1": "chrom_C",
        "c2": "chrom_C",
    }
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a1", "a2"], 0.95),
            "group_C": ("chrom_C", ["c1", "c2"], 0.94),
        }
    )
    groups, membership = _with_singleton_candidates(
        groups,
        membership,
        {"singleton_B": ("chrom_B", "b", 0.93)},
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, _six_views(vectors, modalities)
    )

    selected = result.axis_candidates.loc[
        result.axis_candidates["selection_status"] == "selected"
    ]
    assert len(selected) == 1
    axis = selected.iloc[0]
    assert set(axis["loading_groups"].split(";")) == {"group_A", "singleton_B"}
    assert axis["n_singleton_bridges"] == 1
    assert axis["n_coherent_anchor_groups"] == 1
    bridge = result.group_axis_evidence.loc[
        (result.group_axis_evidence["loading_group"] == "singleton_B")
        & result.group_axis_evidence["bridge__singleton_bridge_selected"]
    ].iloc[0]
    assert bridge["bridge__singleton_projective_null_q"] <= 0.25
    assert bridge["bridge__singleton_axis_rank"] == 1
    assert bridge["bridge__axis_candidate_rank"] == 1
    assert result.sparse_pair_projection.empty
    assert result.provenance["clustering_rule"]["singleton_only_axes_form"] is False


def test_singletons_cannot_form_an_axis_without_a_matching_coherent_anchor() -> None:
    vectors = {
        "a1": _unit([0.0, 1.0, 0.0]),
        "a2": _unit([0.01, 0.99, 0.0]),
        "b": _unit([1.0, 0.0, 0.0]),
        "c": _unit([-1.0, 0.01, 0.0]),
    }
    modalities = {
        "a1": "chrom_A",
        "a2": "chrom_A",
        "b": "chrom_B",
        "c": "chrom_C",
    }
    groups, membership = _groups(
        {"group_A": ("chrom_A", ["a1", "a2"], 0.95)}
    )
    groups, membership = _with_singleton_candidates(
        groups,
        membership,
        {
            "singleton_B": ("chrom_B", "b", 0.94),
            "singleton_C": ("chrom_C", "c", 0.93),
        },
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, _six_views(vectors, modalities)
    )

    assert not (result.axis_candidates["selection_status"] == "selected").any()
    assert not result.axis_candidates["loading_groups"].str.contains(
        "singleton_"
    ).any()
    assert result.provenance["n_selected_singleton_bridges"] == 0


def test_near_tied_singletons_fail_the_axis_candidate_margin() -> None:
    vectors = {
        "a1": _unit([1.0, 0.0, 0.0]),
        "a2": _unit([0.999, 0.01, 0.0]),
        "c1": _unit([0.0, 1.0, 0.0]),
        "c2": _unit([0.01, 0.999, 0.0]),
        "b": _unit([1.0, 0.01, 0.0]),
        "d": _unit([1.0, -0.01, 0.0]),
    }
    modalities = {
        "a1": "chrom_A",
        "a2": "chrom_A",
        "b": "chrom_B",
        "c1": "chrom_C",
        "c2": "chrom_C",
        "d": "chrom_D",
    }
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a1", "a2"], 0.95),
            "group_C": ("chrom_C", ["c1", "c2"], 0.94),
        }
    )
    groups, membership = _with_singleton_candidates(
        groups,
        membership,
        {
            "singleton_B": ("chrom_B", "b", 0.93),
            "singleton_D": ("chrom_D", "d", 0.92),
        },
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, _six_views(vectors, modalities)
    )

    bridge_rows = result.group_axis_evidence.loc[
        result.group_axis_evidence["loading_group"].str.startswith("singleton_")
        & (result.group_axis_evidence["axis_id"] == "latent_axis_001")
    ]
    assert len(bridge_rows) == 2
    assert not bridge_rows["bridge__singleton_bridge_selected"].any()
    assert not bridge_rows["bridge__passes_axis_candidate_margin"].any()
    assert not (result.axis_candidates["selection_status"] == "selected").any()


def test_complete_link_and_axis_margin_abstain_from_similarity_chain() -> None:
    angle = np.deg2rad
    vectors = {
        "a": _unit([1.0, 0.0, 0.0]),
        "b": _unit([np.cos(angle(24)), np.sin(angle(24)), 0.0]),
        "c": _unit([np.cos(angle(48)), np.sin(angle(48)), 0.0]),
    }
    modalities = {"a": "chrom_A", "b": "chrom_B", "c": "chrom_C"}
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a"], 0.90),
            "group_B": ("chrom_B", ["b"], 0.90),
            "group_C": ("chrom_C", ["c"], 0.90),
        }
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, _six_views(vectors, modalities)
    )

    assert not (result.axis_candidates["n_groups"] == 3).any()
    assert (result.axis_candidates["selection_status"] == "unresolved").any()
    bridge_pairs = result.group_pair_evidence.set_index("group_pair_key")
    assert bridge_pairs.loc["group_A|||group_B", "cluster_compatible"]
    assert bridge_pairs.loc["group_B|||group_C", "cluster_compatible"]
    assert not bridge_pairs.loc["group_A|||group_C", "cluster_compatible"]
    assert result.sparse_pair_projection.empty


def test_sparse_projection_requires_unique_external_axis_representatives() -> None:
    angle = np.deg2rad
    vectors = {
        "a1": _unit([1.0, 0.0, 0.0]),
        "a2": _unit([np.cos(angle(30)), np.sin(angle(30)), 0.0]),
        "b1": _unit([1.0, 0.0, 0.0]),
        "b2": _unit([np.cos(angle(30)), -np.sin(angle(30)), 0.0]),
        "expr__g": _unit([1.0, 0.0, 0.0]),
    }
    modalities = {
        "a1": "chrom_A",
        "a2": "chrom_A",
        "b1": "chrom_B",
        "b2": "chrom_B",
        "expr__g": "rna",
    }
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a1", "a2"], 0.92),
            "group_B": ("chrom_B", ["b1", "b2"], 0.91),
            "group_R": ("rna", ["expr__g"], 0.90),
        }
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, _six_views(vectors, modalities)
    )

    assert len(result.axis_candidates.loc[result.axis_candidates["selection_status"] == "selected"]) == 1
    selected_reps = result.representative_evidence.loc[
        result.representative_evidence["representative_selected"]
    ]
    assert set(selected_reps["feature"]) == {"a1", "b1", "expr__g"}
    assert len(result.sparse_pair_projection) == 2
    assert (
        result.sparse_pair_projection["edge_type"]
        == "cross_assay_shared_state_axis_representative"
    ).all()
    assert result.sparse_pair_projection["contains_rna"].any()
    assert result.sparse_pair_projection["interpretation"].str.contains(
        "not a direct causal edge"
    ).all()


def test_low_group_confidence_is_retained_as_evidence_but_abstains() -> None:
    vectors = {
        "a": _unit([1.0, 0.0, 0.0]),
        "b": _unit([-1.0, 0.0, 0.0]),
    }
    modalities = {"a": "chrom_A", "b": "chrom_B"}
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a"], 0.90),
            "group_B": ("chrom_B", ["b"], 0.10),
        }
    )

    result = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, _six_views(vectors, modalities)
    )

    pair = result.group_pair_evidence.iloc[0]
    assert pair["median_absolute_centroid_cosine"] > 0.99
    assert not pair["passes_group_confidence"]
    assert not pair["cluster_compatible"]
    assert (result.axis_candidates["selection_status"] == "unresolved").all()
    assert (result.typed_axis_hyperedges["edge_type"] == "unresolved").all()


def test_result_is_deterministic_under_group_and_membership_row_order() -> None:
    vectors = {
        "a": _unit([1.0, 0.0, 0.0]),
        "b": _unit([-1.0, 0.0, 0.0]),
        "c": _unit([0.0, 1.0, 0.0]),
    }
    modalities = {"a": "chrom_A", "b": "chrom_B", "c": "chrom_C"}
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a"], 0.90),
            "group_B": ("chrom_B", ["b"], 0.90),
            "group_C": ("chrom_C", ["c"], 0.90),
        }
    )
    views = _six_views(vectors, modalities)

    first = axis_clustering.estimate_latent_axis_clustering(
        groups, membership, views
    )
    second = axis_clustering.estimate_latent_axis_clustering(
        groups.sample(frac=1.0, random_state=4),
        membership.sample(frac=1.0, random_state=5),
        list(reversed(views)),
    )

    pd.testing.assert_frame_equal(first.group_pair_evidence, second.group_pair_evidence)
    pd.testing.assert_frame_equal(
        first.axis_candidates.drop(columns=["medoid_anchor_view_support"]),
        second.axis_candidates.drop(columns=["medoid_anchor_view_support"]),
        check_exact=False,
        atol=1e-12,
    )
    assert set(first.typed_axis_hyperedges["loading_groups"]) == set(
        second.typed_axis_hyperedges["loading_groups"]
    )


def test_forbidden_semantic_columns_and_membership_mismatch_are_rejected() -> None:
    vectors = {
        "a": _unit([1.0, 0.0, 0.0]),
        "b": _unit([-1.0, 0.0, 0.0]),
    }
    modalities = {"a": "chrom_A", "b": "chrom_B"}
    groups, membership = _groups(
        {
            "group_A": ("chrom_A", ["a"], 0.90),
            "group_B": ("chrom_B", ["b"], 0.90),
        }
    )
    views = _six_views(vectors, modalities)

    with pytest.raises(ValueError, match="forbidden"):
        axis_clustering.estimate_latent_axis_clustering(
            groups.assign(truth_axis=0), membership, views
        )
    with pytest.raises(ValueError, match="membership rows|Membership disagrees"):
        axis_clustering.estimate_latent_axis_clustering(
            groups, membership.loc[membership["feature"] != "b"], views
        )
