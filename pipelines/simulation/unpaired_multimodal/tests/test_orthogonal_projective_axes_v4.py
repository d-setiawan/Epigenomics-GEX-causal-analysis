from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import loading_group_projection_v4 as loading_projection  # noqa: E402
import orthogonal_projective_axes_v4 as orthogonal_axes  # noqa: E402


def _unit(values: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    return vector / np.linalg.norm(vector)


def _view(
    view_id: str,
    vectors: dict[str, np.ndarray],
    modalities: dict[str, str],
    *,
    perturbation: float = 0.0,
) -> loading_projection.LoadingEvidenceView:
    rng = np.random.default_rng(sum(view_id.encode("utf-8")))
    adjusted = {
        feature: _unit(vector + perturbation * rng.normal(size=len(vector)))
        for feature, vector in vectors.items()
    }
    models: list[dict[str, object]] = []
    loadings: list[dict[str, object]] = []
    for feature in sorted(adjusted):
        modality = modalities[feature]
        channel = "RNA" if modality == "rna" else modality.removeprefix("chrom_")
        models.append(
            {
                "feature": feature,
                "modality": modality,
                "channel": channel,
                "oof_partial_state_fit": 0.15,
                "conditional_loading_norm": 0.80,
                "loading_direction_stability": 0.98,
            }
        )
        for index, coefficient in enumerate(adjusted[feature]):
            loadings.append(
                {
                    "feature": feature,
                    "modality": modality,
                    "channel": channel,
                    "state_coordinate": f"integration__dim{index}",
                    "conditional_coefficient": coefficient,
                }
            )
    pairs: list[dict[str, object]] = []
    features = sorted(adjusted)
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            cosine = abs(float(adjusted[left] @ adjusted[right]))
            pairs.append(
                {
                    "pair_key": f"{left}|||{right}",
                    "node1": left,
                    "node2": right,
                    "modality1": modalities[left],
                    "modality2": modalities[right],
                    "same_assay": modalities[left] == modalities[right],
                    "absolute_loading_cosine": cosine,
                    "node1_oof_partial_state_fit": 0.15,
                    "node2_oof_partial_state_fit": 0.15,
                    "node1_loading_stability": 0.98,
                    "node2_loading_stability": 0.98,
                    "pair_state_score": 0.15 * cosine,
                }
            )
    return loading_projection.make_evidence_view(
        view_id=view_id,
        feature_models=pd.DataFrame(models),
        loadings=pd.DataFrame(loadings),
        pair_evidence=pd.DataFrame(pairs),
    )


def _groups(
    specs: dict[str, tuple[str, list[str]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for group_id, (modality, features) in specs.items():
        group_rows.append(
            {
                "loading_group": group_id,
                "modality": modality,
                "members": ";".join(sorted(features)),
                "group_confidence": 0.95,
                "selection_status": "selected",
            }
        )
        for feature in features:
            member_rows.append(
                {
                    "loading_group": group_id,
                    "feature": feature,
                    "modality": modality,
                }
            )
    return pd.DataFrame(group_rows), pd.DataFrame(member_rows)


def _config(**updates: object) -> orthogonal_axes.OrthogonalProjectiveAxisConfig:
    values: dict[str, object] = {
        "null_directions": 2_000,
        "optimizer_max_iterations": 30,
        "optimizer_population_size": 6,
        "optimizer_tolerance": 1e-5,
        "random_seed": 11,
    }
    values.update(updates)
    return orthogonal_axes.OrthogonalProjectiveAxisConfig(**values)


def test_global_basis_recovers_three_projective_axes_with_arbitrary_signs() -> None:
    vectors = {
        "a1": _unit([1.0, 0.02, 0.01]),
        "a2": _unit([0.98, -0.01, 0.0]),
        "b1": _unit([-1.0, 0.01, 0.0]),
        "b2": _unit([-0.99, -0.02, 0.01]),
        "c1": _unit([0.01, 1.0, 0.02]),
        "c2": _unit([0.0, 0.98, -0.01]),
        "d1": _unit([0.02, -1.0, 0.01]),
        "d2": _unit([-0.01, -0.99, 0.0]),
        "e1": _unit([0.01, 0.02, 1.0]),
        "e2": _unit([0.0, -0.01, 0.99]),
        "expr__g": _unit([0.01, 0.0, -1.0]),
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
            "group_A": ("chrom_A", ["a1", "a2"]),
            "group_B": ("chrom_B", ["b1", "b2"]),
            "group_C": ("chrom_C", ["c1", "c2"]),
            "group_D": ("chrom_D", ["d1", "d2"]),
            "group_E": ("chrom_E", ["e1", "e2"]),
            "group_R": ("rna", ["expr__g"]),
        }
    )
    views = [_view(f"view_{index}", vectors, modalities, perturbation=0.002) for index in range(4)]

    result = orthogonal_axes.estimate_orthogonal_projective_axes(
        groups, membership, views, config=_config()
    )

    selected = result.axis_candidates.loc[
        result.axis_candidates["selection_status"] == "selected"
    ]
    assert len(selected) == 3
    group_sets = {frozenset(value.split(";")) for value in selected["loading_groups"]}
    assert frozenset({"group_A", "group_B"}) in group_sets
    assert frozenset({"group_C", "group_D"}) in group_sets
    assert frozenset({"group_E", "group_R"}) in group_sets
    assert result.group_evidence["group_assignment_selected"].all()
    basis = result.basis.drop(columns="basis_axis_index").to_numpy(dtype=float)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-8)
    assert not result.provenance["information_boundary"]["truth_or_scorer_used"]


def test_global_basis_can_resolve_pair_below_complete_link_threshold() -> None:
    # The A/B pair has cosine 0.76, below the native complete-link 0.85 gate.
    # Two strong orthogonal axes constrain the remaining basis direction.
    vectors = {
        "a": _unit([1.0, 0.0, 0.0]),
        "b": _unit([0.76, 0.45, 0.47]),
        "c": _unit([0.0, 1.0, 0.0]),
        "d": _unit([0.0, 0.99, 0.01]),
        "e": _unit([0.0, 0.0, 1.0]),
        "f": _unit([0.01, 0.0, 0.99]),
    }
    modalities = {feature: f"chrom_{feature.upper()}" for feature in vectors}
    groups, membership = _groups(
        {f"group_{feature}": (modalities[feature], [feature]) for feature in vectors}
    )
    views = [_view(f"view_{index}", vectors, modalities, perturbation=0.001) for index in range(4)]

    result = orthogonal_axes.estimate_orthogonal_projective_axes(
        groups,
        membership,
        views,
        config=_config(
            minimum_leave_one_out_absolute_cosine=0.70,
            maximum_leave_one_out_projective_null_q=0.60,
        ),
    )

    selected_sets = {
        frozenset(value.split(";"))
        for value in result.axis_candidates.loc[
            result.axis_candidates["selection_status"] == "selected", "loading_groups"
        ]
    }
    assert any({"group_a", "group_b"} <= value for value in selected_sets)


def test_ambiguous_held_out_group_abstains() -> None:
    vectors = {
        "a": _unit([1.0, 0.0, 0.0]),
        "b": _unit([0.99, 0.01, 0.0]),
        "c": _unit([0.0, 1.0, 0.0]),
        "d": _unit([0.01, 0.99, 0.0]),
        "ambiguous": _unit([1.0, 1.0, 1.0]),
    }
    modalities = {feature: f"chrom_{feature.upper()}" for feature in vectors}
    groups, membership = _groups(
        {f"group_{feature}": (modalities[feature], [feature]) for feature in vectors}
    )
    views = [_view(f"view_{index}", vectors, modalities, perturbation=0.002) for index in range(4)]

    result = orthogonal_axes.estimate_orthogonal_projective_axes(
        groups, membership, views, config=_config()
    )

    ambiguous = result.group_evidence.set_index("loading_group").loc["group_ambiguous"]
    assert not bool(ambiguous["group_assignment_selected"])
    assert "margin" in str(ambiguous["abstention_reason"])


def test_safe_cli_table_loader_rejects_scorer_path(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    path = scorer / "loading_groups.tsv"
    path.write_text("loading_group\tmodality\tmembers\tgroup_confidence\tselection_status\n")

    with pytest.raises(ValueError, match="scorer/truth"):
        orthogonal_axes._safe_table(path, "loading_groups.tsv")
