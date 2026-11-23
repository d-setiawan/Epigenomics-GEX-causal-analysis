#!/usr/bin/env python3
"""Truth-blind global projective axes for unpaired assay loading groups.

The native CA-BOSS v4 axis layer uses conservative complete-link clustering.
That is an appropriate precision-first default, but it treats each pair gate
locally and therefore can fragment a globally coherent orthogonal state basis.
This standalone prototype fits at most the observed embedding dimension worth
of sign-invariant axes to assay-local loading-group projectors.

Only selected assay-local groups, their observed feature membership, and
complete cell-loading evidence views are consumed.  The fit has no argument
for simulator roles, state-dimension labels, a scorer, paired cells, or feature
correspondence between assays.  A group must retain its assignment in every
algorithmic loading view and when it is excluded from basis fitting.  The
held-out score is calibrated against isotropic projective directions.  Axes
with fewer than two observed modalities remain unresolved.

The repeated loading views use the same observed cells.  They are correlated
algorithmic stability checks, never independent worlds or biological
replicates.  A selected hyperedge means shared response to an estimated state
axis; it is not a direct causal edge or a cell pairing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import differential_evolution, linear_sum_assignment

from loading_group_projection_v4 import LoadingEvidenceView, load_evidence_view


FloatArray = NDArray[np.float64]

_FORBIDDEN_PATH_COMPONENTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
    "counterfactual",
}
_FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "true_",
    "latent",
    "state_dimension",
    "axis_id",
    "role",
    "parent",
    "child",
    "oracle",
    "scorer",
    "correspondence",
    "pairing",
)


@dataclass(frozen=True)
class OrthogonalProjectiveAxisConfig:
    """Frozen global-basis, stability, and abstention settings."""

    objective_power: float = 2.0
    minimum_group_confidence: float = 0.25
    minimum_full_absolute_cosine: float = 0.80
    minimum_view_lower_absolute_cosine: float = 0.78
    minimum_view_assignment_support: float = 0.80
    minimum_view_relative_margin: float = 0.15
    minimum_leave_one_out_absolute_cosine: float = 0.75
    minimum_leave_one_out_relative_margin: float = 0.10
    maximum_leave_one_out_projective_null_q: float = 0.25
    minimum_axis_modalities: int = 2
    one_group_per_modality_per_axis: bool = True
    null_directions: int = 50_000
    random_seed: int = 71
    optimizer_max_iterations: int = 140
    optimizer_population_size: int = 12
    optimizer_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        bounded = (
            self.minimum_group_confidence,
            self.minimum_full_absolute_cosine,
            self.minimum_view_lower_absolute_cosine,
            self.minimum_view_assignment_support,
            self.minimum_view_relative_margin,
            self.minimum_leave_one_out_absolute_cosine,
            self.minimum_leave_one_out_relative_margin,
            self.maximum_leave_one_out_projective_null_q,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("Confidence, cosine, support, margin, and q gates must lie in [0, 1]")
        if self.objective_power < 1.0:
            raise ValueError("objective_power must be at least one")
        if self.minimum_axis_modalities < 2:
            raise ValueError("A shared axis needs at least two modalities")
        if self.null_directions < 1_000:
            raise ValueError("At least 1,000 null directions are required")
        if self.random_seed < 0 or self.optimizer_max_iterations < 1:
            raise ValueError("Optimizer settings are invalid")
        if self.optimizer_population_size < 4 or self.optimizer_tolerance <= 0:
            raise ValueError("Optimizer settings are invalid")


@dataclass
class OrthogonalProjectiveAxisResult:
    basis: pd.DataFrame
    group_evidence: pd.DataFrame
    axis_candidates: pd.DataFrame
    typed_axis_hyperedges: pd.DataFrame
    provenance: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _benjamini_hochberg(values: FloatArray) -> FloatArray:
    clean = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), 1.0)
    if len(clean) == 0:
        return clean
    order = np.argsort(clean)
    ranked = clean[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _unit(vector: FloatArray) -> FloatArray:
    return np.asarray(vector, dtype=np.float64) / max(
        float(np.linalg.norm(vector)), 1e-12
    )


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_COLUMN_TOKENS)
    )


def _safe_table(path: Path | str, expected_name: str) -> tuple[Path, pd.DataFrame]:
    lexical = Path(path).absolute()
    if lexical.name != expected_name:
        raise ValueError(f"Expected a direct {expected_name} input")
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_file():
        raise ValueError(f"{expected_name} must be a direct existing file")
    resolved = lexical.resolve(strict=True)
    forbidden = sorted(
        component.casefold()
        for component in resolved.parts
        if component.casefold() in _FORBIDDEN_PATH_COMPONENTS
    )
    if forbidden:
        raise ValueError(f"Refusing scorer/truth path components: {forbidden}")
    frame = pd.read_csv(resolved, sep="\t", keep_default_na=False)
    forbidden_columns = _forbidden_columns(frame)
    if forbidden_columns:
        raise ValueError(f"Refusing privileged columns: {forbidden_columns}")
    return resolved, frame


def _coordinate_names(views: Sequence[LoadingEvidenceView]) -> list[str]:
    if len(views) < 2:
        raise ValueError("At least two complete loading views are required")
    if len({view.view_id for view in views}) != len(views):
        raise ValueError("Loading view identifiers must be unique")
    names = sorted(
        views[0].loadings["state_coordinate"].astype(str).unique(),
        key=lambda value: int(value.removeprefix("integration__dim")),
    )
    if len(names) < 2:
        raise ValueError("At least two observed embedding coordinates are required")
    expected = [f"integration__dim{index}" for index in range(len(names))]
    if names != expected:
        raise ValueError("Observed embedding coordinates must be sequential")
    reference_features = sorted(views[0].feature_models["feature"].astype(str))
    reference_modalities = (
        views[0].feature_models.set_index("feature")["modality"].astype(str).sort_index()
    )
    for view in views[1:]:
        other_names = sorted(
            view.loadings["state_coordinate"].astype(str).unique(),
            key=lambda value: int(value.removeprefix("integration__dim")),
        )
        if other_names != names:
            raise ValueError("Every loading view must use the same coordinates")
        if sorted(view.feature_models["feature"].astype(str)) != reference_features:
            raise ValueError("Every loading view must contain the same complete features")
        modalities = (
            view.feature_models.set_index("feature")["modality"].astype(str).sort_index()
        )
        if not modalities.equals(reference_modalities):
            raise ValueError("Observable modality labels must agree across views")
    return names


def _validate_groups(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    group_required = {
        "loading_group",
        "modality",
        "members",
        "group_confidence",
        "selection_status",
    }
    membership_required = {"loading_group", "feature", "modality"}
    for label, required, frame in (
        ("groups", group_required, groups),
        ("membership", membership_required, membership),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing columns {missing}")
        forbidden = _forbidden_columns(frame)
        if forbidden:
            raise ValueError(f"{label} contains privileged columns {forbidden}")
    coordinates = _coordinate_names(views)
    selected = groups.loc[groups["selection_status"].astype(str).eq("selected")].copy()
    if len(selected) < 2:
        raise ValueError("At least two selected assay-local groups are required")
    selected["loading_group"] = selected["loading_group"].astype(str)
    selected["modality"] = selected["modality"].astype(str)
    selected["members"] = selected["members"].astype(str)
    selected["group_confidence"] = pd.to_numeric(
        selected["group_confidence"], errors="raise"
    )
    if selected["loading_group"].duplicated().any():
        raise ValueError("Loading group identifiers must be unique")
    members = membership.loc[
        membership["loading_group"].astype(str).isin(selected["loading_group"])
    ].copy()
    members["loading_group"] = members["loading_group"].astype(str)
    members["feature"] = members["feature"].astype(str)
    members["modality"] = members["modality"].astype(str)
    if members.duplicated(["loading_group", "feature"]).any():
        raise ValueError("Group membership rows must be unique")
    feature_modalities = (
        views[0].feature_models.set_index("feature")["modality"].astype(str).to_dict()
    )
    for row in selected.itertuples(index=False):
        subset = members.loc[members["loading_group"] == str(row.loading_group)]
        features = sorted(subset["feature"].astype(str))
        expected = sorted(filter(None, str(row.members).split(";")))
        if not features or features != expected:
            raise ValueError(f"Membership disagrees for {row.loading_group}")
        if set(subset["modality"].astype(str)) != {str(row.modality)}:
            raise ValueError(f"Membership modality disagrees for {row.loading_group}")
        if any(feature_modalities.get(feature) != str(row.modality) for feature in features):
            raise ValueError(f"Loading-view modality disagrees for {row.loading_group}")
    return (
        selected.sort_values("loading_group").reset_index(drop=True),
        members.sort_values(["loading_group", "feature"]).reset_index(drop=True),
        coordinates,
    )


def _group_projectors(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    coordinates: Sequence[str],
) -> FloatArray:
    membership_lookup = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in membership.groupby("loading_group")
    }
    output = np.empty(
        (len(groups), len(views), len(coordinates), len(coordinates)),
        dtype=np.float64,
    )
    for view_index, view in enumerate(views):
        pivot = view.loadings.pivot(
            index="feature",
            columns="state_coordinate",
            values="conditional_coefficient",
        )[list(coordinates)]
        for group_index, group_id in enumerate(groups["loading_group"].astype(str)):
            vectors = np.vstack(
                [_unit(pivot.loc[feature].to_numpy(dtype=np.float64)) for feature in membership_lookup[group_id]]
            )
            similarities = np.abs(vectors @ vectors.T).sum(axis=1)
            reference = vectors[int(np.argmax(similarities))]
            aligned = np.where((vectors @ reference)[:, None] >= 0.0, vectors, -vectors)
            centroid = _unit(np.median(aligned, axis=0))
            output[group_index, view_index] = np.outer(centroid, centroid)
    return output


def _parameters_to_basis(parameters: FloatArray, dimensions: int) -> FloatArray:
    skew = np.zeros((dimensions, dimensions), dtype=np.float64)
    offset = 0
    for left in range(dimensions):
        for right in range(left + 1, dimensions):
            value = float(parameters[offset])
            skew[left, right] = value
            skew[right, left] = -value
            offset += 1
    return np.asarray(expm(skew), dtype=np.float64)


def _basis_objective(
    basis: FloatArray,
    projectors: FloatArray,
    weights: FloatArray,
    power: float,
) -> float:
    squared_cosines = np.einsum(
        "gij,ik,jk->gk", projectors, basis, basis, optimize=True
    )
    explained = np.power(np.clip(np.max(squared_cosines, axis=1), 0.0, 1.0), power)
    return float(np.sum(weights * explained) / max(float(weights.sum()), 1e-12))


def _fit_basis(
    projectors: FloatArray,
    weights: FloatArray,
    *,
    config: OrthogonalProjectiveAxisConfig,
    seed: int,
) -> tuple[FloatArray, float]:
    dimensions = int(projectors.shape[1])
    n_parameters = dimensions * (dimensions - 1) // 2
    if n_parameters == 0:
        return np.eye(dimensions, dtype=np.float64), 1.0

    def loss(parameters: FloatArray) -> float:
        basis = _parameters_to_basis(parameters, dimensions)
        return -_basis_objective(
            basis, projectors, weights, config.objective_power
        )

    result = differential_evolution(
        loss,
        bounds=[(-np.pi, np.pi)] * n_parameters,
        seed=int(seed),
        maxiter=config.optimizer_max_iterations,
        popsize=config.optimizer_population_size,
        tol=config.optimizer_tolerance,
        polish=True,
        updating="immediate",
        workers=1,
    )
    basis = _parameters_to_basis(np.asarray(result.x, dtype=np.float64), dimensions)
    return basis, float(-result.fun)


def _match_basis(candidate: FloatArray, reference: FloatArray) -> FloatArray:
    rows, columns = linear_sum_assignment(-np.abs(candidate.T @ reference))
    matched = np.empty_like(reference)
    for row, column in zip(rows, columns, strict=True):
        vector = candidate[:, int(row)]
        if float(vector @ reference[:, int(column)]) < 0.0:
            vector = -vector
        matched[:, int(column)] = vector
    return np.asarray(matched, dtype=np.float64)


def _cosines(projector: FloatArray, basis: FloatArray) -> FloatArray:
    squared = np.einsum("ij,ik,jk->k", projector, basis, basis, optimize=True)
    return np.sqrt(np.clip(squared, 0.0, 1.0))


def _relative_margin(values: FloatArray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if len(ordered) == 1:
        return 1.0
    return float((ordered[-1] - ordered[-2]) / max(ordered[-1], 1e-12))


def _joint_projective_null_p(
    cosine: float,
    margin: float,
    *,
    dimensions: int,
    null_directions: FloatArray,
) -> float:
    # A complete orthonormal basis has the same isotropic null distribution in
    # every coordinate frame.  The leave-one-group-out basis is independent of
    # the held-out group under the null, so no fitted-axis correction is needed.
    absolute = np.abs(null_directions[:, :dimensions])
    ordered = np.sort(absolute, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2] if dimensions > 1 else np.zeros(len(ordered))
    relative = (maximum - second) / np.maximum(maximum, 1e-12)
    exceed = (maximum >= float(cosine)) & (relative >= float(margin))
    return float((1 + int(exceed.sum())) / (len(exceed) + 1))


def estimate_orthogonal_projective_axes(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    *,
    config: OrthogonalProjectiveAxisConfig | None = None,
) -> OrthogonalProjectiveAxisResult:
    """Fit a global observed-only projective basis and conservative axes."""

    cfg = config or OrthogonalProjectiveAxisConfig()
    selected, members, coordinates = _validate_groups(groups, membership, views)
    projectors = _group_projectors(selected, members, views, coordinates)
    mean_projectors = np.asarray(projectors.mean(axis=1), dtype=np.float64)
    weights = np.clip(
        selected["group_confidence"].to_numpy(dtype=np.float64),
        cfg.minimum_group_confidence,
        1.0,
    )
    basis, global_objective = _fit_basis(
        mean_projectors, weights, config=cfg, seed=cfg.random_seed
    )
    view_bases: list[FloatArray] = []
    for view_index in range(len(views)):
        view_basis, _ = _fit_basis(
            projectors[:, view_index],
            weights,
            config=cfg,
            seed=cfg.random_seed + 1_009 * (view_index + 1),
        )
        view_bases.append(_match_basis(view_basis, basis))
    leave_one_out_bases: list[FloatArray] = []
    for group_index in range(len(selected)):
        keep = np.arange(len(selected)) != group_index
        if int(keep.sum()) < 2:
            leave_one_out_bases.append(basis.copy())
            continue
        held_basis, _ = _fit_basis(
            mean_projectors[keep],
            weights[keep],
            config=cfg,
            seed=cfg.random_seed + 10_007 * (group_index + 1),
        )
        leave_one_out_bases.append(_match_basis(held_basis, basis))

    rng = np.random.default_rng(cfg.random_seed + 9_999_991)
    null_directions = rng.normal(size=(cfg.null_directions, len(coordinates)))
    null_directions /= np.linalg.norm(null_directions, axis=1, keepdims=True)
    rows: list[dict[str, Any]] = []
    raw_null_p: list[float] = []
    for group_index, group in enumerate(selected.itertuples(index=False)):
        full = _cosines(mean_projectors[group_index], basis)
        full_assignment = int(np.argmax(full))
        per_view_cosine: list[float] = []
        per_view_margin: list[float] = []
        per_view_assignment: list[int] = []
        for view_index, view_basis in enumerate(view_bases):
            values = _cosines(projectors[group_index, view_index], view_basis)
            per_view_assignment.append(int(np.argmax(values)))
            per_view_cosine.append(float(np.max(values)))
            per_view_margin.append(_relative_margin(values))
        held = _cosines(mean_projectors[group_index], leave_one_out_bases[group_index])
        held_assignment = int(np.argmax(held))
        held_cosine = float(np.max(held))
        held_margin = _relative_margin(held)
        null_p = _joint_projective_null_p(
            held_cosine,
            held_margin,
            dimensions=len(coordinates),
            null_directions=null_directions,
        )
        raw_null_p.append(null_p)
        rows.append(
            {
                "loading_group": str(group.loading_group),
                "modality": str(group.modality),
                "members": str(group.members),
                "group_confidence": float(group.group_confidence),
                "basis_axis_index": full_assignment,
                "full_absolute_cosine": float(full[full_assignment]),
                "full_relative_margin": _relative_margin(full),
                "median_view_absolute_cosine": float(np.median(per_view_cosine)),
                "lower_view_absolute_cosine": float(np.quantile(per_view_cosine, 0.10)),
                "view_assignment_support": float(
                    np.mean(np.asarray(per_view_assignment) == full_assignment)
                ),
                "median_view_relative_margin": float(np.median(per_view_margin)),
                "leave_one_out_axis_index": held_assignment,
                "leave_one_out_assignment_agrees": bool(
                    held_assignment == full_assignment
                ),
                "leave_one_out_absolute_cosine": held_cosine,
                "leave_one_out_relative_margin": held_margin,
                "leave_one_out_projective_null_p": null_p,
            }
        )
    evidence = pd.DataFrame(rows)
    evidence["leave_one_out_projective_null_q"] = _benjamini_hochberg(
        np.asarray(raw_null_p, dtype=np.float64)
    )
    evidence["passes_group_confidence"] = (
        evidence["group_confidence"] >= cfg.minimum_group_confidence
    )
    evidence["passes_full_cosine"] = (
        evidence["full_absolute_cosine"] >= cfg.minimum_full_absolute_cosine
    )
    evidence["passes_view_lower_cosine"] = (
        evidence["lower_view_absolute_cosine"]
        >= cfg.minimum_view_lower_absolute_cosine
    )
    evidence["passes_view_assignment_support"] = (
        evidence["view_assignment_support"] >= cfg.minimum_view_assignment_support
    )
    evidence["passes_view_margin"] = (
        evidence["median_view_relative_margin"] >= cfg.minimum_view_relative_margin
    )
    evidence["passes_leave_one_out_assignment"] = evidence[
        "leave_one_out_assignment_agrees"
    ]
    evidence["passes_leave_one_out_cosine"] = (
        evidence["leave_one_out_absolute_cosine"]
        >= cfg.minimum_leave_one_out_absolute_cosine
    )
    evidence["passes_leave_one_out_margin"] = (
        evidence["leave_one_out_relative_margin"]
        >= cfg.minimum_leave_one_out_relative_margin
    )
    evidence["passes_projective_null_q"] = (
        evidence["leave_one_out_projective_null_q"]
        <= cfg.maximum_leave_one_out_projective_null_q
    )
    pass_columns = [column for column in evidence if column.startswith("passes_")]
    evidence["group_assignment_selected"] = evidence[pass_columns].all(axis=1)
    evidence["abstention_reason"] = evidence.apply(
        lambda row: "" if bool(row["group_assignment_selected"]) else "failed:"
        + ",".join(
            column.removeprefix("passes_")
            for column in pass_columns
            if not bool(row[column])
        ),
        axis=1,
    )

    membership_lookup = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in members.groupby("loading_group")
    }
    axis_rows: list[dict[str, Any]] = []
    hyperedge_rows: list[dict[str, Any]] = []
    occupied = sorted(set(evidence["basis_axis_index"].astype(int)))
    for output_index, basis_axis in enumerate(occupied, start=1):
        assigned = evidence.loc[
            (evidence["basis_axis_index"] == basis_axis)
            & evidence["group_assignment_selected"]
        ].copy()
        loading_groups = sorted(assigned["loading_group"].astype(str))
        modalities = sorted(set(assigned["modality"].astype(str)))
        modality_conflict = len(modalities) != len(loading_groups)
        enough_modalities = len(modalities) >= cfg.minimum_axis_modalities
        selected_axis = bool(
            enough_modalities
            and (
                not cfg.one_group_per_modality_per_axis or not modality_conflict
            )
        )
        axis_id = f"orthogonal_axis_{output_index:03d}"
        features = sorted(
            feature
            for group_id in loading_groups
            for feature in membership_lookup[group_id]
        )
        reasons: list[str] = []
        if not enough_modalities:
            reasons.append("modalities")
        if cfg.one_group_per_modality_per_axis and modality_conflict:
            reasons.append("duplicate_modality")
        contains_rna = "rna" in modalities
        edge_type = (
            "cross_assay_shared_state_axis_with_rna"
            if selected_axis and contains_rna
            else "cross_assay_shared_state_axis"
            if selected_axis
            else "unresolved"
        )
        confidence = float(
            assigned[
                [
                    "group_confidence",
                    "lower_view_absolute_cosine",
                    "leave_one_out_absolute_cosine",
                ]
            ].to_numpy(dtype=float).min()
        ) if len(assigned) else 0.0
        row = {
            "axis_id": axis_id,
            "basis_axis_index": int(basis_axis),
            "loading_groups": ";".join(loading_groups),
            "modalities": ";".join(modalities),
            "features": ";".join(features),
            "n_groups": len(loading_groups),
            "n_modalities": len(modalities),
            "contains_rna": contains_rna,
            "selection_status": "selected" if selected_axis else "unresolved",
            "abstention_reason": "" if selected_axis else "failed:" + ",".join(reasons),
            "edge_type": edge_type,
            "axis_confidence": confidence,
        }
        for coordinate_index, coordinate in enumerate(coordinates):
            row[f"basis__{coordinate}"] = float(basis[coordinate_index, basis_axis])
        axis_rows.append(row)
        hyperedge_rows.append(
            {
                "hyperedge_id": f"axis__{axis_id}",
                "axis_id": axis_id,
                "edge_type": edge_type,
                "selection_status": row["selection_status"],
                "loading_groups": row["loading_groups"],
                "modalities": row["modalities"],
                "features": row["features"],
                "n_groups": row["n_groups"],
                "n_modalities": row["n_modalities"],
                "contains_rna": contains_rna,
                "confidence": confidence,
                "abstention_reason": row["abstention_reason"],
                "interpretation": (
                    "globally orthogonal groups share an estimated state-response axis; no direct edge or direction is identified"
                    if selected_axis
                    else "global projective assignment is unresolved and is not projected into the graph"
                ),
            }
        )

    basis_rows = []
    for basis_axis in range(len(coordinates)):
        row = {"basis_axis_index": basis_axis}
        for coordinate_index, coordinate in enumerate(coordinates):
            row[coordinate] = float(basis[coordinate_index, basis_axis])
        basis_rows.append(row)
    provenance = {
        "schema_version": "orthogonal_projective_axes_v4.v1",
        "method": "global_sign_invariant_projector_basis_with_view_and_leave_one_group_out_abstention",
        "estimand": "assay-local loading groups sharing an observed-background estimated state axis",
        "config": asdict(cfg),
        "n_embedding_dimensions": len(coordinates),
        "n_loading_views": len(views),
        "n_selected_input_groups": len(selected),
        "n_selected_group_assignments": int(evidence["group_assignment_selected"].sum()),
        "n_selected_axes": int(
            sum(row["selection_status"] == "selected" for row in axis_rows)
        ),
        "global_projective_objective": global_objective,
        "view_semantics": {
            "same_observed_cells_refit": True,
            "algorithmic_stability_repeats": True,
            "independent_worlds": False,
        },
        "information_boundary": {
            "consumed": [
                "selected_assay_local_loading_groups",
                "observed_feature_membership",
                "complete_observed_loading_views",
                "observed_embedding_dimension",
            ],
            "not_consumed": [
                "simulator_roles",
                "true_state_dimensions",
                "scorer",
                "paired_cells",
                "cross_assay_feature_correspondence",
                "external_data",
            ],
            "truth_or_scorer_used": False,
            "direct_causation_claimed": False,
            "cell_pairing_claimed": False,
        },
    }
    return OrthogonalProjectiveAxisResult(
        basis=pd.DataFrame(basis_rows),
        group_evidence=evidence.sort_values("loading_group").reset_index(drop=True),
        axis_candidates=pd.DataFrame(axis_rows),
        typed_axis_hyperedges=pd.DataFrame(hyperedge_rows),
        provenance=provenance,
    )


def write_orthogonal_projective_axes(
    result: OrthogonalProjectiveAxisResult,
    out_dir: Path | str,
    *,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.basis.to_csv(target / "projective_basis.tsv", sep="\t", index=False)
    result.group_evidence.to_csv(
        target / "group_assignment_evidence.tsv", sep="\t", index=False
    )
    result.axis_candidates.to_csv(target / "axis_candidates.tsv", sep="\t", index=False)
    result.typed_axis_hyperedges.to_csv(
        target / "typed_axis_hyperedges.tsv", sep="\t", index=False
    )
    provenance = dict(result.provenance)
    provenance["input_sha256"] = dict(input_hashes or {})
    (target / "axis_provenance.json").write_text(
        json.dumps(_json_ready(provenance), indent=2, sort_keys=True) + "\n"
    )
    files = [
        "projective_basis.tsv",
        "group_assignment_evidence.tsv",
        "axis_candidates.tsv",
        "typed_axis_hyperedges.tsv",
        "axis_provenance.json",
    ]
    manifest = {
        "schema_version": "orthogonal_projective_axes_v4.run.v1",
        "truth_or_scorer_used": False,
        "files": {
            name: {"sha256": _sha256(target / name), "size_bytes": (target / name).stat().st_size}
            for name in files
        },
    }
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--loading-view", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=71)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    group_path, groups = _safe_table(args.groups, "loading_groups.tsv")
    membership_path, membership = _safe_table(
        args.membership, "loading_group_membership.tsv"
    )
    views = [load_evidence_view(path) for path in args.loading_view]
    cfg = OrthogonalProjectiveAxisConfig(random_seed=args.random_seed)
    result = estimate_orthogonal_projective_axes(
        groups, membership, views, config=cfg
    )
    input_hashes = {
        str(group_path): _sha256(group_path),
        str(membership_path): _sha256(membership_path),
    }
    for view in views:
        for name, digest in view.input_hashes.items():
            input_hashes[f"{view.view_id}/{name}"] = str(digest)
    summary = write_orthogonal_projective_axes(
        result, args.out_dir, input_hashes=input_hashes
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OrthogonalProjectiveAxisConfig",
    "OrthogonalProjectiveAxisResult",
    "estimate_orthogonal_projective_axes",
    "write_orthogonal_projective_axes",
]
