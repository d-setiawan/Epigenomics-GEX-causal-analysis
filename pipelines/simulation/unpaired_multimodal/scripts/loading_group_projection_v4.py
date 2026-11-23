#!/usr/bin/env python3
"""Sparse loading-group projection for unpaired multi-assay state structure.

This layer consumes only the complete, observed-only outputs of
``cell_state_loading_groups.py``.  It first finds one coherent loading group
inside each chromatin assay, then compares group centroids across assays and
RNA.  Chromatin groups use mutual global matching, whereas each RNA group has
one asymmetric global best-peak budget.  Cross-assay evidence is represented
as a typed group/hyperedge first.  An ordinary pair edge is emitted only when
both the group match and its representative feature pair are unambiguous.

The procedure is deliberately conservative.  Similar loadings mean that
features respond to the same estimated biological-state axis; they do not
establish direct regulation, direction, or a cell correspondence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import betainc


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
    "role",
    "parent",
    "child",
    "latent",
    "oracle",
    "scorer",
    "correspondence",
    "pairing",
)
_SAFE_INPUT_LAYOUTS = (
    (
        "cell_loading_feature_models.tsv",
        "cell_loading_loadings.tsv",
        "cell_loading_pair_evidence.tsv",
    ),
    (
        "feature_state_models.tsv",
        "conditional_state_loadings.tsv",
        "loading_pair_evidence.tsv",
    ),
)
_COORDINATE_PATTERN = re.compile(r"^integration__dim([0-9]+)$")


@dataclass(frozen=True)
class LoadingProjectionConfig:
    """Frozen group-selection and conservative projection thresholds."""

    coherence_power: float = 3.0
    min_feature_loading_norm: float = 0.10
    min_feature_partial_state_fit: float = 0.0
    min_feature_direction_stability: float = 0.50
    min_positive_fit_view_support: float = 0.50
    min_within_absolute_cosine: float = 0.65
    min_group_score: float = 0.02
    min_group_view_support: float = 0.50
    min_group_top_rank_support: float = 0.50
    min_group_relative_margin: float = 0.025
    max_chromatin_groups_per_assay: int = 1
    cross_evidence_absolute_cosine: float = 0.65
    cross_projection_absolute_cosine: float = 0.90
    cross_projection_lower_cosine: float = 0.85
    cross_projection_view_support: float = 0.80
    cross_projection_empirical_q: float = 0.25
    cross_projection_relative_margin: float = 0.10
    representative_relative_margin: float = 0.05
    min_singleton_oof_partial_state_fit: float = 0.02
    min_singleton_lower_oof_partial_state_fit: float = 0.015
    min_singleton_loading_norm: float = 0.60
    min_singleton_minimum_loading_norm: float = 0.55
    min_singleton_direction_stability: float = 0.95
    min_singleton_minimum_direction_stability: float = 0.90
    min_singleton_positive_fit_view_support: float = 0.80
    allow_cross_assay_group_matching: bool = True

    def __post_init__(self) -> None:
        if self.coherence_power <= 0:
            raise ValueError("coherence_power must be positive")
        if self.min_feature_loading_norm < 0:
            raise ValueError("min_feature_loading_norm may not be negative")
        probabilities = (
            self.min_feature_direction_stability,
            self.min_positive_fit_view_support,
            self.min_within_absolute_cosine,
            self.min_group_view_support,
            self.min_group_top_rank_support,
            self.min_group_relative_margin,
            self.cross_evidence_absolute_cosine,
            self.cross_projection_absolute_cosine,
            self.cross_projection_lower_cosine,
            self.cross_projection_view_support,
            self.cross_projection_empirical_q,
            self.cross_projection_relative_margin,
            self.representative_relative_margin,
            self.min_singleton_direction_stability,
            self.min_singleton_minimum_direction_stability,
            self.min_singleton_positive_fit_view_support,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("probability, cosine, and margin thresholds must lie in [0, 1]")
        nonnegative = (
            self.min_group_score,
            self.min_singleton_oof_partial_state_fit,
            self.min_singleton_lower_oof_partial_state_fit,
            self.min_singleton_loading_norm,
            self.min_singleton_minimum_loading_norm,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("Score, fit, and loading-norm thresholds may not be negative")
        if self.max_chromatin_groups_per_assay < 1:
            raise ValueError("max_chromatin_groups_per_assay must be positive")
        if self.cross_projection_absolute_cosine < self.cross_evidence_absolute_cosine:
            raise ValueError("cross projection cosine must be at least the evidence cosine")


@dataclass(frozen=True)
class LoadingEvidenceView:
    view_id: str
    source_dir: Path | None
    feature_models: pd.DataFrame
    loadings: pd.DataFrame
    pair_evidence: pd.DataFrame
    input_hashes: Mapping[str, str]


@dataclass
class LoadingGroupProjectionResult:
    feature_consensus: pd.DataFrame
    within_group_candidates: pd.DataFrame
    groups: pd.DataFrame
    group_membership: pd.DataFrame
    group_match_evidence: pd.DataFrame
    typed_hyperedges: pd.DataFrame
    sparse_pair_projection: pd.DataFrame
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


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_COLUMN_TOKENS)
    )


def _validate_source_dir(path: Path | str) -> tuple[Path, tuple[str, str, str]]:
    lexical = Path(path).absolute()
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise ValueError("Loading evidence must be an existing direct directory")
    resolved = lexical.resolve(strict=True)
    forbidden = sorted(
        component.casefold()
        for component in resolved.parts
        if component.casefold() in _FORBIDDEN_PATH_COMPONENTS
    )
    if forbidden:
        raise ValueError(f"Refusing scorer/truth loading evidence path: {forbidden}")
    layouts = [layout for layout in _SAFE_INPUT_LAYOUTS if all((resolved / name).is_file() for name in layout)]
    if len(layouts) != 1:
        raise ValueError("Loading evidence directory must contain exactly one supported input layout")
    for filename in layouts[0]:
        if (resolved / filename).is_symlink():
            raise ValueError("Loading evidence files may not be symlinks")
    return resolved, layouts[0]


def _read_regular_bytes_once(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Could not open {label} safely: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_evidence_view(path: Path | str, *, view_id: str | None = None) -> LoadingEvidenceView:
    """Load one whitelisted cell-loading evidence bundle without other files."""

    source, layout = _validate_source_dir(path)
    payloads = {
        filename: _read_regular_bytes_once(source / filename, filename)
        for filename in layout
    }
    frames = [
        pd.read_csv(
            io.BytesIO(payloads[filename]),
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
        for filename in layout
    ]
    return make_evidence_view(
        view_id=view_id or source.name,
        feature_models=frames[0],
        loadings=frames[1],
        pair_evidence=frames[2],
        source_dir=source,
        input_hashes={
            filename: _bytes_sha256(payloads[filename]) for filename in layout
        },
    )


def make_evidence_view(
    *,
    view_id: str,
    feature_models: pd.DataFrame,
    loadings: pd.DataFrame,
    pair_evidence: pd.DataFrame,
    source_dir: Path | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> LoadingEvidenceView:
    """Validate an in-memory complete loading evidence view."""

    if not view_id or not str(view_id).strip():
        raise ValueError("view_id may not be empty")
    models = feature_models.copy()
    coefficient_table = loadings.copy()
    pairs = pair_evidence.copy()
    for name, frame in (
        ("feature_models", models),
        ("loadings", coefficient_table),
        ("pair_evidence", pairs),
    ):
        forbidden = _forbidden_columns(frame)
        if forbidden:
            raise ValueError(f"{name} contains forbidden columns {forbidden}")
    model_required = {
        "feature",
        "modality",
        "channel",
        "oof_partial_state_fit",
        "conditional_loading_norm",
        "loading_direction_stability",
    }
    loading_required = {
        "feature",
        "modality",
        "channel",
        "state_coordinate",
        "conditional_coefficient",
    }
    pair_required = {
        "pair_key",
        "node1",
        "node2",
        "modality1",
        "modality2",
        "same_assay",
        "absolute_loading_cosine",
        "node1_oof_partial_state_fit",
        "node2_oof_partial_state_fit",
        "node1_loading_stability",
        "node2_loading_stability",
        "pair_state_score",
    }
    for label, required, frame in (
        ("feature_models", model_required, models),
        ("loadings", loading_required, coefficient_table),
        ("pair_evidence", pair_required, pairs),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns {missing}")
    models["feature"] = models["feature"].astype(str)
    models["modality"] = models["modality"].astype(str)
    models["channel"] = models["channel"].astype(str)
    if models["feature"].duplicated().any() or len(models) < 2:
        raise ValueError("Feature models must contain unique features")
    if not set(models["modality"]).issubset({"rna"} | {m for m in models["modality"] if m.startswith("chrom_")}):
        raise ValueError("Modalities must be rna or chrom_<mark>")
    features = sorted(models["feature"])
    modalities = models.set_index("feature")["modality"].to_dict()
    for column in (
        "oof_partial_state_fit",
        "conditional_loading_norm",
        "loading_direction_stability",
    ):
        models[column] = pd.to_numeric(models[column], errors="raise")
    if not np.isfinite(models[list(model_required - {"feature", "modality", "channel"})].to_numpy(dtype=float)).all():
        raise ValueError("Feature model values must be finite")

    coefficient_table["feature"] = coefficient_table["feature"].astype(str)
    coefficient_table["state_coordinate"] = coefficient_table["state_coordinate"].astype(str)
    if set(coefficient_table["feature"]) != set(features):
        raise ValueError("Loadings and feature models must contain identical features")
    if coefficient_table.duplicated(["feature", "state_coordinate"]).any():
        raise ValueError("Each feature-coordinate loading must be unique")
    coordinates = sorted(set(coefficient_table["state_coordinate"]), key=_coordinate_sort_key)
    if len(coordinates) < 2 or any(
        _COORDINATE_PATTERN.fullmatch(coordinate) is None for coordinate in coordinates
    ):
        raise ValueError("Loadings require at least two integration__dim coordinates")
    expected_coordinates = [f"integration__dim{index}" for index in range(len(coordinates))]
    if coordinates != expected_coordinates:
        raise ValueError("Loading coordinates must be sequential")
    counts = coefficient_table.groupby("feature")["state_coordinate"].nunique()
    if not (counts == len(coordinates)).all():
        raise ValueError("Every feature must have every loading coordinate")
    coefficient_table["conditional_coefficient"] = pd.to_numeric(
        coefficient_table["conditional_coefficient"], errors="raise"
    )
    if not np.isfinite(coefficient_table["conditional_coefficient"]).all():
        raise ValueError("Loading coefficients must be finite")

    pairs["node1"] = pairs["node1"].astype(str)
    pairs["node2"] = pairs["node2"].astype(str)
    if len(pairs) != len(features) * (len(features) - 1) // 2:
        raise ValueError("Pair evidence must be complete")
    if pairs["pair_key"].duplicated().any() or not (pairs["node1"] < pairs["node2"]).all():
        raise ValueError("Pair evidence must be unique and canonical")
    expected_pairs = {
        (features[left], features[right])
        for left in range(len(features))
        for right in range(left + 1, len(features))
    }
    observed_pairs = set(zip(pairs["node1"], pairs["node2"], strict=False))
    if observed_pairs != expected_pairs:
        raise ValueError("Pair evidence does not contain every feature pair exactly once")
    for row in pairs.itertuples(index=False):
        if str(row.modality1) != modalities[str(row.node1)] or str(row.modality2) != modalities[str(row.node2)]:
            raise ValueError("Pair modality labels disagree with feature models")
        if bool(row.same_assay) != (str(row.modality1) == str(row.modality2)):
            raise ValueError("same_assay label disagrees with modalities")
    numeric_pairs = sorted(
        pair_required
        - {"pair_key", "node1", "node2", "modality1", "modality2", "same_assay"}
    )
    for column in numeric_pairs:
        pairs[column] = pd.to_numeric(pairs[column], errors="raise")
    if not np.isfinite(pairs[numeric_pairs].to_numpy(dtype=float)).all():
        raise ValueError("Pair evidence values must be finite")
    return LoadingEvidenceView(
        view_id=str(view_id),
        source_dir=source_dir,
        feature_models=models,
        loadings=coefficient_table,
        pair_evidence=pairs,
        input_hashes=dict(input_hashes or {}),
    )


def _coordinate_sort_key(value: str) -> tuple[int, str]:
    match = _COORDINATE_PATTERN.fullmatch(str(value))
    return (int(match.group(1)), str(value)) if match else (10**9, str(value))


def _validate_views(views: Sequence[LoadingEvidenceView]) -> tuple[list[str], list[str]]:
    if not views:
        raise ValueError("At least one loading evidence view is required")
    if len({view.view_id for view in views}) != len(views):
        raise ValueError("Loading evidence view IDs must be unique")
    reference_features = sorted(views[0].feature_models["feature"])
    reference_labels = views[0].feature_models.set_index("feature")[["modality", "channel"]].sort_index()
    reference_coordinates = sorted(
        set(views[0].loadings["state_coordinate"]), key=_coordinate_sort_key
    )
    for view in views[1:]:
        if sorted(view.feature_models["feature"]) != reference_features:
            raise ValueError("Every view must contain the same complete feature set")
        labels = view.feature_models.set_index("feature")[["modality", "channel"]].sort_index()
        if not labels.equals(reference_labels):
            raise ValueError("Observable assay labels must agree across views")
        coordinates = sorted(set(view.loadings["state_coordinate"]), key=_coordinate_sort_key)
        if coordinates != reference_coordinates:
            raise ValueError("State coordinates must agree across views")
    return reference_features, reference_coordinates


def _feature_consensus(
    views: Sequence[LoadingEvidenceView],
    config: LoadingProjectionConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for view in views:
        frame = view.feature_models[
            [
                "feature",
                "modality",
                "channel",
                "oof_partial_state_fit",
                "conditional_loading_norm",
                "loading_direction_stability",
            ]
        ].copy()
        frame["view_id"] = view.view_id
        rows.append(frame)
    stacked = pd.concat(rows, ignore_index=True)
    output = (
        stacked.groupby(["feature", "modality", "channel"], as_index=False)
        .agg(
            median_oof_partial_state_fit=("oof_partial_state_fit", "median"),
            lower_oof_partial_state_fit=("oof_partial_state_fit", lambda x: float(np.quantile(x, 0.10))),
            positive_fit_view_support=("oof_partial_state_fit", lambda x: float(np.mean(np.asarray(x) > config.min_feature_partial_state_fit))),
            median_loading_norm=("conditional_loading_norm", "median"),
            minimum_loading_norm=("conditional_loading_norm", "min"),
            median_direction_stability=("loading_direction_stability", "median"),
            minimum_direction_stability=("loading_direction_stability", "min"),
        )
        .sort_values(["modality", "feature"])
        .reset_index(drop=True)
    )
    output["feature_eligible"] = (
        (output["median_oof_partial_state_fit"] > config.min_feature_partial_state_fit)
        & (output["median_loading_norm"] >= config.min_feature_loading_norm)
        & (output["median_direction_stability"] >= config.min_feature_direction_stability)
        & (output["positive_fit_view_support"] >= config.min_positive_fit_view_support)
    )
    return output


def _within_candidates(
    views: Sequence[LoadingEvidenceView],
    feature_consensus: pd.DataFrame,
    config: LoadingProjectionConfig,
    forbidden_pairs: set[tuple[str, str]],
    permitted_pairs: set[tuple[str, str]] | None,
) -> pd.DataFrame:
    eligible = set(feature_consensus.loc[feature_consensus["feature_eligible"], "feature"])
    rows: list[pd.DataFrame] = []
    for view in views:
        frame = view.pair_evidence.loc[
            view.pair_evidence["same_assay"]
            & (view.pair_evidence["modality1"] != "rna")
        ].copy()
        frame["view_id"] = view.view_id
        frame["minimum_partial_state_fit"] = frame[
            ["node1_oof_partial_state_fit", "node2_oof_partial_state_fit"]
        ].min(axis=1).clip(lower=0.0)
        frame["minimum_loading_stability"] = frame[
            ["node1_loading_stability", "node2_loading_stability"]
        ].min(axis=1).clip(lower=0.0)
        frame["coherence_group_score"] = (
            frame["absolute_loading_cosine"].clip(0.0, 1.0) ** config.coherence_power
            * np.sqrt(frame["minimum_partial_state_fit"])
            * np.sqrt(frame["minimum_loading_stability"])
        )
        frame["observable_structure_pair_excluded"] = [
            tuple(sorted((str(left), str(right)))) in forbidden_pairs
            for left, right in zip(frame["node1"], frame["node2"], strict=True)
        ]
        frame["structure_anchor_pair_permitted"] = [
            permitted_pairs is None
            or tuple(sorted((str(left), str(right)))) in permitted_pairs
            for left, right in zip(frame["node1"], frame["node2"], strict=True)
        ]
        generic_feature_gate = frame["node1"].isin(eligible) & frame[
            "node2"
        ].isin(eligible)
        # A formally permitted pair has already passed the independent state
        # sibling and collider-parent gates.  The direct parent need not itself
        # pass state membership, so the old two-endpoint feature-fit gate must
        # not veto that identified decomposition.
        feature_gate = (
            generic_feature_gate
            if permitted_pairs is None
            else frame["structure_anchor_pair_permitted"]
        )
        frame["view_eligible"] = (
            feature_gate
            & (frame["absolute_loading_cosine"] >= config.min_within_absolute_cosine)
            & ~frame["observable_structure_pair_excluded"]
            & frame["structure_anchor_pair_permitted"]
        )
        frame["view_rank"] = 0
        for modality, indices in frame.groupby("modality1").groups.items():
            ordered = frame.loc[indices].sort_values(
                ["view_eligible", "coherence_group_score", "absolute_loading_cosine", "pair_key"],
                ascending=[False, False, False, True],
            )
            ranks = np.arange(1, len(ordered) + 1)
            frame.loc[ordered.index, "view_rank"] = ranks
        rows.append(frame)
    stacked = pd.concat(rows, ignore_index=True)
    grouped_rows: list[dict[str, Any]] = []
    for keys, group in stacked.groupby(
        ["pair_key", "node1", "node2", "modality1"], sort=True
    ):
        pair_key, node1, node2, modality = keys
        grouped_rows.append(
            {
                "pair_key": pair_key,
                "node1": node1,
                "node2": node2,
                "modality": modality,
                "n_views": len(group),
                "median_absolute_loading_cosine": float(group["absolute_loading_cosine"].median()),
                "lower_absolute_loading_cosine": float(group["absolute_loading_cosine"].quantile(0.10)),
                "median_minimum_partial_state_fit": float(group["minimum_partial_state_fit"].median()),
                "median_minimum_loading_stability": float(group["minimum_loading_stability"].median()),
                "median_coherence_group_score": float(group["coherence_group_score"].median()),
                "lower_coherence_group_score": float(group["coherence_group_score"].quantile(0.10)),
                "eligible_view_support": float(group["view_eligible"].mean()),
                "observable_structure_pair_excluded": bool(
                    group["observable_structure_pair_excluded"].all()
                ),
                "structure_anchor_pair_permitted": bool(
                    group["structure_anchor_pair_permitted"].all()
                ),
                "top_rank_view_support": float((group["view_rank"] == 1).mean()),
                "median_source_pair_state_score": float(group["pair_state_score"].median()),
            }
        )
    output = pd.DataFrame(grouped_rows)
    if output.empty:
        return output
    output["aggregate_rank"] = 0
    output["relative_score_margin"] = 0.0
    output["selected_group"] = False
    output["selection_status"] = "not_selected"
    output["selection_reason"] = "lower_rank"
    for modality, indices in output.groupby("modality").groups.items():
        ordered = output.loc[indices].sort_values(
            [
                "eligible_view_support",
                "median_coherence_group_score",
                "median_absolute_loading_cosine",
                "pair_key",
            ],
            ascending=[False, False, False, True],
        )
        output.loc[ordered.index, "aggregate_rank"] = np.arange(1, len(ordered) + 1)
        scores = ordered["median_coherence_group_score"].to_numpy(dtype=float)
        # A forbidden or otherwise ineligible pair is not a competing group
        # hypothesis and must not shrink the margin of the best eligible pair.
        scores = np.where(
            ordered["eligible_view_support"].to_numpy(dtype=float)
            >= config.min_group_view_support,
            scores,
            0.0,
        )
        for rank_index, row_index in enumerate(ordered.index):
            next_score = scores[rank_index + 1] if rank_index + 1 < len(scores) else 0.0
            output.loc[row_index, "relative_score_margin"] = max(
                float((scores[rank_index] - next_score) / max(scores[rank_index], 1e-12)),
                0.0,
            )
        if permitted_pairs is not None:
            permitted_indices = [
                row_index
                for row_index in ordered.index
                if tuple(
                    sorted(
                        (
                            str(output.loc[row_index, "node1"]),
                            str(output.loc[row_index, "node2"]),
                        )
                    )
                )
                in permitted_pairs
            ]
            if len(permitted_indices) > 1:
                raise ValueError(
                    "Structure-anchored projection permits more than one pair "
                    f"in modality {modality}"
                )
            output.loc[indices, ["selected_group", "selection_status", "selection_reason"]] = [
                False,
                "not_selected",
                "not_layered_structure_anchor",
            ]
            if permitted_indices:
                output.loc[
                    permitted_indices[0],
                    ["selected_group", "selection_status", "selection_reason"],
                ] = [
                    True,
                    "selected",
                    "layered_state_sibling_structure_anchor",
                ]
            continue
        for row_index in ordered.index[: config.max_chromatin_groups_per_assay]:
            row = output.loc[row_index]
            checks = {
                "score": row["median_coherence_group_score"] >= config.min_group_score,
                "cosine": row["median_absolute_loading_cosine"] >= config.min_within_absolute_cosine,
                "view_support": row["eligible_view_support"] >= config.min_group_view_support,
                "rank_support": row["top_rank_view_support"] >= config.min_group_top_rank_support,
                "margin": row["relative_score_margin"] >= config.min_group_relative_margin,
            }
            if all(checks.values()):
                output.loc[row_index, ["selected_group", "selection_status", "selection_reason"]] = [
                    True,
                    "selected",
                    "coherent_stable_assay_local_group",
                ]
            else:
                failed = ",".join(key for key, value in checks.items() if not value)
                output.loc[row_index, ["selection_status", "selection_reason"]] = [
                    "unresolved",
                    f"failed:{failed}",
                ]
    return output.sort_values(["modality", "aggregate_rank", "pair_key"]).reset_index(drop=True)


def _view_vectors(
    views: Sequence[LoadingEvidenceView], coordinates: Sequence[str]
) -> dict[tuple[str, str], FloatArray]:
    """Return vectors in the assay-local state-response metric when available."""

    vectors: dict[tuple[str, str], FloatArray] = {}
    for view in views:
        value_column = (
            "response_metric_coefficient"
            if "response_metric_coefficient" in view.loadings.columns
            else "conditional_coefficient"
        )
        pivot = view.loadings.pivot(
            index="feature", columns="state_coordinate", values=value_column
        )
        pivot = pivot[list(coordinates)]
        for feature, row in pivot.iterrows():
            vector = row.to_numpy(dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            vectors[(view.view_id, str(feature))] = vector / max(norm, 1e-12)
    return vectors


def _locally_ambiguous_group_features(
    candidates: pd.DataFrame,
    config: LoadingProjectionConfig,
) -> set[str]:
    """Find assay-local near-ties that must not become singleton seeds.

    This is an observed-only safeguard for the case in which several eligible
    features form an almost exchangeable coloading component.  A singleton
    bridge must not turn failure of the assay-local uniqueness gate into a
    positive cross-assay claim.  Explicit structure exclusions supplied by the
    caller remain the stronger, upstream safeguard.
    """

    ambiguous: set[str] = set()
    if candidates.empty:
        return ambiguous
    for _, frame in candidates.groupby("modality", sort=True):
        top = frame.loc[frame["aggregate_rank"] == 1]
        if len(top) != 1:
            continue
        best = top.iloc[0]
        if str(best["selection_reason"]) != "failed:margin":
            continue
        score = float(best["median_coherence_group_score"])
        qualifying = frame.loc[
            (frame["median_coherence_group_score"] >= config.min_group_score)
            & (frame["median_absolute_loading_cosine"] >= config.min_within_absolute_cosine)
            & (frame["eligible_view_support"] >= config.min_group_view_support)
            & (
                frame["median_coherence_group_score"]
                >= score * (1.0 - config.min_group_relative_margin)
            )
        ]
        nodes = set(qualifying["node1"].astype(str)) | set(
            qualifying["node2"].astype(str)
        )
        if len(qualifying) >= 2 and len(nodes) >= 3:
            ambiguous.update(nodes)
    return ambiguous


def _annotate_singleton_candidates(
    feature_consensus: pd.DataFrame,
    candidates: pd.DataFrame,
    config: LoadingProjectionConfig,
) -> pd.DataFrame:
    """Mark strict, ungrouped feature candidates before group construction."""

    output = feature_consensus.copy()
    ambiguous = _locally_ambiguous_group_features(candidates, config)
    output["locally_ambiguous_group_excluded"] = output["feature"].isin(ambiguous)
    output["singleton_bridge_eligible"] = (
        output["feature_eligible"]
        & (output["modality"] != "rna")
        & ~output["locally_ambiguous_group_excluded"]
        & (
            output["median_oof_partial_state_fit"]
            >= config.min_singleton_oof_partial_state_fit
        )
        & (
            output["lower_oof_partial_state_fit"]
            >= config.min_singleton_lower_oof_partial_state_fit
        )
        & (output["median_loading_norm"] >= config.min_singleton_loading_norm)
        & (
            output["minimum_loading_norm"]
            >= config.min_singleton_minimum_loading_norm
        )
        & (
            output["median_direction_stability"]
            >= config.min_singleton_direction_stability
        )
        & (
            output["minimum_direction_stability"]
            >= config.min_singleton_minimum_direction_stability
        )
        & (
            output["positive_fit_view_support"]
            >= config.min_singleton_positive_fit_view_support
        )
    )
    return output


def _build_groups(
    views: Sequence[LoadingEvidenceView],
    coordinates: Sequence[str],
    feature_consensus: pd.DataFrame,
    candidates: pd.DataFrame,
    config: LoadingProjectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], FloatArray]]:
    vectors = _view_vectors(views, coordinates)
    selected = candidates.loc[candidates["selected_group"]].copy()
    group_specs: list[
        tuple[str, str, list[str], pd.Series | None, str, str]
    ] = []
    for group_index, row in enumerate(selected.itertuples(index=False), start=1):
        group_specs.append(
            (
                f"loading_group_{group_index:03d}",
                str(row.modality),
                [str(row.node1), str(row.node2)],
                pd.Series(row._asdict()),
                "selected",
                "within_assay_coloading",
            )
        )
    rna_features = feature_consensus.loc[
        (feature_consensus["modality"] == "rna") & feature_consensus["feature_eligible"],
        "feature",
    ].astype(str)
    for rna_index, feature in enumerate(sorted(rna_features), start=1):
        group_specs.append(
            (
                f"rna_loading_{rna_index:03d}",
                "rna",
                [feature],
                None,
                "selected",
                "rna_singleton_state_response",
            )
        )

    selected_features = {
        feature for _, _, members, _, _, _ in group_specs for feature in members
    }
    singleton_features = feature_consensus.loc[
        feature_consensus["singleton_bridge_eligible"]
        & ~feature_consensus["feature"].isin(selected_features),
        ["feature", "modality"],
    ].sort_values(["modality", "feature"])
    for singleton_index, row in enumerate(singleton_features.itertuples(index=False), start=1):
        group_specs.append(
            (
                f"singleton_candidate_{singleton_index:03d}",
                str(row.modality),
                [str(row.feature)],
                None,
                "singleton_candidate",
                "ungrouped_stable_singleton_candidate",
            )
        )

    group_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    centroids: dict[tuple[str, str], FloatArray] = {}
    for (
        group_id,
        modality,
        members,
        candidate,
        selection_status,
        group_kind,
    ) in group_specs:
        view_centroids: list[FloatArray] = []
        member_cosines: list[float] = []
        for view in views:
            units = [vectors[(view.view_id, feature)] for feature in members]
            reference = units[0]
            aligned = [unit if float(np.dot(unit, reference)) >= 0 else -unit for unit in units]
            centroid = np.mean(np.vstack(aligned), axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            centroids[(view.view_id, group_id)] = centroid
            view_centroids.append(centroid)
            member_cosines.extend(abs(float(np.dot(unit, centroid))) for unit in units)
        reference = view_centroids[0]
        aligned_views = [
            centroid if float(np.dot(centroid, reference)) >= 0 else -centroid
            for centroid in view_centroids
        ]
        consensus = np.mean(np.vstack(aligned_views), axis=0)
        consensus /= max(float(np.linalg.norm(consensus)), 1e-12)
        view_distances = [
            min(
                float(np.linalg.norm(centroid - consensus)),
                float(np.linalg.norm(centroid + consensus)),
            )
            for centroid in view_centroids
        ]
        if candidate is None:
            selection_margin = 1.0
            top_rank_support = 1.0
            group_score = float(
                feature_consensus.set_index("feature").loc[members[0], "median_oof_partial_state_fit"]
            )
        else:
            selection_margin = float(candidate["relative_score_margin"])
            top_rank_support = float(candidate["top_rank_view_support"])
            group_score = float(candidate["median_coherence_group_score"])
        coherence = float(min(member_cosines)) if member_cosines else 0.0
        confidence = float(
            np.clip(
                coherence
                * (0.5 + 0.5 * top_rank_support)
                * (0.5 + 0.5 * min(selection_margin / 0.20, 1.0))
                * max(1.0 - np.mean(view_distances), 0.0),
                0.0,
                1.0,
            )
        )
        row: dict[str, Any] = {
            "loading_group": group_id,
            "modality": modality,
            "group_kind": group_kind,
            "n_members": len(members),
            "members": ";".join(sorted(members)),
            "group_score": group_score,
            "minimum_member_centroid_cosine": coherence,
            "top_rank_view_support": top_rank_support,
            "relative_selection_margin": selection_margin,
            "mean_centroid_view_distance": float(np.mean(view_distances)),
            "maximum_centroid_view_distance": float(np.max(view_distances)),
            "group_confidence": confidence,
            "selection_status": selection_status,
            "interpretation": (
                "strict observed-only singleton candidate; it is not a group or axis unless a coherent external assay anchor uniquely accepts it"
                if selection_status == "singleton_candidate"
                else "shared response to an estimated state axis; not direct causation"
            ),
        }
        for coordinate, value in zip(coordinates, consensus, strict=True):
            row[f"centroid__{coordinate.removeprefix('integration__')}"] = float(value)
        group_rows.append(row)
        for feature in members:
            per_view_distance: list[float] = []
            per_view_orientation: list[int] = []
            for view in views:
                unit = vectors[(view.view_id, feature)]
                centroid = centroids[(view.view_id, group_id)]
                sign = 1 if float(np.dot(unit, centroid)) >= 0 else -1
                per_view_orientation.append(sign)
                per_view_distance.append(float(np.linalg.norm(sign * unit - centroid)))
            membership_rows.append(
                {
                    "loading_group": group_id,
                    "feature": feature,
                    "modality": modality,
                    "orientation_to_group": 1 if sum(per_view_orientation) >= 0 else -1,
                    "orientation_view_support": float(
                        max(per_view_orientation.count(1), per_view_orientation.count(-1))
                        / len(per_view_orientation)
                    ),
                    "median_unit_distance_to_view_centroid": float(np.median(per_view_distance)),
                    "maximum_unit_distance_to_view_centroid": float(np.max(per_view_distance)),
                }
            )
    group_columns = [
        "loading_group",
        "modality",
        "group_kind",
        "n_members",
        "members",
        "group_score",
        "minimum_member_centroid_cosine",
        "top_rank_view_support",
        "relative_selection_margin",
        "mean_centroid_view_distance",
        "maximum_centroid_view_distance",
        "group_confidence",
        "selection_status",
        "interpretation",
        *[
            f"centroid__{coordinate.removeprefix('integration__')}"
            for coordinate in coordinates
        ],
    ]
    membership_columns = [
        "loading_group",
        "feature",
        "modality",
        "orientation_to_group",
        "orientation_view_support",
        "median_unit_distance_to_view_centroid",
        "maximum_unit_distance_to_view_centroid",
    ]
    return (
        pd.DataFrame(group_rows, columns=group_columns),
        pd.DataFrame(membership_rows, columns=membership_columns),
        centroids,
    )


def _projective_null_p(cosine: float, dimensions: int) -> float:
    squared = float(np.clip(cosine, 0.0, 1.0) ** 2)
    return float(1.0 - betainc(0.5, 0.5 * (dimensions - 1), squared))


def _group_matches(
    views: Sequence[LoadingEvidenceView],
    groups: pd.DataFrame,
    centroids: Mapping[tuple[str, str], FloatArray],
    dimensions: int,
    config: LoadingProjectionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected_groups = groups.loc[
        groups["selection_status"].astype(str).eq("selected")
    ]
    records = list(selected_groups.itertuples(index=False))
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if str(left.modality) == str(right.modality):
                continue
            cosines = np.asarray(
                [
                    abs(
                        float(
                            np.dot(
                                centroids[(view.view_id, str(left.loading_group))],
                                centroids[(view.view_id, str(right.loading_group))],
                            )
                        )
                    )
                    for view in views
                ],
                dtype=np.float64,
            )
            median_cosine = float(np.median(cosines))
            rows.append(
                {
                    "group_pair_key": f"{left.loading_group}|||{right.loading_group}",
                    "group1": str(left.loading_group),
                    "group2": str(right.loading_group),
                    "modality1": str(left.modality),
                    "modality2": str(right.modality),
                    "members1": str(left.members),
                    "members2": str(right.members),
                    "median_absolute_centroid_cosine": median_cosine,
                    "lower_absolute_centroid_cosine": float(np.quantile(cosines, 0.10)),
                    "minimum_absolute_centroid_cosine": float(np.min(cosines)),
                    "centroid_cosine_view_sd": float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0,
                    "cross_evidence_view_support": float(
                        np.mean(cosines >= config.cross_evidence_absolute_cosine)
                    ),
                    "cross_projection_view_support": float(
                        np.mean(cosines >= config.cross_projection_absolute_cosine)
                    ),
                    "projective_empirical_null_p": _projective_null_p(median_cosine, dimensions),
                    "group_confidence_support": float(
                        np.sqrt(float(left.group_confidence) * float(right.group_confidence))
                    ),
                }
            )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["projective_empirical_null_q"] = _benjamini_hochberg(
        output["projective_empirical_null_p"].to_numpy(dtype=np.float64)
    )
    is_peak_gene = (output["modality1"] == "rna") ^ (output["modality2"] == "rna")
    is_chromatin_pair = (output["modality1"] != "rna") & (output["modality2"] != "rna")

    # Chromatin groups and RNA singleton groups are different estimands.  A
    # chromatin--chromatin loading match must be a mutual global neighbor among
    # chromatin assays.  In contrast, requiring a candidate peak to prefer an
    # RNA group is inappropriate: the peak can be even closer to another peak
    # on the same state axis.  Each RNA group therefore receives one global
    # best-peak budget, ranked across all selected chromatin groups.
    output["group1_neighbor_rank"] = 0
    output["group2_neighbor_rank"] = 0
    output["group1_relative_best_margin"] = np.nan
    output["group2_relative_best_margin"] = np.nan

    chromatin_ranks: dict[tuple[str, str], int] = {}
    chromatin_margins: dict[str, float] = {}
    chromatin_matches = output.loc[is_chromatin_pair]
    chromatin_group_ids = sorted(
        set(chromatin_matches["group1"].astype(str))
        | set(chromatin_matches["group2"].astype(str))
    )
    for group_id in chromatin_group_ids:
        candidates = chromatin_matches.loc[
            (chromatin_matches["group1"] == group_id)
            | (chromatin_matches["group2"] == group_id)
        ].sort_values(
            ["median_absolute_centroid_cosine", "group_pair_key"],
            ascending=[False, True],
        )
        values = candidates["median_absolute_centroid_cosine"].to_numpy(dtype=float)
        chromatin_margins[group_id] = (
            float((values[0] - values[1]) / max(values[0], 1e-12))
            if len(values) > 1
            else 1.0
        )
        for rank, row in enumerate(candidates.itertuples(index=False), start=1):
            neighbor = row.group2 if row.group1 == group_id else row.group1
            chromatin_ranks[(group_id, str(neighbor))] = rank

    gene_ranks: dict[tuple[str, str], int] = {}
    gene_margins: dict[str, float] = {}
    gene_matches = output.loc[is_peak_gene]
    rna_group_ids = sorted(
        set(gene_matches.loc[gene_matches["modality1"] == "rna", "group1"].astype(str))
        | set(gene_matches.loc[gene_matches["modality2"] == "rna", "group2"].astype(str))
    )
    for gene_group in rna_group_ids:
        candidates = gene_matches.loc[
            (gene_matches["group1"] == gene_group)
            | (gene_matches["group2"] == gene_group)
        ].sort_values(
            ["median_absolute_centroid_cosine", "group_pair_key"],
            ascending=[False, True],
        )
        values = candidates["median_absolute_centroid_cosine"].to_numpy(dtype=float)
        gene_margins[gene_group] = (
            float((values[0] - values[1]) / max(values[0], 1e-12))
            if len(values) > 1
            else 1.0
        )
        for rank, row in enumerate(candidates.itertuples(index=False), start=1):
            peak_group = row.group2 if row.group1 == gene_group else row.group1
            gene_ranks[(gene_group, str(peak_group))] = rank

    for row_index, row in output.iterrows():
        left = str(row["group1"])
        right = str(row["group2"])
        if bool(is_chromatin_pair.loc[row_index]):
            output.at[row_index, "group1_neighbor_rank"] = chromatin_ranks[(left, right)]
            output.at[row_index, "group2_neighbor_rank"] = chromatin_ranks[(right, left)]
            output.at[row_index, "group1_relative_best_margin"] = chromatin_margins[left]
            output.at[row_index, "group2_relative_best_margin"] = chromatin_margins[right]
        elif bool(is_peak_gene.loc[row_index]):
            gene_group = left if str(row["modality1"]) == "rna" else right
            peak_group = right if gene_group == left else left
            gene_rank = gene_ranks[(gene_group, peak_group)]
            gene_margin = gene_margins[gene_group]
            if gene_group == left:
                output.at[row_index, "group1_neighbor_rank"] = gene_rank
                output.at[row_index, "group1_relative_best_margin"] = gene_margin
            else:
                output.at[row_index, "group2_neighbor_rank"] = gene_rank
                output.at[row_index, "group2_relative_best_margin"] = gene_margin

    output["mutual_best"] = is_chromatin_pair & (
        (output["group1_neighbor_rank"] == 1) & (output["group2_neighbor_rank"] == 1)
    )
    output["gene_peak_neighbor_rank"] = [
        (
            int(row.group1_neighbor_rank)
            if str(row.modality1) == "rna"
            else int(row.group2_neighbor_rank)
            if str(row.modality2) == "rna"
            else 0
        )
        for row in output.itertuples(index=False)
    ]
    output["gene_relative_best_margin"] = [
        (
            float(row.group1_relative_best_margin)
            if str(row.modality1) == "rna"
            else float(row.group2_relative_best_margin)
            if str(row.modality2) == "rna"
            else float("nan")
        )
        for row in output.itertuples(index=False)
    ]
    output["gene_global_best"] = is_peak_gene & (output["gene_peak_neighbor_rank"] == 1)
    output["selection_rule"] = np.where(
        is_peak_gene,
        "asymmetric_global_gene_best_peak",
        "mutual_global_chromatin_best",
    )

    evidence_supported = (
        (output["median_absolute_centroid_cosine"] >= config.cross_projection_absolute_cosine)
        & (output["lower_absolute_centroid_cosine"] >= config.cross_projection_lower_cosine)
        & (output["cross_projection_view_support"] >= config.cross_projection_view_support)
        & (output["projective_empirical_null_q"] <= config.cross_projection_empirical_q)
    )
    supported_chromatin = (
        is_chromatin_pair
        & evidence_supported
        & output["mutual_best"]
        & (output["group1_relative_best_margin"] >= config.cross_projection_relative_margin)
        & (output["group2_relative_best_margin"] >= config.cross_projection_relative_margin)
    )
    supported_peak_gene = (
        is_peak_gene
        & evidence_supported
        & output["gene_global_best"]
        & (output["gene_relative_best_margin"] >= config.cross_projection_relative_margin)
    )
    supported = supported_chromatin | supported_peak_gene
    evidence = output["median_absolute_centroid_cosine"] >= config.cross_evidence_absolute_cosine
    output["selection_status"] = np.where(
        supported,
        "selected",
        np.where(evidence, "unresolved", "unsupported"),
    )
    output["edge_type"] = np.select(
        [supported_peak_gene, supported_chromatin],
        ["prior_constrained_peak_gene", "cross_assay_shared_state"],
        default="unresolved",
    )
    output["group_match_confidence"] = (
        output["median_absolute_centroid_cosine"]
        * output["cross_evidence_view_support"]
        * output["group_confidence_support"]
    ).clip(0.0, 1.0)
    return output.sort_values(
        ["selection_status", "median_absolute_centroid_cosine", "group_pair_key"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _pair_consensus(views: Sequence[LoadingEvidenceView]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for view in views:
        frame = view.pair_evidence[
            ["pair_key", "node1", "node2", "pair_state_score", "absolute_loading_cosine"]
        ].copy()
        rows.append(frame)
    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["pair_key", "node1", "node2"], as_index=False)
        .agg(
            median_pair_state_score=("pair_state_score", "median"),
            median_absolute_loading_cosine=("absolute_loading_cosine", "median"),
        )
    )


def _typed_outputs(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    matches: pd.DataFrame,
    pair_consensus: pd.DataFrame,
    config: LoadingProjectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hyperedges: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    selected_groups = groups.loc[
        groups["selection_status"].astype(str).eq("selected")
    ]
    for group in selected_groups.itertuples(index=False):
        members = str(group.members).split(";")
        hyperedges.append(
            {
                "hyperedge_id": f"within__{group.loading_group}",
                "edge_type": "within_assay_shared_state_group",
                "selection_status": "selected",
                "group1": str(group.loading_group),
                "group2": "",
                "modality1": str(group.modality),
                "modality2": "",
                "members": str(group.members),
                "n_members": len(members),
                "confidence": float(group.group_confidence),
                "interpretation": "features share an assay-local estimated state-response axis",
            }
        )
        if len(members) == 2:
            left, right = sorted(members)
            projections.append(
                {
                    "pair_key": f"{left}|||{right}",
                    "node1": left,
                    "node2": right,
                    "edge_type": "within_assay_shared_state",
                    "selection_status": "selected",
                    "projection_source": "assay_local_loading_group",
                    "group1": str(group.loading_group),
                    "group2": "",
                    "confidence": float(group.group_confidence),
                    "representative_relative_margin": 1.0,
                    "interpretation": "shared state association; not a direct causal edge",
                }
            )
    group_members = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in membership.groupby("loading_group")
    }
    pair_lookup = pair_consensus.set_index(["node1", "node2"])
    for match in matches.itertuples(index=False):
        if str(match.selection_status) not in {"selected", "unresolved"}:
            continue
        members = sorted(group_members[str(match.group1)] + group_members[str(match.group2)])
        hyperedges.append(
            {
                "hyperedge_id": f"cross__{match.group1}__{match.group2}",
                "edge_type": str(match.edge_type),
                "selection_status": str(match.selection_status),
                "group1": str(match.group1),
                "group2": str(match.group2),
                "modality1": str(match.modality1),
                "modality2": str(match.modality2),
                "members": ";".join(members),
                "n_members": len(members),
                "confidence": float(match.group_match_confidence),
                "interpretation": (
                    "RNA group is linked to its global best peak group; the gene relation is prior-constrained, not directed by the data"
                    if str(match.selection_status) == "selected"
                    and str(match.edge_type) == "prior_constrained_peak_gene"
                    else "group centroids share an estimated state-response axis"
                    if str(match.selection_status) == "selected"
                    else "centroid similarity is insufficiently unique for pair projection"
                ),
            }
        )
        if str(match.selection_status) != "selected":
            continue
        feature_candidates: list[tuple[float, float, str, str]] = []
        for left in group_members[str(match.group1)]:
            for right in group_members[str(match.group2)]:
                node1, node2 = sorted((left, right))
                row = pair_lookup.loc[(node1, node2)]
                feature_candidates.append(
                    (
                        float(row["median_pair_state_score"]),
                        float(row["median_absolute_loading_cosine"]),
                        node1,
                        node2,
                    )
                )
        feature_candidates.sort(reverse=True)
        best = feature_candidates[0]
        second_score = feature_candidates[1][0] if len(feature_candidates) > 1 else 0.0
        margin = float((best[0] - second_score) / max(best[0], 1e-12))
        if margin < config.representative_relative_margin:
            continue
        projections.append(
            {
                "pair_key": f"{best[2]}|||{best[3]}",
                "node1": best[2],
                "node2": best[3],
                "edge_type": str(match.edge_type),
                "selection_status": "selected",
                "projection_source": (
                    "asymmetric_global_gene_best_peak_group"
                    if str(match.edge_type) == "prior_constrained_peak_gene"
                    else "unambiguous_group_centroid_match"
                ),
                "group1": str(match.group1),
                "group2": str(match.group2),
                "confidence": float(match.group_match_confidence),
                "representative_relative_margin": margin,
                "interpretation": (
                    "sparse peak-gene representative; gene relation is prior-constrained, not learned"
                    if str(match.edge_type) == "prior_constrained_peak_gene"
                    else "sparse representative of a group-level shared-state match"
                ),
            }
        )
    hyperedge_columns = [
        "hyperedge_id",
        "edge_type",
        "selection_status",
        "group1",
        "group2",
        "modality1",
        "modality2",
        "members",
        "n_members",
        "confidence",
        "interpretation",
    ]
    projection_columns = [
        "pair_key",
        "node1",
        "node2",
        "edge_type",
        "selection_status",
        "projection_source",
        "group1",
        "group2",
        "confidence",
        "representative_relative_margin",
        "interpretation",
    ]
    return (
        pd.DataFrame(hyperedges, columns=hyperedge_columns),
        pd.DataFrame(projections, columns=projection_columns).drop_duplicates(
            ["pair_key", "edge_type"], keep="first"
        ),
    )


def estimate_loading_group_projection(
    views: Sequence[LoadingEvidenceView],
    config: LoadingProjectionConfig | None = None,
    *,
    excluded_features: Sequence[str] = (),
    forbidden_within_pairs: Sequence[tuple[str, str]] = (),
    permitted_within_pairs: Sequence[tuple[str, str]] | None = None,
) -> LoadingGroupProjectionResult:
    """Infer state-response groups and a conservative typed projection.

    ``excluded_features``, ``forbidden_within_pairs``, and
    ``permitted_within_pairs`` are optional observable structure constraints.
    Collider children are excluded because their response mixes their parents.
    The pair formed by the two selected parents of one collider is ineligible
    as a state-only group.  When ``permitted_within_pairs`` is supplied, only
    pairs already selected by the independent state-sibling/collider-parent
    decomposition may form chromatin groups.  None of these constraints uses
    semantic roles or truth.
    """

    cfg = config or LoadingProjectionConfig()
    _, coordinates = _validate_views(views)
    feature_consensus = _feature_consensus(views, cfg)
    excluded = tuple(sorted(set(map(str, excluded_features))))
    unknown_excluded = sorted(set(excluded) - set(feature_consensus["feature"].astype(str)))
    if unknown_excluded:
        raise ValueError(f"Excluded collider features are absent from loading evidence: {unknown_excluded}")
    feature_consensus["direct_collider_excluded"] = feature_consensus["feature"].isin(
        excluded
    )
    feature_consensus["feature_eligible"] = (
        feature_consensus["feature_eligible"]
        & ~feature_consensus["direct_collider_excluded"]
    )
    known_features = set(feature_consensus["feature"].astype(str))
    forbidden_pairs: set[tuple[str, str]] = set()
    for raw_pair in forbidden_within_pairs:
        if len(raw_pair) != 2:
            raise ValueError("Each forbidden within-assay pair must have two nodes")
        left, right = sorted(map(str, raw_pair))
        if left == right:
            raise ValueError("A forbidden within-assay pair cannot be a self-pair")
        if left not in known_features or right not in known_features:
            raise ValueError(
                "Forbidden direct-parent pair is absent from loading evidence: "
                f"{(left, right)}"
            )
        forbidden_pairs.add((left, right))
    permitted_pairs: set[tuple[str, str]] | None = None
    if permitted_within_pairs is not None:
        permitted_pairs = set()
        for raw_pair in permitted_within_pairs:
            if len(raw_pair) != 2:
                raise ValueError("Each permitted within-assay pair must have two nodes")
            left, right = sorted(map(str, raw_pair))
            if left == right:
                raise ValueError("A permitted within-assay pair cannot be a self-pair")
            if left not in known_features or right not in known_features:
                raise ValueError(
                    "Permitted structure-anchored pair is absent from loading evidence: "
                    f"{(left, right)}"
                )
            if (left, right) in forbidden_pairs:
                raise ValueError(
                    "A structure-anchored pair cannot also be structurally forbidden"
                )
            permitted_pairs.add((left, right))
    candidates = _within_candidates(
        views, feature_consensus, cfg, forbidden_pairs, permitted_pairs
    )
    feature_consensus = _annotate_singleton_candidates(
        feature_consensus, candidates, cfg
    )
    groups, membership, centroids = _build_groups(
        views, coordinates, feature_consensus, candidates, cfg
    )
    matches = (
        _group_matches(views, groups, centroids, len(coordinates), cfg)
        if cfg.allow_cross_assay_group_matching
        else pd.DataFrame()
    )
    hyperedges, projection = _typed_outputs(
        groups, membership, matches, _pair_consensus(views), cfg
    )
    provenance = {
        "schema_version": "loading_group_projection_v4.v7",
        "estimand": "typed loading-group response to an observed-background estimated state axis",
        "config": asdict(cfg),
        "view_ids": [view.view_id for view in views],
        "input_file_sha256": {
            view.view_id: dict(view.input_hashes) for view in views
        },
        "coordinate_columns": list(coordinates),
        "within_assay_similarity_metric": (
            "correlation_of_fitted_state_response_scores"
            if all(
                "response_metric_coefficient" in view.loadings.columns
                for view in views
            )
            else "raw_coefficient_cosine_legacy_fallback"
        ),
        "within_assay_similarity_coordinate_invariant": all(
            "response_metric_coefficient" in view.loadings.columns for view in views
        ),
        "cross_assay_group_matching_enabled": bool(
            cfg.allow_cross_assay_group_matching
        ),
        "n_views": len(views),
        "view_semantics": {
            "same_observed_cells_refit": True,
            "views_are_algorithmic_stability_repeats": True,
            "views_are_independent_worlds": False,
            "view_support_not_world_sample_size": True,
        },
        "structure_anchor_semantics": {
            "selection_basis": "layered_state_sibling_structure_anchor",
            "formal_cross_assay_or_assay_local_state_tier_allowed": True,
            "assay_local_tier_does_not_claim_cross_assay_membership": True,
        },
        "n_features": len(feature_consensus),
        "n_selected_groups": int(
            groups["selection_status"].astype(str).eq("selected").sum()
        ),
        "n_singleton_bridge_candidates": int(
            groups["selection_status"].astype(str).eq("singleton_candidate").sum()
        ),
        "n_selected_cross_group_matches": int(
            (matches.get("selection_status", pd.Series(dtype=str)) == "selected").sum()
        ),
        "n_selected_chromatin_group_matches": int(
            (
                (matches.get("selection_status", pd.Series(dtype=str)) == "selected")
                & (matches.get("edge_type", pd.Series(dtype=str)) == "cross_assay_shared_state")
            ).sum()
        ),
        "n_selected_prior_peak_gene_matches": int(
            (
                (matches.get("selection_status", pd.Series(dtype=str)) == "selected")
                & (matches.get("edge_type", pd.Series(dtype=str)) == "prior_constrained_peak_gene")
            ).sum()
        ),
        "n_unresolved_cross_group_matches": int(
            (matches.get("selection_status", pd.Series(dtype=str)) == "unresolved").sum()
        ),
        "n_sparse_pair_projections": len(projection),
        "excluded_direct_collider_features": list(excluded),
        "excluded_observable_structure_features": list(excluded),
        "forbidden_observable_direct_parent_pairs": [
            list(pair) for pair in sorted(forbidden_pairs)
        ],
        "permitted_structure_anchored_within_pairs": (
            None
            if permitted_pairs is None
            else [list(pair) for pair in sorted(permitted_pairs)]
        ),
        "input_policy": {
            "complete_cell_loading_evidence_only": True,
            "observable_assay_labels_used": True,
            "raw_cells_or_metacells_read": False,
            "cell_correspondence_used": False,
            "truth_scorer_roles_or_dag_read": False,
            "external_data_or_priors_used": False,
            "direct_collider_children_excluded_from_state_groups": bool(excluded),
            "direct_collider_parent_pairs_excluded_from_state_groups": bool(
                forbidden_pairs
            ),
            "layered_state_sibling_structure_gate_applied": (
                permitted_pairs is not None
            ),
            "observable_structure_ambiguity_exclusions_respected_by_singletons": True,
            "locally_ambiguous_coloading_components_excluded_from_singletons": True,
            "observable_assay_peak_gene_type_constraint_used": True,
            "cross_assay_group_matching_suppressed_for_unaligned_local_coordinates": (
                not cfg.allow_cross_assay_group_matching
            ),
            "same_observed_cells_refit_across_views": True,
            "correlated_view_support_used_only_for_algorithmic_stability": True,
        },
        "identifiability": {
            "group_match_is_direct_causation": False,
            "cross_assay_direction_identified": False,
            "pair_projection_is_a_sparse_visualization_of_group_evidence": True,
            "ambiguous_cross_group_matches_are_unresolved": True,
            "rna_group_uses_asymmetric_global_best_peak_budget": True,
            "gene_relation_is_prior_constrained_not_learned": True,
            "selected_group_match_attached_post_consensus_once": True,
            "singleton_candidate_is_not_a_selected_local_group": True,
            "singleton_candidate_cannot_seed_an_axis": True,
        },
    }
    return LoadingGroupProjectionResult(
        feature_consensus=feature_consensus,
        within_group_candidates=candidates,
        groups=groups,
        group_membership=membership,
        group_match_evidence=matches,
        typed_hyperedges=hyperedges,
        sparse_pair_projection=projection,
        provenance=provenance,
    )


def run_loading_group_projection(
    evidence_dirs: Sequence[Path | str],
    out_dir: Path | str,
    config: LoadingProjectionConfig | None = None,
) -> dict[str, Any]:
    """Load safe evidence views, estimate the projection, and write artifacts."""

    views = [load_evidence_view(path) for path in evidence_dirs]
    output = Path(out_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    source_dirs = {view.source_dir for view in views}
    resolved = output.resolve(strict=False)
    if resolved in source_dirs:
        raise ValueError("Output must differ from loading evidence inputs")
    output.mkdir(parents=True, exist_ok=True)
    result = estimate_loading_group_projection(views, config)
    tables = {
        "feature_consensus.tsv": result.feature_consensus,
        "within_group_candidates.tsv": result.within_group_candidates,
        "loading_groups.tsv": result.groups,
        "loading_group_membership.tsv": result.group_membership,
        "group_match_evidence.tsv": result.group_match_evidence,
        "typed_group_hyperedges.tsv": result.typed_hyperedges,
        "sparse_pair_projection.tsv": result.sparse_pair_projection,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, sep="\t", index=False)
    (output / "projection_provenance.json").write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": "loading_group_projection_v4.run.v3",
        "n_views": len(views),
        "n_groups": len(result.groups),
        "n_selected_cross_group_matches": int(
            (result.group_match_evidence.get("selection_status", pd.Series(dtype=str)) == "selected").sum()
        ),
        "n_unresolved_cross_group_matches": int(
            (result.group_match_evidence.get("selection_status", pd.Series(dtype=str)) == "unresolved").sum()
        ),
        "n_sparse_pair_projections": len(result.sparse_pair_projection),
        "truth_or_scorer_used": False,
        "files_written": [*tables, "projection_provenance.json", "run_manifest.json"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-group-relative-margin", type=float, default=0.025)
    parser.add_argument("--cross-projection-absolute-cosine", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = LoadingProjectionConfig(
        min_group_relative_margin=args.min_group_relative_margin,
        cross_projection_absolute_cosine=args.cross_projection_absolute_cosine,
    )
    summary = run_loading_group_projection(args.evidence_dir, args.out_dir, config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "LoadingEvidenceView",
    "LoadingGroupProjectionResult",
    "LoadingProjectionConfig",
    "estimate_loading_group_projection",
    "load_evidence_view",
    "make_evidence_view",
    "run_loading_group_projection",
]
