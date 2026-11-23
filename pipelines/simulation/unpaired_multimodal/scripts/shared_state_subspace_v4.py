#!/usr/bin/env python3
"""Rotation-invariant shared-state-subspace discovery for disjoint assays.

This fallback deliberately stops below exact latent-axis matching.  It asks
which focal variables reproducibly respond to an assay-local background-state
*subspace*, then emits one multi-assay ``cross_assay_shared_state_subspace``
hyperedge.  It never pairs features across assays, assigns a common coordinate
axis, or claims direct causal direction.

In addition to strict feature membership, the result exposes a separate
assay-level state-participation table.  A predeclared within-assay omnibus
combines either one Bonferroni-confirmed feature or an order-two
partial-conjunction signal.  Every witness in the winning branch must pass all
six model views.  A structurally singleton assay uses only its one available
route.  Bonferroni combination within assay followed by Holm adjustment over
every observed assay controls the participation-family error rate without
assuming feature- or cross-assay independence.  This assay-only estimand never
promotes a feature into the strict membership table or silently changes the
existing feature-derived hyperedge.

Primary feature evidence is a cross-fitted orthogonal-score Wald test for a
conditional-mean response to the depth-orthogonal assay-local state subspace.
Flexible depth splines estimate both response and joint state nuisances.  A
centered Rademacher multiplier bootstrap, synchronized across features and
assays, calibrates an assay-admission gate, diagnostic feature maxT, and the
conservative multi-assay conjunction.  Within admitted assays, BH and four of
six deterministic cross-fit views select candidate members.  Coordinates are
scalar-normalized per assay, so every formal statistic is invariant to an
assay-specific orthogonal rotation of its state basis.

Chromatin variables below the predeclared event/non-event floor are excluded
before every response fit and multiplicity family.  They remain in the output
as unresolved rows with finite, auditable placeholder diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

import cell_state_loading_groups as loading_groups
import assay_local_state_embedding_v4 as assay_local_embedding
FloatArray = NDArray[np.float64]
_FORBIDDEN_PATH_PARTS = {"scorer", "truth", "ground_truth", "oracle", "counterfactual"}
_DESCRIPTIVE_DECOMPOSITION_FIELDS = (
    "oof_marginal_state_fit",
    "oof_depth_only_fit",
    "oof_joint_fit",
    "oof_partial_state_fit",
)
_DIRECT_CANDIDATE_COLUMNS = (
    "source_assay",
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "edge_type",
    "orientation_source",
    "collider_center",
    "component_id",
    "wedge_score",
    "runner_up_wedge_score",
    "wedge_margin",
    "selected_vote_fraction",
    "mean_empirical_p_positive",
    "mean_pooled_log_odds",
    "interpretation",
)


@dataclass(frozen=True)
class SharedStateSubspaceConfig:
    """Frozen empirical-null and stability selection settings."""

    null_repetitions: int = 999
    random_seed: int = 0
    maximum_empirical_fdr: float = 0.10
    maximum_hyperedge_error_probability: float = 0.10
    minimum_conditional_wald_views: int = 4
    minimum_median_score_direction_stability: float = 0.80
    unstable_feature_maxT_fraction: float = 0.50
    within_assay_strong_core_maxT_p: float = 0.01
    within_assay_strong_core_minimum_members: int = 4
    minimum_modalities: int = 2
    observed_model_views: int = 6
    folds: int = 5
    nuisance_spline_knots: int = 12
    nuisance_spline_degree: int = 3
    nuisance_ridge_alpha: float = 0.0001
    nuisance_logistic_c: float = 0.1
    score_covariance_ridge: float = 1e-6
    minimum_binary_events: int = 20
    minimum_binary_nonevents: int = 20
    assay_state_participation_order: int = 2
    maximum_assay_state_participation_error_probability: float = 0.10
    minimum_assay_state_participation_views: int = 6

    def __post_init__(self) -> None:
        if self.null_repetitions < 199:
            raise ValueError("null_repetitions must be at least 199")
        if self.random_seed < 0 or self.folds < 2:
            raise ValueError("random_seed/folds are invalid")
        for value in (
            self.maximum_empirical_fdr,
            self.maximum_hyperedge_error_probability,
            self.minimum_median_score_direction_stability,
            self.unstable_feature_maxT_fraction,
            self.within_assay_strong_core_maxT_p,
            self.maximum_assay_state_participation_error_probability,
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError("FDR and stability thresholds must lie in (0, 1]")
        if self.minimum_modalities < 2:
            raise ValueError("minimum_modalities must be at least two")
        if self.within_assay_strong_core_minimum_members < 1:
            raise ValueError(
                "within_assay_strong_core_minimum_members must be positive"
            )
        if self.observed_model_views < 2:
            raise ValueError("At least two conditional-score model views are required")
        if not 1 <= self.minimum_conditional_wald_views <= self.observed_model_views:
            raise ValueError("Conditional-Wald view count is outside the fitted views")
        if self.nuisance_spline_knots < 4 or self.nuisance_spline_degree < 1:
            raise ValueError("Spline nuisance model is too small")
        if min(
            self.nuisance_ridge_alpha,
            self.nuisance_logistic_c,
            self.score_covariance_ridge,
        ) <= 0.0:
            raise ValueError("Nuisance/score regularization must be positive")
        if min(self.minimum_binary_events, self.minimum_binary_nonevents) < 10:
            raise ValueError("Binary conditional score needs a nontrivial event floor")
        if self.assay_state_participation_order < 2:
            raise ValueError("Replicated assay participation needs order at least two")
        if not 1 <= self.minimum_assay_state_participation_views <= (
            self.observed_model_views
        ):
            raise ValueError("Assay-participation view count is outside fitted views")


@dataclass
class SharedStateSubspaceResult:
    feature_membership: pd.DataFrame
    typed_subspace_hyperedges: pd.DataFrame
    assay_state_participation: pd.DataFrame
    provenance: dict[str, Any]


@dataclass
class _OrthogonalScoreFit:
    statistic: float
    score_norm: float
    direction_stability: float
    centered_scores: FloatArray
    inverse_score_covariance: FloatArray


@dataclass
class _AssayScoreNuisance:
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]]
    train_bases: list[FloatArray]
    test_bases: list[FloatArray]
    residual_state: FloatArray


def _feature_event_eligibility(
    modality: str,
    assay: loading_groups.AssayData,
    feature: str,
    config: SharedStateSubspaceConfig,
) -> tuple[int, int, bool]:
    """Return observed binary counts and the pre-fit formal-test gate.

    RNA responses are continuous and therefore use the existing ``-1``
    sentinels.  Chromatin eligibility is decided entirely from the observed
    response before fitting any response, descriptive, bootstrap, or
    multiplicity model.  This makes constant and ultra-sparse features a safe
    abstention rather than a logistic-regression failure or a member of a null
    family it cannot validly enter.
    """

    if modality == "rna":
        return -1, -1, True
    response = assay.focal[str(feature)].to_numpy(dtype=np.float64)
    n_events = int(np.rint(response.sum()))
    n_nonevents = int(len(response) - n_events)
    eligible = bool(
        n_events >= config.minimum_binary_events
        and n_nonevents >= config.minimum_binary_nonevents
    )
    return n_events, n_nonevents, eligible


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


def _stable_seed(base_seed: int, *labels: str) -> int:
    value = "\0".join((str(base_seed), *labels))
    digest = hashlib.sha256(f"shared-state-subspace-v4\0{value}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def load_observable_direct_layer_overlap(
    path: Path | str,
) -> tuple[tuple[str, ...], str]:
    """Load centers from frozen, truth-blind direct-layer evidence."""

    lexical = Path(path).absolute()
    allowed_names = {
        "motif_provenance.json",
        "direct_layer_overlap_evidence.json",
    }
    if lexical.name not in allowed_names or lexical.is_symlink() or not lexical.is_file():
        raise ValueError("Direct-layer evidence must be a supported JSON artifact")
    resolved = lexical.resolve(strict=True)
    if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in resolved.parts):
        raise ValueError("Refusing scorer/truth direct-layer evidence")
    evidence = json.loads(resolved.read_text())
    if evidence.get("truth_or_scorer_used") is not False:
        raise ValueError("Direct-layer evidence is not explicitly truth-blind")
    if resolved.name == "direct_layer_overlap_evidence.json":
        if evidence.get("schema_version") != (
            "shared_state_subspace_v4.direct_layer_overlap.v1"
        ):
            raise ValueError("Unsupported direct-layer overlap evidence schema")
        centers_raw = evidence.get("selected_features", [])
        source_hash = str(evidence.get("direct_candidates_sha256", ""))
        if len(source_hash) != 64:
            raise ValueError("Frozen direct-layer evidence lacks its source-table hash")
    else:
        centers_raw = evidence.get("selected_collider_centers", [])
    centers = tuple(sorted(str(value) for value in centers_raw))
    if not centers or len(set(centers)) != len(centers):
        raise ValueError("Direct-layer evidence lacks unique collider centers")
    return centers, _sha256(resolved)


def freeze_observable_direct_layer_overlap(
    direct_candidates_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Validate and freeze benchmark collider centers before any scorer read.

    Standard benchmark ``motif_provenance.json`` files do not currently carry
    a truth-boundary flag or the selected-center list.  This explicit adapter
    accepts only the narrowly typed ``direct_candidates.tsv`` table, validates
    that each component is an observed two-arm collider, and writes the exact
    annotation artifact consumed by this module.  This annotation never alters
    subspace testing or selection; a collider child may belong to both layers.
    """

    source_lexical = Path(direct_candidates_path).absolute()
    if (
        source_lexical.name != "direct_candidates.tsv"
        or source_lexical.is_symlink()
        or not source_lexical.is_file()
    ):
        raise ValueError("Expected a direct, regular direct_candidates.tsv file")
    source = source_lexical.resolve(strict=True)
    if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in source.parts):
        raise ValueError("Refusing scorer/truth direct-candidate evidence")
    frame = pd.read_csv(source, sep="\t")
    if tuple(map(str, frame.columns)) != _DIRECT_CANDIDATE_COLUMNS:
        raise ValueError("Direct-candidate table does not match the frozen schema")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Direct-candidate evidence must be nonempty and complete")
    for column in (
        "source_assay",
        "node1",
        "endpoint1",
        "node2",
        "endpoint2",
        "edge_type",
        "orientation_source",
        "collider_center",
        "component_id",
    ):
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Direct-candidate column {column} contains empty values")
    if not frame["source_assay"].str.startswith("chrom_").all():
        raise ValueError("Only within-chromatin collider children may be annotated")
    if set(frame["edge_type"]) != {"within_assay_direct_candidate"}:
        raise ValueError("Unexpected edge type in direct-candidate evidence")
    if set(frame["orientation_source"]) != {"unique_positive_residual_wedge"}:
        raise ValueError("Unexpected orientation source in direct-candidate evidence")
    if not set(frame["endpoint1"]).union(frame["endpoint2"]) <= {"ARROW", "TAIL"}:
        raise ValueError("Direct candidates require only ARROW/TAIL endpoints")
    if frame.duplicated(["source_assay", "node1", "node2"]).any():
        raise ValueError("Direct-candidate edges must be unique")
    for row in frame.itertuples(index=False):
        center = str(row.collider_center)
        if center == str(row.node1):
            center_endpoint, other_endpoint = str(row.endpoint1), str(row.endpoint2)
        elif center == str(row.node2):
            center_endpoint, other_endpoint = str(row.endpoint2), str(row.endpoint1)
        else:
            raise ValueError("Collider center must be one endpoint of every candidate")
        if center_endpoint != "ARROW" or other_endpoint != "TAIL":
            raise ValueError("Collider candidates must point into their center")
    groups = frame.groupby(
        ["source_assay", "component_id", "collider_center"], sort=True
    )
    for _, group in groups:
        neighbors = {
            str(row.node2) if str(row.node1) == str(row.collider_center) else str(row.node1)
            for row in group.itertuples(index=False)
        }
        if len(group) != 2 or len(neighbors) != 2:
            raise ValueError("Each frozen collider component must have exactly two arms")
    if frame.groupby("component_id")["collider_center"].nunique().max() != 1:
        raise ValueError("A component may not contain multiple collider centers")
    centers = tuple(sorted(set(frame["collider_center"])))
    output_lexical = Path(output_path).absolute()
    if output_lexical.name != "direct_layer_overlap_evidence.json":
        raise ValueError("Frozen direct-layer output must use the canonical filename")
    if output_lexical.exists() or output_lexical.is_symlink():
        raise FileExistsError(f"Refusing to overwrite direct-layer evidence {output_lexical}")
    output_lexical.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "shared_state_subspace_v4.direct_layer_overlap.v1",
        "source_filename": source.name,
        "direct_candidates_sha256": _sha256(source),
        "selected_features": list(centers),
        "n_components": int(groups.ngroups),
        "n_direct_candidate_edges": int(len(frame)),
        "validation": {
            "strict_schema": True,
            "two_arrowhead_collider_arms_per_component": True,
            "within_assay_direct_candidates_only": True,
        },
        "truth_or_scorer_used": False,
    }
    output_lexical.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "evidence_sha256": _sha256(output_lexical)}


def _rotation_invariant_local_inputs(
    inputs: loading_groups.LoadingInputs,
) -> tuple[loading_groups.LoadingInputs, dict[str, float]]:
    """Undo pooled coordinate scaling, then scalar-normalize each assay."""

    assays: dict[str, loading_groups.AssayData] = {}
    assay_scales: dict[str, float] = {}
    dimension = len(inputs.coordinate_columns)
    for modality, assay in sorted(inputs.assays.items()):
        raw = (
            np.asarray(assay.coordinates, dtype=np.float64)
            * inputs.coordinate_scale[None, :]
            + inputs.coordinate_center[None, :]
        )
        centered = raw - raw.mean(axis=0, keepdims=True)
        scalar = float(np.sqrt(np.mean(np.sum(centered**2, axis=1)) / dimension))
        if not np.isfinite(scalar) or scalar <= 1e-12:
            raise ValueError(f"Assay {modality} has a degenerate state subspace")
        assay_scales[modality] = scalar
        assays[modality] = loading_groups.AssayData(
            modality=assay.modality,
            cell_ids=assay.cell_ids,
            coordinates=np.asarray(centered / scalar, dtype=np.float64),
            coordinate_uncertainty=assay.coordinate_uncertainty,
            log_depth=assay.log_depth,
            focal=assay.focal,
        )
    return (
        loading_groups.LoadingInputs(
            focal_root=inputs.focal_root,
            embedding_path=inputs.embedding_path,
            coordinate_columns=inputs.coordinate_columns,
            coordinate_center=np.zeros(dimension, dtype=np.float64),
            coordinate_scale=np.ones(dimension, dtype=np.float64),
            assays=assays,
            input_hashes=inputs.input_hashes,
        ),
        assay_scales,
    )


def _response_independent_assay_folds(
    assay: loading_groups.AssayData,
    *,
    requested: int,
    seed: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Choose one fixed response-blind K-fold partition for an assay."""

    n_splits = min(requested, len(assay.cell_ids))
    if n_splits < 2:
        raise ValueError("Conditional score needs at least two cells")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(np.arange(len(assay.cell_ids)))
    ]


def _formal_feature_eligibility(
    modality: str,
    assay: loading_groups.AssayData,
    feature: str,
    config: SharedStateSubspaceConfig,
) -> tuple[bool, bool, str]:
    """Return count gate, all-scheme fit eligibility, and abstention reason."""

    _, _, binary_event_gate = _feature_event_eligibility(
        modality, assay, feature, config
    )
    if not binary_event_gate:
        return False, False, "insufficient_binary_events"
    response = assay.focal[str(feature)].to_numpy(dtype=np.float64)
    if modality == "rna":
        if float(response.std(ddof=0)) <= 1e-12:
            return True, False, "constant_continuous_response"
        return True, True, ""
    fold_seeds = [
        _stable_seed(
            config.random_seed,
            "orthogonal_score_view_folds",
            str(view_index),
            modality,
        )
        for view_index in range(config.observed_model_views)
    ]
    fold_seeds.append(
        _stable_seed(
            config.random_seed, "conditional_wald_reference_folds", modality
        )
    )
    for seed in fold_seeds:
        folds = _response_independent_assay_folds(
            assay, requested=config.folds, seed=seed
        )
        if any(np.unique(response[train]).size != 2 for train, _ in folds):
            return True, False, "single_class_training_fold"
    return True, True, ""


def _fit_assay_score_nuisance(
    assay: loading_groups.AssayData,
    config: SharedStateSubspaceConfig,
    seed: int,
) -> _AssayScoreNuisance:
    """Fit joint rotation-equivariant E[state|depth] on shared assay folds."""

    folds = _response_independent_assay_folds(
        assay, requested=config.folds, seed=seed
    )
    residual_state = np.empty_like(assay.coordinates, dtype=np.float64)
    train_bases: list[FloatArray] = []
    test_bases: list[FloatArray] = []
    depth = assay.log_depth[:, None]
    for train, test in folds:
        unique_depth = np.unique(depth[train])
        if len(unique_depth) < 4:
            raise ValueError("Conditional score needs at least four unique depth values")
        spline = SplineTransformer(
            n_knots=min(config.nuisance_spline_knots, len(unique_depth)),
            degree=config.nuisance_spline_degree,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        )
        train_basis = np.asarray(spline.fit_transform(depth[train]), dtype=np.float64)
        test_basis = np.asarray(spline.transform(depth[test]), dtype=np.float64)
        state_model = Ridge(
            alpha=config.nuisance_ridge_alpha,
            fit_intercept=True,
        )
        state_model.fit(train_basis, assay.coordinates[train])
        residual_state[test] = assay.coordinates[test] - state_model.predict(test_basis)
        train_bases.append(train_basis)
        test_bases.append(test_basis)
    return _AssayScoreNuisance(
        folds=folds,
        train_bases=train_bases,
        test_bases=test_bases,
        residual_state=np.asarray(residual_state, dtype=np.float64),
    )


def _orthogonal_score_feature(
    modality: str,
    feature: str,
    assay: loading_groups.AssayData,
    nuisance: _AssayScoreNuisance,
    config: SharedStateSubspaceConfig,
    seed: int,
) -> _OrthogonalScoreFit:
    """Cross-fitted conditional-state score with depth spline nuisances."""

    family = "ridge" if modality == "rna" else "logistic"
    raw_response = assay.focal[feature].to_numpy(dtype=np.float64)
    if family == "ridge":
        response_sd = float(raw_response.std(ddof=0))
        if response_sd <= 1e-12:
            raise ValueError(f"Continuous RNA feature {feature} is constant")
        response = raw_response
    else:
        response = raw_response
    residual_response = np.empty(len(response), dtype=np.float64)
    test_folds: list[NDArray[np.int64]] = []
    for (train, test), train_basis, test_basis in zip(
        nuisance.folds,
        nuisance.train_bases,
        nuisance.test_bases,
        strict=True,
    ):
        if family == "ridge":
            response_model = Ridge(
                alpha=config.nuisance_ridge_alpha,
                fit_intercept=True,
            )
            response_model.fit(train_basis, response[train])
            prediction = response_model.predict(test_basis)
        else:
            response_model = LogisticRegression(
                C=config.nuisance_logistic_c,
                fit_intercept=True,
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            )
            response_model.fit(train_basis, response[train].astype(np.int64))
            prediction = response_model.predict_proba(test_basis)[:, 1]
        residual_response[test] = response[test] - prediction
        test_folds.append(np.asarray(test, dtype=np.int64))

    # Remove foldwise intercept error before forming the orthogonal moment.
    # This is response-blind for the state nuisance and prevents shared OOF
    # mean drift in rY/rZ from masquerading as conditional covariance.
    centered_state = nuisance.residual_state.copy()
    for test in test_folds:
        residual_response[test] -= float(residual_response[test].mean())
        centered_state[test] -= centered_state[test].mean(axis=0, keepdims=True)

    scores = residual_response[:, None] * centered_state
    mean_score = scores.mean(axis=0)
    centered = scores - mean_score[None, :]
    covariance = centered.T @ centered / len(centered)
    covariance_scale = max(
        float(np.trace(covariance) / covariance.shape[0]), 1e-12
    )
    regularized = covariance + (
        config.score_covariance_ridge * covariance_scale * np.eye(covariance.shape[0])
    )
    inverse = np.linalg.pinv(regularized, hermitian=True)
    statistic = float(
        max(len(centered) * (mean_score @ inverse @ mean_score), 0.0)
    )
    fold_directions = np.vstack([scores[test].mean(axis=0) for test in test_folds])
    _, direction_stability = loading_groups._direction_stability(
        fold_directions, mean_score
    )
    return _OrthogonalScoreFit(
        statistic=statistic,
        score_norm=float(np.linalg.norm(mean_score)),
        direction_stability=float(direction_stability),
        centered_scores=np.asarray(centered, dtype=np.float64),
        inverse_score_covariance=np.asarray(inverse, dtype=np.float64),
    )


def _fit_observed_model_views(
    inputs: loading_groups.LoadingInputs,
    config: SharedStateSubspaceConfig,
) -> list[pd.DataFrame]:
    """Fit the complete observed feature model under six deterministic views."""

    views: list[pd.DataFrame] = []
    for view_index in range(config.observed_model_views):
        rows: list[dict[str, Any]] = []
        for modality, assay in sorted(inputs.assays.items()):
            feature_metadata = {
                str(feature): (
                    _feature_event_eligibility(
                        modality, assay, str(feature), config
                    )[:2]
                    + _formal_feature_eligibility(
                        modality, assay, str(feature), config
                    )
                )
                for feature in assay.focal.columns
            }
            eligible_features = [
                feature
                for feature, (_, _, _, formally_eligible, _) in feature_metadata.items()
                if formally_eligible
            ]
            nuisance = (
                _fit_assay_score_nuisance(
                    assay,
                    config,
                    _stable_seed(
                        config.random_seed,
                        "orthogonal_score_view_folds",
                        str(view_index),
                        modality,
                    ),
                )
                if eligible_features
                else None
            )
            for feature in assay.focal.columns:
                feature_name = str(feature)
                (
                    n_events,
                    n_nonevents,
                    binary_event_gate,
                    formally_eligible,
                    formal_abstention_reason,
                ) = feature_metadata[feature_name]
                score = (
                    _orthogonal_score_feature(
                        modality,
                        feature_name,
                        assay,
                        nuisance,
                        config,
                        _stable_seed(
                            config.random_seed,
                            "orthogonal_score_view",
                            str(view_index),
                            feature_name,
                        ),
                    )
                    if formally_eligible and nuisance is not None
                    else _OrthogonalScoreFit(
                        statistic=0.0,
                        score_norm=0.0,
                        direction_stability=0.0,
                        centered_scores=np.zeros(
                            (len(assay.cell_ids), assay.coordinates.shape[1]),
                            dtype=np.float64,
                        ),
                        inverse_score_covariance=np.zeros(
                            (assay.coordinates.shape[1], assay.coordinates.shape[1]),
                            dtype=np.float64,
                        ),
                    )
                )
                rows.append(
                    {
                        "feature": feature_name,
                        "modality": modality,
                        "channel": (
                            "RNA" if modality == "rna" else modality.removeprefix("chrom_")
                        ),
                        "n_cells": len(assay.cell_ids),
                        "n_events": n_events,
                        "n_nonevents": n_nonevents,
                        "binary_event_gate": binary_event_gate,
                        "formal_test_eligible": formally_eligible,
                        "formal_test_abstention_reason": formal_abstention_reason,
                        "conditional_wald_statistic": score.statistic,
                        "conditional_score_norm": score.score_norm,
                        "conditional_score_direction_stability": (
                            score.direction_stability
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        frame["view_id"] = f"model_seed{view_index}"
        views.append(frame)
    return views


def _aggregate_observed_models(views: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(views) < 2:
        raise ValueError("At least two observed model views are required")
    rows = []
    for view_index, view in enumerate(views):
        frame = view[
            [
                "feature",
                "modality",
                "channel",
                "n_cells",
                "n_events",
                "n_nonevents",
                "binary_event_gate",
                "formal_test_eligible",
                "formal_test_abstention_reason",
                "conditional_wald_statistic",
                "conditional_score_norm",
                "conditional_score_direction_stability",
                "view_id",
            ]
        ].copy()
        if frame["feature"].astype(str).duplicated().any():
            raise ValueError(f"Observed model view {view_index} repeats features")
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby(["feature", "modality", "channel"], sort=True)
    output = grouped.agg(
        n_views=("view_id", "nunique"),
        n_cells=("n_cells", "first"),
        n_events=("n_events", "first"),
        n_nonevents=("n_nonevents", "first"),
        binary_event_gate=("binary_event_gate", "all"),
        formal_test_eligible=("formal_test_eligible", "all"),
        formal_test_abstention_reason=(
            "formal_test_abstention_reason",
            lambda values: ";".join(
                sorted({str(value) for value in values if str(value)})
            ),
        ),
        median_conditional_wald=("conditional_wald_statistic", "median"),
        lower_quartile_conditional_wald=(
            "conditional_wald_statistic",
            lambda values: float(np.quantile(values, 0.25)),
        ),
        median_conditional_score_norm=("conditional_score_norm", "median"),
        minimum_conditional_score_norm=("conditional_score_norm", "min"),
        median_conditional_score_direction_stability=(
            "conditional_score_direction_stability",
            "median",
        ),
        minimum_conditional_score_direction_stability=(
            "conditional_score_direction_stability",
            "min",
        ),
    ).reset_index()
    view_statistics = combined.pivot(
        index=["feature", "modality", "channel"],
        columns="view_id",
        values="conditional_wald_statistic",
    ).reset_index()
    view_statistics.columns = [
        str(column)
        if str(column) in {"feature", "modality", "channel"}
        else f"conditional_wald__{column}"
        for column in view_statistics.columns
    ]
    output = output.merge(
        view_statistics,
        on=["feature", "modality", "channel"],
        how="left",
        validate="one_to_one",
    )
    if (output["n_views"] != len(views)).any():
        raise ValueError("Every focal feature must occur in every evidence view")
    return output


def _fit_descriptive_state_depth_decomposition(
    inputs: loading_groups.LoadingInputs,
    config: SharedStateSubspaceConfig,
) -> pd.DataFrame:
    """Self-fit one nonselecting OOF state/depth decomposition per feature."""

    descriptive_config = loading_groups.LoadingGroupConfig(
        folds=config.folds,
        bootstrap_repetitions=0,
        seed=config.random_seed,
    )
    rows: list[dict[str, Any]] = []
    for modality, assay in sorted(inputs.assays.items()):
        for feature in assay.focal.columns:
            feature_name = str(feature)
            _, formally_eligible, _ = _formal_feature_eligibility(
                modality, assay, feature_name, config
            )
            if formally_eligible:
                fit_values = loading_groups._estimate_feature(
                    modality,
                    feature_name,
                    assay,
                    inputs.coordinate_columns,
                    descriptive_config,
                    _stable_seed(
                        config.random_seed,
                        "descriptive_state_depth_decomposition",
                        modality,
                        feature_name,
                    ),
                ).feature_row
            else:
                fit_values = {
                    field: 0.0 for field in _DESCRIPTIVE_DECOMPOSITION_FIELDS
                }
            rows.append(
                {
                    "feature": feature_name,
                    **{
                        field: float(fit_values[field])
                        for field in _DESCRIPTIVE_DECOMPOSITION_FIELDS
                    },
                }
            )
    output = pd.DataFrame(rows)
    if output.empty or output["feature"].astype(str).duplicated().any():
        raise ValueError("Descriptive decomposition must contain each feature once")
    values = output[list(_DESCRIPTIVE_DECOMPOSITION_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values > 1.0 + 1e-10):
        raise ValueError("Descriptive OOF fits must be finite and no greater than one")
    return output.sort_values("feature").reset_index(drop=True)


def _fit_observed_evidence(
    inputs: loading_groups.LoadingInputs,
    config: SharedStateSubspaceConfig,
) -> pd.DataFrame:
    """Fit formal multiview evidence plus one nonselecting descriptive model."""

    observed = _aggregate_observed_models(_fit_observed_model_views(inputs, config))
    descriptive = _fit_descriptive_state_depth_decomposition(inputs, config)
    output = observed.merge(
        descriptive,
        on="feature",
        how="left",
        validate="one_to_one",
    )
    if output[list(_DESCRIPTIVE_DECOMPOSITION_FIELDS)].isna().any().any():
        raise ValueError("Descriptive decomposition and formal evidence disagree")
    prefix = [
        "feature",
        "modality",
        "channel",
        "n_views",
        "n_cells",
        "n_events",
        "n_nonevents",
        "binary_event_gate",
        "formal_test_eligible",
        "formal_test_abstention_reason",
        *_DESCRIPTIVE_DECOMPOSITION_FIELDS,
    ]
    return output[
        prefix + [column for column in output.columns if column not in prefix]
    ]


def _validate_safe_embedding_freeze(
    inputs: loading_groups.LoadingInputs,
) -> dict[str, Any]:
    manifest_lexical = inputs.embedding_path.parent / "frozen_evidence_manifest.json"
    if manifest_lexical.is_file() and not manifest_lexical.is_symlink():
        manifest_path = manifest_lexical.resolve(strict=True)
        if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in manifest_path.parts):
            raise ValueError("Refusing scorer/truth embedding provenance")
        manifest = json.loads(manifest_path.read_text())
        manifest_schema = str(manifest.get("schema_version", ""))
        supported_schemas = {
            "component_identified_alignment_v4.frozen_evidence.v1",
            assay_local_embedding.FROZEN_EVIDENCE_SCHEMA_VERSION,
        }
        if manifest_schema not in supported_schemas:
            raise ValueError("Unsupported frozen safe-embedding schema")
        if manifest.get("frozen_before_scorer_read") is not True:
            raise ValueError("Safe embedding was not explicitly frozen before scorer access")
        boundary = manifest.get("information_boundary", {})
        if boundary.get("truth_or_scorer_used") is not False:
            raise ValueError("Safe embedding is not explicitly truth-blind")
        consumed = set(map(str, boundary.get("consumed", [])))
        allowed_consumed = {
            "observed_background_matrix_per_assay",
            "observed_leave_focal_out_depth_per_assay",
            "assay_key",
        }
        if not consumed or not consumed <= allowed_consumed:
            raise ValueError("Frozen embedding consumed forbidden information")
        not_consumed = set(map(str, boundary.get("not_consumed", [])))
        for required in ("scorer", "truth_graph", "external_data"):
            if required not in not_consumed:
                raise ValueError(f"Frozen embedding boundary omits {required}")
        upstream_inputs = manifest.get("input_file_sha256", {})
        if not upstream_inputs or any(
            not str(name).startswith("background_matrix__")
            for name in upstream_inputs
        ):
            raise ValueError("Frozen embedding was not derived only from background matrices")
        output_hashes = manifest.get(
            "output_file_sha256", manifest.get("output_sha256", {})
        )
        expected_hash = inputs.input_hashes.get(inputs.embedding_path.name, "")
        if output_hashes.get(inputs.embedding_path.name) != expected_hash:
            raise ValueError("Safe-embedding manifest hash disagrees with the supplied table")
        accepted = manifest.get("accepted_assays")
        if accepted is not None and set(map(str, accepted)) != set(inputs.assays):
            raise ValueError("Safe-embedding accepted assays disagree with focal inputs")
        sidecar_lexical = manifest_path.with_suffix(".sha256")
        if sidecar_lexical.is_symlink() or not sidecar_lexical.is_file():
            raise ValueError("Frozen embedding lacks its manifest digest sidecar")
        sidecar_digest = sidecar_lexical.read_text().strip().split()[0]
        if sidecar_digest != _sha256(manifest_path):
            raise ValueError("Frozen embedding manifest digest sidecar disagrees")
        if manifest_schema == assay_local_embedding.FROZEN_EVIDENCE_SCHEMA_VERSION:
            local_source = Path(assay_local_embedding.__file__).resolve(strict=True)
            if manifest.get("algorithm_source_sha256") != _sha256(local_source):
                raise ValueError("Assay-local embedding source identity disagrees")
            expected_config = asdict(
                assay_local_embedding.AssayLocalStateEmbeddingConfig()
            )
            if manifest.get("config") != expected_config:
                raise ValueError("Assay-local embedding config is not frozen")
            if int(manifest.get("formal_state_dimension", -1)) != 3 or (
                len(inputs.coordinate_columns) != 3
            ):
                raise ValueError("Formal assay-local embedding must have rank three")
            if manifest.get("rank_policy") != (
                "predeclared_simulator_rank_three_with_parallel_analysis_abstention"
            ):
                raise ValueError("Assay-local rank policy is not the formal contract")
            if manifest.get("zero_padding_used") is not False:
                raise ValueError("Formal assay-local embedding may not pad coordinates")
            if manifest.get("cross_assay_alignment_used") is not False:
                raise ValueError("Formal state-response input may not align assays")
            if manifest.get(
                "all_assays_parallel_rank_support_formal_dimension"
            ) is not True:
                raise ValueError("Parallel analysis did not support formal rank three")
            background_paths = {
                path.name: path
                for path in inputs.focal_root.glob("background_matrix__*.npz")
                if path.is_file() and not path.is_symlink()
            }
            if set(background_paths) != set(upstream_inputs) or any(
                _sha256(background_paths[name]) != str(digest)
                for name, digest in upstream_inputs.items()
            ):
                raise ValueError("Assay-local embedding background hashes disagree")
            output_hashes = manifest.get("output_file_sha256", {})
            provenance_lexical = manifest_path.parent / (
                assay_local_embedding.PROVENANCE_NAME
            )
            if (
                set(output_hashes)
                != {inputs.embedding_path.name, provenance_lexical.name}
                or provenance_lexical.is_symlink()
                or not provenance_lexical.is_file()
                or output_hashes.get(provenance_lexical.name)
                != _sha256(provenance_lexical)
            ):
                raise ValueError("Assay-local embedding provenance hash disagrees")
            provenance = json.loads(provenance_lexical.read_text())
            if (
                provenance.get("schema_version")
                != assay_local_embedding.SCHEMA_VERSION
                or provenance.get("config") != expected_config
                or provenance.get("formal_state_dimension") != 3
                or provenance.get("zero_padding_used") is not False
                or provenance.get("cross_assay_alignment_used") is not False
                or provenance.get("information_boundary") != boundary
            ):
                raise ValueError("Assay-local embedding provenance is inconsistent")
            return {
                "layout": "assay_local_state_embedding_v4",
                "files": {
                    inputs.embedding_path.name: _sha256(inputs.embedding_path),
                    manifest_path.name: _sha256(manifest_path),
                    sidecar_lexical.name: _sha256(sidecar_lexical),
                    provenance_lexical.name: _sha256(provenance_lexical),
                },
                "upstream_algorithm_source_sha256": {
                    local_source.name: _sha256(local_source),
                },
                "formal_state_dimension": 3,
                "rank_policy": manifest["rank_policy"],
            }
        alignment_source = Path(__file__).with_name(
            "component_identified_alignment_v4.py"
        ).resolve(strict=True)
        if manifest.get("algorithm_source_sha256") != _sha256(alignment_source):
            raise ValueError("Frozen embedding algorithm source identity disagrees")
        return {
            "layout": "frozen_evidence_manifest",
            "files": {
                manifest_path.name: _sha256(manifest_path),
                sidecar_lexical.name: _sha256(sidecar_lexical),
            },
            "upstream_algorithm_source_sha256": {
                alignment_source.name: _sha256(alignment_source),
            },
        }

    # Formal HARD preparation writes these two provenance files before any
    # benchmark scorer is opened.  It does not write a frozen-evidence manifest,
    # so validate the actual narrow schema and snapshot all three files here.
    alignment_lexical = inputs.embedding_path.parent / "alignment_provenance.json"
    preparation_lexical = inputs.embedding_path.parent / "preparation_manifest.json"
    for path in (alignment_lexical, preparation_lexical):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Safe embedding lacks supported preparation provenance")
        if any(part.casefold() in _FORBIDDEN_PATH_PARTS for part in path.resolve().parts):
            raise ValueError("Refusing scorer/truth embedding provenance")
    alignment_path = alignment_lexical.resolve(strict=True)
    preparation_path = preparation_lexical.resolve(strict=True)
    alignment = json.loads(alignment_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    if alignment.get("schema_version") != "prepare_ca_boss_v4_inputs.alignment.v1":
        raise ValueError("Unsupported prepared-embedding alignment schema")
    if preparation.get("schema_version") != "prepare_ca_boss_v4_inputs.run.v1":
        raise ValueError("Unsupported prepared-embedding manifest schema")
    if alignment.get("method") != "recomputed_independent_two_feature_fold_consensus":
        raise ValueError("Formal subspace input requires the HARD cross-fitted embedding")
    if preparation.get("alignment_method") != (
        "recomputed_two_feature_fold_background_consensus"
    ):
        raise ValueError("Preparation manifest does not identify the HARD embedding")
    policy = alignment.get("input_policy", {})
    if policy.get("scorer_or_truth_read") is not False:
        raise ValueError("Prepared embedding is not explicitly scorer/truth blind")
    if policy.get("external_data_read") is not False:
        raise ValueError("Prepared embedding used external data")
    for key in (
        "supplied_analysis_embeddings_read",
        "supplied_alignment_provenance_read",
        "supplied_metacells_or_membership_read",
    ):
        if policy.get(key) is not False:
            raise ValueError(f"Prepared embedding violates input policy {key}")
    boundary = alignment.get("information_boundary", {})
    required_false = (
        "cell_rows_compared_across_assays",
        "feature_columns_compared_across_assays",
        "focal_values_used_for_alignment",
    )
    if any(boundary.get(key) is not False for key in required_false):
        raise ValueError("Prepared embedding violates the disjoint-assay boundary")
    if boundary.get("feature_folds_are_within_assay") is not True:
        raise ValueError("Prepared embedding lacks within-assay feature cross-fitting")
    written = set(map(str, preparation.get("files_written", [])))
    if not {"analysis_embeddings.tsv", "alignment_provenance.json"} <= written:
        raise ValueError("Preparation manifest does not declare the safe embedding")
    if int(preparation.get("n_modalities", -1)) != len(inputs.assays):
        raise ValueError("Prepared embedding modality count disagrees with focal inputs")
    if preparation.get("derange") is not False:
        raise ValueError("Deranged diagnostic embeddings are not valid formal inputs")
    upstream_hashes = alignment.get("input_file_sha256", {})
    focal_hashes = {
        filename: digest
        for filename, digest in inputs.input_hashes.items()
        if filename.startswith("observed_focal__")
    }
    if any(upstream_hashes.get(name) != digest for name, digest in focal_hashes.items()):
        raise ValueError("Prepared embedding provenance disagrees with focal inputs")
    prepare_source = Path(__file__).with_name("prepare_ca_boss_v4_inputs.py").resolve(
        strict=True
    )
    return {
        "layout": "prepare_ca_boss_v4_inputs",
        "files": {
            inputs.embedding_path.name: _sha256(inputs.embedding_path),
            alignment_path.name: _sha256(alignment_path),
            preparation_path.name: _sha256(preparation_path),
        },
        "upstream_algorithm_source_sha256": {
            prepare_source.name: _sha256(prepare_source),
        },
    }


def _benjamini_hochberg(values: FloatArray) -> FloatArray:
    p_values = np.asarray(values, dtype=np.float64)
    if p_values.ndim != 1 or not len(p_values) or not np.isfinite(p_values).all():
        raise ValueError("Benjamini-Hochberg requires finite one-dimensional p-values")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("P-values must lie in [0, 1]")
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _holm(values: FloatArray) -> FloatArray:
    """Holm-adjust finite p-values with strong FWER control."""

    p_values = np.asarray(values, dtype=np.float64)
    if p_values.ndim != 1 or not len(p_values) or not np.isfinite(p_values).all():
        raise ValueError("Holm adjustment requires finite one-dimensional p-values")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("P-values must lie in [0, 1]")
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order] * np.arange(len(p_values), 0, -1)
    ranked = np.maximum.accumulate(ranked)
    output = np.empty_like(ranked)
    output[order] = np.minimum(ranked, 1.0)
    return output


def _select_assay_state_participation(
    models: pd.DataFrame,
    config: SharedStateSubspaceConfig,
) -> pd.DataFrame:
    """Call assay-only state participation without changing feature membership.

    Let ``m`` be the number of formally eligible features in an assay and
    ``r`` the predeclared participation order (two, except for a structurally
    singleton assay).  Two routes are computed before looking at their
    results.  The single-feature route is ``m p_(1)`` and requires its witness
    in every model view.  The replicated route is the Bonferroni
    partial-conjunction p-value ``(m-r+1) p_(r)`` and requires every one of its
    ``r`` witnesses in every view.  A failed replication gate sets that
    route's selection p-value to one.  Bonferroni combines the two live routes;
    a structural singleton has one route and therefore no duplicate penalty.
    Holm adjustment of the assay omnibus p-values then controls strong FWER
    for the family of assay complete-null hypotheses.  Every step is valid
    under arbitrary dependence.  Feature p-values retain their existing
    state/depth-preserving conditional wild-null calibration.

    The ordered features are evidence reporters only.  Neither the reporter
    nor any partial-conjunction witness is promoted into strict feature
    membership by this function.
    """

    required = {
        "feature",
        "modality",
        "formal_test_eligible",
        "empirical_null_p",
        "reference_conditional_wald",
        "conditional_wald_view_support",
    }
    missing = sorted(required - set(models))
    if missing:
        raise ValueError(f"Assay participation models are missing columns {missing}")
    rows: list[dict[str, Any]] = []
    for modality, full_group in models.groupby("modality", sort=True):
        full_group = full_group.copy()
        eligible = full_group.loc[
            full_group["formal_test_eligible"].astype(bool)
        ].copy()
        eligible = eligible.sort_values(
            ["empirical_null_p", "reference_conditional_wald", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        n_total = len(full_group)
        n_eligible = len(eligible)
        order = min(config.assay_state_participation_order, n_total)
        enough_eligible = n_eligible >= order and order > 0
        evidence = eligible.head(order) if enough_eligible else eligible.iloc[0:0]
        leading_reporter = (
            str(eligible.iloc[0]["feature"]) if n_eligible else ""
        )
        if n_eligible:
            single_feature_order_p = float(eligible.iloc[0]["empirical_null_p"])
            single_feature_p = min(
                1.0, n_eligible * single_feature_order_p
            )
            single_feature_view_count = int(
                np.rint(
                    float(eligible.iloc[0]["conditional_wald_view_support"])
                    * config.observed_model_views
                )
            )
        else:
            single_feature_order_p = 1.0
            single_feature_p = 1.0
            single_feature_view_count = 0
        single_feature_multiview_gate = bool(
            n_eligible >= 1
            and single_feature_view_count
            >= config.minimum_assay_state_participation_views
        )
        if enough_eligible:
            ordered_p = eligible["empirical_null_p"].to_numpy(dtype=np.float64)
            order_p = float(ordered_p[order - 1])
            partial_conjunction_p = min(
                1.0, (n_eligible - order + 1) * order_p
            )
            view_counts = np.rint(
                evidence["conditional_wald_view_support"].to_numpy(dtype=np.float64)
                * config.observed_model_views
            ).astype(np.int64)
            minimum_view_count = int(np.min(view_counts))
        else:
            order_p = 1.0
            partial_conjunction_p = 1.0
            minimum_view_count = 0
        partial_conjunction_multiview_gate = bool(
            enough_eligible
            and minimum_view_count
            >= config.minimum_assay_state_participation_views
        )
        omnibus_route_count = 1 if n_total == 1 else 2
        live_routes = {
            "single_feature": (
                single_feature_p if single_feature_multiview_gate else 1.0
            )
        }
        if omnibus_route_count == 2:
            live_routes["partial_conjunction"] = (
                partial_conjunction_p
                if partial_conjunction_multiview_gate
                else 1.0
            )
        if any(value < 1.0 for value in live_routes.values()):
            # Prefer the stronger partial-conjunction interpretation on an
            # exact tie; the choice affects only audit labels, never the p-value.
            winning_route = min(
                live_routes,
                key=lambda route: (
                    live_routes[route],
                    0 if route == "partial_conjunction" else 1,
                ),
            )
        else:
            winning_route = "none"
        omnibus_p = min(
            1.0, omnibus_route_count * min(live_routes.values())
        )
        evidence_records = [
            {
                "feature": str(record["feature"]),
                "empirical_null_p": float(record["empirical_null_p"]),
                "reference_conditional_wald": float(
                    record["reference_conditional_wald"]
                ),
                "conditional_wald_view_support": float(
                    record["conditional_wald_view_support"]
                ),
                "single_feature_witness": bool(index == 0),
                "partial_conjunction_witness": bool(
                    enough_eligible and index < order
                ),
            }
            for index, record in enumerate(eligible.to_dict(orient="records"))
        ]
        rows.append(
            {
                "assay_state_id": f"assay_state_participation__{modality}",
                "modality": str(modality),
                "edge_type": "assay_shared_state_participation",
                "n_observed_features": int(n_total),
                "n_formally_eligible_features": int(n_eligible),
                "partial_conjunction_order": int(order),
                "partial_conjunction_order_statistic_p": order_p,
                "partial_conjunction_empirical_p": partial_conjunction_p,
                "assay_participation_holm_p": 1.0,
                "single_feature_order_statistic_p": single_feature_order_p,
                "single_feature_bonferroni_p": single_feature_p,
                "single_feature_witness_feature": leading_reporter,
                "single_feature_witness_view_count": (
                    single_feature_view_count
                ),
                "single_feature_multiview_replication_gate": (
                    single_feature_multiview_gate
                ),
                "minimum_witness_view_count": minimum_view_count,
                "multiview_replication_gate": (
                    partial_conjunction_multiview_gate
                ),
                "partial_conjunction_multiview_replication_gate": (
                    partial_conjunction_multiview_gate
                ),
                "assay_participation_omnibus_route_count": (
                    omnibus_route_count
                ),
                "assay_participation_omnibus_p": omnibus_p,
                "assay_participation_omnibus_holm_p": 1.0,
                "assay_participation_winning_route": winning_route,
                "assay_participation_selection_route": "",
                "leading_reporter": leading_reporter,
                "selected_reporter": "",
                "contributing_feature_count": int(len(evidence)),
                "partial_conjunction_witness_features": ";".join(
                    evidence["feature"].astype(str)
                ),
                "all_eligible_feature_evidence": json.dumps(
                    evidence_records, sort_keys=True, separators=(",", ":")
                ),
                "selection_status": "unresolved",
                "abstention_reason": "",
                "conditional_on_assay_local_state_model": True,
                "feature_membership_claimed": False,
                "exact_axis_pairing_claimed": False,
                "direct_causal_claimed": False,
            }
        )
    participation = pd.DataFrame(rows)
    if participation.empty:
        return participation
    participation["assay_participation_holm_p"] = _holm(
        participation["partial_conjunction_empirical_p"].to_numpy(dtype=np.float64)
    )
    participation["assay_participation_omnibus_holm_p"] = _holm(
        participation["assay_participation_omnibus_p"].to_numpy(
            dtype=np.float64
        )
    )
    statuses: list[str] = []
    reasons: list[str] = []
    for record in participation.to_dict(orient="records"):
        failures = []
        if int(record["n_formally_eligible_features"]) < 1:
            failures.append("insufficient_formally_eligible_features")
        if str(record["assay_participation_winning_route"]) == "none":
            failures.append("no_replicated_omnibus_route")
        if float(record["assay_participation_omnibus_holm_p"]) > (
            config.maximum_assay_state_participation_error_probability
        ):
            failures.append("omnibus_holm_FWER")
        statuses.append("unresolved" if failures else "selected")
        reasons.append(";".join(failures))
    participation["selection_status"] = statuses
    participation["abstention_reason"] = reasons
    participation["selected_reporter"] = np.where(
        participation["selection_status"].eq("selected"),
        participation["leading_reporter"],
        "",
    )
    participation["assay_participation_selection_route"] = np.where(
        participation["selection_status"].eq("selected"),
        participation["assay_participation_winning_route"],
        "",
    )
    return participation


def select_shared_subspace_membership(
    observed_models: pd.DataFrame,
    reference_conditional_wald: Mapping[str, float],
    null_conditional_wald: Mapping[str, FloatArray],
    *,
    direct_layer_features: Sequence[str] = (),
    config: SharedStateSubspaceConfig | None = None,
) -> SharedStateSubspaceResult:
    """Select state-responsive memberships and one non-pairing hyperedge."""

    cfg = config or SharedStateSubspaceConfig()
    required = {
        "feature",
        "modality",
        "channel",
        "n_views",
        "n_cells",
        "n_events",
        "n_nonevents",
        "binary_event_gate",
        "formal_test_eligible",
        "formal_test_abstention_reason",
        *_DESCRIPTIVE_DECOMPOSITION_FIELDS,
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "median_conditional_score_norm",
        "minimum_conditional_score_norm",
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
    }
    missing = sorted(required - set(observed_models))
    if missing:
        raise ValueError(f"Observed models are missing columns {missing}")
    view_statistic_columns = sorted(
        column
        for column in observed_models
        if str(column).startswith("conditional_wald__model_seed")
    )
    if len(view_statistic_columns) < 2:
        raise ValueError("Observed models lack multiview conditional-Wald statistics")
    models = observed_models.copy().sort_values("feature").reset_index(drop=True)
    if models["feature"].astype(str).duplicated().any():
        raise ValueError("Observed feature rows must be unique")
    observed_features = set(models["feature"].astype(str))
    expected_binary_event_gate = (
        models["modality"].astype(str).eq("rna")
        | (
            (models["n_events"] >= cfg.minimum_binary_events)
            & (models["n_nonevents"] >= cfg.minimum_binary_nonevents)
        )
    )
    if not np.array_equal(
        models["binary_event_gate"].to_numpy(dtype=bool),
        expected_binary_event_gate.to_numpy(dtype=bool),
    ):
        raise ValueError("binary_event_gate disagrees with observed event counts")
    eligible_mask = models["formal_test_eligible"].to_numpy(dtype=bool)
    formal_reasons = models["formal_test_abstention_reason"].fillna("").astype(str)
    allowed_formal_reasons = {
        "",
        "insufficient_binary_events",
        "single_class_training_fold",
        "constant_continuous_response",
    }
    if not set(formal_reasons) <= allowed_formal_reasons:
        raise ValueError("Unknown formal-test abstention reason")
    expected_insufficient = ~models["binary_event_gate"].to_numpy(dtype=bool)
    if np.any(eligible_mask & expected_insufficient):
        raise ValueError("Count-ineligible features cannot enter formal tests")
    if np.any(eligible_mask & formal_reasons.ne("").to_numpy(dtype=bool)):
        raise ValueError("Formally eligible features cannot carry an abstention reason")
    if np.any(
        expected_insufficient
        & formal_reasons.ne("insufficient_binary_events").to_numpy(dtype=bool)
    ):
        raise ValueError("Count-ineligible features need the exact abstention reason")
    if np.any(
        (~eligible_mask)
        & (~expected_insufficient)
        & (~formal_reasons.isin(
            {"single_class_training_fold", "constant_continuous_response"}
        ).to_numpy(dtype=bool))
    ):
        raise ValueError("Fit-ineligible features need an auditable abstention reason")
    eligible_features = set(
        models.loc[eligible_mask, "feature"].astype(str)
    )
    placeholder_columns = [
        *_DESCRIPTIVE_DECOMPOSITION_FIELDS,
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "median_conditional_score_norm",
        "minimum_conditional_score_norm",
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
        *view_statistic_columns,
    ]
    models.loc[~eligible_mask, placeholder_columns] = 0.0
    if eligible_features != set(null_conditional_wald) or eligible_features != set(
        reference_conditional_wald
    ):
        raise ValueError(
            "Reference/null statistics must exactly match formally eligible features"
        )
    direct_layer = {str(feature) for feature in direct_layer_features}
    unknown_direct = sorted(direct_layer - observed_features)
    if unknown_direct:
        raise ValueError(f"Direct-layer features are absent: {unknown_direct}")
    descriptive_values = models[list(_DESCRIPTIVE_DECOMPOSITION_FIELDS)].to_numpy(
        dtype=float
    )
    if not np.isfinite(descriptive_values).all() or np.any(
        descriptive_values > 1.0 + 1e-10
    ):
        raise ValueError("Descriptive OOF fits must be finite and no greater than one")
    p_values = []
    null_medians = []
    null_upper = []
    for record in models.to_dict(orient="records"):
        feature = str(record["feature"])
        if not bool(record["formal_test_eligible"]):
            p_values.append(1.0)
            null_medians.append(0.0)
            null_upper.append(0.0)
            continue
        null = np.asarray(null_conditional_wald[feature], dtype=np.float64)
        if null.shape != (cfg.null_repetitions,) or not np.isfinite(null).all():
            raise ValueError(f"Null evidence for {feature} has an invalid shape")
        # The reference is one fixed cross-fitted conditional-Wald statistic.
        # Its empirical null uses centered orthogonal scores with Rademacher
        # multiplier weights shared by every feature in an assay.  The six
        # correlated fold views remain a separate reproducibility gate.
        observed = float(reference_conditional_wald[feature])
        if not np.isfinite(observed):
            raise ValueError(f"Reference statistic for {feature} is not finite")
        p_values.append(
            (1.0 + float(np.sum(null >= observed))) / (len(null) + 1.0)
        )
        null_medians.append(float(np.median(null)))
        null_upper.append(float(np.quantile(null, 0.95)))
    models["empirical_null_p"] = p_values
    models["reference_conditional_wald"] = [
        (
            float(reference_conditional_wald[str(feature)])
            if bool(eligible)
            else 0.0
        )
        for feature, eligible in zip(
            models["feature"], eligible_mask, strict=True
        )
    ]
    # Each assay is a predeclared observable feature family.  Direct collider
    # children remain in this family because a mixed-axis collider child is
    # still genuinely state-responsive.  Layer membership may overlap.
    models["empirical_null_fdr"] = 1.0
    eligible_family_sizes = (
        models.loc[eligible_mask].groupby("modality", sort=True).size().to_dict()
    )
    models["empirical_null_family_size"] = [
        int(eligible_family_sizes.get(str(modality), 0))
        for modality in models["modality"]
    ]
    for _, indices in models.loc[eligible_mask].groupby("modality", sort=True).groups.items():
        family_indices = np.asarray(sorted(indices), dtype=np.int64)
        models.loc[family_indices, "empirical_null_fdr"] = _benjamini_hochberg(
            models.loc[family_indices, "empirical_null_p"].to_numpy(dtype=np.float64)
        )
        models.loc[family_indices, "empirical_null_family_size"] = len(
            family_indices
        )
    models["conditional_wald_null_median"] = null_medians
    models["conditional_wald_null_95pct"] = null_upper
    models["conditional_wald_above_null_95pct"] = (
        models["reference_conditional_wald"]
        - models["conditional_wald_null_95pct"]
    )
    view_passes = (
        eligible_mask[:, None]
        & (
            models[view_statistic_columns].to_numpy(dtype=np.float64)
            >= models["conditional_wald_null_95pct"].to_numpy(dtype=np.float64)[
                :, None
            ]
        )
    )
    conditional_wald_view_count = np.sum(view_passes, axis=1)
    models["conditional_wald_view_support"] = np.mean(view_passes, axis=1)
    assay_state_participation = _select_assay_state_participation(models, cfg)
    participation_by_modality = assay_state_participation.set_index("modality")
    modality_keys = models["modality"].astype(str)
    models["assay_participation_partial_conjunction_p"] = modality_keys.map(
        participation_by_modality["partial_conjunction_empirical_p"]
    ).astype(float)
    models["assay_participation_holm_p"] = modality_keys.map(
        participation_by_modality["assay_participation_holm_p"]
    ).astype(float)
    models["assay_participation_single_feature_order_statistic_p"] = (
        modality_keys.map(
            participation_by_modality["single_feature_order_statistic_p"]
        ).astype(float)
    )
    models["assay_participation_single_feature_p"] = modality_keys.map(
        participation_by_modality["single_feature_bonferroni_p"]
    ).astype(float)
    models["assay_participation_single_feature_witness_view_count"] = (
        modality_keys.map(
            participation_by_modality["single_feature_witness_view_count"]
        ).astype(int)
    )
    models[
        "assay_participation_single_feature_multiview_replication_gate"
    ] = modality_keys.map(
        participation_by_modality[
            "single_feature_multiview_replication_gate"
        ]
    ).astype(bool)
    models[
        "assay_participation_partial_conjunction_multiview_replication_gate"
    ] = modality_keys.map(
        participation_by_modality[
            "partial_conjunction_multiview_replication_gate"
        ]
    ).astype(bool)
    models["assay_participation_omnibus_route_count"] = modality_keys.map(
        participation_by_modality[
            "assay_participation_omnibus_route_count"
        ]
    ).astype(int)
    models["assay_participation_omnibus_p"] = modality_keys.map(
        participation_by_modality["assay_participation_omnibus_p"]
    ).astype(float)
    models["assay_participation_omnibus_holm_p"] = modality_keys.map(
        participation_by_modality["assay_participation_omnibus_holm_p"]
    ).astype(float)
    models["assay_participation_winning_route"] = modality_keys.map(
        participation_by_modality["assay_participation_winning_route"]
    ).astype(str)
    models["assay_participation_selection_route"] = modality_keys.map(
        participation_by_modality["assay_participation_selection_route"]
    ).astype(str)
    models["assay_participation_selected"] = modality_keys.map(
        participation_by_modality["selection_status"].eq("selected")
    ).astype(bool)
    models["assay_participation_order"] = modality_keys.map(
        participation_by_modality["partial_conjunction_order"]
    ).astype(int)
    models["assay_participation_family_size"] = modality_keys.map(
        participation_by_modality["n_formally_eligible_features"]
    ).astype(int)
    models["assay_participation_contributing_feature_count"] = modality_keys.map(
        participation_by_modality["contributing_feature_count"]
    ).astype(int)
    models["assay_participation_minimum_witness_view_count"] = modality_keys.map(
        participation_by_modality["minimum_witness_view_count"]
    ).astype(int)
    models["assay_participation_multiview_replication_gate"] = modality_keys.map(
        participation_by_modality["multiview_replication_gate"]
    ).astype(bool)
    selected_reporters = modality_keys.map(
        participation_by_modality["selected_reporter"]
    ).astype(str)
    models["assay_participation_reporter"] = (
        models["assay_participation_selected"]
        & models["feature"].astype(str).eq(selected_reporters)
    )

    # Strong world-level control for admitting modalities into a multi-assay
    # claim.  One synchronized null repetition contains one shared Rademacher
    # multiplier draw per assay.  Its maximum over every feature and assay is a
    # maxT reference.  Comparing each assay maximum with this same distribution
    # controls the chance that *any* null assay joins a real one.
    modality_null_max: dict[str, FloatArray] = {}
    modality_observed_max: dict[str, float] = {}
    for modality, group in models.loc[eligible_mask].groupby("modality", sort=True):
        features = list(group["feature"].astype(str))
        modality_null_max[str(modality)] = np.max(
            np.vstack([null_conditional_wald[feature] for feature in features]), axis=0
        )
        modality_observed_max[str(modality)] = max(
            float(reference_conditional_wald[feature]) for feature in features
        )
    global_null_max = (
        np.max(np.vstack(list(modality_null_max.values())), axis=0)
        if modality_null_max
        else np.zeros(cfg.null_repetitions, dtype=np.float64)
    )
    models["feature_global_maxT_p"] = [
        (
            (
                1.0
                + float(
                    np.sum(
                        global_null_max
                        >= float(reference_conditional_wald[str(feature)])
                    )
                )
            )
            / (cfg.null_repetitions + 1.0)
            if bool(eligible)
            else 1.0
        )
        for feature, eligible in zip(models["feature"], eligible_mask, strict=True)
    ]
    models["feature_global_maxT_confirmation_gate"] = (
        models["feature_global_maxT_p"]
        <= cfg.maximum_hyperedge_error_probability
    )
    assay_max_p = {
        modality: (1.0 + float(np.sum(global_null_max >= observed)))
        / (cfg.null_repetitions + 1.0)
        for modality, observed in modality_observed_max.items()
    }
    models["assay_global_maxT_p"] = [
        float(assay_max_p.get(str(modality), 1.0)) for modality in models["modality"]
    ]
    models["assay_global_maxT_admission_gate"] = (
        models["assay_global_maxT_p"]
        <= cfg.maximum_hyperedge_error_probability
    )
    models["direction_stability_diagnostic"] = (
        models["median_conditional_score_direction_stability"]
        >= cfg.minimum_median_score_direction_stability
    )
    # A feature near the outer maxT boundary must also reproduce its score
    # direction across model views.  This is a purely subtractive precision
    # guard: p <= alpha/2 retains the original maxT decision, whereas
    # alpha/2 < p <= alpha additionally needs stable direction.  It therefore
    # cannot weaken the existing world-level maxT error control.
    models["feature_maxT_unconditional_confirmation_gate"] = (
        models["feature_global_maxT_p"]
        <= (
            cfg.maximum_hyperedge_error_probability
            * cfg.unstable_feature_maxT_fraction
        )
    )
    models["feature_direction_stability_confirmation_gate"] = (
        models["feature_maxT_unconditional_confirmation_gate"]
        | models["direction_stability_diagnostic"]
    )
    if len(modality_null_max) >= cfg.minimum_modalities:
        ordered_assay_maxima = sorted(modality_observed_max.values(), reverse=True)
        conjunction_statistic = float(
            ordered_assay_maxima[cfg.minimum_modalities - 1]
        )
        # Under the partial-conjunction null, as many as r-1 assays may carry
        # real signal.  The observed r-th maximum can therefore be the maximum
        # of all remaining null assays.  Compare with the full world maximum,
        # not the anti-conservative r-th complete-null order statistic.
        multimodality_p = (
            1.0 + float(np.sum(global_null_max >= conjunction_statistic))
        ) / (cfg.null_repetitions + 1.0)
    else:
        multimodality_p = 1.0
    statuses = []
    reasons = []
    for row_index, record in enumerate(models.to_dict(orient="records")):
        if not bool(record["formal_test_eligible"]):
            statuses.append("unresolved")
            reasons.append(str(record["formal_test_abstention_reason"]))
            continue
        failures = []
        if float(record["empirical_null_fdr"]) > cfg.maximum_empirical_fdr:
            failures.append("empirical_null_fdr")
        if not bool(record["assay_global_maxT_admission_gate"]):
            failures.append("assay_global_maxT_admission")
        if not bool(record["feature_global_maxT_confirmation_gate"]):
            failures.append("feature_global_maxT_confirmation")
        if not bool(record["feature_direction_stability_confirmation_gate"]):
            failures.append("feature_direction_stability_confirmation")
        if int(conditional_wald_view_count[row_index]) < (
            cfg.minimum_conditional_wald_views
        ):
            failures.append("conditional_wald_view_support")
        if failures:
            statuses.append("unresolved")
            reasons.append(";".join(failures))
        else:
            statuses.append("selected")
            reasons.append("")
    models["pre_redundancy_selection_status"] = statuses
    models["pre_redundancy_abstention_reason"] = reasons
    models["selection_status"] = statuses
    models["abstention_reason"] = reasons

    # MaxT controls whether a feature may enter the reported state layer.  It
    # does not, by itself, distinguish one unstable chance hit from several
    # reproducible members in the same admitted assay.  Apply the already
    # declared direction-stability threshold as a second, strictly
    # subtractive confirmation layer.  If at least one preliminary member in
    # an assay is direction-stable, every unstable preliminary member is
    # demoted.  If none is stable, retain only the strongest already-maxT-
    # confirmed member so this precision filter cannot erase an admitted
    # assay or add a feature that failed the original hierarchy.
    models["redundancy_direction_stability_gate"] = models[
        "direction_stability_diagnostic"
    ].astype(bool)
    models["redundancy_fallback_selected"] = False
    models["redundancy_demoted"] = False
    preliminary_mask = models["selection_status"].eq("selected")
    for _, assay_indices in models.loc[preliminary_mask].groupby(
        "modality", sort=True
    ).groups.items():
        assay_indices = list(assay_indices)
        stable_indices = [
            index
            for index in assay_indices
            if bool(models.at[index, "redundancy_direction_stability_gate"])
        ]
        if stable_indices:
            retained_indices = set(stable_indices)
        else:
            ranked = models.loc[assay_indices].sort_values(
                [
                    "feature_global_maxT_p",
                    "reference_conditional_wald",
                    "feature",
                ],
                ascending=[True, False, True],
                kind="mergesort",
            )
            fallback_index = ranked.index[0]
            retained_indices = {fallback_index}
            models.at[fallback_index, "redundancy_fallback_selected"] = True
        demoted_indices = [
            index for index in assay_indices if index not in retained_indices
        ]
        if demoted_indices:
            models.loc[demoted_indices, "selection_status"] = "unresolved"
            models.loc[
                demoted_indices, "abstention_reason"
            ] = "within_assay_redundancy_direction_instability"
            models.loc[demoted_indices, "redundancy_demoted"] = True

    # An assay with a replicated core of world-maxT-strong members does not
    # need a much weaker tail member to establish either assay participation
    # or the shared-state hyperedge.  Remove such tails only after the original
    # hierarchy and redundancy confirmation have selected them.  This is
    # strictly subtractive: it cannot add a feature, remove a strong-core
    # member, erase the last member of an assay, or change the separate assay-
    # participation decision.
    post_redundancy_selected = models["selection_status"].eq("selected")
    models["within_assay_strong_core_member"] = (
        post_redundancy_selected
        & models["feature_global_maxT_p"].le(
            cfg.within_assay_strong_core_maxT_p
        )
    )
    strong_core_counts = (
        models.loc[post_redundancy_selected]
        .groupby("modality", sort=True)["within_assay_strong_core_member"]
        .sum()
        .astype(int)
        .to_dict()
    )
    models["within_assay_strong_core_count"] = (
        models["modality"]
        .astype(str)
        .map(strong_core_counts)
        .fillna(0)
        .astype(int)
    )
    models["within_assay_weak_maxT_tail_demoted"] = False
    weak_tail_mask = (
        post_redundancy_selected
        & models["within_assay_strong_core_count"].ge(
            cfg.within_assay_strong_core_minimum_members
        )
        & models["feature_global_maxT_p"].gt(
            cfg.within_assay_strong_core_maxT_p
        )
    )
    if weak_tail_mask.any():
        models.loc[weak_tail_mask, "selection_status"] = "unresolved"
        models.loc[
            weak_tail_mask, "abstention_reason"
        ] = "within_assay_strong_maxT_core_weak_tail"
        models.loc[
            weak_tail_mask, "within_assay_weak_maxT_tail_demoted"
        ] = True
    models["direct_layer_overlap"] = models["feature"].astype(str).isin(direct_layer)
    models["direct_layer_annotation_available"] = bool(direct_layer)
    selected = models.loc[models["selection_status"] == "selected"]
    selected_modalities = sorted(set(selected["modality"].astype(str)))
    participating_modalities = sorted(
        assay_state_participation.loc[
            assay_state_participation["selection_status"].eq("selected"),
            "modality",
        ].astype(str)
    )
    hyperedge_selected = (
        len(selected_modalities) >= cfg.minimum_modalities
        and multimodality_p <= cfg.maximum_hyperedge_error_probability
    )
    subspace_id = "shared_latent_state_subspace_001"
    models["subspace_id"] = np.where(
        models["selection_status"] == "selected", subspace_id, ""
    )
    hyperedges = pd.DataFrame(
        [
            {
                "subspace_id": subspace_id,
                "edge_type": "cross_assay_shared_state_subspace",
                "modalities": ";".join(selected_modalities),
                "member_features": ";".join(sorted(selected["feature"].astype(str))),
                "n_modalities": len(selected_modalities),
                "n_features": len(selected),
                "selection_status": "selected" if hyperedge_selected else "unresolved",
                "empirical_multimodality_p": multimodality_p,
                "error_control_method": (
                    "synchronized_conditional_score_wild_feature_maxT_"
                    "confirmation_borderline_direction_stability_"
                    "within_assay_redundancy_confirmation_"
                    "within_assay_strong_maxT_core_weak_tail_confirmation_"
                    "assay_maxT_admission_"
                    "within_assay_BH_conservative_order_"
                    f"{cfg.minimum_modalities}_partial_conjunction"
                ),
                "conditional_on_common_state_model": True,
                "causal_interpretation": "hypothesis_only",
                "direction_claimed": False,
                "exact_axis_pairing_claimed": False,
                "direct_causal_claimed": False,
                "abstention_reason": (
                    ""
                    if hyperedge_selected
                    else (
                        "insufficient_modalities"
                        if len(selected_modalities) < cfg.minimum_modalities
                        else "multimodality_maxT"
                    )
                ),
            }
        ]
    )
    provenance = {
        "schema_version": "shared_state_subspace_v4.v10",
        "method": (
            "rotation_invariant_crossfit_depth_orthogonal_score_"
            "synchronized_assay_maxT_hierarchical_BH_multiview_stability_"
            "within_assay_redundancy_confirmation_strong_maxT_core_"
            "weak_tail_confirmation_assay_omnibus_participation"
        ),
        "config": asdict(cfg),
        "n_features": len(models),
        "n_formally_eligible_features": int(models["formal_test_eligible"].sum()),
        "n_prefit_binary_abstentions": int((~models["binary_event_gate"]).sum()),
        "n_prefit_fold_or_continuous_abstentions": int(
            (models["binary_event_gate"] & ~models["formal_test_eligible"]).sum()
        ),
        "n_selected_features": len(selected),
        "selected_modalities": selected_modalities,
        "assay_state_participation": {
            "edge_type": "assay_shared_state_participation",
            "n_selected_assays": len(participating_modalities),
            "selected_modalities": participating_modalities,
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
            "assay_participation_null": (
                "no_state_responsive_features_in_the_assay"
            ),
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
            "minimum_witness_views": (
                cfg.minimum_assay_state_participation_views
            ),
            "affects_feature_membership": False,
            "affects_existing_typed_subspace_hyperedge": False,
            "reporter_or_witness_implies_feature_membership": False,
            "membership_annotation_columns": [
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
            ],
            "records": assay_state_participation.to_dict(orient="records"),
        },
        "direct_layer_overlap_features": sorted(direct_layer),
        "estimand": {
            "edge_type": "cross_assay_shared_state_subspace",
            "exact_cross_assay_feature_pairing": False,
            "exact_latent_axis_identity": False,
            "direct_causal_direction": False,
            "assay_local_state_membership": True,
            "shared_cross_assay_state_identity": "conditional_hypothesis_only",
            "selected_estimand": "depth_orthogonal_state_response",
            "scope": "first_moment_linear_state_response_after_flexible_depth_nuisance",
        },
        "error_control": {
            "binary_eligibility": (
                "pre_fit_exclusion_before_feature_models_BH_maxT_and_conjunction"
            ),
            "ineligible_placeholder_policy": {
                "formal_and_descriptive_statistics": 0.0,
                "empirical_p_fdr_and_feature_maxT_p": 1.0,
                "selection_status": "unresolved",
                "abstention_reasons": [
                    "insufficient_binary_events",
                    "single_class_training_fold",
                    "constant_continuous_response",
                ],
                "included_in_multiplicity_families": False,
            },
            "membership_policy": (
                "feature_world_maxT_confirmation_and_assay_world_maxT_"
                "admission_then_within_assay_BH_and_four_of_six_"
                "multiview_support_then_subtractive_redundancy_and_strong_"
                "core_weak_tail_confirmation"
            ),
            "assay_admission": "synchronized_world_assay_maxT_FWER",
            "within_admitted_assay_feature_level": (
                "within_assay_BH_depth_orthogonal_score"
            ),
            "feature_global_maxT": "membership_confirmation_gate",
            "direction_stability": (
                "required_for_borderline_feature_maxT_and_used_as_a_strictly_"
                "subtractive_within_assay_redundancy_confirmation"
            ),
            "unstable_feature_maxT_fraction": cfg.unstable_feature_maxT_fraction,
            "borderline_direction_gate_is_subtractive": True,
            "redundancy_confirmation": {
                "threshold": cfg.minimum_median_score_direction_stability,
                "stable_alternative_policy": (
                    "retain_only_direction_stable_preliminary_members"
                ),
                "no_stable_member_fallback": (
                    "retain_one_preliminary_member_by_min_feature_global_maxT_"
                    "p_then_max_reference_wald_then_lexical_feature"
                ),
                "may_add_preliminary_members": False,
                "may_remove_last_preliminary_member_from_an_assay": False,
                "n_demoted": int(models["redundancy_demoted"].sum()),
                "n_fallback_selected": int(
                    models["redundancy_fallback_selected"].sum()
                ),
            },
            "strong_core_weak_tail_confirmation": {
                "strong_core_maxT_p": cfg.within_assay_strong_core_maxT_p,
                "minimum_strong_core_members": (
                    cfg.within_assay_strong_core_minimum_members
                ),
                "strong_comparison": "feature_global_maxT_p_at_most_threshold",
                "weak_tail_comparison": "feature_global_maxT_p_above_threshold",
                "evaluated_after_redundancy_confirmation": True,
                "affects_assay_participation": False,
                "may_add_members": False,
                "may_remove_strong_core_members": False,
                "may_remove_last_member_from_an_assay": False,
                "n_demoted": int(
                    models["within_assay_weak_maxT_tail_demoted"].sum()
                ),
            },
            "modality_and_hyperedge_level": (
                "synchronized_world_maxT_conservative_order_2_partial_conjunction"
            ),
            "empirical_multimodality_p": multimodality_p,
        },
        "descriptive_state_depth_decomposition": {
            "method": "cross_fitted_state_plus_orthogonal_depth_predictive_fit",
            "fields": list(_DESCRIPTIVE_DECOMPOSITION_FIELDS),
            "affects_selection": False,
            "shared_state_depth_allocation": "state",
            "calibrated_primary_estimand": "depth_orthogonal_state_response",
        },
    }
    return SharedStateSubspaceResult(
        feature_membership=models,
        typed_subspace_hyperedges=hyperedges,
        assay_state_participation=assay_state_participation,
        provenance=provenance,
    )


def estimate_conditional_wild_null(
    inputs: loading_groups.LoadingInputs,
    *,
    config: SharedStateSubspaceConfig | None = None,
) -> tuple[dict[str, float], dict[str, FloatArray]]:
    """Cross-fitted orthogonal-score statistic and synchronized wild null."""

    cfg = config or SharedStateSubspaceConfig()
    output: dict[str, FloatArray] = {}
    reference: dict[str, float] = {}
    for modality, assay in sorted(inputs.assays.items()):
        eligible_features = [
            str(feature)
            for feature in assay.focal.columns
            if _formal_feature_eligibility(
                modality, assay, str(feature), cfg
            )[1]
        ]
        if not eligible_features:
            continue
        nuisance = _fit_assay_score_nuisance(
            assay,
            cfg,
            _stable_seed(cfg.random_seed, "conditional_wald_reference_folds", modality),
        )
        rng = np.random.default_rng(
            _stable_seed(cfg.random_seed, "wild_multiplier", modality)
        )
        multipliers = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float64),
            size=(cfg.null_repetitions, len(assay.cell_ids)),
            replace=True,
        )
        for feature_name in eligible_features:
            fit = _orthogonal_score_feature(
                modality,
                feature_name,
                assay,
                nuisance,
                cfg,
                _stable_seed(
                    cfg.random_seed, "conditional_wald_reference", feature_name
                ),
            )
            reference[feature_name] = fit.statistic
            bootstrap_means = (
                multipliers @ fit.centered_scores / len(assay.cell_ids)
            )
            bootstrap_statistics = len(assay.cell_ids) * np.einsum(
                "bi,ij,bj->b",
                bootstrap_means,
                fit.inverse_score_covariance,
                bootstrap_means,
                optimize=True,
            )
            output[feature_name] = np.asarray(
                np.maximum(bootstrap_statistics, 0.0), dtype=np.float64
            )
    return reference, output


def run_shared_state_subspace(
    focal_discovery_dir: Path | str,
    safe_embedding_path: Path | str,
    out_dir: Path | str,
    *,
    direct_layer_features: Sequence[str] = (),
    direct_layer_evidence_path: Path | str | None = None,
    config: SharedStateSubspaceConfig | None = None,
) -> dict[str, Any]:
    """Run and hash-lock the dimension-agnostic discovery layer."""

    cfg = config or SharedStateSubspaceConfig()
    direct_layer_evidence_sha256 = ""
    if direct_layer_features and direct_layer_evidence_path is None:
        raise ValueError("Direct-layer annotations require frozen motif evidence")
    if direct_layer_evidence_path is not None:
        evidence_features, direct_layer_evidence_sha256 = (
            load_observable_direct_layer_overlap(direct_layer_evidence_path)
        )
        if direct_layer_features and set(map(str, direct_layer_features)) != set(
            evidence_features
        ):
            raise ValueError("Explicit direct-layer annotations disagree with provenance")
        direct_layer_features = evidence_features
    raw_inputs = loading_groups.load_loading_inputs(
        focal_discovery_dir, safe_embedding_path
    )
    embedding_freeze_evidence = _validate_safe_embedding_freeze(raw_inputs)
    inputs, assay_coordinate_scales = _rotation_invariant_local_inputs(raw_inputs)
    observed = _fit_observed_evidence(inputs, cfg)
    input_features = {
        str(feature)
        for assay in inputs.assays.values()
        for feature in assay.focal.columns
    }
    if set(observed["feature"].astype(str)) != input_features:
        raise ValueError("Self-fitted views and raw observed focal inputs disagree")
    reference, null = estimate_conditional_wild_null(inputs, config=cfg)
    result = select_shared_subspace_membership(
        observed,
        reference,
        null,
        direct_layer_features=direct_layer_features,
        config=cfg,
    )
    result.provenance["information_boundary"] = {
        "consumed": [
            "observed_focal_variables",
            "assay_local_background_state_coordinates",
            "observed_leave_focal_out_depth",
            "self_fitted_crossfit_orthogonal_score_views",
            "self_fitted_nonselecting_state_depth_decomposition",
            "optional_frozen_direct_layer_annotation_if_provided",
        ],
        "not_consumed": [
            "true_latent_state",
            "simulator_roles",
            "truth_graph",
            "scorer",
            "paired_cells",
            "cross_assay_feature_identity",
            "external_data",
        ],
        "cross_assay_rows_paired": False,
        "exact_cross_assay_feature_pairs_inferred": False,
        "direct_causal_claims_emitted": False,
        "truth_or_scorer_used": False,
    }
    result.provenance["direct_layer_annotation"] = {
        "available": bool(direct_layer_features),
        "features": sorted(map(str, direct_layer_features)),
        "affects_subspace_statistics_or_selection": False,
        "source_sha256": direct_layer_evidence_sha256,
    }
    result.provenance["coordinate_normalization"] = {
        "method": "assay_local_center_and_single_scalar_RMS",
        "pooled_axiswise_loader_scaling_undone": True,
        "assay_scalar_scales": assay_coordinate_scales,
        "orthogonal_rotation_invariant": True,
    }
    result.provenance["formal_state_embedding"] = embedding_freeze_evidence
    result.provenance["observed_model_views"] = {
        "n_views": cfg.observed_model_views,
        "minimum_views_for_membership": cfg.minimum_conditional_wald_views,
        "independent_data_replicates": False,
        "purpose": "crossfit_fold_algorithmic_stability",
    }
    result.provenance["conditional_wild_null"] = {
        "method": "centered_orthogonal_score_Rademacher_multiplier_bootstrap",
        "same_weights_shared_by_features_within_assay_and_repetition": True,
        "repetition_index_synchronized_across_assays": True,
        "estimand_note": (
            "depth-orthogonal first-moment state response; state variation "
            "predictable from observed biological library depth is discarded"
        ),
    }
    output = Path(out_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    input_roots = {inputs.focal_root, inputs.embedding_path.parent}
    output_resolved = output.resolve(strict=False)
    if any(
        output_resolved == root or root in output_resolved.parents
        for root in input_roots
    ):
        raise ValueError("Output must differ from raw discovery inputs")
    output.mkdir(parents=True, exist_ok=True)
    membership_path = output / "subspace_feature_membership.tsv"
    hyperedge_path = output / "typed_subspace_hyperedges.tsv"
    provenance_path = output / "subspace_provenance.json"
    result.feature_membership.to_csv(membership_path, sep="\t", index=False)
    result.typed_subspace_hyperedges.to_csv(hyperedge_path, sep="\t", index=False)
    provenance_path.write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    source = Path(__file__).resolve(strict=True)
    estimator_source = Path(loading_groups.__file__).resolve(strict=True)
    dependency_source_sha256 = {
        estimator_source.name: _sha256(estimator_source),
    }
    if embedding_freeze_evidence.get("layout") == "assay_local_state_embedding_v4":
        dependency_source_sha256.update(
            {
                str(name): str(digest)
                for name, digest in embedding_freeze_evidence[
                    "upstream_algorithm_source_sha256"
                ].items()
            }
        )
    manifest = {
        "schema_version": "shared_state_subspace_v4.frozen_discovery.v10",
        "algorithm_source_sha256": _sha256(source),
        "dependency_source_sha256": dependency_source_sha256,
        "runtime_dependency_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": asdict(cfg),
        "raw_input_sha256": inputs.input_hashes,
        "safe_embedding_freeze_evidence": embedding_freeze_evidence,
        "reusable_loading_evidence_sha256": {},
        "reusable_loading_evidence_policy": (
            "not_consumed; all observed views self-fitted from hash-locked raw inputs"
        ),
        "direct_layer_overlap_evidence_sha256": direct_layer_evidence_sha256,
        "output_sha256": {
            membership_path.name: _sha256(membership_path),
            hyperedge_path.name: _sha256(hyperedge_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "frozen_before_scorer_read": True,
        "information_boundary": result.provenance["information_boundary"],
    }
    (output / "frozen_discovery_manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    return {
        "n_selected_features": result.provenance["n_selected_features"],
        "selected_modalities": result.provenance["selected_modalities"],
        "n_assay_state_participation": result.provenance[
            "assay_state_participation"
        ]["n_selected_assays"],
        "assay_state_participation_modalities": result.provenance[
            "assay_state_participation"
        ]["selected_modalities"],
        "hyperedge_status": result.typed_subspace_hyperedges.iloc[0][
            "selection_status"
        ],
        "truth_or_scorer_used": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal-discovery-dir", type=Path, required=True)
    parser.add_argument("--safe-embedding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--direct-layer-evidence", type=Path)
    parser.add_argument("--null-repetitions", type=int, default=999)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_shared_state_subspace(
        args.focal_discovery_dir,
        args.safe_embedding,
        args.out_dir,
        direct_layer_evidence_path=args.direct_layer_evidence,
        config=SharedStateSubspaceConfig(null_repetitions=args.null_repetitions),
    )
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SharedStateSubspaceConfig",
    "SharedStateSubspaceResult",
    "estimate_conditional_wild_null",
    "freeze_observable_direct_layer_overlap",
    "load_observable_direct_layer_overlap",
    "run_shared_state_subspace",
    "select_shared_subspace_membership",
]
