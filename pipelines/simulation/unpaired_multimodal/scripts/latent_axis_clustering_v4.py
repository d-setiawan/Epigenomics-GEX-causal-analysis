#!/usr/bin/env python3
"""Observed-only projective clustering of assay-local state-loading groups.

``loading_group_projection_v4`` deliberately stops at pairwise group matches.
This module provides the next, separately auditable layer: it clusters the
assay-local group centroids into sign-invariant latent-axis hyperedges.  The
inputs are only selected loading groups, their observed feature membership,
and the complete loading evidence views from which those groups were built.

The clustering rule is conservative complete linkage.  Two clusters merge
only when *every* cross-cluster group pair comes from a different assay and
passes the same multi-view projective-similarity gate.  Consequently a chain
of pairwise similarities cannot manufacture an axis.  All per-view pair and
group-to-axis evidence is returned, so thresholds can be recalibrated without
refitting the loading models.  A feature-pair projection is optional and is
emitted only when both assay-local representatives are unique.

An inferred axis means shared response to an estimated biological-state
coordinate.  It is not a direct causal edge, a direction, or a cell pairing.
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
from scipy.special import betainc

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
class LatentAxisClusteringConfig:
    """Frozen projective-clustering and abstention thresholds."""

    min_pair_median_absolute_cosine: float = 0.85
    min_pair_lower_absolute_cosine: float = 0.78
    min_pair_view_support: float = 2.0 / 3.0
    max_pair_view_sd: float = 0.15
    min_group_confidence: float = 0.25
    max_projective_null_q: float = 1.0
    min_axis_modalities: int = 2
    min_member_axis_lower_cosine: float = 0.80
    min_member_axis_view_support: float = 2.0 / 3.0
    min_member_relative_axis_margin: float = 0.05
    max_axis_sign_frustration: float = 0.10
    representative_absolute_cosine: float = 0.80
    representative_view_support: float = 2.0 / 3.0
    representative_relative_margin: float = 0.05
    one_group_per_modality_per_axis: bool = True
    min_singleton_anchor_members: int = 2
    min_singleton_oof_partial_state_fit: float = 0.02
    min_singleton_lower_oof_partial_state_fit: float = 0.015
    min_singleton_loading_norm: float = 0.60
    min_singleton_minimum_loading_norm: float = 0.55
    min_singleton_direction_stability: float = 0.95
    min_singleton_minimum_direction_stability: float = 0.90
    min_singleton_positive_fit_view_support: float = 0.80
    min_singleton_median_absolute_cosine: float = 0.90
    min_singleton_lower_absolute_cosine: float = 0.88
    min_singleton_view_support: float = 0.80
    max_singleton_view_sd: float = 0.08
    max_singleton_projective_null_q: float = 0.25
    min_singleton_axis_relative_margin: float = 0.15
    min_singleton_candidate_relative_margin: float = 0.05

    def __post_init__(self) -> None:
        bounded = (
            self.min_pair_median_absolute_cosine,
            self.min_pair_lower_absolute_cosine,
            self.min_pair_view_support,
            self.max_pair_view_sd,
            self.min_group_confidence,
            self.max_projective_null_q,
            self.min_member_axis_lower_cosine,
            self.min_member_axis_view_support,
            self.min_member_relative_axis_margin,
            self.max_axis_sign_frustration,
            self.representative_absolute_cosine,
            self.representative_view_support,
            self.representative_relative_margin,
            self.min_singleton_direction_stability,
            self.min_singleton_minimum_direction_stability,
            self.min_singleton_positive_fit_view_support,
            self.min_singleton_median_absolute_cosine,
            self.min_singleton_lower_absolute_cosine,
            self.min_singleton_view_support,
            self.max_singleton_view_sd,
            self.max_singleton_projective_null_q,
            self.min_singleton_axis_relative_margin,
            self.min_singleton_candidate_relative_margin,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("Cosine, probability, variability, and margin thresholds must be in [0, 1]")
        if self.min_pair_lower_absolute_cosine > self.min_pair_median_absolute_cosine:
            raise ValueError("The lower-cosine threshold may not exceed the median threshold")
        if self.min_axis_modalities < 2:
            raise ValueError("A selected cross-assay axis requires at least two modalities")
        if self.min_singleton_anchor_members < 2:
            raise ValueError("A singleton bridge requires a coherent group with at least two members")
        nonnegative = (
            self.min_singleton_oof_partial_state_fit,
            self.min_singleton_lower_oof_partial_state_fit,
            self.min_singleton_loading_norm,
            self.min_singleton_minimum_loading_norm,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Singleton fit and loading-norm thresholds may not be negative")


@dataclass
class LatentAxisClusteringResult:
    group_pair_view_evidence: pd.DataFrame
    group_pair_evidence: pd.DataFrame
    axis_candidates: pd.DataFrame
    group_axis_view_evidence: pd.DataFrame
    group_axis_evidence: pd.DataFrame
    typed_axis_hyperedges: pd.DataFrame
    representative_evidence: pd.DataFrame
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


def _projective_null_p(cosine: float, dimensions: int) -> float:
    squared = float(np.clip(cosine, 0.0, 1.0) ** 2)
    return float(1.0 - betainc(0.5, 0.5 * (dimensions - 1), squared))


def _forbidden_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_COLUMN_TOKENS)
    )


def _unit(vector: FloatArray) -> FloatArray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _validate_inputs(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if len(views) < 2:
        raise ValueError("Latent-axis stability requires at least two complete loading views")
    if len({view.view_id for view in views}) != len(views):
        raise ValueError("Loading evidence view IDs must be unique")
    selected = groups.copy()
    members = membership.copy()
    for label, frame in (("groups", selected), ("membership", members)):
        forbidden = _forbidden_columns(frame)
        if forbidden:
            raise ValueError(f"{label} contains forbidden columns {forbidden}")
    group_required = {
        "loading_group",
        "modality",
        "members",
        "group_confidence",
        "selection_status",
    }
    membership_required = {"loading_group", "feature", "modality"}
    for label, required, frame in (
        ("groups", group_required, selected),
        ("membership", membership_required, members),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns {missing}")
    selected["selection_status"] = selected["selection_status"].astype(str)
    selected = selected.loc[
        selected["selection_status"].isin({"selected", "singleton_candidate"})
    ].copy()
    if not selected["selection_status"].eq("selected").any():
        raise ValueError("At least one selected assay-local loading group is required")
    selected["loading_group"] = selected["loading_group"].astype(str)
    selected["modality"] = selected["modality"].astype(str)
    selected["members"] = selected["members"].astype(str)
    selected["group_confidence"] = pd.to_numeric(
        selected["group_confidence"], errors="raise"
    )
    if selected["loading_group"].duplicated().any():
        raise ValueError("Loading-group and singleton-candidate IDs must be unique")
    if not np.isfinite(selected["group_confidence"]).all():
        raise ValueError("Group confidence must be finite")

    members["loading_group"] = members["loading_group"].astype(str)
    members["feature"] = members["feature"].astype(str)
    members["modality"] = members["modality"].astype(str)
    members = members.loc[members["loading_group"].isin(selected["loading_group"])].copy()
    if members.duplicated(["loading_group", "feature"]).any():
        raise ValueError("Group membership must contain unique group-feature rows")
    group_lookup = selected.set_index("loading_group")
    membership_lookup = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in members.groupby("loading_group")
    }
    if set(membership_lookup) != set(group_lookup.index):
        raise ValueError("Every selected loading group must have membership rows")
    for group_id, features in membership_lookup.items():
        expected = sorted(filter(None, str(group_lookup.loc[group_id, "members"]).split(";")))
        if features != expected:
            raise ValueError(f"Membership disagrees with members field for {group_id}")
        modalities = set(members.loc[members["loading_group"] == group_id, "modality"])
        if modalities != {str(group_lookup.loc[group_id, "modality"])}:
            raise ValueError(f"Membership modality disagrees for {group_id}")
        status = str(group_lookup.loc[group_id, "selection_status"])
        if status == "singleton_candidate" and len(features) != 1:
            raise ValueError("A singleton candidate must contain exactly one feature")
        if status == "singleton_candidate" and str(
            group_lookup.loc[group_id, "modality"]
        ) == "rna":
            raise ValueError("RNA responses are selected input groups, not bridge candidates")

    reference_features = sorted(views[0].feature_models["feature"].astype(str))
    reference_coordinates = sorted(
        views[0].loadings["state_coordinate"].astype(str).unique(),
        key=lambda value: int(value.removeprefix("integration__dim")),
    )
    required_features = sorted(set(members["feature"]))
    if not set(required_features).issubset(reference_features):
        raise ValueError("Loading-group members must occur in every loading view")
    reference_labels = views[0].feature_models.set_index("feature")["modality"].astype(str)
    for view in views:
        if sorted(view.feature_models["feature"].astype(str)) != reference_features:
            raise ValueError("Every loading view must have the same complete feature set")
        coordinates = sorted(
            view.loadings["state_coordinate"].astype(str).unique(),
            key=lambda value: int(value.removeprefix("integration__dim")),
        )
        if coordinates != reference_coordinates:
            raise ValueError("Every loading view must have the same state coordinates")
        labels = view.feature_models.set_index("feature")["modality"].astype(str)
        if not labels.sort_index().equals(reference_labels.sort_index()):
            raise ValueError("Observable modality labels must agree across views")
    for row in members.itertuples(index=False):
        if str(reference_labels.loc[str(row.feature)]) != str(row.modality):
            raise ValueError("Group membership modality disagrees with loading evidence")
    return (
        selected.sort_values("loading_group").reset_index(drop=True),
        members.sort_values(["loading_group", "feature"]).reset_index(drop=True),
        reference_coordinates,
        reference_features,
    )


def _feature_vectors(
    views: Sequence[LoadingEvidenceView], coordinates: Sequence[str]
) -> dict[tuple[str, str], FloatArray]:
    output: dict[tuple[str, str], FloatArray] = {}
    for view in views:
        pivot = view.loadings.pivot(
            index="feature", columns="state_coordinate", values="conditional_coefficient"
        )[list(coordinates)]
        for feature, row in pivot.iterrows():
            output[(view.view_id, str(feature))] = _unit(
                row.to_numpy(dtype=np.float64)
            )
    return output


def _group_centroids(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    feature_vectors: Mapping[tuple[str, str], FloatArray],
) -> dict[tuple[str, str], FloatArray]:
    members = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in membership.groupby("loading_group")
    }
    output: dict[tuple[str, str], FloatArray] = {}
    for group_id in groups["loading_group"].astype(str):
        for view in views:
            vectors = [feature_vectors[(view.view_id, feature)] for feature in members[group_id]]
            reference = vectors[0]
            aligned = [vector if float(np.dot(vector, reference)) >= 0 else -vector for vector in vectors]
            output[(view.view_id, group_id)] = _unit(np.mean(np.vstack(aligned), axis=0))
    return output


def _pair_evidence(
    groups: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    centroids: Mapping[tuple[str, str], FloatArray],
    dimensions: int,
    config: LatentAxisClusteringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_lookup = groups.set_index("loading_group")
    group_ids = sorted(group_lookup.index.astype(str))
    view_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(group_ids):
        for right in group_ids[left_index + 1 :]:
            pair_key = f"{left}|||{right}"
            same_modality = str(group_lookup.loc[left, "modality"]) == str(
                group_lookup.loc[right, "modality"]
            )
            for view in views:
                cosine = abs(float(np.dot(centroids[(view.view_id, left)], centroids[(view.view_id, right)])))
                view_rows.append(
                    {
                        "group_pair_key": pair_key,
                        "group1": left,
                        "group2": right,
                        "modality1": str(group_lookup.loc[left, "modality"]),
                        "modality2": str(group_lookup.loc[right, "modality"]),
                        "same_modality": same_modality,
                        "group1_status": str(
                            group_lookup.loc[left, "selection_status"]
                        ),
                        "group2_status": str(
                            group_lookup.loc[right, "selection_status"]
                        ),
                        "view_id": view.view_id,
                        "absolute_centroid_cosine": cosine,
                        "passes_config_pair_cosine": cosine
                        >= config.min_pair_median_absolute_cosine,
                    }
                )
    view_columns = [
        "group_pair_key",
        "group1",
        "group2",
        "modality1",
        "modality2",
        "same_modality",
        "group1_status",
        "group2_status",
        "view_id",
        "absolute_centroid_cosine",
        "passes_config_pair_cosine",
    ]
    view_frame = pd.DataFrame(view_rows, columns=view_columns)
    if view_frame.empty:
        aggregate_columns = [
            "group_pair_key",
            "group1",
            "group2",
            "modality1",
            "modality2",
            "same_modality",
            "group1_status",
            "group2_status",
            "n_views",
            "median_absolute_centroid_cosine",
            "lower_absolute_centroid_cosine",
            "minimum_absolute_centroid_cosine",
            "centroid_cosine_view_sd",
            "pair_cosine_view_support",
            "minimum_group_confidence",
            "projective_null_p",
            "projective_null_q",
            "passes_median_cosine",
            "passes_lower_cosine",
            "passes_view_support",
            "passes_view_sd",
            "passes_group_confidence",
            "passes_projective_null_q",
            "cluster_compatible",
            "singleton_anchor_pair",
            "selection_status",
        ]
        return view_frame, pd.DataFrame(columns=aggregate_columns)
    rows: list[dict[str, Any]] = []
    for keys, frame in view_frame.groupby(
        [
            "group_pair_key",
            "group1",
            "group2",
            "modality1",
            "modality2",
            "same_modality",
            "group1_status",
            "group2_status",
        ],
        sort=True,
    ):
        (
            pair_key,
            left,
            right,
            modality1,
            modality2,
            same_modality,
            group1_status,
            group2_status,
        ) = keys
        cosines = frame["absolute_centroid_cosine"].to_numpy(dtype=np.float64)
        median = float(np.median(cosines))
        rows.append(
            {
                "group_pair_key": pair_key,
                "group1": left,
                "group2": right,
                "modality1": modality1,
                "modality2": modality2,
                "same_modality": bool(same_modality),
                "group1_status": group1_status,
                "group2_status": group2_status,
                "n_views": len(cosines),
                "median_absolute_centroid_cosine": median,
                "lower_absolute_centroid_cosine": float(np.quantile(cosines, 0.10)),
                "minimum_absolute_centroid_cosine": float(np.min(cosines)),
                "centroid_cosine_view_sd": float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0,
                "pair_cosine_view_support": float(
                    np.mean(cosines >= config.min_pair_median_absolute_cosine)
                ),
                "minimum_group_confidence": float(
                    min(
                        group_lookup.loc[left, "group_confidence"],
                        group_lookup.loc[right, "group_confidence"],
                    )
                ),
                "projective_null_p": _projective_null_p(median, dimensions),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate["projective_null_q"] = _benjamini_hochberg(
        aggregate["projective_null_p"].to_numpy(dtype=np.float64)
    )
    aggregate["passes_median_cosine"] = (
        aggregate["median_absolute_centroid_cosine"]
        >= config.min_pair_median_absolute_cosine
    )
    aggregate["passes_lower_cosine"] = (
        aggregate["lower_absolute_centroid_cosine"]
        >= config.min_pair_lower_absolute_cosine
    )
    aggregate["passes_view_support"] = (
        aggregate["pair_cosine_view_support"] >= config.min_pair_view_support
    )
    aggregate["passes_view_sd"] = (
        aggregate["centroid_cosine_view_sd"] <= config.max_pair_view_sd
    )
    aggregate["passes_group_confidence"] = (
        aggregate["minimum_group_confidence"] >= config.min_group_confidence
    )
    aggregate["passes_projective_null_q"] = (
        aggregate["projective_null_q"] <= config.max_projective_null_q
    )
    aggregate["cluster_compatible"] = (
        ~aggregate["same_modality"]
        & aggregate["passes_median_cosine"]
        & aggregate["passes_lower_cosine"]
        & aggregate["passes_view_support"]
        & aggregate["passes_view_sd"]
        & aggregate["passes_group_confidence"]
        & aggregate["passes_projective_null_q"]
    )
    aggregate["singleton_anchor_pair"] = (
        aggregate[["group1_status", "group2_status"]]
        .apply(lambda row: set(map(str, row)) == {"selected", "singleton_candidate"}, axis=1)
    )
    aggregate["selection_status"] = np.select(
        [
            aggregate["cluster_compatible"],
            (~aggregate["same_modality"])
            & (
                aggregate["median_absolute_centroid_cosine"]
                >= config.min_pair_lower_absolute_cosine
            ),
        ],
        ["compatible", "unresolved"],
        default="unsupported",
    )
    return view_frame, aggregate.sort_values("group_pair_key").reset_index(drop=True)


def _complete_link_clusters(
    groups: pd.DataFrame,
    pair_evidence: pd.DataFrame,
    config: LatentAxisClusteringConfig,
) -> list[tuple[str, ...]]:
    clusters: list[tuple[str, ...]] = [
        (group_id,) for group_id in sorted(groups["loading_group"].astype(str))
    ]
    modalities = groups.set_index("loading_group")["modality"].astype(str).to_dict()
    pair_lookup = {
        (str(row.group1), str(row.group2)): row
        for row in pair_evidence.itertuples(index=False)
    }

    def lookup(left: str, right: str) -> Any:
        return pair_lookup[tuple(sorted((left, right)))]

    while True:
        candidates: list[tuple[float, str, int, int]] = []
        for left_index, left_cluster in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right_cluster = clusters[right_index]
                combined_modalities = [modalities[group] for group in (*left_cluster, *right_cluster)]
                if (
                    config.one_group_per_modality_per_axis
                    and len(combined_modalities) != len(set(combined_modalities))
                ):
                    continue
                cross_rows = [
                    lookup(left, right)
                    for left in left_cluster
                    for right in right_cluster
                ]
                if not cross_rows or not all(bool(row.cluster_compatible) for row in cross_rows):
                    continue
                merge_score = min(
                    float(row.median_absolute_centroid_cosine) for row in cross_rows
                )
                merged_key = ";".join(sorted((*left_cluster, *right_cluster)))
                candidates.append((-merge_score, merged_key, left_index, right_index))
        if not candidates:
            break
        _, _, left_index, right_index = min(candidates)
        merged = tuple(sorted((*clusters[left_index], *clusters[right_index])))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort()
    return sorted(clusters, key=lambda cluster: (cluster[0], len(cluster), cluster))


def _axis_centroid(
    cluster: Sequence[str],
    view_id: str,
    group_centroids: Mapping[tuple[str, str], FloatArray],
    group_confidence: Mapping[str, float],
) -> tuple[FloatArray, float, str]:
    # A projective medoid gives a deterministic reference without assigning a
    # biologically meaningful sign to the axis.
    medoid_scores = []
    for candidate in cluster:
        score = sum(
            abs(
                float(
                    np.dot(
                        group_centroids[(view_id, candidate)],
                        group_centroids[(view_id, other)],
                    )
                )
            )
            for other in cluster
        )
        medoid_scores.append((-score, -group_confidence[candidate], candidate))
    anchor = min(medoid_scores)[2]
    reference = group_centroids[(view_id, anchor)]
    aligned: dict[str, FloatArray] = {}
    for group in cluster:
        vector = group_centroids[(view_id, group)]
        aligned[group] = vector if float(np.dot(vector, reference)) >= 0 else -vector
    pair_signs = [
        float(np.dot(aligned[left], aligned[right]))
        for left_index, left in enumerate(cluster)
        for right in cluster[left_index + 1 :]
    ]
    frustration = (
        float(np.mean(np.asarray(pair_signs) < 0.0)) if pair_signs else 0.0
    )
    weights = np.asarray([max(group_confidence[group], 1e-6) for group in cluster])
    matrix = np.vstack([aligned[group] for group in cluster])
    return _unit(np.average(matrix, axis=0, weights=weights)), frustration, anchor


def _singleton_feature_stability(
    singleton_groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    config: LatentAxisClusteringConfig,
) -> pd.DataFrame:
    """Revalidate singleton stability from the complete observed loading views."""

    if singleton_groups.empty:
        return pd.DataFrame()
    feature_by_group = (
        membership.loc[
            membership["loading_group"].isin(singleton_groups["loading_group"])
        ]
        .set_index("loading_group")["feature"]
        .astype(str)
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for group in singleton_groups.itertuples(index=False):
        group_id = str(group.loading_group)
        feature = feature_by_group[group_id]
        models = pd.concat(
            [
                view.feature_models.loc[
                    view.feature_models["feature"].astype(str).eq(feature),
                    [
                        "oof_partial_state_fit",
                        "conditional_loading_norm",
                        "loading_direction_stability",
                    ],
                ].assign(view_id=view.view_id)
                for view in views
            ],
            ignore_index=True,
        )
        fits = models["oof_partial_state_fit"].to_numpy(dtype=float)
        norms = models["conditional_loading_norm"].to_numpy(dtype=float)
        stabilities = models["loading_direction_stability"].to_numpy(dtype=float)
        row = {
            "loading_group": group_id,
            "feature": feature,
            "modality": str(group.modality),
            "singleton_median_oof_partial_state_fit": float(np.median(fits)),
            "singleton_lower_oof_partial_state_fit": float(np.quantile(fits, 0.10)),
            "singleton_positive_fit_view_support": float(np.mean(fits > 0.0)),
            "singleton_median_loading_norm": float(np.median(norms)),
            "singleton_minimum_loading_norm": float(np.min(norms)),
            "singleton_median_direction_stability": float(np.median(stabilities)),
            "singleton_minimum_direction_stability": float(np.min(stabilities)),
        }
        checks = {
            "fit": row["singleton_median_oof_partial_state_fit"]
            >= config.min_singleton_oof_partial_state_fit,
            "lower_fit": row["singleton_lower_oof_partial_state_fit"]
            >= config.min_singleton_lower_oof_partial_state_fit,
            "fit_support": row["singleton_positive_fit_view_support"]
            >= config.min_singleton_positive_fit_view_support,
            "loading_norm": row["singleton_median_loading_norm"]
            >= config.min_singleton_loading_norm,
            "minimum_loading_norm": row["singleton_minimum_loading_norm"]
            >= config.min_singleton_minimum_loading_norm,
            "direction_stability": row["singleton_median_direction_stability"]
            >= config.min_singleton_direction_stability,
            "minimum_direction_stability": row[
                "singleton_minimum_direction_stability"
            ]
            >= config.min_singleton_minimum_direction_stability,
        }
        row["passes_singleton_feature_stability"] = bool(all(checks.values()))
        row["singleton_feature_stability_reason"] = (
            "passed"
            if all(checks.values())
            else "failed:" + ",".join(key for key, passed in checks.items() if not passed)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("loading_group").reset_index(drop=True)


def _singleton_bridge_evidence(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    group_centroids: Mapping[tuple[str, str], FloatArray],
    base_clusters: Sequence[Sequence[str]],
    dimensions: int,
    config: LatentAxisClusteringConfig,
) -> pd.DataFrame:
    """Score singleton-to-axis bridges without allowing a singleton to seed.

    Candidate q-values are adjusted across all base axes for each singleton.
    Axis and candidate margins are separate: the feature must uniquely prefer
    the target axis, and the target must uniquely prefer that feature.  Only a
    one-modality base cluster containing a selected coherent group can be
    rescued, with at most one singleton admitted to that axis.
    """

    singleton_groups = groups.loc[
        groups["selection_status"].astype(str).eq("singleton_candidate")
    ].copy()
    columns = [
        "singleton_axis_key",
        "loading_group",
        "feature",
        "modality",
        "axis_id",
        "anchor_loading_groups",
        "anchor_modalities",
        "anchor_rescue_eligible",
        "anchor_maximum_members",
        "n_views",
        "median_absolute_singleton_axis_cosine",
        "lower_absolute_singleton_axis_cosine",
        "minimum_absolute_singleton_axis_cosine",
        "singleton_axis_cosine_view_sd",
        "singleton_axis_view_support",
        "singleton_projective_null_p",
        "singleton_projective_null_q",
        "singleton_axis_rank",
        "singleton_axis_relative_margin",
        "axis_candidate_rank",
        "axis_candidate_relative_margin",
        "passes_singleton_feature_stability",
        "passes_singleton_median_cosine",
        "passes_singleton_lower_cosine",
        "passes_singleton_view_support",
        "passes_singleton_view_sd",
        "passes_singleton_projective_null_q",
        "passes_singleton_axis_margin",
        "passes_axis_candidate_margin",
        "singleton_bridge_selected",
        "selection_status",
        "abstention_reason",
        "singleton_bridge_confidence",
    ]
    if singleton_groups.empty:
        return pd.DataFrame(columns=columns)

    lookup = groups.set_index("loading_group")
    confidences = lookup["group_confidence"].astype(float).to_dict()
    member_count = membership.groupby("loading_group")["feature"].nunique().to_dict()
    stability = _singleton_feature_stability(
        singleton_groups, membership, views, config
    ).set_index("loading_group")
    rows: list[dict[str, Any]] = []
    for axis_index, cluster_values in enumerate(base_clusters, start=1):
        cluster = tuple(sorted(map(str, cluster_values)))
        axis_id = f"latent_axis_{axis_index:03d}"
        anchor_modalities = sorted(
            {str(lookup.loc[group, "modality"]) for group in cluster}
        )
        maximum_members = max(int(member_count[group]) for group in cluster)
        coherent_anchors = [
            group
            for group in cluster
            if str(lookup.loc[group, "selection_status"]) == "selected"
            and int(member_count[group]) >= config.min_singleton_anchor_members
        ]
        rescue_eligible = bool(
            len(anchor_modalities) == 1 and len(coherent_anchors) >= 1
        )
        axis_vectors = {
            view.view_id: _axis_centroid(
                cluster, view.view_id, group_centroids, confidences
            )[0]
            for view in views
        }
        for singleton in singleton_groups.itertuples(index=False):
            group_id = str(singleton.loading_group)
            singleton_modality = str(singleton.modality)
            cosines = np.asarray(
                [
                    abs(
                        float(
                            np.dot(
                                group_centroids[(view.view_id, group_id)],
                                axis_vectors[view.view_id],
                            )
                        )
                    )
                    for view in views
                ],
                dtype=float,
            )
            stable = stability.loc[group_id]
            median = float(np.median(cosines))
            rows.append(
                {
                    "singleton_axis_key": f"{group_id}|||{axis_id}",
                    "loading_group": group_id,
                    "feature": str(stable["feature"]),
                    "modality": singleton_modality,
                    "axis_id": axis_id,
                    "anchor_loading_groups": ";".join(cluster),
                    "anchor_modalities": ";".join(anchor_modalities),
                    "anchor_rescue_eligible": bool(
                        rescue_eligible and singleton_modality not in anchor_modalities
                    ),
                    "anchor_maximum_members": maximum_members,
                    "n_views": len(cosines),
                    "median_absolute_singleton_axis_cosine": median,
                    "lower_absolute_singleton_axis_cosine": float(
                        np.quantile(cosines, 0.10)
                    ),
                    "minimum_absolute_singleton_axis_cosine": float(np.min(cosines)),
                    "singleton_axis_cosine_view_sd": float(
                        np.std(cosines, ddof=1) if len(cosines) > 1 else 0.0
                    ),
                    "singleton_axis_view_support": float(
                        np.mean(
                            cosines
                            >= config.min_singleton_median_absolute_cosine
                        )
                    ),
                    "singleton_projective_null_p": _projective_null_p(
                        median, dimensions
                    ),
                    "passes_singleton_feature_stability": bool(
                        stable["passes_singleton_feature_stability"]
                    ),
                }
            )
    evidence = pd.DataFrame(rows)
    evidence["singleton_projective_null_q"] = 1.0
    for _, indices in evidence.groupby("loading_group", sort=True).groups.items():
        evidence.loc[indices, "singleton_projective_null_q"] = _benjamini_hochberg(
            evidence.loc[indices, "singleton_projective_null_p"].to_numpy(dtype=float)
        )

    evidence["singleton_axis_rank"] = 0
    evidence["singleton_axis_relative_margin"] = 0.0
    for _, indices in evidence.groupby("loading_group", sort=True).groups.items():
        ordered = evidence.loc[indices].sort_values(
            ["median_absolute_singleton_axis_cosine", "axis_id"],
            ascending=[False, True],
        )
        evidence.loc[ordered.index, "singleton_axis_rank"] = np.arange(
            1, len(ordered) + 1
        )
        best = float(ordered.iloc[0]["median_absolute_singleton_axis_cosine"])
        second = (
            float(ordered.iloc[1]["median_absolute_singleton_axis_cosine"])
            if len(ordered) > 1
            else 0.0
        )
        evidence.loc[ordered.index[0], "singleton_axis_relative_margin"] = (
            best - second
        ) / max(best, 1e-12)

    evidence["axis_candidate_rank"] = 0
    evidence["axis_candidate_relative_margin"] = 0.0
    for _, indices in evidence.loc[
        evidence["anchor_rescue_eligible"]
        & evidence["passes_singleton_feature_stability"]
    ].groupby("axis_id", sort=True).groups.items():
        ordered = evidence.loc[indices].sort_values(
            ["median_absolute_singleton_axis_cosine", "loading_group"],
            ascending=[False, True],
        )
        evidence.loc[ordered.index, "axis_candidate_rank"] = np.arange(
            1, len(ordered) + 1
        )
        best = float(ordered.iloc[0]["median_absolute_singleton_axis_cosine"])
        second = (
            float(ordered.iloc[1]["median_absolute_singleton_axis_cosine"])
            if len(ordered) > 1
            else 0.0
        )
        evidence.loc[ordered.index[0], "axis_candidate_relative_margin"] = (
            best - second
        ) / max(best, 1e-12)

    evidence["passes_singleton_median_cosine"] = (
        evidence["median_absolute_singleton_axis_cosine"]
        >= config.min_singleton_median_absolute_cosine
    )
    evidence["passes_singleton_lower_cosine"] = (
        evidence["lower_absolute_singleton_axis_cosine"]
        >= config.min_singleton_lower_absolute_cosine
    )
    evidence["passes_singleton_view_support"] = (
        evidence["singleton_axis_view_support"]
        >= config.min_singleton_view_support
    )
    evidence["passes_singleton_view_sd"] = (
        evidence["singleton_axis_cosine_view_sd"]
        <= config.max_singleton_view_sd
    )
    evidence["passes_singleton_projective_null_q"] = (
        evidence["singleton_projective_null_q"]
        <= config.max_singleton_projective_null_q
    )
    evidence["passes_singleton_axis_margin"] = (
        (evidence["singleton_axis_rank"] == 1)
        & (
            evidence["singleton_axis_relative_margin"]
            >= config.min_singleton_axis_relative_margin
        )
    )
    evidence["passes_axis_candidate_margin"] = (
        (evidence["axis_candidate_rank"] == 1)
        & (
            evidence["axis_candidate_relative_margin"]
            >= config.min_singleton_candidate_relative_margin
        )
    )
    check_columns = [
        "anchor_rescue_eligible",
        "passes_singleton_feature_stability",
        "passes_singleton_median_cosine",
        "passes_singleton_lower_cosine",
        "passes_singleton_view_support",
        "passes_singleton_view_sd",
        "passes_singleton_projective_null_q",
        "passes_singleton_axis_margin",
        "passes_axis_candidate_margin",
    ]
    evidence["singleton_bridge_selected"] = evidence[check_columns].all(axis=1)
    evidence["selection_status"] = np.where(
        evidence["singleton_bridge_selected"], "selected", "unresolved"
    )
    evidence["abstention_reason"] = [
        ""
        if bool(row["singleton_bridge_selected"])
        else "failed:"
        + ",".join(
            column.removeprefix("passes_")
            for column in check_columns
            if not bool(row[column])
        )
        for _, row in evidence.iterrows()
    ]
    evidence["singleton_bridge_confidence"] = (
        evidence["lower_absolute_singleton_axis_cosine"]
        * evidence["singleton_axis_view_support"]
        * (1.0 - evidence["singleton_projective_null_q"])
        * evidence["singleton_axis_relative_margin"].clip(lower=0.0, upper=1.0)
        * evidence["axis_candidate_relative_margin"].clip(lower=0.0, upper=1.0)
    ).clip(0.0, 1.0)
    return evidence[columns].sort_values(
        ["loading_group", "axis_id"]
    ).reset_index(drop=True)


def _apply_singleton_bridges(
    base_clusters: Sequence[Sequence[str]],
    evidence: pd.DataFrame,
) -> list[tuple[str, ...]]:
    """Attach at most one selected singleton to each pre-existing anchor axis."""

    if evidence.empty:
        return [tuple(sorted(map(str, cluster))) for cluster in base_clusters]
    selected_by_axis = {
        str(row.axis_id): str(row.loading_group)
        for row in evidence.loc[
            evidence.get("singleton_bridge_selected", pd.Series(dtype=bool))
        ].itertuples(index=False)
    }
    output: list[tuple[str, ...]] = []
    for axis_index, cluster_values in enumerate(base_clusters, start=1):
        axis_id = f"latent_axis_{axis_index:03d}"
        members = list(map(str, cluster_values))
        if axis_id in selected_by_axis:
            members.append(selected_by_axis[axis_id])
        output.append(tuple(sorted(members)))
    return output


def _build_axis_candidates(
    clusters: Sequence[Sequence[str]],
    groups: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    group_centroids: Mapping[tuple[str, str], FloatArray],
    coordinates: Sequence[str],
    pair_evidence: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], FloatArray]]:
    group_lookup = groups.set_index("loading_group")
    confidences = group_lookup["group_confidence"].astype(float).to_dict()
    group_status = group_lookup["selection_status"].astype(str).to_dict()
    pair_lookup = pair_evidence.set_index(["group1", "group2"])
    rows: list[dict[str, Any]] = []
    centroids: dict[tuple[str, str], FloatArray] = {}
    for axis_index, cluster_values in enumerate(clusters, start=1):
        cluster = tuple(sorted(map(str, cluster_values)))
        axis_id = f"latent_axis_{axis_index:03d}"
        frustrations: list[float] = []
        anchors: list[str] = []
        view_vectors: list[FloatArray] = []
        for view in views:
            centroid, frustration, anchor = _axis_centroid(
                cluster, view.view_id, group_centroids, confidences
            )
            centroids[(view.view_id, axis_id)] = centroid
            frustrations.append(frustration)
            anchors.append(anchor)
            view_vectors.append(centroid)
        reference = view_vectors[0]
        aligned_views = [
            vector if float(np.dot(vector, reference)) >= 0 else -vector
            for vector in view_vectors
        ]
        consensus = _unit(np.mean(np.vstack(aligned_views), axis=0))
        # The projective axis has no meaningful sign.  Canonicalizing the
        # first non-negligible coordinate makes the serialized centroid
        # invariant to evidence-view order without assigning a causal sign.
        nonzero = np.flatnonzero(np.abs(consensus) > 1e-12)
        if len(nonzero) and consensus[int(nonzero[0])] < 0:
            consensus = -consensus
        pair_rows = []
        for left_index, left in enumerate(cluster):
            for right in cluster[left_index + 1 :]:
                pair_rows.append(pair_lookup.loc[tuple(sorted((left, right)))])
        modalities = sorted({str(group_lookup.loc[group, "modality"]) for group in cluster})
        singleton_groups = sorted(
            group for group in cluster if group_status[group] == "singleton_candidate"
        )
        coherent_anchor_groups = sorted(
            group
            for group in cluster
            if group_status[group] == "selected"
            and len(
                tuple(
                    item
                    for item in str(group_lookup.loc[group, "members"]).split(";")
                    if item
                )
            )
            >= 2
        )
        row: dict[str, Any] = {
            "axis_id": axis_id,
            "loading_groups": ";".join(cluster),
            "modalities": ";".join(modalities),
            "n_groups": len(cluster),
            "n_modalities": len(modalities),
            "contains_rna": "rna" in modalities,
            "n_singleton_bridges": len(singleton_groups),
            "singleton_bridge_groups": ";".join(singleton_groups),
            "n_coherent_anchor_groups": len(coherent_anchor_groups),
            "coherent_anchor_groups": ";".join(coherent_anchor_groups),
            "singleton_only_axis": bool(
                singleton_groups and not coherent_anchor_groups
            ),
            "minimum_pair_median_absolute_cosine": (
                min(float(pair["median_absolute_centroid_cosine"]) for pair in pair_rows)
                if pair_rows
                else 1.0
            ),
            "minimum_pair_lower_absolute_cosine": (
                min(float(pair["lower_absolute_centroid_cosine"]) for pair in pair_rows)
                if pair_rows
                else 1.0
            ),
            "minimum_pair_view_support": (
                min(float(pair["pair_cosine_view_support"]) for pair in pair_rows)
                if pair_rows
                else 1.0
            ),
            "minimum_group_confidence": min(confidences[group] for group in cluster),
            "mean_sign_frustration": float(np.mean(frustrations)),
            "maximum_sign_frustration": float(np.max(frustrations)),
            "medoid_anchor_view_support": float(
                max(anchors.count(anchor) for anchor in set(anchors)) / len(anchors)
            ),
        }
        for coordinate, value in zip(coordinates, consensus, strict=True):
            row[f"axis_centroid__{coordinate.removeprefix('integration__')}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows), centroids


def _group_axis_evidence(
    axes: pd.DataFrame,
    clusters: Sequence[Sequence[str]],
    groups: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    group_centroids: Mapping[tuple[str, str], FloatArray],
    axis_centroids: Mapping[tuple[str, str], FloatArray],
    config: LatentAxisClusteringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_lookup = groups.set_index("loading_group")
    confidence = group_lookup["group_confidence"].astype(float).to_dict()
    cluster_lookup = {
        str(row.axis_id): tuple(sorted(map(str, cluster)))
        for row, cluster in zip(axes.itertuples(index=False), clusters, strict=True)
    }
    view_rows: list[dict[str, Any]] = []
    for group in sorted(group_lookup.index.astype(str)):
        for axis in axes["axis_id"].astype(str):
            cluster = cluster_lookup[axis]
            is_member = group in cluster
            for view in views:
                leave_one_out = is_member and len(cluster) > 1
                if leave_one_out:
                    comparison_members = tuple(item for item in cluster if item != group)
                    comparison, _, _ = _axis_centroid(
                        comparison_members,
                        view.view_id,
                        group_centroids,
                        confidence,
                    )
                else:
                    comparison = axis_centroids[(view.view_id, axis)]
                cosine = abs(
                    float(np.dot(group_centroids[(view.view_id, group)], comparison))
                )
                view_rows.append(
                    {
                        "group_axis_key": f"{group}|||{axis}",
                        "loading_group": group,
                        "group_modality": str(group_lookup.loc[group, "modality"]),
                        "axis_id": axis,
                        "is_cluster_member": is_member,
                        "leave_one_group_out_for_member": leave_one_out,
                        "modality_conflict_with_axis": (
                            not is_member
                            and str(group_lookup.loc[group, "modality"])
                            in set(axes.set_index("axis_id").loc[axis, "modalities"].split(";"))
                        ),
                        "view_id": view.view_id,
                        "absolute_group_axis_cosine": cosine,
                        "passes_config_member_cosine": cosine
                        >= config.min_member_axis_lower_cosine,
                    }
                )
    view_frame = pd.DataFrame(view_rows)
    rows: list[dict[str, Any]] = []
    for keys, frame in view_frame.groupby(
        [
            "group_axis_key",
            "loading_group",
            "group_modality",
            "axis_id",
            "is_cluster_member",
            "leave_one_group_out_for_member",
            "modality_conflict_with_axis",
        ],
        sort=True,
    ):
        (
            key,
            group,
            modality,
            axis,
            is_member,
            leave_one_out,
            modality_conflict,
        ) = keys
        cosines = frame["absolute_group_axis_cosine"].to_numpy(dtype=float)
        rows.append(
            {
                "group_axis_key": key,
                "loading_group": group,
                "group_modality": modality,
                "axis_id": axis,
                "is_cluster_member": bool(is_member),
                "leave_one_group_out_for_member": bool(leave_one_out),
                "modality_conflict_with_axis": bool(modality_conflict),
                "median_absolute_group_axis_cosine": float(np.median(cosines)),
                "lower_absolute_group_axis_cosine": float(np.quantile(cosines, 0.10)),
                "minimum_absolute_group_axis_cosine": float(np.min(cosines)),
                "group_axis_cosine_view_sd": float(np.std(cosines, ddof=1)) if len(cosines) > 1 else 0.0,
                "group_axis_view_support": float(
                    np.mean(cosines >= config.min_member_axis_lower_cosine)
                ),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate["best_competing_axis_median_cosine"] = 0.0
    aggregate["relative_axis_margin"] = 0.0
    for group, indices in aggregate.groupby("loading_group").groups.items():
        group_rows = aggregate.loc[indices]
        for row_index in indices:
            row = aggregate.loc[row_index]
            competitors = group_rows.loc[group_rows["axis_id"] != row["axis_id"]]
            competing = (
                float(competitors["median_absolute_group_axis_cosine"].max())
                if not competitors.empty
                else 0.0
            )
            own = float(row["median_absolute_group_axis_cosine"])
            aggregate.at[row_index, "best_competing_axis_median_cosine"] = competing
            aggregate.at[row_index, "relative_axis_margin"] = float(
                (own - competing) / max(own, 1e-12)
            )
    aggregate["passes_member_lower_cosine"] = (
        aggregate["lower_absolute_group_axis_cosine"]
        >= config.min_member_axis_lower_cosine
    )
    aggregate["passes_member_view_support"] = (
        aggregate["group_axis_view_support"] >= config.min_member_axis_view_support
    )
    aggregate["passes_relative_axis_margin"] = (
        aggregate["relative_axis_margin"] >= config.min_member_relative_axis_margin
    )
    return view_frame, aggregate.sort_values(
        ["loading_group", "axis_id"]
    ).reset_index(drop=True)


def _attach_singleton_bridge_evidence(
    group_axis: pd.DataFrame,
    bridge_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen pre-merge singleton decisions to axis evidence rows."""

    output = group_axis.copy()
    bridge_columns = [
        "loading_group",
        "axis_id",
        "anchor_rescue_eligible",
        "median_absolute_singleton_axis_cosine",
        "lower_absolute_singleton_axis_cosine",
        "singleton_axis_cosine_view_sd",
        "singleton_axis_view_support",
        "singleton_projective_null_p",
        "singleton_projective_null_q",
        "singleton_axis_rank",
        "singleton_axis_relative_margin",
        "axis_candidate_rank",
        "axis_candidate_relative_margin",
        "passes_singleton_feature_stability",
        "passes_singleton_median_cosine",
        "passes_singleton_lower_cosine",
        "passes_singleton_view_support",
        "passes_singleton_view_sd",
        "passes_singleton_projective_null_q",
        "passes_singleton_axis_margin",
        "passes_axis_candidate_margin",
        "singleton_bridge_selected",
        "abstention_reason",
        "singleton_bridge_confidence",
    ]
    if bridge_evidence.empty:
        for column in bridge_columns[2:]:
            output[f"bridge__{column}"] = False if column.startswith("passes_") or column in {
                "anchor_rescue_eligible",
                "singleton_bridge_selected",
            } else "" if column == "abstention_reason" else np.nan
        return output
    renamed = bridge_evidence[bridge_columns].rename(
        columns={
            column: f"bridge__{column}"
            for column in bridge_columns
            if column not in {"loading_group", "axis_id"}
        }
    )
    output = output.merge(
        renamed, on=["loading_group", "axis_id"], how="left", validate="one_to_one"
    )
    boolean_columns = [
        column
        for column in renamed.columns
        if column.startswith("bridge__passes_")
        or column
        in {
            "bridge__anchor_rescue_eligible",
            "bridge__singleton_bridge_selected",
        }
    ]
    for column in boolean_columns:
        output[column] = output[column].map(
            lambda value: bool(value) if pd.notna(value) else False
        ).astype(bool)
    output["bridge__abstention_reason"] = output[
        "bridge__abstention_reason"
    ].fillna("")
    return output


def _select_axes(
    axes: pd.DataFrame,
    group_axis: pd.DataFrame,
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    config: LatentAxisClusteringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_axes = axes.copy()
    selected_axes["minimum_member_axis_lower_cosine"] = 0.0
    selected_axes["minimum_member_axis_view_support"] = 0.0
    selected_axes["minimum_member_relative_axis_margin"] = 0.0
    selected_axes["selection_status"] = "unresolved"
    selected_axes["abstention_reason"] = ""
    selected_axes["edge_type"] = "unresolved"
    selected_axes["axis_confidence"] = 0.0
    selected_axes["minimum_singleton_bridge_confidence"] = 1.0
    for row_index, axis in selected_axes.iterrows():
        members = group_axis.loc[
            (group_axis["axis_id"] == axis["axis_id"])
            & group_axis["is_cluster_member"]
        ]
        lower = float(members["lower_absolute_group_axis_cosine"].min())
        support = float(members["group_axis_view_support"].min())
        margin = float(members["relative_axis_margin"].min())
        selected_axes.at[row_index, "minimum_member_axis_lower_cosine"] = lower
        selected_axes.at[row_index, "minimum_member_axis_view_support"] = support
        selected_axes.at[row_index, "minimum_member_relative_axis_margin"] = margin
        bridge_members = members.loc[
            members.get(
                "bridge__singleton_bridge_selected", pd.Series(False, index=members.index)
            ).astype(bool)
        ]
        bridge_confidence = (
            float(bridge_members["bridge__singleton_bridge_confidence"].min())
            if not bridge_members.empty
            else 1.0
        )
        selected_axes.at[
            row_index, "minimum_singleton_bridge_confidence"
        ] = bridge_confidence
        checks = {
            "modalities": int(axis["n_modalities"]) >= config.min_axis_modalities,
            "coherent_anchor": (
                int(axis.get("n_singleton_bridges", 0)) == 0
                or int(axis.get("n_coherent_anchor_groups", 0)) >= 1
            ),
            "member_cosine": lower >= config.min_member_axis_lower_cosine,
            "view_support": support >= config.min_member_axis_view_support,
            "axis_margin": margin >= config.min_member_relative_axis_margin,
            "sign_frustration": float(axis["maximum_sign_frustration"])
            <= config.max_axis_sign_frustration,
        }
        confidence = float(
            np.clip(
                lower
                * support
                * float(axis["minimum_group_confidence"])
                * max(1.0 - float(axis["maximum_sign_frustration"]), 0.0)
                * min(max(margin, 0.0) / max(config.min_member_relative_axis_margin, 1e-12), 1.0),
                0.0,
                1.0,
            )
        )
        selected_axes.at[row_index, "axis_confidence"] = min(
            confidence, bridge_confidence
        )
        if all(checks.values()):
            selected_axes.at[row_index, "selection_status"] = "selected"
            selected_axes.at[row_index, "abstention_reason"] = ""
            selected_axes.at[row_index, "edge_type"] = (
                "cross_assay_shared_state_axis_with_rna"
                if bool(axis["contains_rna"])
                else "cross_assay_shared_state_axis"
            )
        else:
            selected_axes.at[row_index, "abstention_reason"] = "failed:" + ",".join(
                key for key, passed in checks.items() if not passed
            )

    group_axis = group_axis.copy()
    status_lookup = selected_axes.set_index("axis_id")["selection_status"].to_dict()
    group_axis["selection_status"] = [
        "selected"
        if bool(row.is_cluster_member) and status_lookup[str(row.axis_id)] == "selected"
        else "member_unresolved"
        if bool(row.is_cluster_member)
        else "candidate"
        for row in group_axis.itertuples(index=False)
    ]
    group_members = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in membership.groupby("loading_group")
    }
    hyperedge_rows: list[dict[str, Any]] = []
    for axis in selected_axes.itertuples(index=False):
        axis_groups = str(axis.loading_groups).split(";")
        features = sorted(
            feature
            for group in axis_groups
            for feature in group_members[group]
        )
        hyperedge_rows.append(
            {
                "hyperedge_id": f"axis__{axis.axis_id}",
                "axis_id": str(axis.axis_id),
                "edge_type": str(axis.edge_type),
                "selection_status": str(axis.selection_status),
                "loading_groups": str(axis.loading_groups),
                "modalities": str(axis.modalities),
                "features": ";".join(features),
                "n_groups": int(axis.n_groups),
                "n_modalities": int(axis.n_modalities),
                "contains_rna": bool(axis.contains_rna),
                "n_singleton_bridges": int(
                    getattr(axis, "n_singleton_bridges", 0)
                ),
                "singleton_bridge_groups": str(
                    getattr(axis, "singleton_bridge_groups", "")
                ),
                "coherent_anchor_groups": str(
                    getattr(axis, "coherent_anchor_groups", "")
                ),
                "confidence": float(axis.axis_confidence),
                "abstention_reason": str(axis.abstention_reason),
                "interpretation": (
                    "a coherent assay-local group anchors a shared-state axis; any singleton member passed stability, null, and uniqueness gates; no direct edge or direction is identified"
                    if str(axis.selection_status) == "selected"
                    and int(getattr(axis, "n_singleton_bridges", 0)) > 0
                    else "groups share an estimated state-response axis; no direct edge or direction is identified"
                    if str(axis.selection_status) == "selected"
                    else "axis membership is unresolved and is not projected into the graph"
                ),
            }
        )
    return selected_axes, group_axis, pd.DataFrame(hyperedge_rows)


def _representative_projection(
    axes: pd.DataFrame,
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    feature_vectors: Mapping[tuple[str, str], FloatArray],
    group_centroids: Mapping[tuple[str, str], FloatArray],
    config: LatentAxisClusteringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_lookup = groups.set_index("loading_group")
    confidences = group_lookup["group_confidence"].astype(float).to_dict()
    members = {
        group_id: sorted(frame["feature"].astype(str))
        for group_id, frame in membership.groupby("loading_group")
    }
    evidence_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for axis in axes.loc[axes["selection_status"] == "selected"].itertuples(index=False):
        # A singleton bridge is itself the sparse representation of the axis.
        # Expanding it into an ordinary feature-pair projection would invite a
        # shared-state membership claim to be misread as direct regulation.
        if int(getattr(axis, "n_singleton_bridges", 0)) > 0:
            continue
        axis_groups = tuple(str(axis.loading_groups).split(";"))
        for group in axis_groups:
            other_groups = tuple(item for item in axis_groups if item != group)
            feature_scores: list[dict[str, Any]] = []
            for feature in members[group]:
                cosines: list[float] = []
                for view in views:
                    external_centroid, _, _ = _axis_centroid(
                        other_groups,
                        view.view_id,
                        group_centroids,
                        confidences,
                    )
                    cosines.append(
                        abs(
                            float(
                                np.dot(
                                    feature_vectors[(view.view_id, feature)],
                                    external_centroid,
                                )
                            )
                        )
                    )
                feature_scores.append(
                    {
                        "axis_id": str(axis.axis_id),
                        "loading_group": group,
                        "modality": str(group_lookup.loc[group, "modality"]),
                        "feature": feature,
                        "median_external_axis_cosine": float(np.median(cosines)),
                        "lower_external_axis_cosine": float(np.quantile(cosines, 0.10)),
                        "external_axis_view_support": float(
                            np.mean(
                                np.asarray(cosines)
                                >= config.representative_absolute_cosine
                            )
                        ),
                    }
                )
            feature_scores.sort(
                key=lambda row: (
                    -row["median_external_axis_cosine"],
                    -row["lower_external_axis_cosine"],
                    row["feature"],
                )
            )
            best_score = float(feature_scores[0]["median_external_axis_cosine"])
            second_score = (
                float(feature_scores[1]["median_external_axis_cosine"])
                if len(feature_scores) > 1
                else 0.0
            )
            margin = (best_score - second_score) / max(best_score, 1e-12)
            for rank, row in enumerate(feature_scores, start=1):
                row["within_group_rank"] = rank
                row["representative_relative_margin"] = margin if rank == 1 else 0.0
                row["representative_selected"] = bool(
                    rank == 1
                    and row["lower_external_axis_cosine"]
                    >= config.representative_absolute_cosine
                    and row["external_axis_view_support"]
                    >= config.representative_view_support
                    and margin >= config.representative_relative_margin
                )
                row["selection_status"] = (
                    "selected" if row["representative_selected"] else "unresolved"
                )
                evidence_rows.append(row)
        selected_representatives = [
            row
            for row in evidence_rows
            if row["axis_id"] == str(axis.axis_id) and row["representative_selected"]
        ]
        if len(selected_representatives) < 2:
            continue
        selected_representatives.sort(
            key=lambda row: (
                -row["median_external_axis_cosine"],
                row["loading_group"],
                row["feature"],
            )
        )
        anchor = selected_representatives[0]
        for target in selected_representatives[1:]:
            node1, node2 = sorted((str(anchor["feature"]), str(target["feature"])))
            projection_rows.append(
                {
                    "pair_key": f"{node1}|||{node2}",
                    "node1": node1,
                    "node2": node2,
                    "axis_id": str(axis.axis_id),
                    "edge_type": "cross_assay_shared_state_axis_representative",
                    "selection_status": "selected",
                    "projection_source": "unique_external_axis_representatives",
                    "confidence": float(
                        min(
                            axis.axis_confidence,
                            anchor["lower_external_axis_cosine"],
                            target["lower_external_axis_cosine"],
                        )
                    ),
                    "contains_rna": bool(
                        anchor["modality"] == "rna" or target["modality"] == "rna"
                    ),
                    "interpretation": "sparse visualization of a group-level shared-state axis; not a direct causal edge",
                }
            )
    representative_columns = [
        "axis_id",
        "loading_group",
        "modality",
        "feature",
        "median_external_axis_cosine",
        "lower_external_axis_cosine",
        "external_axis_view_support",
        "within_group_rank",
        "representative_relative_margin",
        "representative_selected",
        "selection_status",
    ]
    projection_columns = [
        "pair_key",
        "node1",
        "node2",
        "axis_id",
        "edge_type",
        "selection_status",
        "projection_source",
        "confidence",
        "contains_rna",
        "interpretation",
    ]
    return (
        pd.DataFrame(evidence_rows, columns=representative_columns),
        pd.DataFrame(projection_rows, columns=projection_columns).drop_duplicates(
            ["pair_key", "axis_id"], keep="first"
        ),
    )


def estimate_latent_axis_clustering(
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    views: Sequence[LoadingEvidenceView],
    config: LatentAxisClusteringConfig | None = None,
) -> LatentAxisClusteringResult:
    """Cluster selected assay-local groups into typed projective axes."""

    cfg = config or LatentAxisClusteringConfig()
    available, members, coordinates, _ = _validate_inputs(groups, membership, views)
    selected = available.loc[
        available["selection_status"].astype(str).eq("selected")
    ].copy()
    singleton_candidates = available.loc[
        available["selection_status"].astype(str).eq("singleton_candidate")
    ].copy()
    feature_vectors = _feature_vectors(views, coordinates)
    group_centroids = _group_centroids(available, members, views, feature_vectors)
    _, base_pairs = _pair_evidence(
        selected, views, group_centroids, len(coordinates), cfg
    )
    base_clusters = _complete_link_clusters(selected, base_pairs, cfg)
    bridge_evidence = _singleton_bridge_evidence(
        available,
        members,
        views,
        group_centroids,
        base_clusters,
        len(coordinates),
        cfg,
    )
    clusters = _apply_singleton_bridges(base_clusters, bridge_evidence)
    pair_views, pairs = _pair_evidence(
        available, views, group_centroids, len(coordinates), cfg
    )
    axes, axis_centroids = _build_axis_candidates(
        clusters, available, views, group_centroids, coordinates, pairs
    )
    group_axis_views, group_axis = _group_axis_evidence(
        axes,
        clusters,
        available,
        views,
        group_centroids,
        axis_centroids,
        cfg,
    )
    group_axis = _attach_singleton_bridge_evidence(
        group_axis, bridge_evidence
    )
    axes, group_axis, hyperedges = _select_axes(
        axes, group_axis, available, members, cfg
    )
    representative_evidence, sparse_projection = _representative_projection(
        axes,
        available,
        members,
        views,
        feature_vectors,
        group_centroids,
        cfg,
    )
    provenance = {
        "schema_version": "latent_axis_clustering_v4.v2",
        "estimand": "typed assay-spanning response groups on an observed-background estimated state axis",
        "config": asdict(cfg),
        "view_ids": [view.view_id for view in views],
        "input_file_sha256": {
            view.view_id: dict(view.input_hashes) for view in views
        },
        "coordinate_columns": list(coordinates),
        "n_views": len(views),
        "n_groups": len(selected),
        "n_singleton_candidates": len(singleton_candidates),
        "n_selected_singleton_bridges": int(
            bridge_evidence.get(
                "singleton_bridge_selected", pd.Series(dtype=bool)
            ).sum()
        ),
        "n_axis_candidates": len(axes),
        "n_selected_axes": int((axes["selection_status"] == "selected").sum()),
        "n_unresolved_axes": int((axes["selection_status"] == "unresolved").sum()),
        "n_sparse_pair_projections": len(sparse_projection),
        "clustering_rule": {
            "method": "deterministic_complete_link_plus_anchored_singleton_bridge",
            "all_cross_cluster_pairs_must_pass": True,
            "transitive_similarity_chains_merge": False,
            "group_sign_has_biological_interpretation": False,
            "thresholds_recoverable_from_long_evidence": True,
            "singleton_only_axes_form": False,
            "singleton_bridge_rescues_only_one_modality_anchor_axes": True,
            "maximum_singletons_per_rescued_axis": 1,
            "singleton_bridge_requires_coherent_two_feature_anchor": True,
            "singleton_bridge_uses_per_candidate_projective_null_bh": True,
            "singleton_bridge_requires_axis_and_candidate_uniqueness": True,
        },
        "input_policy": {
            "selected_loading_groups_strict_singleton_candidates_and_complete_loading_views_only": True,
            "observable_assay_labels_used": True,
            "raw_cells_or_metacells_read": False,
            "cell_correspondence_used": False,
            "truth_scorer_roles_or_dag_read": False,
            "external_data_or_biological_prior_used": False,
        },
        "identifiability": {
            "selected_axis_is_direct_causation": False,
            "cross_assay_direction_identified": False,
            "rna_membership_implies_peak_gene_regulation": False,
            "ambiguous_axes_are_abstained": True,
            "pair_projection_is_optional_sparse_visualization": True,
            "selected_axis_attaches_post_consensus_once": True,
            "singleton_bridge_expanded_into_pair_projection": False,
            "singleton_only_axis_permitted": False,
        },
    }
    return LatentAxisClusteringResult(
        group_pair_view_evidence=pair_views,
        group_pair_evidence=pairs,
        axis_candidates=axes,
        group_axis_view_evidence=group_axis_views,
        group_axis_evidence=group_axis,
        typed_axis_hyperedges=hyperedges,
        representative_evidence=representative_evidence,
        sparse_pair_projection=sparse_projection,
        provenance=provenance,
    )


def _validate_projection_dir(path: Path | str) -> tuple[Path, dict[str, str]]:
    lexical = Path(path).absolute()
    if lexical.is_symlink() or not lexical.exists() or not lexical.is_dir():
        raise ValueError("Loading-group projection must be an existing direct directory")
    resolved = lexical.resolve(strict=True)
    forbidden = sorted(
        component.casefold()
        for component in resolved.parts
        if component.casefold() in _FORBIDDEN_PATH_COMPONENTS
    )
    if forbidden:
        raise ValueError(f"Refusing scorer/truth projection path: {forbidden}")
    required = ("loading_groups.tsv", "loading_group_membership.tsv")
    if any(not (resolved / filename).is_file() for filename in required):
        raise ValueError("Projection directory lacks loading groups or membership")
    if any((resolved / filename).is_symlink() for filename in required):
        raise ValueError("Projection input files may not be symlinks")
    return resolved, {filename: _sha256(resolved / filename) for filename in required}


def run_latent_axis_clustering(
    projection_dir: Path | str,
    evidence_dirs: Sequence[Path | str],
    out_dir: Path | str,
    config: LatentAxisClusteringConfig | None = None,
) -> dict[str, Any]:
    """Load observed-only artifacts, infer axes, and write auditable tables."""

    projection, projection_hashes = _validate_projection_dir(projection_dir)
    views = [load_evidence_view(path) for path in evidence_dirs]
    groups = pd.read_csv(projection / "loading_groups.tsv", sep="\t")
    membership = pd.read_csv(projection / "loading_group_membership.tsv", sep="\t")
    result = estimate_latent_axis_clustering(groups, membership, views, config)
    result.provenance["projection_input_sha256"] = projection_hashes
    output = Path(out_dir).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    if output.resolve(strict=False) == projection:
        raise ValueError("Axis output must differ from loading-group projection input")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "group_pair_view_evidence.tsv": result.group_pair_view_evidence,
        "group_pair_evidence.tsv": result.group_pair_evidence,
        "axis_candidates.tsv": result.axis_candidates,
        "group_axis_view_evidence.tsv": result.group_axis_view_evidence,
        "group_axis_evidence.tsv": result.group_axis_evidence,
        "typed_axis_hyperedges.tsv": result.typed_axis_hyperedges,
        "representative_evidence.tsv": result.representative_evidence,
        "sparse_pair_projection.tsv": result.sparse_pair_projection,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, sep="\t", index=False)
    (output / "axis_provenance.json").write_text(
        json.dumps(_json_ready(result.provenance), indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "schema_version": "latent_axis_clustering_v4.run.v1",
        "n_views": len(views),
        "n_groups": int(result.provenance["n_groups"]),
        "n_selected_axes": int(
            (result.axis_candidates["selection_status"] == "selected").sum()
        ),
        "n_unresolved_axes": int(
            (result.axis_candidates["selection_status"] == "unresolved").sum()
        ),
        "n_sparse_pair_projections": len(result.sparse_pair_projection),
        "truth_or_scorer_used": False,
        "files_written": [*tables, "axis_provenance.json", "run_manifest.json"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-pair-cosine", type=float, default=0.85)
    parser.add_argument("--min-axis-margin", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = LatentAxisClusteringConfig(
        min_pair_median_absolute_cosine=args.min_pair_cosine,
        min_member_relative_axis_margin=args.min_axis_margin,
    )
    summary = run_latent_axis_clustering(
        args.projection_dir, args.evidence_dir, args.out_dir, config
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "LatentAxisClusteringConfig",
    "LatentAxisClusteringResult",
    "estimate_latent_axis_clustering",
    "run_latent_axis_clustering",
]
