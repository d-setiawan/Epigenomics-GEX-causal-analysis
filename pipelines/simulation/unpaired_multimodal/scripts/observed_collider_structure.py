#!/usr/bin/env python3
"""Observed-only structural refinement for sparse within-assay colliders.

The pairwise beta-binomial layer is deliberately conservative.  It can leave
three observable graph patterns unresolved:

* an over-complete star, where a proposed center has more than two arms;
* a short path, where two centers define competing unshielded wedges; and
* a dual-supported single arm whose second arm is too sparse to pass the
  ordinary pairwise multiplicity gate.

This module resolves only those graph patterns.  It consumes evidence already
computed before any scorer is opened: residual pair correlations from each
aggregation view, cross-fitted state-response diagnostics, preliminary direct
graphs, and beta-binomial votes.  It accepts no feature roles, simulator
parameters, paired cells, or external priors.

For residual correlations ``r_xy``, ``r_xc``, and ``r_yc``, the conditional
outer correlation for wedge ``x-c-y`` is

``(r_xy - r_xc*r_yc) / sqrt((1-r_xc**2)*(1-r_yc**2))``.

A collider can make this quantity negative after conditioning on ``c``.  The
module uses view-wise ranks and a four-of-six stability gate rather than a
new loose pairwise p-value threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from structure_aware_collider_rescue import aggregate_vote_evidence, canonical_edge


_DIRECT_EDGE_TYPE = "within_assay_direct_candidate"
_DIRECT_COLUMNS = (
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
    "observed_refinement_action",
    "precentrality_center",
    "precentrality_outer1",
    "precentrality_outer2",
    "centrality_selected_new_parent",
    "centrality_mean_center_vs_outer_lor_contrast",
    "centrality_minimum_center_vs_outer_lor_contrast",
    "completion_anchor_node1",
    "completion_anchor_node2",
    "completion_winner_view_count",
    "completion_n_views",
    "interpretation",
)
_UNRESOLVED_COLUMNS = (
    "source_assay",
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "edge_type",
    "component_id",
    "was_base_selected",
    "was_rescue_selected",
    "mean_empirical_p_positive",
    "mean_pooled_log_odds",
    "reason",
)
_WEDGE_COLUMNS = (
    "source_assay",
    "component_id",
    "center",
    "outer1",
    "outer2",
    "arm1_log_odds",
    "arm2_log_odds",
    "wedge_score",
    "outer_pair_is_candidate",
    "edge1_node1",
    "edge1_node2",
    "edge2_node1",
    "edge2_node2",
    "rank",
    "accepted",
)
_FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "role",
    "oracle",
    "generative",
    "counterfactual",
    "paired_cell",
)


@dataclass(frozen=True)
class ObservedColliderStructureConfig:
    """Frozen TRAIN-stage structural and stability settings."""

    minimum_base_view_count: int = 4
    minimum_winner_view_count: int = 4
    same_center_minimum_pairwise_wedge_margin_exclusive: float = 0.0
    minimum_rescue_vote_fraction: float = 5.0 / 6.0
    maximum_rescue_mean_empirical_p_positive: float = 0.020
    minimum_rescue_mean_pooled_log_odds: float = 1.00
    minimum_completion_mean_pooled_log_odds: float = 0.0
    minimum_completion_center_contrast_view_count: int = 4
    minimum_unique_dual_base_winner_view_fraction: float = 0.50
    maximum_rescue_only_loading_empirical_null_p: float = 0.125
    required_rescue_only_base_graph_view_count: int = 6
    required_rescue_only_loading_view_count: int = 6
    required_rescue_only_beta_vote_count: int = 6
    require_unique_rescue_only_candidate: bool = True
    correlation_clip: float = 1.0 - 1e-9

    def validate(self) -> None:
        if self.minimum_base_view_count < 1:
            raise ValueError("minimum_base_view_count must be positive")
        if self.minimum_winner_view_count < 1:
            raise ValueError("minimum_winner_view_count must be positive")
        if self.minimum_completion_center_contrast_view_count < 1:
            raise ValueError(
                "minimum_completion_center_contrast_view_count must be positive"
            )
        if self.required_rescue_only_loading_view_count < 1:
            raise ValueError(
                "required_rescue_only_loading_view_count must be positive"
            )
        if self.required_rescue_only_base_graph_view_count < 1:
            raise ValueError(
                "required_rescue_only_base_graph_view_count must be positive"
            )
        if self.required_rescue_only_beta_vote_count < 1:
            raise ValueError(
                "required_rescue_only_beta_vote_count must be positive"
            )
        for name, value in (
            ("minimum_rescue_vote_fraction", self.minimum_rescue_vote_fraction),
            (
                "maximum_rescue_mean_empirical_p_positive",
                self.maximum_rescue_mean_empirical_p_positive,
            ),
            (
                "minimum_unique_dual_base_winner_view_fraction",
                self.minimum_unique_dual_base_winner_view_fraction,
            ),
            (
                "maximum_rescue_only_loading_empirical_null_p",
                self.maximum_rescue_only_loading_empirical_null_p,
            ),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not np.isfinite(self.minimum_rescue_mean_pooled_log_odds):
            raise ValueError("minimum_rescue_mean_pooled_log_odds must be finite")
        if not np.isfinite(self.minimum_completion_mean_pooled_log_odds):
            raise ValueError("minimum_completion_mean_pooled_log_odds must be finite")
        if not np.isfinite(
            self.same_center_minimum_pairwise_wedge_margin_exclusive
        ):
            raise ValueError(
                "same_center_minimum_pairwise_wedge_margin_exclusive must be finite"
            )
        if not np.isfinite(self.correlation_clip) or not 0.0 < self.correlation_clip < 1.0:
            raise ValueError("correlation_clip must lie in (0, 1)")
        if not isinstance(self.require_unique_rescue_only_candidate, bool):
            raise TypeError("require_unique_rescue_only_candidate must be boolean")


@dataclass
class ObservedColliderStructureResult:
    """Refinement decisions and the complete observed evidence audit trail."""

    decisions: pd.DataFrame
    wedge_view_evidence: pd.DataFrame
    wedge_evidence: pd.DataFrame
    completion_view_evidence: pd.DataFrame
    completion_evidence: pd.DataFrame
    metadata: dict[str, Any]


def _assert_observed_only(frame: pd.DataFrame, source: str) -> None:
    forbidden = sorted(
        {
            str(column)
            for column in frame.columns
            if any(
                token in str(column).lower()
                for token in _FORBIDDEN_COLUMN_TOKENS
            )
        }
    )
    if forbidden:
        raise ValueError(f"{source} contains privileged columns: {forbidden}")


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def conditional_outer_correlation(
    outer_r: float,
    first_arm_r: float,
    second_arm_r: float,
    *,
    correlation_clip: float = 1.0 - 1e-9,
) -> float:
    """Return the Gaussian partial correlation between outers given center."""

    values = np.asarray(
        [outer_r, first_arm_r, second_arm_r], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Collider correlations must be finite")
    if not 0.0 < correlation_clip < 1.0:
        raise ValueError("correlation_clip must lie in (0, 1)")
    outer, first, second = np.clip(
        values, -correlation_clip, correlation_clip
    )
    denominator = np.sqrt(max((1.0 - first**2) * (1.0 - second**2), 1e-18))
    return float(np.clip((outer - first * second) / denominator, -1.0, 1.0))


def _pair_map(frame: pd.DataFrame, value_column: str, source: str) -> dict[tuple[str, str], float]:
    _assert_observed_only(frame, source)
    _require_columns(frame, {"node1", "node2", value_column}, source)
    result: dict[tuple[str, str], float] = {}
    for record in frame.to_dict(orient="records"):
        edge = canonical_edge(str(record["node1"]), str(record["node2"]))
        if edge in result:
            raise ValueError(f"{source} contains a duplicate pair: {edge}")
        value = float(record[value_column])
        if not np.isfinite(value):
            raise ValueError(f"{source} contains a non-finite {value_column}")
        result[edge] = value
    return result


def _selected_pairs(frame: pd.DataFrame, source: str) -> set[tuple[str, str]]:
    _assert_observed_only(frame, source)
    _require_columns(frame, {"node1", "node2"}, source)
    selected = frame.copy()
    if "selected" in selected:
        selected = selected.loc[selected["selected"].astype(bool)]
    if "edge_type" in selected:
        selected = selected.loc[selected["edge_type"].astype(str) == _DIRECT_EDGE_TYPE]
    return {
        canonical_edge(str(record["node1"]), str(record["node2"]))
        for record in selected.to_dict(orient="records")
    }


def _base_support(
    base_edge_views: Mapping[str, pd.DataFrame],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], tuple[str, ...]]]:
    counts: dict[tuple[str, str], int] = {}
    view_ids: dict[tuple[str, str], list[str]] = {}
    for view_id, frame in sorted(base_edge_views.items()):
        for edge in _selected_pairs(frame, f"base_edge_views[{view_id!r}]"):
            counts[edge] = counts.get(edge, 0) + 1
            view_ids.setdefault(edge, []).append(str(view_id))
    return counts, {edge: tuple(ids) for edge, ids in view_ids.items()}


def _validate_vote_evidence_rows(vote_evidence: pd.DataFrame) -> None:
    """Reject duplicate assay/vote/pair records before any aggregation."""

    _assert_observed_only(vote_evidence, "vote_evidence")
    _require_columns(
        vote_evidence,
        {"source_assay", "node1", "node2", "vote"},
        "vote_evidence",
    )
    seen: set[tuple[str, str, str, str]] = set()
    duplicates: set[tuple[str, str, str, str]] = set()
    for record in vote_evidence.to_dict(orient="records"):
        edge = canonical_edge(str(record["node1"]), str(record["node2"]))
        key = (
            str(record["source_assay"]),
            str(record["vote"]),
            edge[0],
            edge[1],
        )
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(
            "vote_evidence contains duplicate assay/vote/pair rows: "
            f"{sorted(duplicates)}"
        )


def _state_score_map(frame: pd.DataFrame, source: str) -> dict[str, float]:
    _assert_observed_only(frame, source)
    _require_columns(
        frame,
        {
            "feature",
            "modality",
            "oof_partial_state_fit",
            "loading_direction_stability",
        },
        source,
    )
    if frame["feature"].astype(str).duplicated().any():
        raise ValueError(f"{source} contains duplicate features")
    result: dict[str, float] = {}
    for record in frame.to_dict(orient="records"):
        partial_fit = max(float(record["oof_partial_state_fit"]), 0.0)
        stability = max(float(record["loading_direction_stability"]), 0.0)
        if not np.isfinite(partial_fit) or not np.isfinite(stability):
            raise ValueError(f"{source} contains non-finite state evidence")
        result[str(record["feature"])] = partial_fit * stability
    return result


def _feature_assay_map(feature_model_views: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    assay_map: dict[str, str] = {}
    for view_id, frame in sorted(feature_model_views.items()):
        _assert_observed_only(frame, f"feature_model_views[{view_id!r}]")
        _require_columns(frame, {"feature", "modality"}, f"feature_model_views[{view_id!r}]")
        for record in frame[["feature", "modality"]].to_dict(orient="records"):
            feature = str(record["feature"])
            assay = str(record["modality"])
            previous = assay_map.setdefault(feature, assay)
            if previous != assay:
                raise ValueError(f"Feature {feature!r} changes assay across views")
    return assay_map


def _wedge_view_rows(
    wedges: pd.DataFrame,
    residual_views: Mapping[str, pd.DataFrame],
    feature_model_views: Mapping[str, pd.DataFrame],
    base_counts: Mapping[tuple[str, str], int],
    config: ObservedColliderStructureConfig,
) -> pd.DataFrame:
    if set(residual_views) != set(feature_model_views):
        raise ValueError("Residual and feature-model view identifiers must match exactly")
    residual_maps = {
        str(view_id): _pair_map(frame, "residual_r", f"residual_views[{view_id!r}]")
        for view_id, frame in sorted(residual_views.items())
    }
    state_maps = {
        str(view_id): _state_score_map(frame, f"feature_model_views[{view_id!r}]")
        for view_id, frame in sorted(feature_model_views.items())
    }
    rows: list[dict[str, Any]] = []
    for wedge in wedges.to_dict(orient="records"):
        center = str(wedge["center"])
        outer1 = str(wedge["outer1"])
        outer2 = str(wedge["outer2"])
        arm1 = canonical_edge(center, outer1)
        arm2 = canonical_edge(center, outer2)
        outer_pair = canonical_edge(outer1, outer2)
        for view_id in sorted(residual_maps):
            correlations = residual_maps[view_id]
            missing = [edge for edge in (arm1, arm2, outer_pair) if edge not in correlations]
            if missing:
                raise ValueError(
                    f"Residual view {view_id!r} lacks wedge pairs: {missing}"
                )
            outer_r = correlations[outer_pair]
            first_arm_r = correlations[arm1]
            second_arm_r = correlations[arm2]
            conditional_r = conditional_outer_correlation(
                outer_r,
                first_arm_r,
                second_arm_r,
                correlation_clip=config.correlation_clip,
            )
            states = state_maps[view_id]
            if outer1 not in states or outer2 not in states:
                raise ValueError(f"Feature-model view {view_id!r} lacks a wedge outer")
            rows.append(
                {
                    "source_assay": str(wedge["source_assay"]),
                    "component_id": str(wedge["component_id"]),
                    "center": center,
                    "outer1": outer1,
                    "outer2": outer2,
                    "view_id": view_id,
                    "base_arm_count": int(
                        (base_counts.get(arm1, 0) >= config.minimum_base_view_count)
                        + (base_counts.get(arm2, 0) >= config.minimum_base_view_count)
                    ),
                    "outer_residual_r": outer_r,
                    "first_arm_residual_r": first_arm_r,
                    "second_arm_residual_r": second_arm_r,
                    "conditional_outer_residual_r": conditional_r,
                    "collider_separation_score": -conditional_r,
                    "collider_opening_delta": outer_r - conditional_r,
                    "outer_state_support_floor": min(states[outer1], states[outer2]),
                    "pairwise_wedge_score": float(wedge["wedge_score"]),
                }
            )
    return pd.DataFrame(rows)


def _rank_wedge_views(
    components: pd.DataFrame,
    wedges: pd.DataFrame,
    view_evidence: pd.DataFrame,
    config: ObservedColliderStructureConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    key_columns = ["component_id", "center", "outer1", "outer2"]
    if view_evidence.empty:
        return pd.DataFrame(columns=[*key_columns, "stable_view_winner"]), decisions
    component_lookup = components.set_index("component_id")
    for key, group in view_evidence.groupby(key_columns, sort=True):
        component_id, center, outer1, outer2 = key
        summaries.append(
            {
                "component_id": component_id,
                "center": center,
                "outer1": outer1,
                "outer2": outer2,
                "n_views": len(group),
                "base_arm_count": int(group["base_arm_count"].iloc[0]),
                "median_conditional_outer_residual_r": float(
                    group["conditional_outer_residual_r"].median()
                ),
                "median_collider_separation_score": float(
                    group["collider_separation_score"].median()
                ),
                "median_collider_opening_delta": float(
                    group["collider_opening_delta"].median()
                ),
                "median_outer_state_support_floor": float(
                    group["outer_state_support_floor"].median()
                ),
                "pairwise_wedge_score": float(group["pairwise_wedge_score"].iloc[0]),
                "winner_view_count": 0,
                "stable_view_winner": False,
                "stable_view_winner_pairwise_margin": np.nan,
                "stable_view_winner_pairwise_compatible": False,
                "stable_view_winner_pairwise_conflict_abstained": False,
            }
        )
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return summary, decisions
    for component_id, component_wedges in wedges.groupby("component_id", sort=True):
        if len(component_wedges) <= 1:
            continue
        component = component_lookup.loc[component_id]
        same_center_star = component_wedges["center"].astype(str).nunique() == 1
        status = str(component["status"])
        if not same_center_star and status != "unresolved_competing_wedges":
            continue
        winner_keys: list[tuple[str, str, str, str]] = []
        component_views = view_evidence.loc[
            view_evidence["component_id"].astype(str) == str(component_id)
        ]
        for _, candidates in component_views.groupby("view_id", sort=True):
            records = candidates.to_dict(orient="records")
            if same_center_star:
                winner = max(
                    records,
                    key=lambda record: (
                        float(record["collider_separation_score"]),
                        float(record["outer_state_support_floor"]),
                        float(record["pairwise_wedge_score"]),
                        str(record["outer1"]),
                        str(record["outer2"]),
                    ),
                )
            else:
                winner = max(
                    records,
                    key=lambda record: (
                        int(record["base_arm_count"]),
                        float(record["outer_state_support_floor"]),
                        float(record["collider_separation_score"]),
                        float(record["pairwise_wedge_score"]),
                        str(record["center"]),
                        str(record["outer1"]),
                        str(record["outer2"]),
                    ),
                )
            winner_keys.append(
                (
                    str(component_id),
                    str(winner["center"]),
                    str(winner["outer1"]),
                    str(winner["outer2"]),
                )
            )
        counts: dict[tuple[str, str, str, str], int] = {}
        for winner_key in winner_keys:
            counts[winner_key] = counts.get(winner_key, 0) + 1
        for candidate_key, candidate_count in counts.items():
            candidate_mask = (
                summary["component_id"].astype(str).eq(candidate_key[0])
                & summary["center"].astype(str).eq(candidate_key[1])
                & summary["outer1"].astype(str).eq(candidate_key[2])
                & summary["outer2"].astype(str).eq(candidate_key[3])
            )
            summary.loc[
                candidate_mask, "winner_view_count"
            ] = int(candidate_count)
        stable_key, winner_count = max(
            counts.items(),
            key=lambda item: (item[1], item[0]),
        )
        mask = (
            summary["component_id"].astype(str).eq(stable_key[0])
            & summary["center"].astype(str).eq(stable_key[1])
            & summary["outer1"].astype(str).eq(stable_key[2])
            & summary["outer2"].astype(str).eq(stable_key[3])
        )
        stable = winner_count >= config.minimum_winner_view_count
        summary.loc[mask, "stable_view_winner"] = stable
        stable_pairwise_score = float(
            summary.loc[mask, "pairwise_wedge_score"].iloc[0]
        )
        component_mask = summary["component_id"].astype(str).eq(str(component_id))
        runner_scores = summary.loc[component_mask & ~mask, "pairwise_wedge_score"]
        stable_pairwise_margin = (
            stable_pairwise_score - float(runner_scores.max())
            if len(runner_scores)
            else math.inf
        )
        pairwise_compatible = bool(
            not same_center_star
            or stable_pairwise_margin
            > config.same_center_minimum_pairwise_wedge_margin_exclusive
        )
        summary.loc[mask, "stable_view_winner_pairwise_margin"] = (
            stable_pairwise_margin
        )
        summary.loc[mask, "stable_view_winner_pairwise_compatible"] = (
            pairwise_compatible
        )
        pairwise_conflict_abstained = bool(
            stable and same_center_star and not pairwise_compatible
        )
        summary.loc[
            mask, "stable_view_winner_pairwise_conflict_abstained"
        ] = pairwise_conflict_abstained
        if stable and pairwise_compatible:
            decisions.append(
                {
                    "source_assay": str(component["source_assay"]),
                    "component_id": str(component_id),
                    "action": (
                        "reselect_overcomplete_same_center_star"
                        if same_center_star
                        else "resolve_competing_multicenter_path"
                    ),
                    "center": stable_key[1],
                    "outer1": stable_key[2],
                    "outer2": stable_key[3],
                    "winner_view_count": int(winner_count),
                    "n_views": len(winner_keys),
                    "anchor_node1": "",
                    "anchor_node2": "",
                    "new_arm_node1": "",
                    "new_arm_node2": "",
                }
            )
    return summary.sort_values(key_columns).reset_index(drop=True), decisions


def _add_rejected_dual_base_half_view_decisions(
    components: pd.DataFrame,
    wedge_summary: pd.DataFrame,
    decisions: list[dict[str, Any]],
    config: ObservedColliderStructureConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Recover one uniquely dominant two-base-arm rejected same-center wedge."""

    if wedge_summary.empty:
        return wedge_summary, decisions
    summary = wedge_summary.copy()
    for column in (
        "unique_two_base_arm_wedge",
        "runners_at_most_one_base_arm",
        "half_view_support",
        "positive_pairwise_margin",
        "selected_by_half_view_dual_base_rule",
    ):
        summary[column] = False
    decided_components = {
        str(decision["component_id"]) for decision in decisions
    }
    additions: list[dict[str, Any]] = []
    for component in components.to_dict(orient="records"):
        component_id = str(component["component_id"])
        if component_id in decided_components:
            continue
        status = str(component["status"])
        accepted = bool(
            component.get("accepted", status.startswith("accepted"))
        )
        if accepted:
            continue
        candidates = summary.loc[
            summary["component_id"].astype(str).eq(component_id)
        ]
        if (
            len(candidates) <= 1
            or candidates["center"].astype(str).nunique() != 1
        ):
            continue
        maximum_base_arms = int(candidates["base_arm_count"].max())
        leaders = candidates.loc[
            candidates["base_arm_count"].astype(int).eq(maximum_base_arms)
        ]
        if maximum_base_arms != 2 or len(leaders) != 1:
            continue
        leader_index = leaders.index[0]
        leader = leaders.iloc[0]
        runners = candidates.drop(index=leader_index)
        runners_at_most_one_base_arm = bool(
            (runners["base_arm_count"].astype(int) <= 1).all()
        )
        half_view_support = bool(
            int(leader["winner_view_count"])
            >= math.ceil(
                config.minimum_unique_dual_base_winner_view_fraction
                * int(leader["n_views"])
            )
        )
        positive_pairwise_margin = bool(
            float(leader["pairwise_wedge_score"])
            > float(runners["pairwise_wedge_score"].max())
        )
        summary.loc[
            leader_index,
            [
                "unique_two_base_arm_wedge",
                "runners_at_most_one_base_arm",
                "half_view_support",
                "positive_pairwise_margin",
            ],
        ] = [
            True,
            runners_at_most_one_base_arm,
            half_view_support,
            positive_pairwise_margin,
        ]
        if not (
            runners_at_most_one_base_arm
            and half_view_support
            and positive_pairwise_margin
        ):
            continue
        summary.loc[
            leader_index, "selected_by_half_view_dual_base_rule"
        ] = True
        additions.append(
            {
                "source_assay": str(component["source_assay"]),
                "component_id": component_id,
                "action": "accept_rejected_unique_dual_base_half_view_wedge",
                "center": str(leader["center"]),
                "outer1": str(leader["outer1"]),
                "outer2": str(leader["outer2"]),
                "winner_view_count": int(leader["winner_view_count"]),
                "n_views": int(leader["n_views"]),
                "anchor_node1": "",
                "anchor_node2": "",
                "new_arm_node1": "",
                "new_arm_node2": "",
            }
        )
    return summary, [*decisions, *additions]


def _completion_rows(
    components: pd.DataFrame,
    unresolved: pd.DataFrame,
    aggregated: pd.DataFrame,
    vote_evidence: pd.DataFrame,
    base_counts: Mapping[tuple[str, str], int],
    feature_model_views: Mapping[str, pd.DataFrame],
    config: ObservedColliderStructureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if unresolved.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    assay_map = _feature_assay_map(feature_model_views)
    state_maps = [
        _state_score_map(frame, f"feature_model_views[{view_id!r}]")
        for view_id, frame in sorted(feature_model_views.items())
    ]
    median_state = {
        feature: float(np.median([mapping[feature] for mapping in state_maps]))
        for feature in sorted(assay_map)
    }
    evidence_map = {
        canonical_edge(str(record["node1"]), str(record["node2"])): record
        for record in aggregated.to_dict(orient="records")
    }
    rescue_pairs = {
        edge
        for edge, record in evidence_map.items()
        if float(record["selected_vote_fraction"])
        >= config.minimum_rescue_vote_fraction
        and float(record["mean_empirical_p_positive"])
        <= config.maximum_rescue_mean_empirical_p_positive
        and float(record["mean_pooled_log_odds"])
        >= config.minimum_rescue_mean_pooled_log_odds
    }
    base_pairs = {
        edge
        for edge, count in base_counts.items()
        if count >= config.minimum_base_view_count
    }
    candidate_pairs = base_pairs | rescue_pairs
    votes = vote_evidence.copy()
    _assert_observed_only(votes, "vote_evidence")
    _require_columns(
        votes,
        {
            "source_assay",
            "node1",
            "node2",
            "vote",
            "selected",
            "pooled_log_odds",
        },
        "vote_evidence",
    )
    vote_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for component in components.to_dict(orient="records"):
        if str(component["status"]) != "unresolved_no_complete_wedge":
            continue
        component_id = str(component["component_id"])
        component_unresolved = unresolved.loc[
            unresolved["component_id"].astype(str) == component_id
        ]
        if len(component_unresolved) != 1:
            continue
        record = component_unresolved.iloc[0]
        anchor = canonical_edge(str(record["node1"]), str(record["node2"]))
        anchor_base_count = int(base_counts.get(anchor, 0))
        if anchor_base_count < config.minimum_base_view_count or anchor not in rescue_pairs:
            continue
        assay = str(component["source_assay"])
        assay_features = sorted(
            feature for feature, feature_assay in assay_map.items() if feature_assay == assay
        )
        candidates: list[dict[str, Any]] = []
        for center, anchor_outer in (anchor, anchor[::-1]):
            for new_outer in assay_features:
                if new_outer in anchor:
                    continue
                new_arm = canonical_edge(center, new_outer)
                outer_pair = canonical_edge(anchor_outer, new_outer)
                if outer_pair in candidate_pairs or new_arm not in evidence_map:
                    continue
                aggregate_record = evidence_map[new_arm]
                candidates.append(
                    {
                        "source_assay": assay,
                        "component_id": component_id,
                        "center": center,
                        "anchor_outer": anchor_outer,
                        "new_outer": new_outer,
                        "anchor_node1": anchor[0],
                        "anchor_node2": anchor[1],
                        "new_arm_node1": new_arm[0],
                        "new_arm_node2": new_arm[1],
                        "outer_pair_node1": outer_pair[0],
                        "outer_pair_node2": outer_pair[1],
                        "anchor_base_view_count": anchor_base_count,
                        "anchor_rescue_selected": True,
                        "mean_pooled_log_odds": float(
                            aggregate_record["mean_pooled_log_odds"]
                        ),
                        "mean_selected_vote_fraction": float(
                            aggregate_record["selected_vote_fraction"]
                        ),
                        "outer_state_support_floor": min(
                            median_state[anchor_outer], median_state[new_outer]
                        ),
                    }
                )
        if not candidates:
            continue
        component_votes = sorted(
            votes.loc[votes["source_assay"].astype(str) == assay, "vote"].unique()
        )
        winner_counts = [0] * len(candidates)
        for vote in component_votes:
            vote_frame = votes.loc[
                (votes["source_assay"].astype(str) == assay)
                & (votes["vote"] == vote)
            ]
            vote_map = {
                canonical_edge(str(row["node1"]), str(row["node2"])): row
                for row in vote_frame.to_dict(orient="records")
            }
            scores: list[tuple[Any, ...]] = []
            for candidate in candidates:
                edge = (
                    str(candidate["new_arm_node1"]),
                    str(candidate["new_arm_node2"]),
                )
                if edge not in vote_map:
                    raise ValueError(f"Vote {vote!r} lacks completion edge {edge}")
                vote_record = vote_map[edge]
                score = (
                    int(bool(vote_record["selected"])),
                    float(vote_record["pooled_log_odds"]),
                    float(candidate["outer_state_support_floor"]),
                    str(candidate["center"]),
                    str(candidate["new_outer"]),
                )
                scores.append(score)
                vote_rows.append(
                    {
                        **candidate,
                        "vote": vote,
                        "vote_selected": bool(vote_record["selected"]),
                        "vote_pooled_log_odds": float(
                            vote_record["pooled_log_odds"]
                        ),
                        "view_winner": False,
                    }
                )
            winner_index = max(range(len(candidates)), key=lambda index: scores[index])
            winner_counts[winner_index] += 1
            for row in reversed(vote_rows):
                if row["component_id"] != component_id or row["vote"] != vote:
                    continue
                if (
                    row["center"] == candidates[winner_index]["center"]
                    and row["new_outer"] == candidates[winner_index]["new_outer"]
                ):
                    row["view_winner"] = True
                    break
        for index, candidate in enumerate(candidates):
            stable = winner_counts[index] >= config.minimum_winner_view_count
            summary_rows.append(
                {
                    **candidate,
                    "winner_view_count": winner_counts[index],
                    "n_views": len(component_votes),
                    "stable_view_winner": stable,
                }
            )
            if (
                stable
                and float(candidate["mean_pooled_log_odds"])
                > config.minimum_completion_mean_pooled_log_odds
            ):
                decisions.append(
                    {
                        "source_assay": assay,
                        "component_id": component_id,
                        "action": "complete_dual_supported_unshielded_anchor",
                        "center": str(candidate["center"]),
                        "outer1": str(candidate["anchor_outer"]),
                        "outer2": str(candidate["new_outer"]),
                        "winner_view_count": winner_counts[index],
                        "n_views": len(component_votes),
                        "anchor_node1": anchor[0],
                        "anchor_node2": anchor[1],
                        "new_arm_node1": str(candidate["new_arm_node1"]),
                        "new_arm_node2": str(candidate["new_arm_node2"]),
                    }
                )
    return (
        pd.DataFrame(vote_rows),
        pd.DataFrame(summary_rows),
        decisions,
    )


def _reorient_completion_decisions_by_beta_centrality(
    decisions: list[dict[str, Any]],
    aggregated: pd.DataFrame,
    vote_evidence: pd.DataFrame,
    feature_model_views: Mapping[str, pd.DataFrame],
    config: ObservedColliderStructureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Correct a completion's center using assay-wide beta centrality."""

    if not decisions:
        return pd.DataFrame(), pd.DataFrame(), []
    assay_map = _feature_assay_map(feature_model_views)
    features_by_assay: dict[str, list[str]] = {}
    for feature, assay in assay_map.items():
        features_by_assay.setdefault(assay, []).append(feature)
    features_by_assay = {
        assay: sorted(set(features))
        for assay, features in features_by_assay.items()
    }
    view_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    refined: list[dict[str, Any]] = []
    for decision in decisions:
        if str(decision["action"]) != "complete_dual_supported_unshielded_anchor":
            refined.append(dict(decision))
            continue
        assay = str(decision["source_assay"])
        component_id = str(decision["component_id"])
        anchor = canonical_edge(
            str(decision["anchor_node1"]), str(decision["anchor_node2"])
        )
        assay_aggregated = aggregated.loc[
            aggregated["source_assay"].astype(str).eq(assay)
        ]
        centrality: dict[str, tuple[float, float]] = {}
        for endpoint in anchor:
            incident = assay_aggregated.loc[
                assay_aggregated["node1"].astype(str).eq(endpoint)
                | assay_aggregated["node2"].astype(str).eq(endpoint)
            ]
            centrality[endpoint] = (
                float(incident["selected_vote_fraction"].sum()),
                float(incident["mean_pooled_log_odds"].sum()),
            )
        selected_center = max(
            anchor,
            key=lambda endpoint: (
                centrality[endpoint][0],
                centrality[endpoint][1],
                endpoint,
            ),
        )
        anchor_outer = anchor[1] if selected_center == anchor[0] else anchor[0]
        current_center = str(decision["center"])
        candidate_features = [
            feature
            for feature in features_by_assay[assay]
            if feature not in anchor
        ]
        if not candidate_features:
            raise ValueError(
                f"Completion component {component_id!r} has no third feature"
            )
        winner_counts = {feature: 0 for feature in candidate_features}
        contrasts: dict[str, list[float]] = {
            feature: [] for feature in candidate_features
        }
        assay_votes = vote_evidence.loc[
            vote_evidence["source_assay"].astype(str).eq(assay)
        ]
        for vote, vote_frame in assay_votes.groupby("vote", sort=True):
            vote_map = {
                canonical_edge(str(row["node1"]), str(row["node2"])): float(
                    row["pooled_log_odds"]
                )
                for row in vote_frame.to_dict(orient="records")
            }
            vote_contrasts: dict[str, float] = {}
            for feature in candidate_features:
                center_edge = canonical_edge(selected_center, feature)
                outer_edge = canonical_edge(anchor_outer, feature)
                if center_edge not in vote_map or outer_edge not in vote_map:
                    raise ValueError(
                        f"Vote {vote!r} lacks beta-centrality contrast pairs"
                    )
                vote_contrasts[feature] = (
                    vote_map[center_edge] - vote_map[outer_edge]
                )
            winner = max(
                candidate_features,
                key=lambda feature: (vote_contrasts[feature], feature),
            )
            winner_counts[winner] += 1
            for feature, contrast in vote_contrasts.items():
                contrasts[feature].append(float(contrast))
                view_rows.append(
                    {
                        "source_assay": assay,
                        "component_id": component_id,
                        "proposal_family": "completion_beta_centrality",
                        "current_center": current_center,
                        "selected_center": selected_center,
                        "anchor_outer": anchor_outer,
                        "candidate_new_parent": feature,
                        "anchor_node1": anchor[0],
                        "anchor_node2": anchor[1],
                        "vote": vote,
                        "center_vs_outer_lor_contrast": float(contrast),
                        "view_winner": feature == winner,
                    }
                )
        selected_new_parent = max(
            candidate_features,
            key=lambda feature: (
                winner_counts[feature],
                float(np.mean(contrasts[feature])),
                feature,
            ),
        )
        for feature in candidate_features:
            summary_rows.append(
                {
                    "source_assay": assay,
                    "component_id": component_id,
                    "proposal_family": "completion_beta_centrality",
                    "current_center": current_center,
                    "selected_center": selected_center,
                    "anchor_outer": anchor_outer,
                    "candidate_new_parent": feature,
                    "anchor_node1": anchor[0],
                    "anchor_node2": anchor[1],
                    "anchor_node1_selected_fraction_centrality": centrality[
                        anchor[0]
                    ][0],
                    "anchor_node1_lor_centrality": centrality[anchor[0]][1],
                    "anchor_node2_selected_fraction_centrality": centrality[
                        anchor[1]
                    ][0],
                    "anchor_node2_lor_centrality": centrality[anchor[1]][1],
                    "center_changes": selected_center != current_center,
                    "contrast_winner_view_count": winner_counts[feature],
                    "n_contrast_views": len(contrasts[feature]),
                    "mean_center_vs_outer_lor_contrast": float(
                        np.mean(contrasts[feature])
                    ),
                    "minimum_center_vs_outer_lor_contrast": float(
                        np.min(contrasts[feature])
                    ),
                    "selected_new_parent": feature == selected_new_parent,
                }
            )
        if (
            selected_center != current_center
            and winner_counts[selected_new_parent]
            >= config.minimum_completion_center_contrast_view_count
        ):
            refined.append(
                {
                    **dict(decision),
                    "action": "reorient_completion_by_assay_beta_centrality",
                    "precentrality_center": current_center,
                    "precentrality_outer1": str(decision["outer1"]),
                    "precentrality_outer2": str(decision["outer2"]),
                    "center": selected_center,
                    "outer1": anchor_outer,
                    "outer2": selected_new_parent,
                    "centrality_selected_new_parent": selected_new_parent,
                    "centrality_mean_center_vs_outer_lor_contrast": float(
                        np.mean(contrasts[selected_new_parent])
                    ),
                    "centrality_minimum_center_vs_outer_lor_contrast": float(
                        np.min(contrasts[selected_new_parent])
                    ),
                    "winner_view_count": winner_counts[selected_new_parent],
                    "n_views": len(contrasts[selected_new_parent]),
                    "new_arm_node1": canonical_edge(
                        selected_center, selected_new_parent
                    )[0],
                    "new_arm_node2": canonical_edge(
                        selected_center, selected_new_parent
                    )[1],
                }
            )
        else:
            refined.append(dict(decision))
    return pd.DataFrame(view_rows), pd.DataFrame(summary_rows), refined


def _loading_pair_maps(
    loading_pair_views: Mapping[str, pd.DataFrame],
) -> tuple[
    dict[str, dict[tuple[str, str], float]],
    dict[str, dict[tuple[str, str], float]],
]:
    p_maps: dict[str, dict[tuple[str, str], float]] = {}
    cosine_maps: dict[str, dict[tuple[str, str], float]] = {}
    for view_id, frame in sorted(loading_pair_views.items()):
        p_maps[str(view_id)] = _pair_map(
            frame,
            "loading_empirical_null_p",
            f"loading_pair_views[{view_id!r}]",
        )
        cosine_maps[str(view_id)] = _pair_map(
            frame,
            "absolute_loading_cosine",
            f"loading_pair_views[{view_id!r}]",
        )
        invalid_p = [
            value
            for value in p_maps[str(view_id)].values()
            if not 0.0 <= value <= 1.0
        ]
        if invalid_p:
            raise ValueError(
                f"loading_pair_views[{view_id!r}] contains p-values outside [0, 1]"
            )
    return p_maps, cosine_maps


def _rescue_only_completion_rows(
    components: pd.DataFrame,
    unresolved: pd.DataFrame,
    aggregated: pd.DataFrame,
    vote_evidence: pd.DataFrame,
    base_counts: Mapping[tuple[str, str], int],
    feature_model_views: Mapping[str, pd.DataFrame],
    loading_pair_views: Mapping[str, pd.DataFrame] | None,
    state_membership: pd.DataFrame | None,
    config: ObservedColliderStructureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Complete one rescue-only arm only under state/loading unanimity."""

    if loading_pair_views is None and state_membership is None:
        return pd.DataFrame(), pd.DataFrame(), []
    if loading_pair_views is None or state_membership is None:
        raise ValueError(
            "Rescue-only completion requires both loading-pair views and "
            "frozen state membership"
        )
    if set(loading_pair_views) != set(feature_model_views):
        raise ValueError(
            "Loading-pair and feature-model view identifiers must match exactly"
        )
    _assert_observed_only(state_membership, "state_membership")
    _require_columns(
        state_membership,
        {"feature", "selection_status"},
        "state_membership",
    )
    if state_membership["feature"].astype(str).duplicated().any():
        raise ValueError("state_membership contains duplicate features")
    state_selected = {
        str(record["feature"]): str(record["selection_status"]) == "selected"
        for record in state_membership.to_dict(orient="records")
    }
    assay_map = _feature_assay_map(feature_model_views)
    missing_state_features = sorted(set(assay_map) - set(state_selected))
    if missing_state_features:
        raise ValueError(
            "state_membership lacks observed focal features: "
            f"{missing_state_features}"
        )
    features_by_assay: dict[str, list[str]] = {}
    for feature, assay in assay_map.items():
        features_by_assay.setdefault(assay, []).append(feature)
    features_by_assay = {
        assay: sorted(set(features))
        for assay, features in features_by_assay.items()
    }
    aggregated_map = {
        canonical_edge(str(record["node1"]), str(record["node2"])): record
        for record in aggregated.to_dict(orient="records")
    }
    p_maps, cosine_maps = _loading_pair_maps(loading_pair_views)
    n_loading_views = len(p_maps)
    if n_loading_views != config.required_rescue_only_loading_view_count:
        raise ValueError(
            "Rescue-only completion requires exactly "
            f"{config.required_rescue_only_loading_view_count} loading views; "
            f"received {n_loading_views}"
        )
    required_loading_support = config.required_rescue_only_loading_view_count
    vote_groups = list(vote_evidence.groupby("vote", sort=True))
    n_vote_views = len(vote_groups)
    if n_vote_views != config.required_rescue_only_beta_vote_count:
        raise ValueError(
            "Rescue-only completion requires exactly "
            f"{config.required_rescue_only_beta_vote_count} beta votes; "
            f"received {n_vote_views}"
        )
    required_contrast_support = config.required_rescue_only_beta_vote_count
    evidence_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for component in components.to_dict(orient="records"):
        if str(component["status"]) != "unresolved_no_complete_wedge":
            continue
        component_id = str(component["component_id"])
        assay = str(component["source_assay"])
        component_unresolved = unresolved.loc[
            unresolved["component_id"].astype(str).eq(component_id)
        ]
        if len(component_unresolved) != 1:
            continue
        unresolved_record = component_unresolved.iloc[0]
        anchor = canonical_edge(
            str(unresolved_record["node1"]),
            str(unresolved_record["node2"]),
        )
        anchor_base_view_count = int(base_counts.get(anchor, 0))
        if anchor not in aggregated_map:
            raise ValueError(f"Rescue-only anchor {anchor} lacks aggregate evidence")
        anchor_evidence = aggregated_map[anchor]
        anchor_rescue_selected = bool(
            float(anchor_evidence["selected_vote_fraction"])
            >= config.minimum_rescue_vote_fraction
            and float(anchor_evidence["mean_empirical_p_positive"])
            <= config.maximum_rescue_mean_empirical_p_positive
            and float(anchor_evidence["mean_pooled_log_odds"])
            >= config.minimum_rescue_mean_pooled_log_odds
        )
        rescue_only_component = bool(
            anchor_rescue_selected
            and anchor_base_view_count < config.minimum_base_view_count
        )
        if not rescue_only_component:
            continue
        passing: list[dict[str, Any]] = []
        assay_votes = vote_evidence.loc[
            vote_evidence["source_assay"].astype(str).eq(assay)
        ]
        assay_vote_count = int(assay_votes["vote"].nunique())
        if assay_vote_count != config.required_rescue_only_beta_vote_count:
            raise ValueError(
                f"Rescue-only assay {assay!r} requires exactly "
                f"{config.required_rescue_only_beta_vote_count} beta votes; "
                f"received {assay_vote_count}"
            )
        for center, anchor_outer in (anchor, anchor[::-1]):
            for new_parent in features_by_assay[assay]:
                if new_parent in anchor:
                    continue
                new_arm = canonical_edge(center, new_parent)
                outer_pair = canonical_edge(anchor_outer, new_parent)
                loading_p_values: list[float] = []
                loading_cosines: list[float] = []
                for view_id in sorted(p_maps):
                    if (
                        new_arm not in p_maps[view_id]
                        or new_arm not in cosine_maps[view_id]
                    ):
                        raise ValueError(
                            f"Loading view {view_id!r} lacks rescue-only arm {new_arm}"
                        )
                    p_value = float(p_maps[view_id][new_arm])
                    cosine = float(cosine_maps[view_id][new_arm])
                    loading_p_values.append(p_value)
                    loading_cosines.append(cosine)
                    view_rows.append(
                        {
                            "source_assay": assay,
                            "component_id": component_id,
                            "proposal_family": (
                                "rescue_only_state_loading_contrast"
                            ),
                            "center": center,
                            "anchor_outer": anchor_outer,
                            "candidate_new_parent": new_parent,
                            "evidence_view_type": "loading",
                            "view_id": view_id,
                            "loading_empirical_null_p": p_value,
                            "absolute_loading_cosine": cosine,
                            "view_gate_passes": p_value
                            <= config.maximum_rescue_only_loading_empirical_null_p,
                        }
                    )
                contrasts: list[float] = []
                for vote, vote_frame in assay_votes.groupby("vote", sort=True):
                    vote_map = {
                        canonical_edge(
                            str(record["node1"]), str(record["node2"])
                        ): float(record["pooled_log_odds"])
                        for record in vote_frame.to_dict(orient="records")
                    }
                    if new_arm not in vote_map or outer_pair not in vote_map:
                        raise ValueError(
                            f"Vote {vote!r} lacks rescue-only contrast pairs"
                        )
                    contrast = vote_map[new_arm] - vote_map[outer_pair]
                    contrasts.append(float(contrast))
                    view_rows.append(
                        {
                            "source_assay": assay,
                            "component_id": component_id,
                            "proposal_family": (
                                "rescue_only_state_loading_contrast"
                            ),
                            "center": center,
                            "anchor_outer": anchor_outer,
                            "candidate_new_parent": new_parent,
                            "evidence_view_type": "beta_contrast",
                            "vote": vote,
                            "direct_vs_outer_lor_contrast": float(contrast),
                            "view_gate_passes": contrast > 0.0,
                        }
                    )
                loading_support = sum(
                    value
                    <= config.maximum_rescue_only_loading_empirical_null_p
                    for value in loading_p_values
                )
                contrast_support = sum(value > 0.0 for value in contrasts)
                candidate_state_selected = bool(
                    state_selected.get(new_parent, False)
                )
                passes = bool(
                    candidate_state_selected
                    and loading_support >= required_loading_support
                    and contrast_support >= required_contrast_support
                )
                record = {
                    "source_assay": assay,
                    "component_id": component_id,
                    "proposal_family": "rescue_only_state_loading_contrast",
                    "center": center,
                    "anchor_outer": anchor_outer,
                    "candidate_new_parent": new_parent,
                    "anchor_node1": anchor[0],
                    "anchor_node2": anchor[1],
                    "anchor_base_view_count": anchor_base_view_count,
                    "anchor_rescue_selected": anchor_rescue_selected,
                    "rescue_only_component": rescue_only_component,
                    "new_parent_formal_state_selected": (
                        candidate_state_selected
                    ),
                    "loading_empirical_null_view_support": loading_support,
                    "required_loading_view_support": required_loading_support,
                    "maximum_loading_empirical_null_p": max(loading_p_values),
                    "minimum_absolute_loading_cosine": min(loading_cosines),
                    "positive_contrast_view_support": contrast_support,
                    "required_positive_contrast_view_support": (
                        required_contrast_support
                    ),
                    "mean_direct_vs_outer_lor_contrast": float(
                        np.mean(contrasts)
                    ),
                    "minimum_direct_vs_outer_lor_contrast": float(
                        np.min(contrasts)
                    ),
                    "passes_all_candidate_gates": passes,
                }
                evidence_rows.append(record)
                if passes:
                    passing.append(record)
        unique = len(passing) == 1
        for record in evidence_rows:
            if record["component_id"] == component_id:
                record["n_component_gate_passing_candidates"] = len(passing)
                record["unique_component_candidate"] = unique
        should_select = unique if config.require_unique_rescue_only_candidate else bool(passing)
        if not should_select:
            continue
        winner = passing[0]
        decisions.append(
            {
                "source_assay": assay,
                "component_id": component_id,
                "action": "complete_unique_rescue_only_state_loading_contrast",
                "center": str(winner["center"]),
                "outer1": str(winner["anchor_outer"]),
                "outer2": str(winner["candidate_new_parent"]),
                "winner_view_count": int(
                    winner["positive_contrast_view_support"]
                ),
                "n_views": n_vote_views,
                "anchor_node1": anchor[0],
                "anchor_node2": anchor[1],
                "new_arm_node1": canonical_edge(
                    str(winner["center"]),
                    str(winner["candidate_new_parent"]),
                )[0],
                "new_arm_node2": canonical_edge(
                    str(winner["center"]),
                    str(winner["candidate_new_parent"]),
                )[1],
            }
        )
    return pd.DataFrame(view_rows), pd.DataFrame(evidence_rows), decisions


def _abstain_overlapping_refinement_arms(
    decisions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Abstain globally when independent wedges claim a common direct arm."""

    if decisions.empty:
        return decisions.copy(), []
    output = decisions.reset_index(drop=True).copy()
    arm_to_decisions: dict[tuple[str, tuple[str, str]], list[int]] = {}
    for index, record in output.iterrows():
        assay = str(record["source_assay"])
        center = str(record["center"])
        for outer in (str(record["outer1"]), str(record["outer2"])):
            arm_to_decisions.setdefault(
                (assay, canonical_edge(center, outer)), []
            ).append(int(index))
    conflicting_indices = {
        index
        for indices in arm_to_decisions.values()
        if len(indices) > 1
        for index in indices
    }
    if not conflicting_indices:
        return output, []
    conflicting_arms_by_index: dict[int, list[str]] = {
        index: [] for index in conflicting_indices
    }
    for (assay, arm), indices in sorted(arm_to_decisions.items()):
        if len(indices) <= 1:
            continue
        label = f"{assay}:{arm[0]}|{arm[1]}"
        for index in indices:
            conflicting_arms_by_index[index].append(label)
    abstentions: list[dict[str, Any]] = []
    for index in sorted(conflicting_indices):
        record = output.loc[index]
        abstentions.append(
            {
                "source_assay": str(record["source_assay"]),
                "component_id": str(record["component_id"]),
                "action": str(record["action"]),
                "conflicting_selected_arms": sorted(
                    conflicting_arms_by_index[index]
                ),
                "reason": "overlapping_refinement_arm_requires_global_abstention",
            }
        )
    return output.drop(index=sorted(conflicting_indices)).copy(), abstentions


def select_observed_collider_refinements(
    components: pd.DataFrame,
    wedges: pd.DataFrame,
    unresolved: pd.DataFrame,
    vote_evidence: pd.DataFrame,
    base_edge_views: Mapping[str, pd.DataFrame],
    residual_views: Mapping[str, pd.DataFrame],
    feature_model_views: Mapping[str, pd.DataFrame],
    config: ObservedColliderStructureConfig | None = None,
    *,
    loading_pair_views: Mapping[str, pd.DataFrame] | None = None,
    state_membership: pd.DataFrame | None = None,
) -> ObservedColliderStructureResult:
    """Select stable structural refinements without opening scorer information."""

    cfg = config or ObservedColliderStructureConfig()
    cfg.validate()
    for source, frame, required in (
        (
            "components",
            components,
            {"source_assay", "component_id", "status"},
        ),
        (
            "wedges",
            wedges,
            {
                "source_assay",
                "component_id",
                "center",
                "outer1",
                "outer2",
                "wedge_score",
            },
        ),
        (
            "unresolved",
            unresolved,
            {"source_assay", "component_id", "node1", "node2"},
        ),
    ):
        _assert_observed_only(frame, source)
        if source == "components" or not frame.empty:
            _require_columns(frame, required, source)
    if len(residual_views) < cfg.minimum_winner_view_count:
        raise ValueError("Too few residual views for the winner-stability gate")
    if len(base_edge_views) < cfg.minimum_base_view_count:
        raise ValueError("Too few base graph views for the anchor-stability gate")
    optional_inputs_supplied = (
        loading_pair_views is not None or state_membership is not None
    )
    if (
        optional_inputs_supplied
        and len(base_edge_views)
        != cfg.required_rescue_only_base_graph_view_count
    ):
        raise ValueError(
            "Rescue-only completion requires exactly "
            f"{cfg.required_rescue_only_base_graph_view_count} base graph views; "
            f"received {len(base_edge_views)}"
        )
    _validate_vote_evidence_rows(vote_evidence)
    aggregated = aggregate_vote_evidence(vote_evidence)
    base_counts, base_view_ids = _base_support(base_edge_views)
    wedge_views = _wedge_view_rows(
        wedges,
        residual_views,
        feature_model_views,
        base_counts,
        cfg,
    )
    wedge_summary, wedge_decisions = _rank_wedge_views(
        components,
        wedges,
        wedge_views,
        cfg,
    )
    wedge_summary, wedge_decisions = (
        _add_rejected_dual_base_half_view_decisions(
            components,
            wedge_summary,
            wedge_decisions,
            cfg,
        )
    )
    completion_views, completion_summary, completion_decisions = _completion_rows(
        components,
        unresolved,
        aggregated,
        vote_evidence,
        base_counts,
        feature_model_views,
        cfg,
    )
    centrality_views, centrality_summary, completion_decisions = (
        _reorient_completion_decisions_by_beta_centrality(
            completion_decisions,
            aggregated,
            vote_evidence,
            feature_model_views,
            cfg,
        )
    )
    rescue_only_views, rescue_only_summary, rescue_only_decisions = (
        _rescue_only_completion_rows(
            components,
            unresolved,
            aggregated,
            vote_evidence,
            base_counts,
            feature_model_views,
            loading_pair_views,
            state_membership,
            cfg,
        )
    )
    completion_view_frames = [
        frame
        for frame in (
            (
                completion_views.assign(
                    proposal_family="dual_supported_unshielded_anchor"
                )
                if not completion_views.empty
                else completion_views
            ),
            centrality_views,
            rescue_only_views,
        )
        if not frame.empty
    ]
    completion_summary_frames = [
        frame
        for frame in (
            (
                completion_summary.assign(
                    proposal_family="dual_supported_unshielded_anchor"
                )
                if not completion_summary.empty
                else completion_summary
            ),
            centrality_summary,
            rescue_only_summary,
        )
        if not frame.empty
    ]
    completion_views = (
        pd.concat(completion_view_frames, ignore_index=True, sort=False)
        if completion_view_frames
        else pd.DataFrame()
    )
    completion_summary = (
        pd.concat(completion_summary_frames, ignore_index=True, sort=False)
        if completion_summary_frames
        else pd.DataFrame()
    )
    decisions = pd.DataFrame(
        [
            *wedge_decisions,
            *completion_decisions,
            *rescue_only_decisions,
        ]
    )
    pre_overlap_decision_count = len(decisions)
    overlap_abstentions: list[dict[str, Any]] = []
    if not decisions.empty:
        if decisions["component_id"].astype(str).duplicated().any():
            raise ValueError("Structural refinement selected more than one decision per component")
        decisions, overlap_abstentions = (
            _abstain_overlapping_refinement_arms(decisions)
        )
        decisions = decisions.sort_values(
            ["source_assay", "component_id", "action"]
        ).reset_index(drop=True)
    metadata = {
        "method": "observed_only_stable_collider_structure_refinement",
        "schema_version": "observed_collider_structure.v4",
        "config": asdict(cfg),
        "n_base_views": len(base_edge_views),
        "n_residual_views": len(residual_views),
        "n_feature_model_views": len(feature_model_views),
        "n_vote_views": int(vote_evidence["vote"].nunique()),
        "n_decisions": len(decisions),
        "n_pre_overlap_abstention_decisions": pre_overlap_decision_count,
        "n_overlapping_arm_conflict_abstentions": len(overlap_abstentions),
        "n_same_center_pairwise_conflict_abstentions": int(
            wedge_summary.get(
                "stable_view_winner_pairwise_conflict_abstained",
                pd.Series(dtype=bool),
            )
            .fillna(False)
            .astype(bool)
            .sum()
        ),
        "same_center_view_winner_policy": (
            "view-stable winner must also have a strictly positive pairwise "
            "wedge margin"
        ),
        "overlapping_arm_conflict_policy": (
            "abstain_all_refinement_decisions_incident_to_a_shared_arm"
        ),
        "overlapping_arm_conflict_abstentions": overlap_abstentions,
        "decision_action_counts": (
            {
                str(action): int(count)
                for action, count in decisions["action"].value_counts().items()
            }
            if not decisions.empty
            else {}
        ),
        "loading_pair_views_supplied": loading_pair_views is not None,
        "formal_state_membership_supplied": state_membership is not None,
        "loading_views_reuse_same_observed_cells": True,
        "loading_views_are_algorithmic_stability_repeats": True,
        "loading_views_are_independent_worlds": False,
        "loading_view_support_not_world_sample_size": True,
        "beta_votes_reuse_same_observed_cells": True,
        "beta_votes_are_algorithmic_stability_repeats": True,
        "beta_votes_are_independent_experiments": False,
        "beta_vote_support_not_world_sample_size": True,
        "completion_center_centrality_formula": (
            "lexicographic(sum_incident_selected_vote_fraction, "
            "sum_incident_mean_pooled_log_odds, endpoint_id)"
        ),
        "completion_new_parent_contrast_formula": (
            "pooled_log_odds(center,new_parent) - "
            "pooled_log_odds(anchor_outer,new_parent)"
        ),
        "rescue_only_loading_gate": (
            "formal_state_selected AND loading_empirical_null_p stability "
            "AND positive beta contrast stability AND unique candidate"
        ),
        "base_support_view_ids": {
            "|".join(edge): list(view_ids)
            for edge, view_ids in sorted(base_view_ids.items())
        },
        "scorer_information_used": False,
        "paired_cells_used": False,
        "external_prior_used": False,
        "selection_is_structural_not_global_pair_threshold_relaxation": True,
    }
    return ObservedColliderStructureResult(
        decisions=decisions,
        wedge_view_evidence=wedge_views,
        wedge_evidence=wedge_summary,
        completion_view_evidence=completion_views,
        completion_evidence=completion_summary,
        metadata=metadata,
    )


def _aggregated_record_map(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    _assert_observed_only(frame, "motif.aggregated_evidence")
    _require_columns(
        frame,
        {
            "node1",
            "node2",
            "selected_vote_fraction",
            "mean_empirical_p_positive",
            "mean_pooled_log_odds",
        },
        "motif.aggregated_evidence",
    )
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        edge = canonical_edge(str(record["node1"]), str(record["node2"]))
        if edge in records:
            raise ValueError(f"Aggregated evidence contains duplicate pair {edge}")
        records[edge] = record
    return records


def _direct_row_from_evidence(
    *,
    assay: str,
    edge: tuple[str, str],
    center: str,
    component_id: str,
    wedge_score: float,
    runner_up_score: float,
    evidence: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    left, right = edge
    if center not in edge:
        raise ValueError("A selected collider center must be an edge endpoint")
    center_is_left = center == left
    margin = (
        wedge_score - runner_up_score if np.isfinite(runner_up_score) else np.inf
    )
    return {
        "source_assay": assay,
        "node1": left,
        "endpoint1": "ARROW" if center_is_left else "TAIL",
        "node2": right,
        "endpoint2": "TAIL" if center_is_left else "ARROW",
        "edge_type": _DIRECT_EDGE_TYPE,
        "orientation_source": "observed_structural_view_consensus",
        "collider_center": center,
        "component_id": component_id,
        "wedge_score": wedge_score,
        "runner_up_wedge_score": runner_up_score,
        "wedge_margin": margin,
        "selected_vote_fraction": float(evidence["selected_vote_fraction"]),
        "mean_empirical_p_positive": float(evidence["mean_empirical_p_positive"]),
        "mean_pooled_log_odds": float(evidence["mean_pooled_log_odds"]),
        "observed_refinement_action": action,
        "interpretation": (
            "edge in a stable observed-only within-assay collider candidate "
            f"({action})"
        ),
    }


def _rescue_selected(
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    return bool(
        float(evidence["selected_vote_fraction"])
        >= float(config["minimum_rescue_vote_fraction"])
        and float(evidence["mean_empirical_p_positive"])
        <= float(config["maximum_rescue_mean_empirical_p_positive"])
        and float(evidence["mean_pooled_log_odds"])
        >= float(config["minimum_rescue_mean_pooled_log_odds"])
    )


def _base_selected(
    edge: tuple[str, str],
    refinement_metadata: Mapping[str, Any],
) -> bool:
    config = refinement_metadata.get("config", {})
    minimum = int(config.get("minimum_base_view_count", 4))
    view_ids = refinement_metadata.get("base_support_view_ids", {}).get(
        "|".join(edge), []
    )
    return len(view_ids) >= minimum


def _unresolved_row_from_evidence(
    *,
    assay: str,
    edge: tuple[str, str],
    component_id: str,
    evidence: Mapping[str, Any],
    refinement_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    config = refinement_metadata.get("config", {})
    return {
        "source_assay": assay,
        "node1": edge[0],
        "endpoint1": "CIRCLE",
        "node2": edge[1],
        "endpoint2": "CIRCLE",
        "edge_type": "unresolved",
        "component_id": component_id,
        "was_base_selected": _base_selected(edge, refinement_metadata),
        "was_rescue_selected": _rescue_selected(evidence, config),
        "mean_empirical_p_positive": float(evidence["mean_empirical_p_positive"]),
        "mean_pooled_log_odds": float(evidence["mean_pooled_log_odds"]),
        "reason": "not_in_stable_observed_structural_winner",
    }


def _empty_with_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if not frame.empty:
        return frame.copy()
    return pd.DataFrame(columns=list(frame.columns) or list(columns))


def _with_observed_refinement_action(frame: pd.DataFrame) -> pd.DataFrame:
    """Return direct candidates with explicit, structured action provenance."""

    output = frame.copy()
    if "observed_refinement_action" not in output.columns:
        output["observed_refinement_action"] = ""
    else:
        output["observed_refinement_action"] = output[
            "observed_refinement_action"
        ].fillna("").astype(str)
    return output


def apply_observed_collider_refinements(
    motif: Any,
    refinement: ObservedColliderStructureResult,
) -> Any:
    """Apply each structural decision once and return a consistent motif result.

    The returned object has the same runtime type as ``motif`` (normally
    :class:`structure_aware_collider_motif.ColliderMotifResult`).  A second
    application is rejected explicitly so provenance cannot silently record
    the same recovery twice.
    """

    required_attributes = {
        "direct_candidates",
        "demoted_or_unresolved",
        "component_diagnostics",
        "wedge_diagnostics",
        "aggregated_evidence",
        "metadata",
    }
    missing_attributes = sorted(
        attribute for attribute in required_attributes if not hasattr(motif, attribute)
    )
    if missing_attributes:
        raise TypeError(f"motif lacks required result fields: {missing_attributes}")
    if bool(motif.metadata.get("observed_structural_refinement_applied", False)):
        raise ValueError("Observed collider refinements have already been applied")
    decisions = refinement.decisions.copy()
    _assert_observed_only(decisions, "refinement.decisions")
    if not decisions.empty:
        _require_columns(
            decisions,
            {
                "source_assay",
                "component_id",
                "action",
                "center",
                "outer1",
                "outer2",
            },
            "refinement.decisions",
        )
        if decisions["component_id"].astype(str).duplicated().any():
            raise ValueError("Refinement decisions must be unique by component")

    if decisions.empty:
        metadata = {
            **dict(motif.metadata),
            "method": "unique_positive_residual_collider_motif_plus_observed_structure",
            "schema_version": "collider_motif_selector.v4",
            "observed_structural_refinement_applied": True,
            "active_direct_observed_refinement_action_field": (
                "observed_refinement_action"
            ),
            "observed_structural_refinement": {
                **dict(refinement.metadata),
                "decisions": [],
            },
            "scorer_information_used": False,
        }
        result_type = type(motif)
        return result_type(
            direct_candidates=_with_observed_refinement_action(
                motif.direct_candidates
            ),
            demoted_or_unresolved=motif.demoted_or_unresolved.copy(),
            component_diagnostics=motif.component_diagnostics.copy(),
            wedge_diagnostics=motif.wedge_diagnostics.copy(),
            aggregated_evidence=motif.aggregated_evidence.copy(),
            metadata=metadata,
        )

    direct = _with_observed_refinement_action(
        _empty_with_columns(motif.direct_candidates, _DIRECT_COLUMNS)
    )
    unresolved = _empty_with_columns(
        motif.demoted_or_unresolved, _UNRESOLVED_COLUMNS
    )
    components = motif.component_diagnostics.copy()
    wedges = _empty_with_columns(motif.wedge_diagnostics, _WEDGE_COLUMNS)
    aggregated = motif.aggregated_evidence.copy()
    evidence_map = _aggregated_record_map(aggregated)
    component_ids = set(components["component_id"].astype(str))
    new_direct_rows: list[dict[str, Any]] = []
    new_unresolved_rows: list[dict[str, Any]] = []
    new_wedge_rows: list[dict[str, Any]] = []
    decision_audit: list[dict[str, Any]] = []

    for decision in decisions.to_dict(orient="records"):
        component_id = str(decision["component_id"])
        if component_id not in component_ids:
            raise ValueError(f"Refinement names unknown component {component_id!r}")
        assay = str(decision["source_assay"])
        center = str(decision["center"])
        outer1 = str(decision["outer1"])
        outer2 = str(decision["outer2"])
        action = str(decision["action"])
        if outer1 == outer2 or center in {outer1, outer2}:
            raise ValueError("A refinement decision must describe a three-node wedge")
        arm1 = canonical_edge(center, outer1)
        arm2 = canonical_edge(center, outer2)
        selected_arms = {arm1, arm2}
        if any(edge not in evidence_map for edge in selected_arms):
            raise ValueError("A selected refinement arm lacks aggregated evidence")
        arm1_log_odds = float(evidence_map[arm1]["mean_pooled_log_odds"])
        arm2_log_odds = float(evidence_map[arm2]["mean_pooled_log_odds"])
        wedge_score = min(arm1_log_odds, arm2_log_odds)

        component_wedges = wedges.loc[
            wedges["component_id"].astype(str) == component_id
        ]
        component_edges: set[tuple[str, str]] = set()
        for record in component_wedges.to_dict(orient="records"):
            component_edges.add(
                canonical_edge(str(record["center"]), str(record["outer1"]))
            )
            component_edges.add(
                canonical_edge(str(record["center"]), str(record["outer2"]))
            )
        component_unresolved = unresolved.loc[
            unresolved["component_id"].astype(str) == component_id
        ]
        component_edges.update(
            canonical_edge(str(record["node1"]), str(record["node2"]))
            for record in component_unresolved.to_dict(orient="records")
        )
        component_direct = direct.loc[
            direct["component_id"].astype(str) == component_id
        ]
        component_edges.update(
            canonical_edge(str(record["node1"]), str(record["node2"]))
            for record in component_direct.to_dict(orient="records")
        )
        component_edges.update(selected_arms)

        other_scores = [
            float(record["wedge_score"])
            for record in component_wedges.to_dict(orient="records")
            if {
                canonical_edge(str(record["center"]), str(record["outer1"])),
                canonical_edge(str(record["center"]), str(record["outer2"])),
            }
            != selected_arms
        ]
        runner_up_score = max(other_scores) if other_scores else -np.inf
        direct = direct.loc[
            direct["component_id"].astype(str) != component_id
        ].copy()
        unresolved = unresolved.loc[
            unresolved["component_id"].astype(str) != component_id
        ].copy()
        for edge in sorted(selected_arms):
            direct_row = _direct_row_from_evidence(
                assay=assay,
                edge=edge,
                center=center,
                component_id=component_id,
                wedge_score=wedge_score,
                runner_up_score=runner_up_score,
                evidence=evidence_map[edge],
                action=action,
            )
            if action == "reorient_completion_by_assay_beta_centrality":
                for field in (
                    "precentrality_center",
                    "precentrality_outer1",
                    "precentrality_outer2",
                    "centrality_selected_new_parent",
                    "centrality_mean_center_vs_outer_lor_contrast",
                    "centrality_minimum_center_vs_outer_lor_contrast",
                ):
                    direct_row[field] = decision[field]
            elif action == "complete_dual_supported_unshielded_anchor":
                direct_row["completion_anchor_node1"] = decision["anchor_node1"]
                direct_row["completion_anchor_node2"] = decision["anchor_node2"]
                direct_row["completion_winner_view_count"] = decision[
                    "winner_view_count"
                ]
                direct_row["completion_n_views"] = decision["n_views"]
            new_direct_rows.append(direct_row)
        for edge in sorted(component_edges - selected_arms):
            if edge not in evidence_map:
                raise ValueError("A demoted component edge lacks aggregated evidence")
            new_unresolved_rows.append(
                _unresolved_row_from_evidence(
                    assay=assay,
                    edge=edge,
                    component_id=component_id,
                    evidence=evidence_map[edge],
                    refinement_metadata=refinement.metadata,
                )
            )

        selected_wedge_mask = (
            wedges["component_id"].astype(str).eq(component_id)
            & wedges["center"].astype(str).eq(center)
            & wedges.apply(
                lambda row: {
                    canonical_edge(str(row["center"]), str(row["outer1"])),
                    canonical_edge(str(row["center"]), str(row["outer2"])),
                }
                == selected_arms,
                axis=1,
            )
        )
        wedges.loc[
            wedges["component_id"].astype(str).eq(component_id), "accepted"
        ] = False
        if selected_wedge_mask.any():
            wedges.loc[selected_wedge_mask, "accepted"] = True
        else:
            edge1, edge2 = sorted(selected_arms)
            new_wedge_rows.append(
                {
                    "source_assay": assay,
                    "component_id": component_id,
                    "center": center,
                    "outer1": outer1,
                    "outer2": outer2,
                    "arm1_log_odds": arm1_log_odds,
                    "arm2_log_odds": arm2_log_odds,
                    "wedge_score": wedge_score,
                    "outer_pair_is_candidate": False,
                    "edge1_node1": edge1[0],
                    "edge1_node2": edge1[1],
                    "edge2_node1": edge2[0],
                    "edge2_node2": edge2[1],
                    "rank": 1,
                    "accepted": True,
                }
            )

        component_mask = components["component_id"].astype(str).eq(component_id)
        # Keep the component inventory aligned with every retained candidate
        # edge, including spokes demoted by a structural reselection.  The
        # selected wedge is recorded separately by ``best_center`` and the
        # accepted wedge diagnostic; narrowing ``nodes`` to that wedge would
        # make ``n_nodes`` inconsistent with ``n_candidate_edges``.
        nodes = sorted({node for edge in component_edges for node in edge})
        components.loc[component_mask, "nodes"] = ";".join(nodes)
        components.loc[component_mask, "n_nodes"] = len(nodes)
        components.loc[component_mask, "n_candidate_edges"] = len(component_edges)
        components.loc[component_mask, "n_wedges"] = max(
            1, int(component_wedges.shape[0])
        )
        components.loc[component_mask, "best_center"] = center
        components.loc[component_mask, "best_wedge_score"] = wedge_score
        components.loc[component_mask, "runner_up_wedge_score"] = (
            runner_up_score if np.isfinite(runner_up_score) else np.nan
        )
        components.loc[component_mask, "accepted"] = True
        components.loc[
            component_mask, "status"
        ] = "accepted_observed_structural_wedge"
        decision_audit.append(
            {
                "component_id": component_id,
                "action": action,
                "center": center,
                "outer1": outer1,
                "outer2": outer2,
                "winner_view_count": int(decision.get("winner_view_count", 0)),
                "n_views": int(decision.get("n_views", 0)),
                **(
                    {
                        field: decision[field]
                        for field in (
                            "precentrality_center",
                            "precentrality_outer1",
                            "precentrality_outer2",
                            "centrality_selected_new_parent",
                            "centrality_mean_center_vs_outer_lor_contrast",
                            "centrality_minimum_center_vs_outer_lor_contrast",
                        )
                    }
                    if action == "reorient_completion_by_assay_beta_centrality"
                    else {}
                ),
                **(
                    {
                        "completion_anchor_node1": decision["anchor_node1"],
                        "completion_anchor_node2": decision["anchor_node2"],
                        "completion_winner_view_count": decision[
                            "winner_view_count"
                        ],
                        "completion_n_views": decision["n_views"],
                    }
                    if action == "complete_dual_supported_unshielded_anchor"
                    else {}
                ),
            }
        )

    if new_direct_rows:
        additions = pd.DataFrame(new_direct_rows)
        direct = (
            additions
            if direct.empty
            else pd.concat([direct, additions], ignore_index=True, sort=False)
        )
    if new_unresolved_rows:
        additions = pd.DataFrame(new_unresolved_rows)
        unresolved = (
            additions
            if unresolved.empty
            else pd.concat(
                [unresolved, additions], ignore_index=True, sort=False
            )
        )
    if new_wedge_rows:
        additions = pd.DataFrame(new_wedge_rows)
        wedges = (
            additions
            if wedges.empty
            else pd.concat([wedges, additions], ignore_index=True, sort=False)
        )
    if not direct.empty:
        direct_pairs = direct.apply(
            lambda row: canonical_edge(str(row["node1"]), str(row["node2"])),
            axis=1,
        )
        if direct_pairs.duplicated().any():
            raise ValueError("Applied refinement produced duplicate direct arms")
        direct = direct.sort_values(["source_assay", "node1", "node2"]).reset_index(
            drop=True
        )
    if not unresolved.empty:
        unresolved_pairs = unresolved.apply(
            lambda row: canonical_edge(str(row["node1"]), str(row["node2"])),
            axis=1,
        )
        if unresolved_pairs.duplicated().any():
            raise ValueError("Applied refinement produced duplicate unresolved edges")
        unresolved = unresolved.sort_values(
            ["source_assay", "component_id", "node1", "node2"]
        ).reset_index(drop=True)
    if not wedges.empty:
        wedges = wedges.sort_values(
            ["source_assay", "component_id", "rank", "center", "outer1", "outer2"]
        ).reset_index(drop=True)
    components = components.sort_values(
        ["source_assay", "component_id"]
    ).reset_index(drop=True)
    metadata = {
        **dict(motif.metadata),
        "method": "unique_positive_residual_collider_motif_plus_observed_structure",
        "schema_version": "collider_motif_selector.v4",
        "observed_structural_refinement_applied": True,
        "active_direct_observed_refinement_action_field": (
            "observed_refinement_action"
        ),
        "observed_structural_refinement": {
            **dict(refinement.metadata),
            "decisions": decision_audit,
        },
        "n_direct_candidate_edges": int(len(direct)),
        "n_unresolved_or_demoted_edges": int(len(unresolved)),
        "n_accepted_components": int(
            components["accepted"].astype(bool).sum() if not components.empty else 0
        ),
        "n_unresolved_components": int(
            (~components["accepted"].astype(bool)).sum()
            if not components.empty
            else 0
        ),
        "scorer_information_used": False,
    }
    result_type = type(motif)
    return result_type(
        direct_candidates=direct,
        demoted_or_unresolved=unresolved,
        component_diagnostics=components,
        wedge_diagnostics=wedges,
        aggregated_evidence=aggregated,
        metadata=metadata,
    )
