#!/usr/bin/env python3
"""Select assay-local P-like--reporter structure without privileged labels.

The observable direct layer supplies one collider center and two parent
candidates.  The formal maxT state layer nominates a nonmotif reporter, and
assay-local loading evidence chooses which parent shares that reporter's state
axis.  Parent membership is deliberately not required: weak direct parents
are identified structurally, not rescued by relabeling the state-membership
test.

Selection is uncertainty-aware and abstention-first.  A parent must win the
within-reporter comparison stably and by a practical absolute-similarity
margin.  The pair must pass an empirical signed-permutation null in at least
four of six refits, unless all six parent comparisons agree and a stricter
stability/uncertainty fallback is met.  The output is within-assay shared-state
support, never a direct edge or cross-assay pair claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructureAnchoredStateReporterConfig:
    """Frozen TRAIN-stage thresholds for the structure/state decomposition."""

    minimum_lower_absolute_loading_cosine: float = 0.65
    minimum_loading_stability: float = 0.80
    minimum_eligible_view_support: float = 4.0 / 6.0
    minimum_parent_win_view_support: float = 4.0 / 6.0
    minimum_parent_similarity_advantage: float = 0.15
    maximum_relative_loading_uncertainty: float = 0.75
    empirical_null_p_threshold: float = 0.125
    minimum_empirical_null_view_support: float = 4.0 / 6.0
    fallback_maximum_relative_loading_uncertainty: float = 0.50
    fallback_minimum_loading_stability: float = 0.90
    fallback_parent_win_view_support: float = 1.00
    minimum_best_coherence_margin: float = 0.05

    def __post_init__(self) -> None:
        bounded = (
            self.minimum_lower_absolute_loading_cosine,
            self.minimum_loading_stability,
            self.minimum_eligible_view_support,
            self.minimum_parent_win_view_support,
            self.maximum_relative_loading_uncertainty,
            self.empirical_null_p_threshold,
            self.minimum_empirical_null_view_support,
            self.fallback_maximum_relative_loading_uncertainty,
            self.fallback_minimum_loading_stability,
            self.fallback_parent_win_view_support,
            self.minimum_best_coherence_margin,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("All probability, cosine, uncertainty, and support thresholds must lie in [0, 1]")
        if (
            not np.isfinite(self.minimum_parent_similarity_advantage)
            or self.minimum_parent_similarity_advantage < 0.0
        ):
            raise ValueError("minimum_parent_similarity_advantage must be non-negative")
        if self.empirical_null_p_threshold not in {0.125, 0.25, 0.50}:
            raise ValueError("empirical_null_p_threshold must be one of 0.125, 0.25, 0.50")
        if (
            self.fallback_maximum_relative_loading_uncertainty
            > self.maximum_relative_loading_uncertainty
        ):
            raise ValueError("The uncertainty fallback must be at least as strict as the global cap")
        if self.fallback_minimum_loading_stability < self.minimum_loading_stability:
            raise ValueError("The stability fallback must be at least as strict as the base gate")
        if self.fallback_parent_win_view_support < self.minimum_parent_win_view_support:
            raise ValueError("The parent-win fallback must be at least as strict as the base gate")


@dataclass
class StructureAnchoredStateReporterResult:
    candidate_evidence: pd.DataFrame
    selected_groups: pd.DataFrame
    assay_decisions: pd.DataFrame
    provenance: dict[str, Any]


_REQUIRED = {
    "pair_key",
    "node1",
    "node2",
    "modality",
    "collider_center",
    "parent_candidate",
    "other_parent_candidate",
    "nonmotif_sibling_candidate",
    "median_coherence_group_score",
    "lower_absolute_loading_cosine",
    "median_minimum_loading_stability",
    "eligible_view_support",
    "parent_win_view_support",
    "median_parent_similarity_advantage",
    "upper_maximum_relative_loading_uncertainty",
    "sibling_formal_state_selected",
    "parent_membership_used_as_gate",
    "center_excluded",
    "parent_parent_pair_forbidden",
    "empirical_null_support_p_le_0_125",
    "empirical_null_support_p_le_0_25",
    "empirical_null_support_p_le_0_50",
}
_FORBIDDEN_TOKENS = (
    "truth",
    "role",
    "scorer",
    "oracle",
    "generative",
    "counterfactual",
)


def _null_support_column(config: StructureAnchoredStateReporterConfig) -> str:
    return {
        0.125: "empirical_null_support_p_le_0_125",
        0.25: "empirical_null_support_p_le_0_25",
        0.50: "empirical_null_support_p_le_0_50",
    }[config.empirical_null_p_threshold]


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in _FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Structural candidates contain privileged columns: {forbidden}")
    missing = sorted(_REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"Structural candidates are missing columns: {missing}")
    output = frame.copy()
    for column in (
        "pair_key",
        "node1",
        "node2",
        "modality",
        "collider_center",
        "parent_candidate",
        "other_parent_candidate",
        "nonmotif_sibling_candidate",
    ):
        output[column] = output[column].astype(str)
    if output["pair_key"].duplicated().any():
        raise ValueError("Structural pair keys must be unique within one world")
    numeric = (
        "median_coherence_group_score",
        "lower_absolute_loading_cosine",
        "median_minimum_loading_stability",
        "eligible_view_support",
        "parent_win_view_support",
        "median_parent_similarity_advantage",
        "upper_maximum_relative_loading_uncertainty",
        "empirical_null_support_p_le_0_125",
        "empirical_null_support_p_le_0_25",
        "empirical_null_support_p_le_0_50",
    )
    for column in numeric:
        output[column] = pd.to_numeric(output[column], errors="raise")
    if not np.isfinite(output[list(numeric)].to_numpy(dtype=float)).all():
        raise ValueError("Structural evidence contains non-finite numeric values")
    for column in (
        "sibling_formal_state_selected",
        "parent_membership_used_as_gate",
        "center_excluded",
        "parent_parent_pair_forbidden",
    ):
        if not output[column].isin([True, False]).all():
            raise ValueError(f"{column} must be boolean")
    if output["parent_membership_used_as_gate"].astype(bool).any():
        raise ValueError("Direct-parent state membership may not be used as a gate")
    if not output["center_excluded"].astype(bool).all():
        raise ValueError("Every structural candidate must exclude the collider center")
    if not output["parent_parent_pair_forbidden"].astype(bool).all():
        raise ValueError("Every structural candidate must forbid the parent--parent pair")
    if (output["parent_candidate"] == output["other_parent_candidate"]).any():
        raise ValueError("A refined motif must have two distinct parent candidates")
    if (
        output["nonmotif_sibling_candidate"].eq(output["collider_center"])
        | output["nonmotif_sibling_candidate"].eq(output["parent_candidate"])
        | output["nonmotif_sibling_candidate"].eq(output["other_parent_candidate"])
    ).any():
        raise ValueError("The nominated reporter must be outside the refined direct motif")
    return output


def estimate_structure_anchored_state_reporters(
    structural_candidates: pd.DataFrame,
    config: StructureAnchoredStateReporterConfig | None = None,
) -> StructureAnchoredStateReporterResult:
    """Select at most one parent--state-reporter pair per chromatin assay."""

    cfg = config or StructureAnchoredStateReporterConfig()
    evidence = _validate_input(structural_candidates)
    null_column = _null_support_column(cfg)
    evidence["passes_sibling_state_membership_gate"] = evidence[
        "sibling_formal_state_selected"
    ].astype(bool)
    evidence["passes_lower_cosine_gate"] = (
        evidence["lower_absolute_loading_cosine"]
        >= cfg.minimum_lower_absolute_loading_cosine
    )
    evidence["passes_loading_stability_gate"] = (
        evidence["median_minimum_loading_stability"]
        >= cfg.minimum_loading_stability
    )
    evidence["passes_eligible_view_support_gate"] = (
        evidence["eligible_view_support"] >= cfg.minimum_eligible_view_support
    )
    evidence["passes_parent_win_support_gate"] = (
        evidence["parent_win_view_support"]
        >= cfg.minimum_parent_win_view_support
    )
    evidence["passes_parent_advantage_gate"] = (
        evidence["median_parent_similarity_advantage"]
        >= cfg.minimum_parent_similarity_advantage
    )
    evidence["passes_global_uncertainty_gate"] = (
        evidence["upper_maximum_relative_loading_uncertainty"]
        <= cfg.maximum_relative_loading_uncertainty
    )
    evidence["passes_empirical_null_gate"] = (
        evidence[null_column] >= cfg.minimum_empirical_null_view_support
    )
    evidence["passes_low_uncertainty_fallback"] = (
        (
            evidence["upper_maximum_relative_loading_uncertainty"]
            <= cfg.fallback_maximum_relative_loading_uncertainty
        )
        & (
            evidence["median_minimum_loading_stability"]
            >= cfg.fallback_minimum_loading_stability
        )
        & (
            evidence["parent_win_view_support"]
            >= cfg.fallback_parent_win_view_support
        )
    )
    base_gates = (
        "passes_sibling_state_membership_gate",
        "passes_lower_cosine_gate",
        "passes_loading_stability_gate",
        "passes_eligible_view_support_gate",
        "passes_parent_win_support_gate",
        "passes_parent_advantage_gate",
        "passes_global_uncertainty_gate",
    )
    evidence["passes_all_pre_margin_gates"] = evidence[list(base_gates)].all(
        axis=1
    ) & (
        evidence["passes_empirical_null_gate"]
        | evidence["passes_low_uncertainty_fallback"]
    )
    evidence["candidate_rank"] = 0
    evidence["relative_best_coherence_margin"] = 0.0
    evidence["selected"] = False

    decisions: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for modality, indices in evidence.groupby("modality", sort=True).groups.items():
        eligible = evidence.loc[
            indices
        ].loc[evidence.loc[indices, "passes_all_pre_margin_gates"]].sort_values(
            ["median_coherence_group_score", "pair_key"],
            ascending=[False, True],
        )
        if eligible.empty:
            failed_counts = {
                gate: int((~evidence.loc[indices, gate].astype(bool)).sum())
                for gate in base_gates
            }
            failed_counts["passes_empirical_null_or_fallback"] = int(
                (~(
                    evidence.loc[indices, "passes_empirical_null_gate"]
                    | evidence.loc[indices, "passes_low_uncertainty_fallback"]
                )).sum()
            )
            decisions.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "abstention_reason": "no_structural_candidate_passed_gates",
                    "n_structural_candidates": len(indices),
                    "n_gate_eligible_candidates": 0,
                    "relative_best_coherence_margin": 0.0,
                    "selected_pair_key": "",
                    **{f"n_failed__{key}": value for key, value in failed_counts.items()},
                }
            )
            continue
        for rank, row_index in enumerate(eligible.index, start=1):
            evidence.loc[row_index, "candidate_rank"] = rank
        best_index = eligible.index[0]
        best_score = float(eligible.iloc[0]["median_coherence_group_score"])
        runner_up = (
            float(eligible.iloc[1]["median_coherence_group_score"])
            if len(eligible) > 1
            else 0.0
        )
        margin = max((best_score - runner_up) / max(best_score, 1e-12), 0.0)
        evidence.loc[best_index, "relative_best_coherence_margin"] = margin
        if margin < cfg.minimum_best_coherence_margin:
            decisions.append(
                {
                    "modality": modality,
                    "selection_status": "unresolved",
                    "abstention_reason": "ambiguous_best_coherence_margin",
                    "n_structural_candidates": len(indices),
                    "n_gate_eligible_candidates": len(eligible),
                    "relative_best_coherence_margin": margin,
                    "selected_pair_key": "",
                }
            )
            continue
        evidence.loc[best_index, "selected"] = True
        best = evidence.loc[best_index]
        decisions.append(
            {
                "modality": modality,
                "selection_status": "selected",
                "abstention_reason": "",
                "n_structural_candidates": len(indices),
                "n_gate_eligible_candidates": len(eligible),
                "relative_best_coherence_margin": margin,
                "selected_pair_key": str(best["pair_key"]),
            }
        )
        members = sorted(
            (str(best["parent_candidate"]), str(best["nonmotif_sibling_candidate"]))
        )
        selected_rows.append(
            {
                "relation_id": f"structure_state_reporter__{modality}",
                "edge_type": "within_assay_shared_state_support",
                "selection_status": "selected",
                "modality": modality,
                "members": ";".join(members),
                "n_members": 2,
                "direct_parent_anchor": str(best["parent_candidate"]),
                "state_nominated_nonmotif_reporter": str(
                    best["nonmotif_sibling_candidate"]
                ),
                "excluded_collider_center": str(best["collider_center"]),
                "other_direct_parent": str(best["other_parent_candidate"]),
                "relative_best_coherence_margin": margin,
                "empirical_null_or_fallback": (
                    "empirical_null"
                    if bool(best["passes_empirical_null_gate"])
                    else "low_uncertainty_stability_fallback"
                ),
                "interpretation": (
                    "assay-local state sharing between one direct parent and a "
                    "formal maxT-nominated nonmotif reporter; not direct causation"
                ),
                "direct_causal_claimed": False,
                "cross_assay_pair_claimed": False,
                "parent_membership_required": False,
                "sibling_membership_required": True,
            }
        )

    decisions_frame = pd.DataFrame(decisions).sort_values("modality").reset_index(
        drop=True
    )
    selected_frame = pd.DataFrame(selected_rows)
    if not selected_frame.empty:
        selected_frame = selected_frame.sort_values("modality").reset_index(drop=True)
    provenance = {
        "schema_version": "structure_anchored_state_reporter_v4.v1",
        "config": asdict(cfg),
        "method": "refined_direct_parents_plus_formal_state_reporter_plus_parent_loading_contrast",
        "empirical_null_support_column": null_column,
        "counts": {
            "n_assays": int(evidence["modality"].nunique()),
            "n_structural_candidates": len(evidence),
            "n_selected_groups": len(selected_frame),
            "n_abstained_assays": int(
                decisions_frame["selection_status"].astype(str).ne("selected").sum()
            ),
        },
        "information_boundary": {
            "refined_direct_structure_used": True,
            "formal_maxT_state_membership_used_for_nonmotif_sibling_only": True,
            "direct_parent_membership_used": False,
            "collider_center_excluded": True,
            "parent_parent_pair_forbidden": True,
            "truth_roles_scorer_or_external_data_used": False,
            "cross_assay_rows_paired": False,
        },
        "semantics": {
            "within_assay_shared_state_support_only": True,
            "direct_causal_claimed": False,
            "cross_assay_pair_claimed": False,
            "cross_assay_direction_claimed": False,
        },
    }
    return StructureAnchoredStateReporterResult(
        candidate_evidence=evidence.sort_values(
            ["modality", "selected", "candidate_rank", "pair_key"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True),
        selected_groups=selected_frame,
        assay_decisions=decisions_frame,
        provenance=provenance,
    )


__all__ = [
    "StructureAnchoredStateReporterConfig",
    "StructureAnchoredStateReporterResult",
    "estimate_structure_anchored_state_reporters",
]

