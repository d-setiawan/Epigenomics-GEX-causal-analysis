#!/usr/bin/env python3
"""Truth-blind reporter-only synchronization evidence for CA-BOSS v4.47.

This phase deliberately removes within-assay direct-parent resolution from
the eligibility condition for cross-assay shared-state synchronization.
Reporter components enter each context independently when their assay-local
bootstrap assignment is supported in that context.  Exact multi-assay
permutation synchronization then supplies edge-specific co-assignment and
exclusion-margin evidence.  No threshold is fitted and no hypothesis is
selected here; a separate cross-block empirical-null calibration stage owns
that decision.
"""

from __future__ import annotations

import argparse
from itertools import combinations, product
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import score_unpaired_mdcr_ks_baseline_v4 as mdcr_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.relaxed_reporter_synchronization_evidence.v1"
ALGORITHM = "CA-BOSS v4.47 reporter-only context-specific synchronization evidence"
CONTEXT_COLUMNS = [
    "context_index",
    "context_label",
    "eligible_modalities",
    "eligible_modality_count",
    "rank",
    "assignments_evaluated",
    "best_total_scaled_ks_ratio",
    "second_best_total_scaled_ks_ratio",
    "global_assignment_relative_margin",
    "best_global_to_local_components_json",
    "reporter_groups_json",
]
PAIR_COLUMNS = [
    "left_modality",
    "right_modality",
    "modalities",
    "candidate_assay_pairs_json",
    "candidate_assay_pair_count",
    "external_state_reporter_members",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "statistical_context_support_pattern",
    "statistical_context_support_count",
    "statistical_context_support_fraction",
    "reporter_bootstrap_context_support_pattern",
    "global_reporter_coassignment_context_pattern",
    "global_reporter_coassignment_context_count",
    "heldout_global_reporter_coassignment_pattern",
    "heldout_global_reporter_coassignment_count",
    "reporter_pair_scaled_ks_ratio_json",
    "minimum_reporter_pair_scaled_ks_ratio",
    "mean_reporter_pair_scaled_ks_ratio",
    "maximum_reporter_pair_scaled_ks_ratio",
    "pair_exclusion_relative_margin_json",
    "minimum_coassigned_pair_exclusion_relative_margin",
    "mean_coassigned_pair_exclusion_relative_margin",
    "minimum_heldout_coassigned_pair_exclusion_relative_margin",
    "mean_heldout_coassigned_pair_exclusion_relative_margin",
    "both_assays_within_assay_pair_resolved_by_component",
    "both_reporters_all_context_bootstrap_supported",
    "v444_assay_pair_already_represented",
    "eligible_for_cross_block_empirical_null_calibration",
    "hypothesis_selected",
    "hypothesis_type",
    "within_assay_direct_structure_claimed",
    "exact_cross_assay_pair_claimed",
    "cross_assay_direction_claimed",
    "candidate_pairs_may_be_expanded_as_edges",
]


def _reporter_contexts(
    row: pd.Series, fold_ids: Sequence[int]
) -> tuple[list[str], list[int | None], list[bool]]:
    """Return reporter features, context components, and bootstrap support."""

    reporters = [
        value
        for value in str(row["external_state_reporter_members"]).split(";")
        if value
    ]
    if not reporters:
        raise ValueError("Assay has no external state reporter")
    components: list[int | None] = []
    full_identified = synchronization._as_bool(
        row["reporter_component_identified"]
    )
    components.append(
        int(row["reporter_local_component"]) if full_identified else None
    )
    supports = [
        bool(
            full_identified
            and synchronization._as_bool(
                row["reporter_bootstrap_component_assignment_supported"]
            )
        )
    ]
    fold_components = json.loads(
        str(row["reporter_feature_fold_components_json"])
    )
    fold_support = json.loads(
        str(row["reporter_feature_fold_bootstrap_supported_json"])
    )
    for fold in fold_ids:
        values: set[int] = set()
        for reporter in reporters:
            try:
                values.add(int(fold_components[reporter][str(fold)]))
            except (KeyError, TypeError, ValueError):
                values.clear()
                break
        components.append(next(iter(values)) if len(values) == 1 else None)
        supports.append(
            bool(
                len(values) == 1
                and synchronization._as_bool(
                    fold_support.get(str(fold), False)
                )
            )
        )
    return reporters, components, supports


def _json_floats(values: Sequence[float]) -> str:
    payload = [None if not np.isfinite(value) else float(value) for value in values]
    return json.dumps(payload, separators=(",", ":"))


def _finite_summary(values: Sequence[float]) -> tuple[float, float, float]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(finite.min()), float(finite.mean()), float(finite.max())


def build_relaxed_synchronization_evidence(
    assay: pd.DataFrame,
    hyperedges: pd.DataFrame,
    pair_decisions: pd.DataFrame,
    context_tables: Sequence[pd.DataFrame],
    context_labels: Sequence[str],
    fold_ids: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct reporter-only global assignments and pair audit evidence."""

    if len(context_tables) != 3 or len(context_labels) != 3:
        raise ValueError("Expected full fit and exactly two feature folds")
    if context_labels[0] != "full_fit":
        raise ValueError("The first synchronization context must be full_fit")
    if assay["modality"].astype(str).duplicated().any():
        raise ValueError("Assay assignment table contains duplicate modalities")
    assay_index = assay.set_index(assay["modality"].astype(str), drop=False)
    reporters: dict[str, list[str]] = {}
    components: dict[str, list[int | None]] = {}
    supports: dict[str, list[bool]] = {}
    for modality, row in assay_index.iterrows():
        reporter_values, component_values, support_values = _reporter_contexts(
            row, fold_ids
        )
        reporters[str(modality)] = reporter_values
        components[str(modality)] = component_values
        supports[str(modality)] = support_values

    solutions: list[
        synchronization.MultiwayPermutationSolution | None
    ] = []
    context_rows: list[dict[str, Any]] = []
    for context_index, (label, table) in enumerate(
        zip(context_labels, context_tables, strict=True)
    ):
        eligible = sorted(
            modality
            for modality in assay_index.index.astype(str)
            if supports[modality][context_index]
            and components[modality][context_index] is not None
        )
        solution = (
            synchronization.solve_multiway_permutation_synchronization(
                table, eligible
            )
            if len(eligible) >= 2
            else None
        )
        solutions.append(solution)
        best_payload: dict[str, list[int]] = {}
        reporter_groups: set[frozenset[str]] = set()
        if solution is not None:
            groups: dict[int, list[str]] = {}
            for modality_index, modality in enumerate(solution.modalities):
                order = [
                    int(value) for value in solution.best_orders[modality_index]
                ]
                best_payload[modality] = order
                reporter_component = components[modality][context_index]
                if reporter_component is None:
                    raise RuntimeError("Eligible reporter lacks a component")
                global_labels = [
                    index
                    for index, local_component in enumerate(order)
                    if local_component == reporter_component
                ]
                if len(global_labels) != 1:
                    raise RuntimeError("Reporter lacks a unique global label")
                groups.setdefault(global_labels[0], []).append(modality)
            reporter_groups = {
                frozenset(values)
                for values in groups.values()
                if len(values) >= 2
            }
        context_rows.append(
            {
                "context_index": context_index,
                "context_label": str(label),
                "eligible_modalities": ";".join(eligible),
                "eligible_modality_count": len(eligible),
                "rank": (
                    int(solution.orders.shape[2]) if solution is not None else 0
                ),
                "assignments_evaluated": (
                    int(len(solution.objectives)) if solution is not None else 0
                ),
                "best_total_scaled_ks_ratio": (
                    solution.best_objective if solution is not None else np.nan
                ),
                "second_best_total_scaled_ks_ratio": (
                    solution.second_best_objective
                    if solution is not None
                    else np.nan
                ),
                "global_assignment_relative_margin": (
                    solution.global_relative_margin
                    if solution is not None
                    else np.nan
                ),
                "best_global_to_local_components_json": json.dumps(
                    best_payload, sort_keys=True, separators=(",", ":")
                ),
                "reporter_groups_json": json.dumps(
                    [
                        sorted(group)
                        for group in sorted(
                            reporter_groups,
                            key=lambda value: tuple(sorted(value)),
                        )
                    ],
                    separators=(",", ":"),
                ),
            }
        )

    represented = synchronization._represented_assay_pairs(hyperedges)
    context_count = len(context_tables)
    strict_majority_count = context_count // 2 + 1
    pair_rows: list[dict[str, Any]] = []
    for decision in pair_decisions.itertuples(index=False):
        left, right = sorted(
            (str(decision.left_modality), str(decision.right_modality))
        )
        pair = frozenset((left, right))
        statistical_pattern = synchronization._context_support_pattern(
            decision.statistical_context_support_pattern, context_count
        )
        bootstrap_pattern = synchronization._context_support_pattern(
            decision.reporter_bootstrap_context_support_pattern, context_count
        )
        coassignments: list[bool] = []
        exclusion_margins: list[float] = []
        reporter_ratios: list[float] = []
        for context_index, (table, solution) in enumerate(
            zip(context_tables, solutions, strict=True)
        ):
            left_component = components[left][context_index]
            right_component = components[right][context_index]
            if left_component is None or right_component is None:
                reporter_ratios.append(float("nan"))
            else:
                matrix, critical = synchronization.pair_cost_matrix(
                    table, left, right
                )
                reporter_ratios.append(
                    float(matrix[left_component, right_component] / critical)
                )
            if (
                solution is not None
                and left in solution.modalities
                and right in solution.modalities
                and left_component is not None
                and right_component is not None
            ):
                diagnostics = synchronization.reporter_pair_diagnostics(
                    solution,
                    table,
                    left,
                    right,
                    left_component,
                    right_component,
                )
                coassignments.append(
                    bool(diagnostics["coassigned_in_best_solution"])
                )
                exclusion_margins.append(
                    float(diagnostics["pair_exclusion_relative_margin"])
                )
            else:
                coassignments.append(False)
                exclusion_margins.append(float("nan"))
        coassignment_pattern = "".join(
            "1" if value else "0" for value in coassignments
        )
        heldout_pattern = coassignment_pattern[1:]
        coassigned_margins = [
            margin
            for margin, coassigned in zip(
                exclusion_margins, coassignments, strict=True
            )
            if coassigned and np.isfinite(margin)
        ]
        heldout_coassigned_margins = [
            margin
            for margin, coassigned in zip(
                exclusion_margins[1:], coassignments[1:], strict=True
            )
            if coassigned and np.isfinite(margin)
        ]
        ratio_min, ratio_mean, ratio_max = _finite_summary(reporter_ratios)
        coassigned_min, coassigned_mean, _ = _finite_summary(
            coassigned_margins
        )
        heldout_min, heldout_mean, _ = _finite_summary(
            heldout_coassigned_margins
        )
        already_represented = pair in represented
        statistical_count = statistical_pattern.count("1")
        eligible_for_calibration = bool(
            not already_represented
            and statistical_count >= strict_majority_count
            and heldout_pattern.count("1") >= 1
            and np.isfinite(heldout_min)
        )
        left_reporters = reporters[left]
        right_reporters = reporters[right]
        feature_pairs = sorted(
            {
                tuple(sorted((left_feature, right_feature)))
                for left_feature, right_feature in product(
                    left_reporters, right_reporters
                )
            }
        )
        pair_rows.append(
            {
                "left_modality": left,
                "right_modality": right,
                "modalities": f"{left};{right}",
                "candidate_assay_pairs_json": json.dumps(
                    [[left, right]], separators=(",", ":")
                ),
                "candidate_assay_pair_count": 1,
                "external_state_reporter_members": ";".join(
                    sorted(set(left_reporters + right_reporters))
                ),
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    feature_pairs, separators=(",", ":")
                ),
                "candidate_cross_assay_feature_pair_count": len(feature_pairs),
                "statistical_context_support_pattern": statistical_pattern,
                "statistical_context_support_count": statistical_count,
                "statistical_context_support_fraction": float(
                    statistical_count / context_count
                ),
                "reporter_bootstrap_context_support_pattern": bootstrap_pattern,
                "global_reporter_coassignment_context_pattern": (
                    coassignment_pattern
                ),
                "global_reporter_coassignment_context_count": (
                    coassignment_pattern.count("1")
                ),
                "heldout_global_reporter_coassignment_pattern": heldout_pattern,
                "heldout_global_reporter_coassignment_count": (
                    heldout_pattern.count("1")
                ),
                "reporter_pair_scaled_ks_ratio_json": _json_floats(
                    reporter_ratios
                ),
                "minimum_reporter_pair_scaled_ks_ratio": ratio_min,
                "mean_reporter_pair_scaled_ks_ratio": ratio_mean,
                "maximum_reporter_pair_scaled_ks_ratio": ratio_max,
                "pair_exclusion_relative_margin_json": _json_floats(
                    exclusion_margins
                ),
                "minimum_coassigned_pair_exclusion_relative_margin": (
                    coassigned_min
                ),
                "mean_coassigned_pair_exclusion_relative_margin": (
                    coassigned_mean
                ),
                "minimum_heldout_coassigned_pair_exclusion_relative_margin": (
                    heldout_min
                ),
                "mean_heldout_coassigned_pair_exclusion_relative_margin": (
                    heldout_mean
                ),
                "both_assays_within_assay_pair_resolved_by_component": all(
                    synchronization._as_bool(value)
                    for value in (
                        assay_index.loc[left][
                            "within_assay_pair_resolved_by_component"
                        ],
                        assay_index.loc[right][
                            "within_assay_pair_resolved_by_component"
                        ],
                    )
                ),
                "both_reporters_all_context_bootstrap_supported": all(
                    supports[modality][index]
                    for modality in (left, right)
                    for index in range(context_count)
                ),
                "v444_assay_pair_already_represented": already_represented,
                "eligible_for_cross_block_empirical_null_calibration": (
                    eligible_for_calibration
                ),
                "hypothesis_selected": False,
                "hypothesis_type": "cross_assay_shared_state_equivalence",
                "within_assay_direct_structure_claimed": False,
                "exact_cross_assay_pair_claimed": False,
                "cross_assay_direction_claimed": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
            }
        )
    return (
        pd.DataFrame(pair_rows, columns=PAIR_COLUMNS),
        pd.DataFrame(context_rows, columns=CONTEXT_COLUMNS),
    )


def freeze_evidence(
    *, bridge_dir: Path, evidence_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Freeze one world's reporter-only synchronization evidence."""

    bridge, bridge_provenance = synchronization.verify_bridge(bridge_dir)
    evidence = synchronization._forbid_truth_path(evidence_dir)
    evidence_manifest = mdcr_scorer.verify_frozen_evidence(evidence)
    if str(bridge_provenance.get("mdcr_manifest_payload_sha256")) != str(
        evidence_manifest.get("manifest_payload_sha256")
    ):
        raise ValueError("Bridge and KS evidence manifests disagree")
    assay = pd.read_csv(
        bridge / "assay_axis_assignments.tsv", sep="\t", keep_default_na=False
    )
    hyperedges = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    pair_decisions = pd.read_csv(
        bridge / "context_majority_pair_decisions.tsv",
        sep="\t",
        keep_default_na=False,
    )
    full_pairwise = pd.read_csv(
        evidence / "pairwise_component_matches.tsv", sep="\t"
    )
    fold_pairwise = pd.read_csv(
        evidence / "fold_pairwise_component_matches.tsv", sep="\t"
    )
    fold_pairwise["feature_fold"] = pd.to_numeric(
        fold_pairwise["feature_fold"], errors="raise"
    ).astype(int)
    fold_ids = sorted(fold_pairwise["feature_fold"].unique())
    if len(fold_ids) != 2:
        raise ValueError("Exactly two held-out background-feature folds are required")
    context_tables = [full_pairwise] + [
        frame.reset_index(drop=True)
        for _, frame in fold_pairwise.groupby("feature_fold", sort=True)
    ]
    context_labels = ["full_fit", *[f"feature_fold_{fold}" for fold in fold_ids]]
    pairs, contexts = build_relaxed_synchronization_evidence(
        assay,
        hyperedges,
        pair_decisions,
        context_tables,
        context_labels,
        fold_ids,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "relaxed_reporter_synchronization_pairs.tsv": pairs,
        "relaxed_reporter_synchronization_contexts.tsv": contexts,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "phase": "pretruth_evidence_only",
        "block": bridge_provenance["block"],
        "seed": int(bridge_provenance["seed"]),
        "information_boundary": {
            "reporter_only_context_specific_eligibility": True,
            "within_assay_direct_structure_required": False,
            "frozen_pairwise_ks_evidence_used": True,
            "full_fit_plus_two_disjoint_background_feature_folds_used": True,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "scorer_truth_used": False,
            "external_features_or_priors_used": False,
        },
        "formal_pretruth_barrier_satisfied": bool(
            bridge_provenance["formal_pretruth_barrier_satisfied"]
        ),
        "new_worlds_or_datasets_created": False,
        "hypothesis_selected": False,
        "calibration_eligible_pair_count": int(
            pairs[
                "eligible_for_cross_block_empirical_null_calibration"
            ].astype(bool).sum()
        ),
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": synchronization._sha256(
            bridge / "provenance.json"
        ),
        "mdcr_manifest_payload_sha256": evidence_manifest[
            "manifest_payload_sha256"
        ],
        "source_sha256": {
            Path(__file__).name: synchronization._sha256(
                Path(__file__).resolve()
            )
        },
        "input_sha256": {
            "assay_axis_assignments.tsv": synchronization._sha256(
                bridge / "assay_axis_assignments.tsv"
            ),
            "stable_cross_assay_hyperedges.tsv": synchronization._sha256(
                bridge / "stable_cross_assay_hyperedges.tsv"
            ),
            "context_majority_pair_decisions.tsv": synchronization._sha256(
                bridge / "context_majority_pair_decisions.tsv"
            ),
            "pairwise_component_matches.tsv": synchronization._sha256(
                evidence / "pairwise_component_matches.tsv"
            ),
            "fold_pairwise_component_matches.tsv": synchronization._sha256(
                evidence / "fold_pairwise_component_matches.tsv"
            ),
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "permutation_synchronization": (
                "Pachauri, Kondor, and Singh, NeurIPS 2013"
            ),
            "multiway_matching": "Tang and Jebara, AISTATS 2017",
            "unpaired_multi_domain_causal_representation": (
                "Sturma, Squires, Drton, and Uhler, NeurIPS 2023"
            ),
        },
    }
    result["manifest_payload_sha256"] = synchronization._canonical_sha256(
        result
    )
    (out_dir / "evidence_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = freeze_evidence(
        bridge_dir=args.bridge_dir,
        evidence_dir=args.evidence_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
