#!/usr/bin/env python3
"""Cycle-consistent multi-assay component synchronization for CA-BOSS v4.

This pre-truth diagnostic reuses a frozen component-conditioned bridge and its
frozen pairwise KS evidence.  In each statistical context it assigns every
eligible assay's local components to one common permutation frame by minimizing
the total pairwise KS cost.  Reporter groups must repeat exactly across the
full fit and both held-out background-feature folds.  The script has no scorer
or truth input and never expands a synchronized group into ordinary edges.

The formulation adapts permutation synchronization / multi-way matching to
the rank-three unpaired-domain component-registration problem.  Exact
enumeration is feasible here because each assay has only 3! permutations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_unpaired_mdcr_ks_baseline_v4 as mdcr_scorer  # noqa: E402


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SCHEMA = "ca_boss_v4.multi_assay_permutation_synchronization.v1"
EXPECTED_BRIDGE_SCHEMA = (
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.v15"
)
ALGORITHM = "CA-BOSS v4.45 exact multi-assay KS permutation synchronization"
FIXED_SELECTION_RULE = (
    "synchronize_all_component_resolved_all_context_reporter_bootstrap_"
    "supported_assays_by_the_exact_minimum_total_pairwise_scaled_ks_over_"
    "rank_three_permutations_in_the_full_fit_and_two_disjoint_background_"
    "feature_folds_then_retain_an_exactly_repeated_reporter_group_only_when_"
    "every_within_group_pair_has_positive_global_pair_exclusion_margin_"
    "reporter_pair_ks_fit_in_every_context_at_least_one_ordinary_statistical_"
    "supporting_context_and_no_pair_is_already_represented_by_the_v444_"
    "core_or_ambiguity_output"
)
MAXIMUM_ASSIGNMENTS = 100_000

CANDIDATE_COLUMNS = [
    "synchronized_hyperedge_id",
    "modalities",
    "n_modalities",
    "candidate_assay_pairs_json",
    "candidate_assay_pair_count",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "context_count",
    "reporter_group_repeated_exactly_in_all_contexts",
    "all_reporters_bootstrap_supported_in_all_contexts",
    "minimum_pair_statistical_context_support_count",
    "minimum_pair_statistical_context_support_fraction",
    "minimum_context_pair_exclusion_relative_margin",
    "maximum_context_reporter_pair_scaled_ks_ratio",
    "minimum_context_global_assignment_relative_margin",
    "maximum_context_global_assignment_objective",
    "any_candidate_pair_already_represented_by_v444",
    "hypothesis_selected",
    "selection_status",
    "hypothesis_type",
    "exact_assay_participation_claimed",
    "cross_assay_direction_claimed",
    "candidate_pairs_may_be_expanded_as_edges",
    "fixed_selection_rule",
]

CONTEXT_COLUMNS = [
    "context_index",
    "context_label",
    "modalities",
    "n_modalities",
    "rank",
    "assignments_evaluated",
    "best_total_scaled_ks_ratio",
    "second_best_total_scaled_ks_ratio",
    "global_assignment_relative_margin",
    "best_global_to_local_components_json",
    "reporter_groups_json",
]


@dataclass(frozen=True)
class MultiwayPermutationSolution:
    """Exact rank-small multi-way assignment and its complete audit surface."""

    modalities: tuple[str, ...]
    orders: IntArray
    objectives: FloatArray
    best_index: int
    best_objective: float
    second_best_objective: float
    global_relative_margin: float

    @property
    def best_orders(self) -> IntArray:
        return self.orders[self.best_index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _forbid_truth_path(path: Path) -> Path:
    resolved = path.expanduser().absolute().resolve(strict=True)
    if any(
        part.casefold() in {"scorer", "truth", "ground_truth", "oracle"}
        for part in resolved.parts
    ):
        raise ValueError(f"Pre-truth synchronization refuses path: {path}")
    return resolved


def verify_bridge(bridge_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Verify a frozen v4.44 bridge without opening any truth resource."""

    directory = _forbid_truth_path(bridge_dir)
    provenance_path = directory / "provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != EXPECTED_BRIDGE_SCHEMA:
        raise ValueError("Unexpected component-conditioned bridge schema")
    observed = str(provenance.get("provenance_payload_sha256", ""))
    unsigned = dict(provenance)
    unsigned.pop("provenance_payload_sha256", None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError("Bridge provenance payload hash mismatch")
    boundary = provenance.get("information_boundary", {})
    if boundary.get("scorer_truth_used") is not False:
        raise ValueError("Bridge did not certify the pre-truth boundary")
    for name, expected in provenance.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"Bridge output hash mismatch: {name}")
    return directory, provenance


def _oriented_pair_table(
    table: pd.DataFrame, left: str, right: str
) -> pd.DataFrame:
    forward = table.loc[
        table["left_modality"].astype(str).eq(left)
        & table["right_modality"].astype(str).eq(right)
    ].copy()
    reverse = table.loc[
        table["left_modality"].astype(str).eq(right)
        & table["right_modality"].astype(str).eq(left)
    ].copy()
    if not forward.empty and not reverse.empty:
        raise ValueError("Pairwise evidence contains both assay orientations")
    if forward.empty and reverse.empty:
        raise ValueError(f"Pairwise evidence lacks assay pair {left!r}, {right!r}")
    if forward.empty:
        oriented = reverse.copy()
        left_values = oriented["right_component"].copy()
        right_values = oriented["left_component"].copy()
        oriented["left_component"] = left_values
        oriented["right_component"] = right_values
        return oriented
    return forward


def pair_cost_matrix(
    table: pd.DataFrame, left: str, right: str
) -> tuple[FloatArray, float]:
    """Return a complete orientation-normalized KS cost matrix and bound."""

    pair = _oriented_pair_table(table, str(left), str(right))
    required = {
        "left_component",
        "right_component",
        "scaled_ks_statistic",
        "asymptotic_critical_value",
    }
    if missing := sorted(required.difference(pair.columns)):
        raise ValueError(f"Pairwise evidence is missing columns: {missing}")
    left_components = sorted(
        pd.to_numeric(pair["left_component"], errors="raise")
        .astype(int)
        .unique()
    )
    right_components = sorted(
        pd.to_numeric(pair["right_component"], errors="raise")
        .astype(int)
        .unique()
    )
    if (
        left_components != list(range(len(left_components)))
        or right_components != left_components
        or len(left_components) < 2
    ):
        raise ValueError("Component labels must be a common consecutive rank")
    rank = len(left_components)
    matrix = np.full((rank, rank), np.nan, dtype=np.float64)
    for row in pair.itertuples(index=False):
        left_component = int(row.left_component)
        right_component = int(row.right_component)
        if np.isfinite(matrix[left_component, right_component]):
            raise ValueError("Pairwise evidence contains a duplicate component pair")
        matrix[left_component, right_component] = float(
            row.scaled_ks_statistic
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Pairwise evidence is not a finite complete cost matrix")
    critical_values = pd.to_numeric(
        pair["asymptotic_critical_value"], errors="raise"
    ).astype(float)
    if not np.isfinite(critical_values).all() or (critical_values <= 0.0).any():
        raise ValueError("Pairwise KS critical values must be finite and positive")
    return matrix, float(critical_values.min())


def solve_multiway_permutation_synchronization(
    table: pd.DataFrame, modalities: Sequence[str]
) -> MultiwayPermutationSolution:
    """Exactly minimize all pairwise critical-value-scaled KS assignment cost.

    ``orders[a, m, g]`` is the local component in modality ``m`` assigned to
    global component ``g`` for candidate assignment ``a``.  The first sorted
    modality is fixed to the identity to remove the unidentifiable common
    label permutation.
    """

    ordered_modalities = tuple(sorted(str(value) for value in modalities))
    if len(ordered_modalities) < 2 or len(set(ordered_modalities)) != len(
        ordered_modalities
    ):
        raise ValueError("Synchronization requires at least two unique assays")
    matrices: dict[tuple[str, str], FloatArray] = {}
    rank: int | None = None
    for left, right in combinations(ordered_modalities, 2):
        matrix, critical = pair_cost_matrix(table, left, right)
        if rank is None:
            rank = int(matrix.shape[0])
        if matrix.shape != (rank, rank):
            raise ValueError("All assay pairs must have the same component rank")
        matrices[(left, right)] = matrix / critical
    if rank is None:
        raise RuntimeError("No assay-pair costs were constructed")
    permutation_values = np.asarray(
        list(permutations(range(rank))), dtype=np.int64
    )
    assignment_count = len(permutation_values) ** (
        len(ordered_modalities) - 1
    )
    if assignment_count > MAXIMUM_ASSIGNMENTS:
        raise ValueError(
            f"Exact synchronization would evaluate {assignment_count} assignments"
        )
    choice_rows = np.asarray(
        list(
            product(
                range(len(permutation_values)),
                repeat=len(ordered_modalities) - 1,
            )
        ),
        dtype=np.int64,
    )
    orders = np.empty(
        (assignment_count, len(ordered_modalities), rank), dtype=np.int64
    )
    orders[:, 0, :] = np.arange(rank, dtype=np.int64)
    for modality_index in range(1, len(ordered_modalities)):
        orders[:, modality_index, :] = permutation_values[
            choice_rows[:, modality_index - 1]
        ]
    objectives = np.zeros(assignment_count, dtype=np.float64)
    modality_index = {
        modality: index for index, modality in enumerate(ordered_modalities)
    }
    for (left, right), matrix in matrices.items():
        left_orders = orders[:, modality_index[left], :]
        right_orders = orders[:, modality_index[right], :]
        objectives += matrix[left_orders, right_orders].sum(axis=1)
    ranking = np.argsort(objectives, kind="stable")
    best_index = int(ranking[0])
    best = float(objectives[best_index])
    second = float(objectives[int(ranking[1])])
    relative_margin = float((second - best) / max(abs(best), 1e-12))
    return MultiwayPermutationSolution(
        modalities=ordered_modalities,
        orders=orders,
        objectives=objectives,
        best_index=best_index,
        best_objective=best,
        second_best_objective=second,
        global_relative_margin=relative_margin,
    )


def reporter_pair_diagnostics(
    solution: MultiwayPermutationSolution,
    table: pd.DataFrame,
    left: str,
    right: str,
    left_component: int,
    right_component: int,
) -> dict[str, Any]:
    """Measure support and cost of one reporter pair in a global solution."""

    try:
        left_index = solution.modalities.index(str(left))
        right_index = solution.modalities.index(str(right))
    except ValueError as error:
        raise ValueError("Reporter assay is absent from synchronization") from error
    rank = solution.orders.shape[2]
    if not 0 <= int(left_component) < rank or not 0 <= int(right_component) < rank:
        raise ValueError("Reporter component lies outside the synchronized rank")
    left_labels = np.argmax(
        solution.orders[:, left_index, :] == int(left_component), axis=1
    )
    right_labels = np.argmax(
        solution.orders[:, right_index, :] == int(right_component), axis=1
    )
    coassigned = left_labels == right_labels
    best_coassigned = bool(coassigned[solution.best_index])
    breaking = solution.objectives[~coassigned]
    if breaking.size == 0:
        raise RuntimeError("Reporter co-assignment has no competing solution")
    best_breaking = float(breaking.min())
    exclusion_margin = float(
        (best_breaking - solution.best_objective)
        / max(abs(solution.best_objective), 1e-12)
    )
    matrix, critical = pair_cost_matrix(table, str(left), str(right))
    reporter_ratio = float(
        matrix[int(left_component), int(right_component)] / critical
    )
    return {
        "coassigned_in_best_solution": best_coassigned,
        "pair_exclusion_relative_margin": exclusion_margin,
        "reporter_pair_scaled_ks_ratio": reporter_ratio,
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}")


def _context_support_pattern(value: Any, context_count: int) -> str:
    """Normalize TSV-inferred binary patterns without losing leading zeroes."""

    if pd.isna(value):
        raise ValueError("Pair statistical support pattern is missing")
    if isinstance(value, (int, np.integer)):
        pattern = str(int(value))
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        pattern = str(int(value))
    else:
        pattern = str(value).strip()
    if pattern.isdigit() and len(pattern) <= context_count:
        pattern = pattern.zfill(context_count)
    if len(pattern) != context_count or set(pattern) - {"0", "1"}:
        raise ValueError("Pair statistical support pattern is malformed")
    return pattern


def _reporter_components(
    row: Any, fold_ids: Sequence[int]
) -> tuple[list[str], list[int]]:
    reporters = [
        value
        for value in str(row.external_state_reporter_members).split(";")
        if value
    ]
    if not reporters:
        raise ValueError("A synchronized assay has no reporter feature")
    fold_payload = json.loads(str(row.reporter_feature_fold_components_json))
    components = [int(row.reporter_local_component)]
    for fold in fold_ids:
        fold_components = {
            int(fold_payload[reporter][str(fold)]) for reporter in reporters
        }
        if len(fold_components) != 1:
            raise ValueError("An assay's reporters disagree within a fold")
        components.append(next(iter(fold_components)))
    return reporters, components


def _represented_assay_pairs(hyperedges: pd.DataFrame) -> set[frozenset[str]]:
    represented: set[frozenset[str]] = set()
    for row in hyperedges.itertuples(index=False):
        for pair in json.loads(str(row.candidate_assay_pairs_json)):
            if len(pair) != 2:
                raise ValueError("A v4.44 candidate assay pair is malformed")
            represented.add(frozenset((str(pair[0]), str(pair[1]))))
    return represented


def build_synchronized_candidates(
    assay: pd.DataFrame,
    hyperedges: pd.DataFrame,
    pair_decisions: pd.DataFrame,
    context_tables: Sequence[pd.DataFrame],
    context_labels: Sequence[str],
    fold_ids: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fixed-rule multi-assay reporter groups from frozen evidence."""

    if len(context_tables) != len(context_labels) or len(context_tables) < 3:
        raise ValueError("Synchronization requires full plus two fold contexts")
    eligible_mask = pd.Series(True, index=assay.index, dtype=bool)
    for column in (
        "within_assay_pair_resolved_by_component",
        "reporter_component_identified",
        "reporter_context_components_identified",
        "reporter_all_context_bootstrap_component_assignments_supported",
    ):
        if column not in assay.columns:
            raise ValueError(f"Assay assignment table is missing {column!r}")
        eligible_mask &= assay[column].map(_as_bool)
    eligible = assay.loc[eligible_mask].copy()
    if eligible["modality"].astype(str).duplicated().any():
        raise ValueError("Assay assignment table contains duplicate modalities")
    modalities = sorted(eligible["modality"].astype(str))
    if len(modalities) < 2:
        return (
            pd.DataFrame(columns=CANDIDATE_COLUMNS),
            pd.DataFrame(columns=CONTEXT_COLUMNS),
        )
    reporters_by_modality: dict[str, list[str]] = {}
    components_by_modality: dict[str, list[int]] = {}
    for row in eligible.itertuples(index=False):
        reporters, components = _reporter_components(row, fold_ids)
        reporters_by_modality[str(row.modality)] = reporters
        components_by_modality[str(row.modality)] = components
    solutions: list[MultiwayPermutationSolution] = []
    groups_by_context: list[set[frozenset[str]]] = []
    context_rows: list[dict[str, Any]] = []
    for context_index, (label, table) in enumerate(
        zip(context_labels, context_tables, strict=True)
    ):
        solution = solve_multiway_permutation_synchronization(table, modalities)
        solutions.append(solution)
        best_orders = solution.best_orders
        groups: dict[int, list[str]] = {}
        best_payload: dict[str, list[int]] = {}
        for modality_index, modality in enumerate(solution.modalities):
            order = [int(value) for value in best_orders[modality_index]]
            best_payload[modality] = order
            reporter_component = components_by_modality[modality][context_index]
            labels = [
                global_component
                for global_component, local_component in enumerate(order)
                if local_component == reporter_component
            ]
            if len(labels) != 1:
                raise RuntimeError("A reporter component lacks a global label")
            groups.setdefault(labels[0], []).append(modality)
        reporter_groups = {
            frozenset(values) for values in groups.values() if len(values) >= 2
        }
        groups_by_context.append(reporter_groups)
        context_rows.append(
            {
                "context_index": context_index,
                "context_label": str(label),
                "modalities": ";".join(solution.modalities),
                "n_modalities": len(solution.modalities),
                "rank": int(solution.orders.shape[2]),
                "assignments_evaluated": int(len(solution.objectives)),
                "best_total_scaled_ks_ratio": solution.best_objective,
                "second_best_total_scaled_ks_ratio": (
                    solution.second_best_objective
                ),
                "global_assignment_relative_margin": (
                    solution.global_relative_margin
                ),
                "best_global_to_local_components_json": json.dumps(
                    best_payload, sort_keys=True, separators=(",", ":")
                ),
                "reporter_groups_json": json.dumps(
                    [
                        sorted(group)
                        for group in sorted(
                            reporter_groups, key=lambda value: tuple(sorted(value))
                        )
                    ],
                    separators=(",", ":"),
                ),
            }
        )
    repeated_groups = set.intersection(*groups_by_context)
    represented_pairs = _represented_assay_pairs(hyperedges)
    decision_lookup = {
        frozenset((str(row.left_modality), str(row.right_modality))): row
        for row in pair_decisions.itertuples(index=False)
    }
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, group in enumerate(
        sorted(repeated_groups, key=lambda value: tuple(sorted(value))), start=1
    ):
        group_modalities = sorted(group)
        assay_pairs = [
            tuple(pair) for pair in combinations(group_modalities, 2)
        ]
        pair_diagnostics: list[dict[str, Any]] = []
        statistical_counts: list[int] = []
        for left, right in assay_pairs:
            pair = frozenset((left, right))
            decision = decision_lookup.get(pair)
            pattern = (
                _context_support_pattern(
                    decision.statistical_context_support_pattern,
                    len(context_tables),
                )
                if decision is not None
                else "0" * len(context_tables)
            )
            statistical_counts.append(pattern.count("1"))
            for context_index, (solution, table) in enumerate(
                zip(solutions, context_tables, strict=True)
            ):
                diagnostics = reporter_pair_diagnostics(
                    solution,
                    table,
                    left,
                    right,
                    components_by_modality[left][context_index],
                    components_by_modality[right][context_index],
                )
                pair_diagnostics.append(diagnostics)
        minimum_statistical_count = min(statistical_counts)
        minimum_exclusion_margin = min(
            float(value["pair_exclusion_relative_margin"])
            for value in pair_diagnostics
        )
        maximum_reporter_ratio = max(
            float(value["reporter_pair_scaled_ks_ratio"])
            for value in pair_diagnostics
        )
        all_coassigned = all(
            bool(value["coassigned_in_best_solution"])
            for value in pair_diagnostics
        )
        already_represented = any(
            frozenset(pair) in represented_pairs for pair in assay_pairs
        )
        selected = bool(
            all_coassigned
            and minimum_statistical_count >= 1
            and minimum_exclusion_margin > 0.0
            and maximum_reporter_ratio <= 1.0
            and not already_represented
        )
        reporters = {
            modality: reporters_by_modality[modality]
            for modality in group_modalities
        }
        feature_pairs = sorted(
            {
                tuple(sorted((left_feature, right_feature)))
                for left, right in assay_pairs
                for left_feature, right_feature in product(
                    reporters[left], reporters[right]
                )
            }
        )
        if selected:
            status = "selected_as_multi_assay_synchronized_shared_state_group"
        elif already_represented:
            status = "diagnostic_duplicate_of_v444_output"
        elif minimum_statistical_count < 1:
            status = "unresolved_without_ordinary_statistical_context_support"
        elif minimum_exclusion_margin <= 0.0:
            status = "unresolved_nonunique_pair_under_global_assignment"
        elif maximum_reporter_ratio > 1.0:
            status = "unresolved_reporter_pair_outside_ks_bound"
        else:
            status = "unresolved_multi_assay_synchronization"
        candidate_rows.append(
            {
                "synchronized_hyperedge_id": (
                    f"multi_assay_synchronized_shared_state_{candidate_index:03d}"
                ),
                "modalities": ";".join(group_modalities),
                "n_modalities": len(group_modalities),
                "candidate_assay_pairs_json": json.dumps(
                    assay_pairs, separators=(",", ":")
                ),
                "candidate_assay_pair_count": len(assay_pairs),
                "candidate_cross_assay_feature_pairs_json": json.dumps(
                    feature_pairs, separators=(",", ":")
                ),
                "candidate_cross_assay_feature_pair_count": len(feature_pairs),
                "context_count": len(context_tables),
                "reporter_group_repeated_exactly_in_all_contexts": True,
                "all_reporters_bootstrap_supported_in_all_contexts": True,
                "minimum_pair_statistical_context_support_count": (
                    minimum_statistical_count
                ),
                "minimum_pair_statistical_context_support_fraction": float(
                    minimum_statistical_count / len(context_tables)
                ),
                "minimum_context_pair_exclusion_relative_margin": float(
                    minimum_exclusion_margin
                ),
                "maximum_context_reporter_pair_scaled_ks_ratio": float(
                    maximum_reporter_ratio
                ),
                "minimum_context_global_assignment_relative_margin": float(
                    min(value.global_relative_margin for value in solutions)
                ),
                "maximum_context_global_assignment_objective": float(
                    max(value.best_objective for value in solutions)
                ),
                "any_candidate_pair_already_represented_by_v444": (
                    already_represented
                ),
                "hypothesis_selected": selected,
                "selection_status": status,
                "hypothesis_type": "cross_assay_shared_state_equivalence",
                "exact_assay_participation_claimed": len(group_modalities) == 2,
                "cross_assay_direction_claimed": False,
                "candidate_pairs_may_be_expanded_as_edges": False,
                "fixed_selection_rule": FIXED_SELECTION_RULE,
            }
        )
    return (
        pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS),
        pd.DataFrame(context_rows, columns=CONTEXT_COLUMNS),
    )


def freeze_synchronization(
    *, bridge_dir: Path, evidence_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Freeze one pre-truth synchronization diagnostic and its provenance."""

    bridge, bridge_provenance = verify_bridge(bridge_dir)
    evidence = _forbid_truth_path(evidence_dir)
    evidence_manifest = mdcr_scorer.verify_frozen_evidence(evidence)
    if (
        str(bridge_provenance.get("mdcr_manifest_payload_sha256"))
        != str(evidence_manifest.get("manifest_payload_sha256"))
    ):
        raise ValueError("Bridge and KS evidence manifests disagree")
    assay = pd.read_csv(bridge / "assay_axis_assignments.tsv", sep="\t")
    hyperedges = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv", sep="\t"
    )
    pair_decisions = pd.read_csv(
        bridge / "context_majority_pair_decisions.tsv", sep="\t"
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
    candidates, contexts = build_synchronized_candidates(
        assay,
        hyperedges,
        pair_decisions,
        context_tables,
        context_labels,
        fold_ids,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "multi_assay_synchronization_candidates.tsv": candidates,
        "multi_assay_synchronization_contexts.tsv": contexts,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "block": str(bridge_provenance["block"]),
        "seed": int(bridge_provenance["seed"]),
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "information_boundary": {
            "frozen_v444_bridge_used": True,
            "frozen_pairwise_ks_evidence_used": True,
            "full_fit_plus_two_disjoint_background_feature_folds_used": True,
            "within_assay_reporter_bootstrap_used": True,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "scorer_truth_used": False,
            "external_features_or_priors_used": False,
        },
        "new_worlds_or_datasets_created": False,
        "formal_pretruth_barrier_satisfied": bool(
            bridge_provenance["formal_pretruth_barrier_satisfied"]
        ),
        "rank_three_exact_assignment": True,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "cross_assay_direction_claimed": False,
        "selected_hypothesis_count": int(
            candidates["hypothesis_selected"].astype(bool).sum()
        ),
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": _sha256(
            bridge / "provenance.json"
        ),
        "mdcr_manifest_payload_sha256": evidence_manifest[
            "manifest_payload_sha256"
        ],
        "source_sha256": {Path(__file__).name: _sha256(Path(__file__).resolve())},
        "input_sha256": {
            "assay_axis_assignments.tsv": _sha256(
                bridge / "assay_axis_assignments.tsv"
            ),
            "stable_cross_assay_hyperedges.tsv": _sha256(
                bridge / "stable_cross_assay_hyperedges.tsv"
            ),
            "context_majority_pair_decisions.tsv": _sha256(
                bridge / "context_majority_pair_decisions.tsv"
            ),
            "pairwise_component_matches.tsv": _sha256(
                evidence / "pairwise_component_matches.tsv"
            ),
            "fold_pairwise_component_matches.tsv": _sha256(
                evidence / "fold_pairwise_component_matches.tsv"
            ),
        },
        "output_sha256": {
            name: _sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "permutation_synchronization": (
                "Pachauri, Kondor, and Singh, NeurIPS 2013"
            ),
            "multiway_coordinate_optimization": (
                "Tang and Jebara, AISTATS 2017"
            ),
            "unpaired_multi_domain_causal_representation": (
                "Sturma, Squires, Drton, and Uhler, NeurIPS 2023"
            ),
        },
    }
    result["manifest_payload_sha256"] = _canonical_sha256(result)
    (out_dir / "synchronization_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    result = freeze_synchronization(
        bridge_dir=args.bridge_dir,
        evidence_dir=args.evidence_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
