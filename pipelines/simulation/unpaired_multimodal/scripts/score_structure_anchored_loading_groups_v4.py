#!/usr/bin/env python3
"""Score frozen structure-anchored assay-local loading pairs.

This scorer is intentionally separate from candidate generation.  It reads
truth only after the observable candidate files have been frozen, and scores
the emitted pairs solely against ``within_assay_shared_state`` truth edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


_GROUP_COLUMNS = {
    "relation_id",
    "edge_type",
    "selection_status",
    "modality",
    "members",
    "n_members",
    "direct_causal_claimed",
    "cross_assay_membership_claimed",
}
_TRUTH_COLUMNS = {"node1", "node2", "truth_type", "mark"}


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def _pair_key(nodes: frozenset[str]) -> str:
    return ";".join(sorted(nodes))


def score_structure_anchored_groups(
    typed_groups: pd.DataFrame,
    truth_edges: pd.DataFrame,
    world_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Score one frozen world and return metrics, pairs, and assay metrics."""

    if missing := sorted(_GROUP_COLUMNS.difference(typed_groups.columns)):
        raise ValueError(f"Typed groups are missing required columns: {missing}")
    if missing := sorted(_TRUTH_COLUMNS.difference(truth_edges.columns)):
        raise ValueError(f"Truth edges are missing required columns: {missing}")

    selected = typed_groups.loc[
        typed_groups["selection_status"].astype(str).eq("selected")
    ].copy()
    unexpected_types = sorted(
        set(selected["edge_type"].astype(str))
        - {"within_assay_shared_state_support"}
    )
    if unexpected_types:
        raise ValueError(f"Selected groups contain unexpected edge types: {unexpected_types}")
    if selected["relation_id"].astype(str).duplicated().any():
        raise ValueError("Selected relation identifiers must be unique")
    if any(_as_bool(value) for value in selected["direct_causal_claimed"]):
        raise ValueError("An assay-local loading group may not claim direct causation")
    if any(_as_bool(value) for value in selected["cross_assay_membership_claimed"]):
        raise ValueError("An assay-local loading group may not claim cross-assay membership")

    predicted: dict[frozenset[str], str] = {}
    for row in selected.itertuples(index=False):
        members = frozenset(
            member.strip() for member in str(row.members).split(";") if member.strip()
        )
        if len(members) != 2 or int(row.n_members) != 2:
            raise ValueError("Every selected structure-anchored group must be one pair")
        modality = str(row.modality)
        if any(not member.endswith(f"__{modality.removeprefix('chrom_')}") for member in members):
            raise ValueError("A selected group contains a member from another assay")
        if members in predicted:
            raise ValueError("A predicted pair is duplicated")
        predicted[members] = modality
    if len(set(predicted.values())) != len(predicted):
        raise ValueError("At most one selected structure-anchored pair is allowed per assay")

    shared_truth = truth_edges.loc[
        truth_edges["truth_type"].astype(str).eq("within_assay_shared_state")
    ]
    truth: dict[frozenset[str], str] = {}
    for row in shared_truth.itertuples(index=False):
        pair = frozenset((str(row.node1), str(row.node2)))
        if len(pair) != 2:
            raise ValueError("Shared-state truth edges must contain two distinct nodes")
        modality = f"chrom_{row.mark}"
        if pair in truth:
            raise ValueError("Shared-state truth contains a duplicate pair")
        truth[pair] = modality

    predicted_pairs = set(predicted)
    truth_pairs = set(truth)
    true_pairs = predicted_pairs & truth_pairs
    false_pairs = predicted_pairs - truth_pairs
    missed_pairs = truth_pairs - predicted_pairs

    pair_rows = []
    for pair in sorted(predicted_pairs | truth_pairs, key=_pair_key):
        if pair in true_pairs:
            classification = "true_positive"
        elif pair in false_pairs:
            classification = "false_positive"
        else:
            classification = "false_negative"
        pair_rows.append(
            {
                "pair_key": _pair_key(pair),
                "modality": predicted.get(pair, truth.get(pair, "")),
                "classification": classification,
                "predicted": pair in predicted_pairs,
                "truth": pair in truth_pairs,
            }
        )
    pair_classification = pd.DataFrame(
        pair_rows,
        columns=["pair_key", "modality", "classification", "predicted", "truth"],
    )

    truth_modalities = sorted(set(truth.values()))
    assay_rows = []
    for modality in truth_modalities:
        assay_predicted = {pair for pair, assay in predicted.items() if assay == modality}
        assay_truth = {pair for pair, assay in truth.items() if assay == modality}
        assay_true = assay_predicted & assay_truth
        assay_rows.append(
            {
                "modality": modality,
                "n_predicted": len(assay_predicted),
                "n_truth": len(assay_truth),
                "n_true_positive": len(assay_true),
                "n_false_positive": len(assay_predicted - assay_truth),
                "n_false_negative": len(assay_truth - assay_predicted),
                "complete_without_false_pair": bool(
                    assay_predicted == assay_truth and bool(assay_truth)
                ),
            }
        )
    per_assay = pd.DataFrame(assay_rows)

    precision = _rate(len(true_pairs), len(predicted_pairs))
    recall = _rate(len(true_pairs), len(truth_pairs))
    metrics = {
        "schema_version": "structure_anchored_loading_groups_v4.scorer.v1",
        "world_seed": int(world_seed),
        "estimand": "assay_local_shared_state_pair_support",
        "n_predicted_pairs": len(predicted_pairs),
        "n_truth_pairs": len(truth_pairs),
        "n_true_positive_pairs": len(true_pairs),
        "n_false_positive_pairs": len(false_pairs),
        "n_false_negative_pairs": len(missed_pairs),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": _rate(2 * len(true_pairs), 2 * len(true_pairs) + len(false_pairs) + len(missed_pairs)),
        "n_assays_complete_without_false_pair": int(
            per_assay["complete_without_false_pair"].sum()
        ),
        "n_truth_assays": len(truth_modalities),
        "assay_coverage": _rate(
            int(per_assay["complete_without_false_pair"].sum()), len(truth_modalities)
        ),
        "zero_false_pair_gate": len(false_pairs) == 0,
        "recall_at_least_0_80_gate": recall >= 0.80,
        "complete_recovery_gate": predicted_pairs == truth_pairs,
        "direct_causal_claimed": False,
        "cross_assay_membership_claimed": False,
    }
    return metrics, pair_classification, per_assay


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("WORLD_SEED", "TYPED_GROUPS", "TRUTH_EDGES"),
        required=True,
        help="Repeat once per frozen world.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for seed_text, group_path_text, truth_path_text in args.case:
        seed = int(seed_text)
        metrics, pairs, assays = score_structure_anchored_groups(
            pd.read_csv(Path(group_path_text), sep="\t"),
            pd.read_csv(Path(truth_path_text), sep="\t"),
            seed,
        )
        world_dir = args.output_dir / f"world_seed{seed}"
        world_dir.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(world_dir / "pair_classification.tsv", sep="\t", index=False)
        assays.to_csv(world_dir / "per_assay_metrics.tsv", sep="\t", index=False)
        (world_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        records.append(metrics)

    total_predicted = sum(record["n_predicted_pairs"] for record in records)
    total_truth = sum(record["n_truth_pairs"] for record in records)
    total_true = sum(record["n_true_positive_pairs"] for record in records)
    total_false = sum(record["n_false_positive_pairs"] for record in records)
    summary = {
        "schema_version": "structure_anchored_loading_groups_v4.batch_scorer.v1",
        "n_worlds": len(records),
        "development_validity": "post_truth_known_TRAIN_diagnostic_only_not_blind_validation",
        "candidate_generation_frozen_before_scoring": True,
        "estimand": "assay_local_shared_state_pair_support_not_cross_assay_membership",
        "total_predicted_pairs": total_predicted,
        "total_truth_pairs": total_truth,
        "total_true_positive_pairs": total_true,
        "total_false_positive_pairs": total_false,
        "micro_pair_precision": _rate(total_true, total_predicted),
        "micro_pair_recall": _rate(total_true, total_truth),
        "mean_pair_recall": float(
            sum(record["pair_recall"] for record in records) / len(records)
        ),
        "n_zero_false_pair_worlds": sum(record["zero_false_pair_gate"] for record in records),
        "n_recall_at_least_0_80_worlds": sum(
            record["recall_at_least_0_80_gate"] for record in records
        ),
        "n_complete_recovery_worlds": sum(record["complete_recovery_gate"] for record in records),
        "records": records,
    }
    (args.output_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
