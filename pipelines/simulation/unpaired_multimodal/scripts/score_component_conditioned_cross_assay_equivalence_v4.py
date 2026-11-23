#!/usr/bin/env python3
"""Truth-separated scorer for v4.44 component-conditioned hyperedges."""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402


SCHEMA = "ca_boss_v4.component_conditioned_cross_assay_equivalence.scorer.v16"
EXPECTED_BRIDGE_SCHEMA = (
    "ca_boss_v4.component_conditioned_cross_assay_equivalence.v15"
)


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


def _pair(left: str, right: str) -> frozenset[str]:
    return frozenset((str(left), str(right)))


def maximum_truth_matching(
    candidates: Sequence[set[frozenset[str]]],
    truth_sets: Sequence[set[frozenset[str]]],
) -> dict[int, int]:
    """Return a deterministic maximum one-to-one candidate/truth matching."""

    adjacency = [
        [truth_index for truth_index, truth in enumerate(truth_sets) if candidate & truth]
        for candidate in candidates
    ]
    truth_owner: dict[int, int] = {}

    def augment(prediction: int, seen: set[int]) -> bool:
        for truth_index in adjacency[prediction]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            owner = truth_owner.get(truth_index)
            if owner is None or augment(owner, seen):
                truth_owner[truth_index] = prediction
                return True
        return False

    for prediction in range(len(candidates)):
        augment(prediction, set())
    return {
        prediction: truth_index
        for truth_index, prediction in truth_owner.items()
    }


def _verify_bridge(bridge_dir: Path) -> tuple[Path, dict[str, Any]]:
    directory = bridge_dir.expanduser().absolute().resolve(strict=True)
    provenance_path = directory / "provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != EXPECTED_BRIDGE_SCHEMA:
        raise ValueError("Unexpected bridge schema")
    observed = str(provenance.get("provenance_payload_sha256", ""))
    unsigned = dict(provenance)
    unsigned.pop("provenance_payload_sha256", None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError("Bridge provenance payload hash mismatch")
    if provenance.get("information_boundary", {}).get("scorer_truth_used") is not False:
        raise ValueError("Bridge did not certify scorer separation")
    for name, expected in provenance.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"Bridge output hash mismatch: {name}")
    return directory, provenance


def _truth_hyperedges(truth_path: Path) -> list[dict[str, Any]]:
    truth = pd.read_csv(truth_path, sep="\t", keep_default_na=False)
    cross = truth.loc[
        truth["truth_type"].astype(str).eq("cross_assay_shared_state")
    ]
    rows: list[dict[str, Any]] = []
    for row in cross.itertuples(index=False):
        left_set = [value for value in str(row.node_set1).split(";") if value]
        right_set = [value for value in str(row.node_set2).split(";") if value]
        rows.append(
            {
                "truth_id": str(row.label),
                "equivalence_pairs": {
                    _pair(left, right)
                    for left, right in product(left_set, right_set)
                },
                "exact_pair": _pair(str(row.node1), str(row.node2)),
                "modalities": {
                    f"chrom_{value}" for value in str(row.mark).split("|")
                },
                "direction_identifiable": str(row.orientation_credit).casefold()
                in {"true", "1"},
            }
        )
    return rows


def score_bridge(
    *, bridge_dir: Path, scorer_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Verify frozen observable evidence, then compute the four metric families."""

    bridge, provenance = _verify_bridge(bridge_dir)
    scorer = scorer_dir.expanduser().absolute().resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    truth_path = scorer / "truth_typed_edges.tsv"
    roles_path = scorer / "role_assignments.tsv"
    if any(path.is_symlink() or not path.is_file() for path in (truth_path, roles_path)):
        raise FileNotFoundError("Scorer truth files are incomplete")
    truths = _truth_hyperedges(truth_path)
    predictions = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if "hypothesis_selected" not in predictions:
        raise ValueError("Bridge output lacks a frozen selection decision")
    roles = pd.read_csv(roles_path, sep="\t", keep_default_na=False)
    sentinel_features = set(
        roles.loc[roles["role"].astype(str).isin({"N1", "N2"}), "node"]
        .astype(str)
        .tolist()
    )

    candidate_sets: list[set[frozenset[str]]] = []
    exact_candidate_sets: list[set[frozenset[str]]] = []
    for row in predictions.itertuples(index=False):
        parsed = json.loads(str(row.candidate_cross_assay_feature_pairs_json))
        pairs = {_pair(left, right) for left, right in parsed}
        candidate_sets.append(pairs)
        exact_candidate_sets.append(pairs if len(pairs) == 1 else set())
    truth_equivalence_sets = [
        set(row["equivalence_pairs"]) for row in truths
    ]
    selected_indices = predictions.index[
        predictions["hypothesis_selected"].astype(bool)
    ].tolist()
    selected_equivalence_matching = maximum_truth_matching(
        [candidate_sets[index] for index in selected_indices],
        truth_equivalence_sets,
    )
    equivalence_matching = {
        selected_indices[position]: truth_index
        for position, truth_index in selected_equivalence_matching.items()
    }
    exact_selected_indices = [
        index for index in selected_indices if len(candidate_sets[index]) == 1
    ]
    selected_exact_matching = maximum_truth_matching(
        [exact_candidate_sets[index] for index in exact_selected_indices],
        [{row["exact_pair"]} for row in truths],
    )
    exact_matching = {
        exact_selected_indices[position]: truth_index
        for position, truth_index in selected_exact_matching.items()
    }

    score_rows: list[dict[str, Any]] = []
    selected_sentinel_features: set[str] = set()
    participation_true_positives = 0
    for index, row in enumerate(predictions.itertuples(index=False)):
        is_selected = bool(row.hypothesis_selected)
        members = set().union(*candidate_sets[index]) if candidate_sets[index] else set()
        sentinels = members & sentinel_features
        if is_selected:
            selected_sentinel_features.update(sentinels)
        truth_index = equivalence_matching.get(index)
        predicted_modalities = {
            value for value in str(row.modalities).split(";") if value
        }
        exact_assay_participation = bool(
            row.exact_assay_participation_claimed
        )
        if (
            is_selected
            and exact_assay_participation
            and truth_index is not None
        ):
            participation_true_positives += len(
                predicted_modalities & truths[truth_index]["modalities"]
            )
        score_rows.append(
            {
                "hyperedge_id": str(row.hyperedge_id),
                "hypothesis_selected": is_selected,
                "matched_to_unique_truth_pair": truth_index is not None,
                "matched_truth_id": (
                    truths[truth_index]["truth_id"]
                    if truth_index is not None
                    else ""
                ),
                "candidate_pair_count": int(
                    row.candidate_cross_assay_feature_pair_count
                ),
                "contains_sentinel_candidate": bool(sentinels),
                "sentinel_candidate_feature_count": len(sentinels),
                "sentinel_null_feature_opportunities": len(sentinel_features),
                "predicted_modalities": ";".join(sorted(predicted_modalities)),
                "predicted_assay_participation_count": (
                    len(predicted_modalities)
                    if exact_assay_participation
                    else 0
                ),
                "assay_participation_is_set_valued": bool(
                    row.assay_participation_is_set_valued
                ),
                "candidate_assay_pair_count": int(
                    row.candidate_assay_pair_count
                ),
                "diagnostic_exact_pair_prediction": bool(
                    is_selected and len(candidate_sets[index]) == 1
                ),
                "diagnostic_exact_pair_true_positive": index in exact_matching,
                "direction_claimed": False,
            }
        )
    scored = pd.DataFrame(
        score_rows,
        columns=[
            "hyperedge_id",
            "hypothesis_selected",
            "matched_to_unique_truth_pair",
            "matched_truth_id",
            "candidate_pair_count",
            "contains_sentinel_candidate",
            "sentinel_candidate_feature_count",
            "sentinel_null_feature_opportunities",
            "predicted_modalities",
            "predicted_assay_participation_count",
            "assay_participation_is_set_valued",
            "candidate_assay_pair_count",
            "diagnostic_exact_pair_prediction",
            "diagnostic_exact_pair_true_positive",
            "direction_claimed",
        ],
    )
    total_truth_participation = sum(len(row["modalities"]) for row in truths)
    predicted_participation = int(
        scored.loc[
            scored["hypothesis_selected"].astype(bool),
            "predicted_assay_participation_count",
        ].sum()
    )
    ambiguous_selected = scored["hypothesis_selected"].astype(bool) & scored[
        "assay_participation_is_set_valued"
    ].astype(bool)
    ambiguous_participation_hypotheses = int(ambiguous_selected.sum())
    ambiguous_assay_pair_alternatives = int(
        scored.loc[ambiguous_selected, "candidate_assay_pair_count"].sum()
    )
    coverage = benchmark.coverage_metrics(
        scored,
        total_truth=len(truths),
        truth_assay_participations=total_truth_participation,
        predicted_assay_participations=predicted_participation,
        true_positive_assay_participations=participation_true_positives,
        ambiguous_assay_participation_hypotheses=(
            ambiguous_participation_hypotheses
        ),
        candidate_assay_pair_alternatives=(
            ambiguous_assay_pair_alternatives
        ),
        assay_participation_point_abstentions=(
            ambiguous_participation_hypotheses
        ),
        sentinel_null_feature_opportunities=len(sentinel_features),
        sentinel_false_positive_features=len(selected_sentinel_features),
    )
    pair_predictions = int(
        scored["diagnostic_exact_pair_prediction"].astype(bool).sum()
    )
    pair_true_positives = int(
        scored["diagnostic_exact_pair_true_positive"].astype(bool).sum()
    )
    stress = benchmark.exact_cross_assay_stress(
        cross_truth_pairs=len(truths),
        direction_truths=sum(
            int(row["direction_identifiable"]) for row in truths
        ),
        pair_predictions=pair_predictions,
        pair_true_positives=pair_true_positives,
        interpretation=(
            "singleton_reporter_pair_is_scored_diagnostically_but_the_v444_"
            "algorithm_emits_shared_state_equivalence_or_an_unexpanded_"
            "alternative_set_not_a_direct_edge"
        ),
    )
    metric_families = benchmark.metric_families(coverage, stress)
    reliability_metrics = {
        "shared_state_hyperedge_precision_is_one": (
            coverage["shared_state_hyperedge_precision"] == 1.0
        ),
        "sentinel_false_positive_rate_is_zero": (
            coverage["sentinel_false_positive_rate"] == 0.0
        ),
        "candidate_alternatives_never_expanded": True,
        "exact_pair_and_direction_stress_not_a_gate": True,
    }
    out_dir.mkdir(parents=True)
    scored.to_csv(out_dir / "hyperedge_scores.tsv", sep="\t", index=False)
    summary = {
        "schema_version": SCHEMA,
        "bridge_schema_version": provenance["schema_version"],
        "bridge_provenance_payload_sha256": provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": _sha256(
            bridge / "provenance.json"
        ),
        "bridge_output_sha256": provenance["output_sha256"],
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "block": provenance["block"],
        "seed": provenance["seed"],
        "truth_opened_only_after_bridge_hash_verification": True,
        "algorithm_used_truth_for_fit_or_selection": False,
        "formal_pretruth_barrier_satisfied": bool(
            provenance["formal_pretruth_barrier_satisfied"]
        ),
        "eligible_for_reliability_claim": bool(
            provenance["formal_pretruth_barrier_satisfied"]
        ),
        "retrospective_cached_world_exploration": True,
        "new_worlds_or_datasets_created": False,
        "benchmark_metrics": metric_families,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "reliability_claim_withheld_due_to_upstream_container_boundary": (
            not provenance["formal_pretruth_barrier_satisfied"]
        ),
        "output_sha256": {
            "hyperedge_scores.tsv": _sha256(out_dir / "hyperedge_scores.tsv")
        },
        "truth_input_sha256": {
            "truth_typed_edges.tsv": _sha256(truth_path),
            "role_assignments.tsv": _sha256(roles_path),
        },
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    (out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = score_bridge(
        bridge_dir=args.bridge_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
