#!/usr/bin/env python3
"""Truth-separated scorer for the combined CA-BOSS v4.45 output.

The scorer first verifies the frozen v4.44 bridge, the pre-truth v4.45
synchronization manifest, every bound output hash, and the binding between the
two stages.  Only then does it open a world's scorer partition.  Selected
synchronization groups augment—not replace—the v4.44 hyperedges, and all
candidate alternatives remain unexpanded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.multi_assay_permutation_synchronization.scorer.v1"
ALGORITHM = "CA-BOSS v4.45 combined v4.44 plus multi-assay synchronization"
EXPECTED_SYNCHRONIZATION_SCHEMA = (
    "ca_boss_v4.multi_assay_permutation_synchronization.v1"
)
PREDICTION_COLUMNS = [
    "hyperedge_id",
    "prediction_source",
    "selection_status",
    "hypothesis_selected",
    "modalities",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "candidate_assay_pair_count",
    "exact_assay_participation_claimed",
    "assay_participation_is_set_valued",
    "candidate_pairs_may_be_expanded_as_edges",
]
SCORE_COLUMNS = [
    "hyperedge_id",
    "prediction_source",
    "selection_status",
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
]


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value {value!r}")


def _verify_synchronization(
    synchronization_dir: Path,
    bridge: Path,
    bridge_provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Verify the pre-truth stage and its exact v4.44 bridge binding."""

    directory = synchronization_dir.expanduser().absolute().resolve(strict=True)
    manifest_path = directory / "synchronization_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_SYNCHRONIZATION_SCHEMA:
        raise ValueError("Unexpected synchronization schema")
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError("Synchronization manifest payload hash mismatch")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("scorer_truth_used") is not False:
        raise ValueError("Synchronization did not certify truth separation")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Synchronization created a new world or dataset")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("Synchronization did not satisfy the pre-truth barrier")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != str(expected):
            raise ValueError(f"Synchronization output hash mismatch: {name}")
    if str(manifest.get("bridge_provenance_payload_sha256")) != str(
        bridge_provenance["provenance_payload_sha256"]
    ):
        raise ValueError("Synchronization and bridge payload hashes disagree")
    if str(manifest.get("bridge_provenance_file_sha256")) != _sha256(
        bridge / "provenance.json"
    ):
        raise ValueError("Synchronization and bridge file hashes disagree")
    if (
        str(manifest.get("block")) != str(bridge_provenance.get("block"))
        or int(manifest.get("seed")) != int(bridge_provenance.get("seed"))
    ):
        raise ValueError("Synchronization and bridge world identities disagree")
    return directory, manifest


def normalize_predictions(
    core: pd.DataFrame, synchronized: pd.DataFrame
) -> pd.DataFrame:
    """Normalize v4.44 and incremental v4.45 rows without expanding sets."""

    core_status_column = (
        "selection_status"
        if "selection_status" in core.columns
        else "selection_route"
    )
    if core_status_column not in core.columns:
        raise ValueError("v4.44 output lacks a selection audit column")
    rows: list[dict[str, Any]] = []
    for row in core.itertuples(index=False):
        rows.append(
            {
                "hyperedge_id": str(row.hyperedge_id),
                "prediction_source": "v444_component_conditioned_bridge",
                "selection_status": str(getattr(row, core_status_column)),
                "hypothesis_selected": _as_bool(row.hypothesis_selected),
                "modalities": str(row.modalities),
                "candidate_cross_assay_feature_pairs_json": str(
                    row.candidate_cross_assay_feature_pairs_json
                ),
                "candidate_cross_assay_feature_pair_count": int(
                    row.candidate_cross_assay_feature_pair_count
                ),
                "candidate_assay_pair_count": int(
                    row.candidate_assay_pair_count
                ),
                "exact_assay_participation_claimed": _as_bool(
                    row.exact_assay_participation_claimed
                ),
                "assay_participation_is_set_valued": _as_bool(
                    row.assay_participation_is_set_valued
                ),
                "candidate_pairs_may_be_expanded_as_edges": _as_bool(
                    row.candidate_pairs_may_be_expanded_as_edges
                ),
            }
        )
    for row in synchronized.itertuples(index=False):
        exact = _as_bool(row.exact_assay_participation_claimed)
        rows.append(
            {
                "hyperedge_id": str(row.synchronized_hyperedge_id),
                "prediction_source": "v445_multi_assay_synchronization",
                "selection_status": str(row.selection_status),
                "hypothesis_selected": _as_bool(row.hypothesis_selected),
                "modalities": str(row.modalities),
                "candidate_cross_assay_feature_pairs_json": str(
                    row.candidate_cross_assay_feature_pairs_json
                ),
                "candidate_cross_assay_feature_pair_count": int(
                    row.candidate_cross_assay_feature_pair_count
                ),
                "candidate_assay_pair_count": int(
                    row.candidate_assay_pair_count
                ),
                "exact_assay_participation_claimed": exact,
                "assay_participation_is_set_valued": not exact,
                "candidate_pairs_may_be_expanded_as_edges": _as_bool(
                    row.candidate_pairs_may_be_expanded_as_edges
                ),
            }
        )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    if predictions["hyperedge_id"].duplicated().any():
        raise ValueError("Combined output contains duplicate hyperedge IDs")
    if predictions.loc[
        predictions["hypothesis_selected"].astype(bool),
        "candidate_pairs_may_be_expanded_as_edges",
    ].astype(bool).any():
        raise ValueError("A selected candidate attempted pair expansion")
    return predictions


def score_predictions(
    predictions: pd.DataFrame,
    truths: Sequence[Mapping[str, Any]],
    sentinel_features: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the four metric families for already-frozen predictions."""

    candidate_sets: list[set[frozenset[str]]] = []
    for row in predictions.itertuples(index=False):
        parsed = json.loads(str(row.candidate_cross_assay_feature_pairs_json))
        pairs = {core_scorer._pair(left, right) for left, right in parsed}
        if len(pairs) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("Candidate feature-pair count does not match payload")
        candidate_sets.append(pairs)
    truth_equivalence_sets = [
        set(row["equivalence_pairs"]) for row in truths
    ]
    selected_indices = predictions.index[
        predictions["hypothesis_selected"].astype(bool)
    ].tolist()
    selected_matching = core_scorer.maximum_truth_matching(
        [candidate_sets[index] for index in selected_indices],
        truth_equivalence_sets,
    )
    equivalence_matching = {
        selected_indices[position]: truth_index
        for position, truth_index in selected_matching.items()
    }
    singleton_indices = [
        index for index in selected_indices if len(candidate_sets[index]) == 1
    ]
    singleton_matching = core_scorer.maximum_truth_matching(
        [candidate_sets[index] for index in singleton_indices],
        [{row["exact_pair"]} for row in truths],
    )
    exact_matching = {
        singleton_indices[position]: truth_index
        for position, truth_index in singleton_matching.items()
    }

    rows: list[dict[str, Any]] = []
    selected_sentinels: set[str] = set()
    true_positive_participations = 0
    for index, row in enumerate(predictions.itertuples(index=False)):
        selected = bool(row.hypothesis_selected)
        members = (
            set().union(*candidate_sets[index])
            if candidate_sets[index]
            else set()
        )
        sentinels = members & sentinel_features
        if selected:
            selected_sentinels.update(sentinels)
        truth_index = equivalence_matching.get(index)
        modalities = {
            value for value in str(row.modalities).split(";") if value
        }
        exact_assay = bool(row.exact_assay_participation_claimed)
        if selected and exact_assay and truth_index is not None:
            true_positive_participations += len(
                modalities & set(truths[truth_index]["modalities"])
            )
        rows.append(
            {
                "hyperedge_id": str(row.hyperedge_id),
                "prediction_source": str(row.prediction_source),
                "selection_status": str(row.selection_status),
                "hypothesis_selected": selected,
                "matched_to_unique_truth_pair": truth_index is not None,
                "matched_truth_id": (
                    str(truths[truth_index]["truth_id"])
                    if truth_index is not None
                    else ""
                ),
                "candidate_pair_count": len(candidate_sets[index]),
                "contains_sentinel_candidate": bool(sentinels),
                "sentinel_candidate_feature_count": len(sentinels),
                "sentinel_null_feature_opportunities": len(sentinel_features),
                "predicted_modalities": ";".join(sorted(modalities)),
                "predicted_assay_participation_count": (
                    len(modalities) if exact_assay else 0
                ),
                "assay_participation_is_set_valued": bool(
                    row.assay_participation_is_set_valued
                ),
                "candidate_assay_pair_count": int(
                    row.candidate_assay_pair_count
                ),
                "diagnostic_exact_pair_prediction": bool(
                    selected and len(candidate_sets[index]) == 1
                ),
                "diagnostic_exact_pair_true_positive": index in exact_matching,
                "direction_claimed": False,
            }
        )
    scored = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    selected_mask = scored["hypothesis_selected"].astype(bool)
    predicted_participations = int(
        scored.loc[selected_mask, "predicted_assay_participation_count"].sum()
    )
    ambiguous = selected_mask & scored[
        "assay_participation_is_set_valued"
    ].astype(bool)
    coverage = benchmark.coverage_metrics(
        scored,
        total_truth=len(truths),
        truth_assay_participations=sum(
            len(set(row["modalities"])) for row in truths
        ),
        predicted_assay_participations=predicted_participations,
        true_positive_assay_participations=true_positive_participations,
        ambiguous_assay_participation_hypotheses=int(ambiguous.sum()),
        candidate_assay_pair_alternatives=int(
            scored.loc[ambiguous, "candidate_assay_pair_count"].sum()
        ),
        assay_participation_point_abstentions=int(ambiguous.sum()),
        sentinel_null_feature_opportunities=len(sentinel_features),
        sentinel_false_positive_features=len(selected_sentinels),
    )
    stress = benchmark.exact_cross_assay_stress(
        cross_truth_pairs=len(truths),
        direction_truths=sum(
            int(bool(row["direction_identifiable"])) for row in truths
        ),
        pair_predictions=int(
            scored["diagnostic_exact_pair_prediction"].astype(bool).sum()
        ),
        pair_true_positives=int(
            scored["diagnostic_exact_pair_true_positive"].astype(bool).sum()
        ),
        interpretation=(
            "singleton_reporter_pairs_are_scored_diagnostically_but_v445_"
            "emits_shared_state_equivalence_groups_not_direct_or_directed_edges"
        ),
    )
    return scored, benchmark.metric_families(coverage, stress)


def score_combined(
    *,
    bridge_dir: Path,
    synchronization_dir: Path,
    scorer_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Verify both pre-truth stages, then score their combined output."""

    bridge, bridge_provenance = core_scorer._verify_bridge(bridge_dir)
    synchronized_dir, manifest = _verify_synchronization(
        synchronization_dir, bridge, bridge_provenance
    )
    scorer = scorer_dir.expanduser().absolute().resolve(strict=True)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    truth_path = scorer / "truth_typed_edges.tsv"
    roles_path = scorer / "role_assignments.tsv"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (truth_path, roles_path)
    ):
        raise FileNotFoundError("Scorer truth files are incomplete")

    core = pd.read_csv(
        bridge / "stable_cross_assay_hyperedges.tsv",
        sep="\t",
        keep_default_na=False,
    )
    synchronized = pd.read_csv(
        synchronized_dir / "multi_assay_synchronization_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    predictions = normalize_predictions(core, synchronized)
    selected_synchronizations = int(
        synchronized["hypothesis_selected"].map(_as_bool).sum()
    )
    if selected_synchronizations != int(manifest["selected_hypothesis_count"]):
        raise ValueError("Synchronization selected-count audit failed")

    truths = core_scorer._truth_hyperedges(truth_path)
    roles = pd.read_csv(roles_path, sep="\t", keep_default_na=False)
    sentinel_features = set(
        roles.loc[roles["role"].astype(str).isin({"N1", "N2"}), "node"]
        .astype(str)
        .tolist()
    )
    scored, metric_families = score_predictions(
        predictions, truths, sentinel_features
    )
    shared = metric_families["shared_state_hyperedge"]
    sentinel = metric_families[
        "assay_participation_and_sentinel_false_positive"
    ]
    stress = metric_families["identifiability_stress_test"]
    reliability_metrics = {
        "shared_state_hyperedge_precision_is_one": (
            shared["shared_state_hyperedge_precision"] == 1.0
        ),
        "sentinel_false_positive_rate_is_zero": (
            sentinel["sentinel_false_positive_rate"] == 0.0
        ),
        "candidate_alternatives_never_expanded": bool(
            not predictions["candidate_pairs_may_be_expanded_as_edges"]
            .astype(bool)
            .any()
        ),
        "exact_pair_and_direction_stress_not_a_gate": (
            stress["included_in_reliability_gate"] is False
        ),
    }
    out_dir.mkdir(parents=True)
    score_path = out_dir / "hyperedge_scores.tsv"
    scored.to_csv(score_path, sep="\t", index=False)
    summary = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "bridge_schema_version": bridge_provenance["schema_version"],
        "synchronization_schema_version": manifest["schema_version"],
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": _sha256(
            bridge / "provenance.json"
        ),
        "bridge_output_sha256": bridge_provenance["output_sha256"],
        "synchronization_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "synchronization_manifest_file_sha256": _sha256(
            synchronized_dir / "synchronization_manifest.json"
        ),
        "synchronization_output_sha256": manifest["output_sha256"],
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "block": bridge_provenance["block"],
        "seed": bridge_provenance["seed"],
        "truth_opened_only_after_all_pretruth_hash_verification": True,
        "algorithm_used_truth_for_fit_or_selection": False,
        "formal_pretruth_barrier_satisfied": bool(
            bridge_provenance["formal_pretruth_barrier_satisfied"]
            and manifest["formal_pretruth_barrier_satisfied"]
        ),
        "eligible_for_reliability_claim": bool(
            bridge_provenance["formal_pretruth_barrier_satisfied"]
            and manifest["formal_pretruth_barrier_satisfied"]
        ),
        "retrospective_cached_world_exploration": True,
        "new_worlds_or_datasets_created": False,
        "selected_v445_incremental_hypotheses": selected_synchronizations,
        "benchmark_metrics": metric_families,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "reliability_claim_withheld_due_to_upstream_container_boundary": (
            not bool(bridge_provenance["formal_pretruth_barrier_satisfied"])
        ),
        "source_sha256": {
            Path(__file__).name: _sha256(Path(__file__).resolve())
        },
        "output_sha256": {"hyperedge_scores.tsv": _sha256(score_path)},
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
    parser.add_argument("--synchronization-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = score_combined(
        bridge_dir=args.bridge_dir,
        synchronization_dir=args.synchronization_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
