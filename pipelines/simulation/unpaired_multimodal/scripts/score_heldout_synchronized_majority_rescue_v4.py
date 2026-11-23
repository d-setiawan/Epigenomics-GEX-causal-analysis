#!/usr/bin/env python3
"""Truth-separated scorer for combined CA-BOSS v4.46 hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cross_assay_benchmark_metrics_v4 as benchmark  # noqa: E402
import heldout_synchronized_majority_rescue_v4 as rescue  # noqa: E402
import score_component_conditioned_cross_assay_equivalence_v4 as core_scorer  # noqa: E402
import score_multi_assay_permutation_synchronization_v4 as combined_scorer  # noqa: E402


SCHEMA = "ca_boss_v4.heldout_synchronized_majority_rescue.scorer.v1"
ALGORITHM = "CA-BOSS v4.46 combined v4.44 plus held-out synchronized majority"
EXPECTED_RESCUE_SCHEMA = "ca_boss_v4.heldout_synchronized_majority_rescue.v1"


def _verify_rescue(
    rescue_dir: Path,
    bridge: Path,
    bridge_provenance: Mapping[str, Any],
    synchronized: Path,
    synchronization_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Verify the complete pre-truth rescue chain before truth is opened."""

    directory = rescue_dir.expanduser().absolute().resolve(strict=True)
    manifest_path = directory / "rescue_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_RESCUE_SCHEMA:
        raise ValueError("Unexpected rescue schema")
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != combined_scorer._canonical_sha256(unsigned):
        raise ValueError("Rescue manifest payload hash mismatch")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("scorer_truth_used") is not False:
        raise ValueError("Rescue did not certify truth separation")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("Rescue did not satisfy the pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Rescue created a new world or dataset")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = directory / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if combined_scorer._sha256(path) != str(expected):
            raise ValueError(f"Rescue output hash mismatch: {name}")
    bindings = {
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": combined_scorer._sha256(
            bridge / "provenance.json"
        ),
        "synchronization_manifest_payload_sha256": synchronization_manifest[
            "manifest_payload_sha256"
        ],
        "synchronization_manifest_file_sha256": combined_scorer._sha256(
            synchronized / "synchronization_manifest.json"
        ),
    }
    for field, expected in bindings.items():
        if str(manifest.get(field)) != str(expected):
            raise ValueError(f"Rescue upstream binding mismatch: {field}")
    if (
        str(manifest.get("block")) != str(bridge_provenance.get("block"))
        or int(manifest.get("seed")) != int(bridge_provenance.get("seed"))
    ):
        raise ValueError("Rescue and bridge world identities disagree")
    return directory, manifest


def normalize_rescue_predictions(
    core: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Adapt v4.46 rows to the shared unexpanded-hyperedge scorer."""

    incremental_ids = set(
        candidates["synchronized_majority_hyperedge_id"].astype(str)
    )
    incremental = candidates.rename(
        columns={
            "synchronized_majority_hyperedge_id": "synchronized_hyperedge_id"
        }
    ).copy()
    predictions = combined_scorer.normalize_predictions(core, incremental)
    predictions.loc[
        predictions["hyperedge_id"].astype(str).isin(incremental_ids),
        "prediction_source",
    ] = "v446_heldout_synchronized_majority"
    return predictions


def score_combined(
    *,
    bridge_dir: Path,
    synchronization_dir: Path,
    rescue_dir: Path,
    scorer_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Verify v4.44→v4.45→v4.46, then score the combined hypotheses."""

    bridge, bridge_provenance = core_scorer._verify_bridge(bridge_dir)
    synchronized, synchronization_manifest = rescue.verify_synchronization(
        synchronization_dir, bridge, bridge_provenance
    )
    rescue_path, rescue_manifest = _verify_rescue(
        rescue_dir,
        bridge,
        bridge_provenance,
        synchronized,
        synchronization_manifest,
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
    candidates = pd.read_csv(
        rescue_path / "heldout_synchronized_majority_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    predictions = normalize_rescue_predictions(core, candidates)
    selected_incremental = int(
        candidates["hypothesis_selected"].map(combined_scorer._as_bool).sum()
    )
    if selected_incremental != int(rescue_manifest["selected_hypothesis_count"]):
        raise ValueError("Rescue selected-count audit failed")

    truths = core_scorer._truth_hyperedges(truth_path)
    roles = pd.read_csv(roles_path, sep="\t", keep_default_na=False)
    sentinel_features = set(
        roles.loc[roles["role"].astype(str).isin({"N1", "N2"}), "node"]
        .astype(str)
        .tolist()
    )
    scored, metrics = combined_scorer.score_predictions(
        predictions, truths, sentinel_features
    )
    shared = metrics["shared_state_hyperedge"]
    sentinel = metrics["assay_participation_and_sentinel_false_positive"]
    stress = metrics["identifiability_stress_test"]
    stress["interpretation"] = (
        "singleton_reporter_pairs_are_scored_diagnostically_but_v446_"
        "emits_shared_state_equivalence_hypotheses_not_direct_or_directed_edges"
    )
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
        "synchronization_schema_version": synchronization_manifest[
            "schema_version"
        ],
        "rescue_schema_version": rescue_manifest["schema_version"],
        "bridge_provenance_payload_sha256": bridge_provenance[
            "provenance_payload_sha256"
        ],
        "bridge_provenance_file_sha256": combined_scorer._sha256(
            bridge / "provenance.json"
        ),
        "bridge_output_sha256": bridge_provenance["output_sha256"],
        "synchronization_manifest_payload_sha256": synchronization_manifest[
            "manifest_payload_sha256"
        ],
        "synchronization_manifest_file_sha256": combined_scorer._sha256(
            synchronized / "synchronization_manifest.json"
        ),
        "synchronization_output_sha256": synchronization_manifest[
            "output_sha256"
        ],
        "rescue_manifest_payload_sha256": rescue_manifest[
            "manifest_payload_sha256"
        ],
        "rescue_manifest_file_sha256": combined_scorer._sha256(
            rescue_path / "rescue_manifest.json"
        ),
        "rescue_output_sha256": rescue_manifest["output_sha256"],
        "benchmark_metric_schema_version": benchmark.SCHEMA,
        "block": bridge_provenance["block"],
        "seed": bridge_provenance["seed"],
        "truth_opened_only_after_all_pretruth_hash_verification": True,
        "algorithm_used_truth_for_fit_or_selection": False,
        "formal_pretruth_barrier_satisfied": bool(
            bridge_provenance["formal_pretruth_barrier_satisfied"]
            and synchronization_manifest["formal_pretruth_barrier_satisfied"]
            and rescue_manifest["formal_pretruth_barrier_satisfied"]
        ),
        "eligible_for_reliability_claim": True,
        "retrospective_cached_world_exploration": True,
        "new_worlds_or_datasets_created": False,
        "selected_v446_incremental_hypotheses": selected_incremental,
        "benchmark_metrics": metrics,
        "reliability_metrics": reliability_metrics,
        "all_reliability_metrics_pass": all(reliability_metrics.values()),
        "reliability_claim_withheld_due_to_upstream_container_boundary": False,
        "source_sha256": {
            Path(__file__).name: combined_scorer._sha256(
                Path(__file__).resolve()
            )
        },
        "output_sha256": {
            "hyperedge_scores.tsv": combined_scorer._sha256(score_path)
        },
        "truth_input_sha256": {
            "truth_typed_edges.tsv": combined_scorer._sha256(truth_path),
            "role_assignments.tsv": combined_scorer._sha256(roles_path),
        },
    }
    summary["summary_payload_sha256"] = combined_scorer._canonical_sha256(
        summary
    )
    (out_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    parser.add_argument("--synchronization-dir", type=Path, required=True)
    parser.add_argument("--rescue-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = score_combined(
        bridge_dir=args.bridge_dir,
        synchronization_dir=args.synchronization_dir,
        rescue_dir=args.rescue_dir,
        scorer_dir=args.scorer_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
