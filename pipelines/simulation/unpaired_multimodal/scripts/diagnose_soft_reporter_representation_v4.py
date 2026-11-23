#!/usr/bin/env python3
"""Freeze and score the v4.63 soft reporter representation experiment.

The combined initialization/cell-refit mode is primary. The other modes are
ablations, not alternative candidates selected for their benchmark result.
The primary score is the minimum over feature contexts of mean signed
association-weighted concordance across all attempted representation refits.
Null calibration compares only worlds with the same declared assay count.
This diagnostic does not promote edges into the production benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_representation_bootstrap_v4 as hard
import soft_reporter_representation_ensemble_v4 as soft

geometry = hard.geometry
SCHEMA = "ca_boss_v4.soft_reporter_representation_diagnostic.v1"
PRIMARY_SCORE = "minimum_context_mean_weighted_concordance"


def select_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """Use only non-target null labels, and abstain with no matching nulls."""
    output = frame.copy()
    output["selected_diagnostic_proposal"] = False
    output["null_reference_available"] = False
    output["null_upper_envelope"] = float("nan")
    for (mode, block, assays), target in output.groupby(["mode", "block", "expected_assay_count"]):
        nulls = output[
            output["mode"].eq(mode) & ~output["block"].eq(block)
            & output["expected_assay_count"].eq(assays) & ~output["truth_pair"]
        ]
        if nulls.empty:
            continue
        ceiling = float(nulls[PRIMARY_SCORE].max())
        output.loc[target.index, "null_reference_available"] = True
        output.loc[target.index, "null_upper_envelope"] = ceiling
        output.loc[target.index, "selected_diagnostic_proposal"] = (
            target[PRIMARY_SCORE].gt(0.0) & target[PRIMARY_SCORE].gt(ceiling))
    return output


def freeze(*, output_root: Path, formal_root: Path, out_dir: Path,
           worlds_per_block: int, random_seed: int, replicates: int, max_workers: int) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    if min(worlds_per_block, replicates, max_workers) < 1:
        raise ValueError("Counts must be positive")
    output_root = output_root.resolve(strict=True)
    formal_root = formal_root.resolve(strict=True)
    frames, _, parents = geometry._discover_v453(output_root)
    sampled = hard.depth._sample_keys(frames, worlds_per_block=worlds_per_block,
                                     random_seed=random_seed, excluded=set())
    keys = sorted(k for group in sampled.values() for k in group)
    artifacts = geometry._resolve_artifacts(output_root, keys)
    contract = {
        "schema_version": SCHEMA, "selected_world_keys": keys,
        "random_seed": random_seed, "replicates": replicates,
        "modes": list(hard.MODES), "primary_mode": "both", "primary_score": PRIMARY_SCORE,
        "selection_rule": "positive and strictly exceeds the maximum other-block null with the same assay count",
        "no_comparable_null_policy": "abstain",
        "missing_representation_policy": "zero for all pairs of an incomplete context; retain denominator",
        "source_sha256": hard._source_hashes(),
        "new_worlds_or_datasets_created": False, "fresh_locked_certification": False,
    }
    out_dir.mkdir(parents=True)
    (out_dir / "launch_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    completed = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(soft.compute_world_soft_evidence, key, artifacts[key], formal_root,
                               replicates=replicates, random_seed=random_seed): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            completed[key] = future.result()
            print(f"Completed soft representation worlds: {len(completed)}/{len(keys)}", flush=True)
    pairs = pd.concat([completed[k][0] for k in keys], ignore_index=True)
    reporters = pd.concat([completed[k][1] for k in keys], ignore_index=True)
    aggregate = soft.aggregate_soft_evidence(pairs, replicates)
    files = {"soft_pair_context_evidence.tsv": pairs, "soft_reporter_fit_evidence.tsv": reporters,
             "soft_pair_evidence.tsv": aggregate}
    for name, table in files.items():
        table.to_csv(out_dir / name, sep="\t", index=False)
    manifest = {
        **contract, "phase": "pretruth_evidence_only", "world_count": len(keys),
        "input_sha256_by_world": {k: completed[k][2] for k in keys},
        "parent_payloads": {k: p["manifest_payload_sha256"] for k, p in parents.items()},
        "information_boundary": {
            "truth_read": False, "external_data_used": False,
            "paired_cells_used": False, "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False, "focal_features_used_for_ica": False,
            "reporter_feature_identities_fixed_from_cached_parent": True,
            "normalized_association_weights_are_posterior_probabilities": False,
        },
        "limitations": [
            "Shared state association only; no causal direction claim.",
            "The one-third centering constant is algebraic, not the calibrated null of optimized synchronization.",
            "Negative replicate evidence is preserved; failure zeros express abstention, not absence.",
            "Reference and reporter selection used the full existing data; this is retrospective sensitivity evidence.",
            "Partial correlations and ICA retain linear-model assumptions on transformed count observations.",
            "Other modes are ablations; only both is the predeclared primary.",
        ],
        "output_sha256": {name: geometry._sha256(out_dir / name) for name in ["launch_contract.json", *files]},
    }
    manifest["manifest_payload_sha256"] = geometry._canonical_sha256(manifest)
    (out_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def score(*, formal_root: Path, evidence_dir: Path) -> dict[str, Any]:
    if (evidence_dir / "diagnostic_summary.json").exists() or (evidence_dir / "scored_pair_evidence.tsv").exists():
        raise FileExistsError("Refusing to overwrite prior scores")
    manifest = geometry._verified_manifest(evidence_dir / "evidence_manifest.json", SCHEMA)
    for name, expected in manifest["source_sha256"].items():
        if geometry._sha256(SCRIPT_DIR / name) != expected:
            raise ValueError(f"Changed source {name}")
    for name, expected in manifest["output_sha256"].items():
        if geometry._sha256(evidence_dir / name) != expected:
            raise ValueError(f"Changed evidence {name}")
    for inputs in manifest["input_sha256_by_world"].values():
        for path, expected in inputs.items():
            if geometry._sha256(Path(path)) != expected:
                raise ValueError(f"Changed input {path}")
    table = pd.read_csv(evidence_dir / "soft_pair_evidence.tsv", sep="\t")
    truth_hashes = {}
    for key in manifest["selected_world_keys"]:
        block, seed = key.split("__world_seed", 1)
        path = formal_root / block / f"world_seed{int(seed)}" / "scorer/truth_typed_edges.tsv"
        truth_hashes[str(path.resolve())] = geometry._sha256(path)
    truth = hard.depth._load_truth(formal_root, manifest["selected_world_keys"])
    table["truth_pair"] = geometry._truth_labels(table, truth)
    scored = select_diagnostic(table)
    summaries = []
    for mode, group in scored.groupby("mode", sort=True):
        selected = group["selected_diagnostic_proposal"]
        labels = group["truth_pair"]
        summaries.append({
            "mode": mode, "primary": mode == "both",
            "diagnostic_true_assay_pairs": int((selected & labels).sum()),
            "diagnostic_false_assay_pairs": int((selected & ~labels).sum()),
            "truth_assay_pairs_with_candidate_reporters": int(labels.sum()),
            "no_comparable_null_abstentions": int((~group["null_reference_available"]).sum()),
            "mean_true_score": float(group.loc[labels, PRIMARY_SCORE].mean()),
            "mean_null_score": float(group.loc[~labels, PRIMARY_SCORE].mean()),
        })
    path = evidence_dir / "scored_pair_evidence.tsv"
    scored.to_csv(path, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA, "phase": "retrospective_diagnostic",
        "evidence_payload_sha256": manifest["manifest_payload_sha256"],
        "primary_mode": "both", "world_count": manifest["world_count"],
        "replicates": manifest["replicates"], "results": summaries,
        "not_a_production_four_family_benchmark": True, "fresh_locked_certification": False,
        "truth_input_sha256": truth_hashes, "source_sha256": hard._source_hashes(),
        "output_sha256": {path.name: geometry._sha256(path)},
    }
    result["summary_payload_sha256"] = geometry._canonical_sha256(result)
    (evidence_dir / "diagnostic_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--worlds-per-block", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=463000)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "freeze":
        result = freeze(output_root=args.output_root, formal_root=args.formal_root, out_dir=args.out_dir,
                        worlds_per_block=args.worlds_per_block, random_seed=args.random_seed,
                        replicates=args.replicates, max_workers=args.max_workers)
        print(json.dumps({k: result[k] for k in ("world_count", "replicates", "manifest_payload_sha256")}, indent=2))
    else:
        result = score(formal_root=args.formal_root, evidence_dir=args.out_dir)
        print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
