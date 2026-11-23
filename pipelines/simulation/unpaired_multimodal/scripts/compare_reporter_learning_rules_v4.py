#!/usr/bin/env python3
"""Paired v4.64 comparison of marginal and joint reporter learning rules.

ICA fits, within-assay evaluation rows, and background-distribution alignment
are computed once and shared by both models. The only model change is reporter
axis evidence: marginal depth-adjusted squared correlations versus joint OLS
unique contributions (delta R-squared). Both retain the v4.63 signed pair
statistic and mean-across-refits then minimum-across-contexts aggregation.

This freezes observed evidence before scoring existing truth partitions. The
joint model with combined initialization/cell refits is primary; the marginal
model and cached representation are fixed controls. This is a retrospective
diagnostic, not a production graph or a four-family benchmark promotion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import diagnose_representation_bootstrap_v4 as hard
import diagnose_soft_reporter_representation_v4 as marginal_diagnostic
import soft_reporter_representation_ensemble_v4 as soft
import joint_reporter_regression_evidence_v4 as joint_reporter

geometry = hard.geometry
SCHEMA = "ca_boss_v4.paired_reporter_learning_comparison.v1"
REPORTER_MODELS = ("marginal_squared_correlation", "joint_unique_delta_r2")
DEFAULT_MODES = ("cached_reference_subset", "both")
PRIMARY_MODEL = "joint_unique_delta_r2"
PRIMARY_MODE = "both"
PRIMARY_SCORE = marginal_diagnostic.PRIMARY_SCORE
MODEL_DEFINITIONS = {
    "marginal_squared_correlation": "squared marginal correlation after adjusting reporter and each ICA score for observed depth",
    "joint_unique_delta_r2": "unique reporter variance explained by each ICA score after adjusting for depth and all other ICA scores",
}


def compute_array_comparison(
    key: str, arrays: Mapping[str, Mapping[str, Any]], *, replicates: int,
    random_seed: int, modes: Sequence[str] = DEFAULT_MODES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both reporter models on one shared representation experiment."""
    if replicates < 1 or not modes or len(set(modes)) != len(modes) or any(m not in hard.MODES for m in modes):
        raise ValueError("Positive replicates and unique supported modes are required")
    modalities = sorted(arrays)
    if len(modalities) < 2:
        raise ValueError("At least two independent assay arrays are required")
    block = key.split("__world_seed", 1)[0]
    models = {
        REPORTER_MODELS[0]: soft.reporter_axis_evidence,
        REPORTER_MODELS[1]: joint_reporter.reporter_axis_evidence,
    }
    pairs, fits = [], []
    with threadpool_limits(limits=1):
        for mode in modes:
            for replicate in range(replicates):
                seed = hard.mdcr._stable_seed(random_seed, f"{key}:{replicate}")
                context_scores = {context: {} for context in hard.CONTEXTS}
                reporters = {model: {context: {} for context in hard.CONTEXTS} for model in REPORTER_MODELS}
                evaluation_hashes = {}
                for modality in modalities:
                    a = arrays[modality]
                    evaluation = hard._evaluation_indices(len(a["counts"]), modality, seed)
                    evaluation_hash = geometry._canonical_sha256({"indices": evaluation.tolist()})
                    evaluation_hashes[modality] = evaluation_hash
                    try:
                        if len(evaluation) < 8:
                            raise hard.representation.RepresentationFitError("Fewer than eight common evaluation cells")
                        fitted = None if mode == "cached_reference_subset" else hard.representation.bootstrap_fit_assay_contexts(
                            a["counts"], a["depth"], a["reference"], modality=modality,
                            reference_seed=a["reference_seed"], resample_seed=seed,
                            initialization_seed=seed, mode=mode, config=a["config"])
                    except hard.representation.RepresentationFitError as error:
                        for context in hard.CONTEXTS:
                            for model in REPORTER_MODELS:
                                fits.append({"world_key": key, "mode": mode, "replicate": replicate,
                                             "modality": modality, "context": context, "reporter_model": model,
                                             "fit_ok": False, "failure": str(error),
                                             "evaluation_cells": len(evaluation), "evaluation_indices_sha256": evaluation_hash})
                        continue
                    if fitted is not None and mode in ("cell_resample_only", "both"):
                        if not np.array_equal(evaluation, np.flatnonzero(fitted.oob_mask)):
                            raise ValueError("Representation and evaluation bootstrap rows disagree")
                    for context in hard.CONTEXTS:
                        scores = (a["reference"][context] if fitted is None else fitted.scores[context])[evaluation]
                        context_scores[context][modality] = scores
                        for model, compute_reporter in models.items():
                            reporter = compute_reporter(a["focal"][evaluation], scores, a["depth"][evaluation])
                            reporters[model][context][modality] = reporter
                            row = {
                                "world_key": key, "mode": mode, "replicate": replicate,
                                "modality": modality, "context": context, "reporter_model": model,
                                "fit_ok": True, "failure": "", "evaluation_cells": len(evaluation),
                                "evaluation_indices_sha256": evaluation_hash,
                                "evaluation_cells_out_of_bag_for_ica": mode in ("cell_resample_only", "both"),
                                "representation_refitted": fitted is not None,
                                "reporter_usable": reporter.usable,
                                "reporter_count": int(a["focal"].shape[1]),
                                "nonzero_reporters": reporter.nonzero_reporters,
                                "total_axis_strength": reporter.total_strength,
                                "reporter_axis_information": reporter.information,
                                "reporter_weight_margin": reporter.weight_margin,
                                "minimum_reporter_best_axis_strength": reporter.minimum_reporter_best_squared_correlation,
                                "signed_axis_evidence_json": json.dumps(reporter.partial_correlations.tolist()),
                                "per_reporter_axis_weights_json": json.dumps(reporter.per_reporter_weights.tolist()),
                                "mean_axis_strength_json": json.dumps(reporter.mean_squared_correlations.tolist()),
                                "axis_weights_json": json.dumps(reporter.weights.tolist()),
                                "axis_evidence_definition": MODEL_DEFINITIONS[model],
                                "alignment_diagnostics_json": json.dumps({"reference_scores_reused": True} if fitted is None else fitted.contexts[context].__dict__, sort_keys=True),
                            }
                            if model == PRIMARY_MODEL:
                                row.update(
                                    reporter_model_identified=bool(reporter.identified),
                                    reporter_unresolved_reason=reporter.unresolved_reason,
                                    residual_score_rank=int(reporter.residual_score_rank),
                                    residual_score_condition_number=float(reporter.residual_score_condition_number),
                                    unique_axis_variance_json=json.dumps(reporter.unique_axis_variance.tolist()),
                                    joint_coefficients_json=json.dumps(reporter.coefficients.tolist()),
                                )
                            fits.append(row)
                shared_evaluation_hash = geometry._canonical_sha256(evaluation_hashes)
                for context in hard.CONTEXTS:
                    current = context_scores[context]
                    complete = len(current) == len(modalities)
                    fingerprints = {m: hard.joint._projection_quantiles(x, copula_only=False) for m, x in current.items()} if complete else {}
                    matrices = {(a, b): hard.joint._precomputed_cost_matrix(fingerprints[a], fingerprints[b]) for a, b in combinations(sorted(fingerprints), 2)}
                    table = geometry._normalized_cost_table(current, matrices) if matrices else None
                    solution = geometry.synchronization.solve_multiway_permutation_synchronization(table, sorted(current)) if table is not None else None
                    orders = {} if solution is None else dict(zip(solution.modalities, solution.best_orders))
                    geometry_hash = geometry._canonical_sha256({
                        "complete": complete, "orders": {m: order.tolist() for m, order in orders.items()},
                        "global_relative_margin": None if solution is None else float(solution.global_relative_margin)})
                    for left, right in combinations(modalities, 2):
                        for model in REPORTER_MODELS:
                            row = {
                                "world_key": key, "block": block, "left_modality": left,
                                "right_modality": right, "mode": mode, "replicate": replicate,
                                "context": context, "reporter_model": model,
                                "fit_pair_available": False, "eligible": False,
                                "synchronized_assay_count": len(current), "expected_assay_count": len(modalities),
                                "complete_context": complete,
                                "shared_evaluation_sha256": shared_evaluation_hash,
                                "shared_geometry_sha256": geometry_hash,
                                **{metric: 0.0 for metric in soft.PAIR_METRICS},
                            }
                            if left in orders and right in orders:
                                row.update(soft.soft_pair_evidence(reporters[model][context][left], reporters[model][context][right], orders[left], orders[right]))
                                row.update(fit_pair_available=True, global_relative_margin=float(solution.global_relative_margin),
                                           left_global_order_json=json.dumps(orders[left].tolist()), right_global_order_json=json.dumps(orders[right].tolist()))
                            pairs.append(row)
    return pd.DataFrame(pairs), pd.DataFrame(fits)


def aggregate_comparison(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    """Apply the unchanged v4.63 aggregation separately to both models."""
    if set(frame["reporter_model"].astype(str)) != set(REPORTER_MODELS):
        raise ValueError("Both declared reporter models are required")
    identity = [*soft.IDENTITY, "mode", "replicate", "context"]
    for _, group in frame.groupby(identity, sort=False):
        if len(group) != len(REPORTER_MODELS) or set(group["reporter_model"]) != set(REPORTER_MODELS):
            raise ValueError("Missing or duplicate paired-model context")
        for field in ("shared_evaluation_sha256", "shared_geometry_sha256", "expected_assay_count", "complete_context", "fit_pair_available"):
            if group[field].nunique(dropna=False) != 1:
                raise ValueError(f"Reporter models do not share {field}")
    aggregates = []
    for model in REPORTER_MODELS:
        subset = frame[frame["reporter_model"].eq(model)].drop(columns="reporter_model")
        aggregate = soft.aggregate_soft_evidence(subset, replicates)
        aggregate["reporter_model"] = model
        aggregates.append(aggregate)
    return pd.concat(aggregates, ignore_index=True)


def select_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep empirical-null calibration separated by reporter model and mode."""
    if set(frame["reporter_model"].astype(str)) != set(REPORTER_MODELS):
        raise ValueError("Both declared reporter models are required")
    outputs = []
    for model in REPORTER_MODELS:
        subset = frame[frame["reporter_model"].eq(model)].copy()
        result = marginal_diagnostic.select_diagnostic(subset)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def _compute_world(key: str, artifacts: Any, formal_root: Path, *, replicates: int,
                   random_seed: int, modes: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    arrays, hashes = hard._load_world(key, artifacts, formal_root)
    pairs, reporters = compute_array_comparison(key, arrays, replicates=replicates, random_seed=random_seed, modes=modes)
    return pairs, reporters, hashes


def freeze(*, output_root: Path, formal_root: Path, out_dir: Path, worlds_per_block: int = 2,
           random_seed: int = 463000, replicates: int = 16, max_workers: int = 6,
           modes: Sequence[str] = DEFAULT_MODES) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    if min(worlds_per_block, replicates, max_workers) < 1 or not modes or len(set(modes)) != len(modes) or any(m not in hard.MODES for m in modes):
        raise ValueError("Counts must be positive and modes unique/supported")
    if PRIMARY_MODE not in modes:
        raise ValueError("The predeclared primary mode must be included")
    output_root, formal_root = output_root.resolve(strict=True), formal_root.resolve(strict=True)
    frames, _, parents = geometry._discover_v453(output_root)
    sampled = hard.depth._sample_keys(frames, worlds_per_block=worlds_per_block, random_seed=random_seed, excluded=set())
    keys = sorted(key for block_keys in sampled.values() for key in block_keys)
    artifacts = geometry._resolve_artifacts(output_root, keys)
    contract = {
        "schema_version": SCHEMA, "selected_world_keys": keys, "random_seed": random_seed,
        "replicates": replicates, "modes": list(modes), "reporter_models": list(REPORTER_MODELS),
        "reporter_model_definitions": MODEL_DEFINITIONS, "primary_reporter_model": PRIMARY_MODEL,
        "primary_mode": PRIMARY_MODE, "primary_score": PRIMARY_SCORE,
        "score_definition": "mean signed strength-centered concordance over all requested refits in each context, then minimum across three contexts",
        "legacy_metric_field_alias": "minimum_reporter_best_squared_correlation denotes minimum reporter best unique delta R-squared for the joint model",
        "selection_rule": "positive and strictly exceeds maximum other-block null with the same reporter model, representation mode and assay count",
        "no_comparable_null_policy": "abstain",
        "missing_representation_policy": "zero for all pairs of an incomplete context; retain denominator",
        "both_models_share_ica_fits_evaluation_rows_and_synchronization": True,
        "source_sha256": hard._source_hashes(), "new_worlds_or_datasets_created": False,
        "fresh_locked_certification": False,
    }
    out_dir.mkdir(parents=True)
    (out_dir / "launch_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    completed = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_world, key, artifacts[key], formal_root, replicates=replicates,
                               random_seed=random_seed, modes=tuple(modes)): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            completed[key] = future.result()
            print(f"Completed paired reporter worlds: {len(completed)}/{len(keys)}", flush=True)
    pairs = pd.concat([completed[key][0] for key in keys], ignore_index=True)
    reporters = pd.concat([completed[key][1] for key in keys], ignore_index=True)
    aggregate = aggregate_comparison(pairs, replicates)
    files = {"paired_reporter_context_evidence.tsv": pairs, "paired_reporter_fit_evidence.tsv": reporters,
             "paired_reporter_pair_evidence.tsv": aggregate}
    for name, table in files.items():
        table.to_csv(out_dir / name, sep="\t", index=False)
    if hard._source_hashes() != contract["source_sha256"]:
        raise ValueError("Source closure changed during evidence computation")
    manifest = {
        **contract, "phase": "pretruth_evidence_only", "world_count": len(keys),
        "input_sha256_by_world": {key: completed[key][2] for key in keys},
        "parent_payloads": {key: parent["manifest_payload_sha256"] for key, parent in parents.items()},
        "information_boundary": {
            "truth_read": False, "external_data_used": False, "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False, "simulator_latent_state_used": False,
            "focal_features_used_for_ica_or_synchronization": False,
            "reporter_feature_identities_fixed_from_cached_parent": True,
            "normalized_association_weights_are_posterior_probabilities": False,
        },
        "limitations": [
            "Shared-state association diagnostic only; no direct or directed causal structure claim.",
            "Cached references and reporter identities used the full existing data; evaluation is retrospective sensitivity.",
            "Joint OLS uses unique linear explained variance on transformed observations, not a sparse-count likelihood.",
            "The centered score is unchanged; centering is algebraic, not the empirical null of optimized synchronization.",
            "Negative evidence is preserved and failed contexts retain zero contributions in the requested denominator.",
            "All model/mode combinations are reported; only joint_unique_delta_r2 with both is primary.",
        ],
        "output_sha256": {name: geometry._sha256(out_dir / name) for name in ["launch_contract.json", *files]},
    }
    manifest["manifest_payload_sha256"] = geometry._canonical_sha256(manifest)
    (out_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _verify_evidence(evidence_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = geometry._verified_manifest(evidence_dir / "evidence_manifest.json", SCHEMA)
    if manifest.get("primary_reporter_model") != PRIMARY_MODEL or manifest.get("primary_mode") != PRIMARY_MODE or manifest.get("primary_score") != PRIMARY_SCORE:
        raise ValueError("Predeclared primary comparison changed")
    if manifest.get("information_boundary", {}).get("truth_read") is not False:
        raise ValueError("Evidence lacks a truth-free barrier")
    for name, expected in manifest["source_sha256"].items():
        if geometry._sha256(SCRIPT_DIR / name) != expected:
            raise ValueError(f"Changed source {name}")
    for name, expected in manifest["output_sha256"].items():
        if geometry._sha256(evidence_dir / name) != expected:
            raise ValueError(f"Changed evidence {name}")
    for inputs in manifest["input_sha256_by_world"].values():
        for name, expected in inputs.items():
            if geometry._sha256(Path(name)) != expected:
                raise ValueError(f"Changed input {name}")
    contexts = pd.read_csv(evidence_dir / "paired_reporter_context_evidence.tsv", sep="\t")
    expected_table = aggregate_comparison(contexts, int(manifest["replicates"]))
    table = pd.read_csv(evidence_dir / "paired_reporter_pair_evidence.tsv", sep="\t")
    pd.testing.assert_frame_equal(table, expected_table, check_dtype=False, rtol=1e-12, atol=1e-12)
    if set(table["world_key"].astype(str)) != set(manifest["selected_world_keys"]) or set(table["mode"]) != set(manifest["modes"]):
        raise ValueError("Evidence world or mode coverage changed")
    return manifest, table


def score(*, formal_root: Path, evidence_dir: Path) -> dict[str, Any]:
    if (evidence_dir / "diagnostic_summary.json").exists() or (evidence_dir / "scored_pair_evidence.tsv").exists():
        raise FileExistsError("Refusing to overwrite prior scores")
    manifest, table = _verify_evidence(evidence_dir)
    formal_root = formal_root.resolve(strict=True)
    # Every source, input, output, paired-context and aggregate check precedes truth access.
    truth_hashes = {}
    for key in manifest["selected_world_keys"]:
        block, seed = key.split("__world_seed", 1)
        path = formal_root / block / f"world_seed{int(seed)}" / "scorer/truth_typed_edges.tsv"
        truth_hashes[str(path.resolve())] = geometry._sha256(path)
    truth = hard.depth._load_truth(formal_root, manifest["selected_world_keys"])
    table["truth_pair"] = geometry._truth_labels(table, truth)
    scored = select_diagnostic(table)
    summaries = []
    for (model, mode), group in scored.groupby(["reporter_model", "mode"], sort=True):
        selected, labels = group["selected_diagnostic_proposal"], group["truth_pair"]
        summaries.append({
            "reporter_model": model, "mode": mode, "primary": model == PRIMARY_MODEL and mode == PRIMARY_MODE,
            "diagnostic_true_assay_pairs": int((selected & labels).sum()),
            "diagnostic_false_assay_pairs": int((selected & ~labels).sum()),
            "truth_assay_pairs_with_candidate_reporters": int(labels.sum()),
            "no_comparable_null_abstentions": int((~group["null_reference_available"]).sum()),
            "mean_true_score": float(group.loc[labels, PRIMARY_SCORE].mean()),
            "mean_null_score": float(group.loc[~labels, PRIMARY_SCORE].mean()),
        })
    deltas = []
    for mode, group in scored.groupby("mode", sort=True):
        marginal = group[group["reporter_model"].eq(REPORTER_MODELS[0])]
        joint = group[group["reporter_model"].eq(REPORTER_MODELS[1])]
        paired = marginal.merge(joint, on=soft.IDENTITY, suffixes=("_marginal", "_joint"), validate="one_to_one")
        gained = ~paired.selected_diagnostic_proposal_marginal & paired.selected_diagnostic_proposal_joint
        lost = paired.selected_diagnostic_proposal_marginal & ~paired.selected_diagnostic_proposal_joint
        labels = paired.truth_pair_joint
        deltas.append({"mode": mode, "joint_only_true_pairs": int((gained & labels).sum()),
                       "joint_only_false_pairs": int((gained & ~labels).sum()),
                       "marginal_only_true_pairs": int((lost & labels).sum()),
                       "marginal_only_false_pairs": int((lost & ~labels).sum())})
    path = evidence_dir / "scored_pair_evidence.tsv"
    scored.to_csv(path, sep="\t", index=False)
    result = {
        "schema_version": SCHEMA, "phase": "retrospective_paired_model_diagnostic",
        "evidence_payload_sha256": manifest["manifest_payload_sha256"],
        "primary_reporter_model": PRIMARY_MODEL, "primary_mode": PRIMARY_MODE,
        "world_count": manifest["world_count"], "replicates": manifest["replicates"],
        "results": summaries, "paired_model_differences": deltas,
        "not_a_production_four_family_benchmark": True, "promoted_to_checkpoint": False,
        "fresh_locked_certification": False, "target_block_truth_used_for_target_selection": False,
        "other_block_truth_used_only_for_empirical_null": True,
        "truth_input_sha256": truth_hashes, "source_sha256": manifest["source_sha256"],
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
    parser.add_argument("--modes", nargs="+", choices=hard.MODES, default=list(DEFAULT_MODES))
    args = parser.parse_args()
    if args.phase == "freeze":
        result = freeze(output_root=args.output_root, formal_root=args.formal_root, out_dir=args.out_dir,
                        worlds_per_block=args.worlds_per_block, random_seed=args.random_seed,
                        replicates=args.replicates, max_workers=args.max_workers, modes=args.modes)
        print(json.dumps({key: result[key] for key in ("world_count", "replicates", "manifest_payload_sha256")}, indent=2))
    else:
        result = score(formal_root=args.formal_root, evidence_dir=args.out_dir)
        print(json.dumps({"results": result["results"], "paired_model_differences": result["paired_model_differences"]}, indent=2))


if __name__ == "__main__":
    main()
