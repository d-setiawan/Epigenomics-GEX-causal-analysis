#!/usr/bin/env python3
"""CA-BOSS v4.62 cached-world ICA representation uncertainty experiment.

This is a crossed sensitivity experiment, not a production selector.  Refit
held-out-background normalization and ICA, recompute depth-adjusted reporter
assignments, then align unpaired score distributions with the existing joint
sliced-W2 synchronizer. All modes evaluate the same within-assay cell subset;
it is out of bag only for cell-refitted models. Cached-reference and
initialization-only fits have seen those cells. A cached-reference control
separates evaluation-subset variation from refitting the representation.
Resampling is shared within an assay's contexts, never across assays.

``freeze`` writes evidence and its source/input hashes without opening truth.
``score`` checks that freeze and evaluates a fixed majority-plus-other-block-
null rule.  Frequencies are sensitivity summaries, not posterior probabilities.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bootstrap_background_ica_representation_v4 as representation
import diagnose_depth_conditioned_geometry_rescue_v4 as depth
import diagnose_joint_sliced_geometry_rescue_v4 as joint
import diagnose_quantile_geometry_rescue_v4 as geometry
import unpaired_mdcr_ks_baseline_v4 as mdcr

SCHEMA = "ca_boss_v4.representation_bootstrap_diagnostic.v1"
MODES = ("cached_reference_subset", "initialization_only", "cell_resample_only", "both")
CONTEXTS = ("full", "fold0", "fold1")
IDENTITY = ["world_key", "block", "left_modality", "right_modality"]
def _source_hashes() -> dict[str, str]:
    """Bind the loaded local helper closure, including artifact resolvers."""
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == SCRIPT_DIR and path.suffix == ".py":
                paths.add(path)
    return {path.name: geometry._sha256(path) for path in sorted(paths)}


def _reporter_assignment(
    focal: np.ndarray, scores: np.ndarray, observed_depth: np.ndarray
) -> tuple[int | None, float, float]:
    """Require a unique shared winner across an assay's reporter alternatives."""
    if len(scores) < 8:
        return None, 0.0, 0.0
    design = np.column_stack([np.ones(len(scores)), np.log1p(observed_depth)])
    y = np.log1p(focal)
    y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x = scores - design @ np.linalg.lstsq(design, scores, rcond=None)[0]
    denominator = np.linalg.norm(y, axis=0)[:, None] * np.linalg.norm(x, axis=0)
    correlation = np.divide(
        np.abs(y.T @ x), denominator,
        out=np.zeros((y.shape[1], x.shape[1])), where=denominator > 1e-12,
    )
    order = np.argsort(-correlation, axis=1, kind="stable")
    best = correlation[np.arange(len(order)), order[:, 0]]
    runner = correlation[np.arange(len(order)), order[:, 1]]
    if np.any(best <= 1e-12) or np.any(best - runner <= 1e-12):
        return None, float(best.min()), 0.0
    winners = set(order[:, 0].tolist())
    margin = float(np.min((best - runner) / best))
    return (int(next(iter(winners))) if len(winners) == 1 else None,
            float(best.min()), margin)


def _evaluation_indices(n_cells: int, modality: str, seed: int) -> np.ndarray:
    """Use the same evaluation subset across all four modes and three contexts."""
    _, oob = representation._resample_indices(
        n_cells, modality=modality, resample_seed=seed, mode="both")
    return np.flatnonzero(oob)


def _load_world(
    key: str, artifacts: geometry._WorldArtifacts, formal_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Verify every consumed score/archive; join IDs only inside each assay."""
    discovery = mdcr._safe_discovery_dir(depth._formal_discovery(formal_root, key))
    manifest_path = artifacts.mdcr_dir / "frozen_evidence_manifest.json"
    manifest = geometry._verified_manifest(manifest_path, mdcr.SCHEMA)
    hashes: dict[str, str] = {}
    for name in ("component_scores.tsv", "fold_component_scores.tsv"):
        path = artifacts.mdcr_dir / name
        actual = geometry._sha256(path)
        if actual != manifest["output_file_sha256"][name]:
            raise ValueError(f"Changed reference score input: {key}/{name}")
        hashes[str(path.resolve())] = actual
    hashes[str(manifest_path.resolve())] = geometry._sha256(manifest_path)
    hashes[str(artifacts.assignment_path.resolve())] = geometry._sha256(artifacts.assignment_path)
    full = pd.read_csv(artifacts.mdcr_dir / "component_scores.tsv", sep="\t")
    folds = pd.read_csv(artifacts.mdcr_dir / "fold_component_scores.tsv", sep="\t")
    assignments = pd.read_csv(artifacts.assignment_path, sep="\t", keep_default_na=False)
    arrays: dict[str, dict[str, Any]] = {}
    for row in assignments.itertuples(index=False):
        modality = str(row.modality)
        reporters = [x for x in str(row.external_state_reporter_members).split(";") if x]
        if not reporters:
            continue
        archive_path = discovery / f"background_matrix__{modality}.npz"
        actual = geometry._sha256(archive_path)
        if actual != manifest["input_file_sha256"][archive_path.name]:
            raise ValueError(f"Changed background archive: {key}/{modality}")
        hashes[str(archive_path)] = actual
        with np.load(archive_path, allow_pickle=False) as archive:
            counts = np.asarray(archive["matrix"], dtype=float)
            observed_depth = np.asarray(archive["background_depth"], dtype=float)
            cells = list(map(str, archive["cell_ids"]))
        if len(set(cells)) != len(cells):
            raise ValueError("Duplicate within-assay cell IDs")
        focal_path = discovery / f"observed_focal__{modality}.tsv"
        hashes[str(focal_path)] = geometry._sha256(focal_path)
        focal = pd.read_csv(focal_path, sep="\t").set_index("cell", verify_integrity=True)
        if set(focal.index.astype(str)) != set(cells):
            raise ValueError("Focal and background cells do not match")
        if set(focal["modality"].astype(str)) != {modality}:
            raise ValueError("Focal assay identity mismatch")
        if not np.allclose(focal.loc[cells, "observed_background_library_size"], observed_depth):
            raise ValueError("Focal and background depth do not match")
        reference: dict[str, np.ndarray] = {}
        for label, table in [("full", full), *[(f"fold{i}", folds[folds["feature_fold"].eq(i)]) for i in range(2)]]:
            local = table[table["modality"].astype(str).eq(modality)].set_index("cell_id", verify_integrity=True)
            if set(local.index.astype(str)) != set(cells):
                raise ValueError("Reference and archive cells do not match")
            reference[label] = local.loc[cells, [f"component_{i}" for i in range(3)]].to_numpy(dtype=float)
        arrays[modality] = {
            "counts": counts, "depth": observed_depth, "reference": reference,
            "focal": focal.loc[cells, reporters].to_numpy(dtype=float),
            "reporters": reporters, "reference_seed": int(manifest["config"]["random_seed"]),
            "config": mdcr.UnpairedMDCRConfig(**manifest["config"]),
        }
    return arrays, hashes


def _compute_world(
    key: str, artifacts: geometry._WorldArtifacts, formal_root: Path,
    *, replicates: int, random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    arrays, hashes = _load_world(key, artifacts, formal_root)
    block = key.split("__world_seed", 1)[0]
    modalities = sorted(arrays)
    pair_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for mode in MODES:
            for replicate in range(replicates):
                scores: dict[str, dict[str, np.ndarray]] = {label: {} for label in CONTEXTS}
                reporters: dict[str, dict[str, int]] = {label: {} for label in CONTEXTS}
                for modality in modalities:
                    a = arrays[modality]
                    # Pair the two cell-resampling ablations; optimizer seeds
                    # likewise pair initialization-only with combined refits.
                    seed = mdcr._stable_seed(random_seed, f"{key}:{replicate}")
                    evaluation = _evaluation_indices(len(a["counts"]), modality, seed)
                    try:
                        fitted = (None if mode == "cached_reference_subset" else
                            representation.bootstrap_fit_assay_contexts(
                            a["counts"], a["depth"], a["reference"],
                            modality=modality, reference_seed=a["reference_seed"],
                            resample_seed=seed, initialization_seed=seed,
                            mode=mode, config=a["config"],
                        ))
                    except representation.RepresentationFitError as error:
                        for context in CONTEXTS:
                            fit_rows.append({"world_key": key, "mode": mode, "replicate": replicate,
                                             "modality": modality, "context": context, "fit_ok": False,
                                             "failure": str(error)})
                        continue
                    if fitted is not None and mode in ("both", "cell_resample_only"):
                        if not np.array_equal(evaluation, np.flatnonzero(fitted.oob_mask)):
                            raise ValueError("Refit and evaluation bootstrap samples disagree")
                    for context in CONTEXTS:
                        current = (a["reference"][context] if fitted is None else fitted.scores[context])[evaluation]
                        component, association, margin = _reporter_assignment(
                            a["focal"][evaluation], current, a["depth"][evaluation],
                        )
                        alignment = ({"reference_scores_reused": True} if fitted is None
                                     else fitted.contexts[context].__dict__)
                        fit_rows.append({
                            "world_key": key, "mode": mode, "replicate": replicate,
                            "modality": modality, "context": context, "fit_ok": True,
                            "evaluation_cells": len(evaluation), "reporter_component": component,
                            "representation_refitted": fitted is not None,
                            "evaluation_cells_out_of_bag_for_ica": mode in ("both", "cell_resample_only"),
                            "reporter_absolute_partial_correlation": association,
                            "reporter_relative_margin": margin,
                            "alignment_diagnostics_json": json.dumps(alignment, sort_keys=True),
                            "failure": "",
                        })
                        if component is not None:
                            scores[context][modality] = current
                            reporters[context][modality] = component
                for context in CONTEXTS:
                    current = scores[context]
                    fingerprints = {m: joint._projection_quantiles(x, copula_only=False)
                                    for m, x in current.items()}
                    matrices = {(a, b): joint._precomputed_cost_matrix(fingerprints[a], fingerprints[b])
                                for a, b in combinations(sorted(current), 2)}
                    table = geometry._normalized_cost_table(current, matrices) if matrices else None
                    solution = (geometry.synchronization.solve_multiway_permutation_synchronization(table, sorted(current))
                                if table is not None else None)
                    for left, right in combinations(modalities, 2):
                        row: dict[str, Any] = {
                            "world_key": key, "block": block, "left_modality": left,
                            "right_modality": right, "mode": mode, "replicate": replicate,
                            "context": context, "eligible": False, "local_mutual": False,
                            "global_coassigned": False, "reporter_cost_ratio": np.nan,
                            "global_exclusion_margin": 0.0,
                        }
                        if left in current and right in current and solution is not None:
                            li, ri = reporters[context][left], reporters[context][right]
                            cost, _, mutual = geometry._local_diagnostics(matrices[(left, right)], li, ri)
                            global_row = geometry.synchronization.reporter_pair_diagnostics(solution, table, left, right, li, ri)
                            row.update(eligible=True, local_mutual=mutual,
                                       global_coassigned=bool(global_row["coassigned_in_best_solution"]),
                                       reporter_cost_ratio=cost,
                                       global_exclusion_margin=float(global_row["pair_exclusion_relative_margin"]))
                        pair_rows.append(row)
    return pd.DataFrame(pair_rows), pd.DataFrame(fit_rows), hashes


def _aggregate_replicates(frame: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, group in frame.groupby([*IDENTITY, "mode"], sort=True):
        expected = {(r, c) for r in range(replicates) for c in CONTEXTS}
        observed = list(zip(group["replicate"].astype(int), group["context"].astype(str)))
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("Missing or duplicate replicate/context evidence")
        row = dict(zip([*IDENTITY, "mode"], identity))
        for column in ("eligible", "local_mutual", "global_coassigned"):
            values = group.groupby("replicate")[column].sum()
            row[f"{column}_all_context_count"] = int(values.eq(3).sum())
            row[f"{column}_all_context_frequency"] = float(values.eq(3).mean())
            row[f"{column}_mean_context_frequency"] = float(values.mean() / 3)
        row["replicates"] = replicates
        rows.append(row)
    return pd.DataFrame(rows)


def freeze(*, output_root: Path, formal_root: Path, out_dir: Path,
           worlds_per_block: int, random_seed: int, replicates: int, max_workers: int) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    if replicates < 1 or max_workers < 1 or worlds_per_block < 1:
        raise ValueError("Replicates, workers, and worlds per block must be positive")
    output_root = output_root.resolve(strict=True)
    formal_root = formal_root.resolve(strict=True)
    frames, _, parents = geometry._discover_v453(output_root)
    selected = depth._sample_keys(frames, worlds_per_block=worlds_per_block,
                                  random_seed=random_seed, excluded=set())
    keys = sorted(k for group in selected.values() for k in group)
    artifacts = geometry._resolve_artifacts(output_root, keys)
    # A launch contract is persisted before any estimator fit or outcome read.
    out_dir.mkdir(parents=True)
    contract = {"schema_version": SCHEMA, "selected_world_keys": keys,
                "worlds_per_block": worlds_per_block, "random_seed": random_seed,
                "replicates": replicates, "modes": list(MODES),
                "evaluation_rule": "same within-assay subset across modes; OOB only for cell-resampling fits",
                "sources": _source_hashes(),
                "selection_rule": "all-context frequency > 0.5 and strictly > maximum other-block null frequency",
                "positive_labels_used_for_selection": False, "fresh_locked_certification": False}
    (out_dir / "launch_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_compute_world, key, artifacts[key], formal_root,
                               replicates=replicates, random_seed=random_seed): key for key in keys}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f"Completed representation refits: {len(results)}/{len(keys)}", flush=True)
    pairs = pd.concat([results[k][0] for k in keys], ignore_index=True)
    fits = pd.concat([results[k][1] for k in keys], ignore_index=True)
    evidence = _aggregate_replicates(pairs, replicates)
    for name, table in (("replicate_pair_evidence.tsv", pairs), ("fit_diagnostics.tsv", fits),
                        ("representation_pair_evidence.tsv", evidence)):
        table.to_csv(out_dir / name, sep="\t", index=False)
    manifest = {
        **contract, "phase": "pretruth_evidence_only", "world_count": len(keys),
        "input_sha256_by_world": {k: results[k][2] for k in keys},
        "parent_payloads": {k: p["manifest_payload_sha256"] for k, p in parents.items()},
        "information_boundary": {"truth_read": False, "existing_worlds_only": True,
            "normalization_and_ica_refitted": True, "reporter_assignments_recomputed": True,
            "cross_assay_cell_correspondence_used": False, "paired_cells_used": False,
            "simulator_latent_state_used": False, "external_data_used": False,
            "focal_features_used_in_embedding_fit": False,
            "reference_fit_used_all_cells": True,
            "out_of_bag_labels_are_coordinate_alignment_only": True},
        "limitations": ["Fixed rank-three log-linear count approximation remains.",
            "Reference scores already used all cells; this is no independent OOB validation of the full pipeline.",
            "Existing assay-local reporter feature identities are fixed, while component assignments are refitted.",
            "Full and two feature folds share cells and are dependent.",
            "Frequencies are Monte Carlo sensitivity scores, not calibrated posterior probabilities.",
            "All modes evaluate identical rows within an assay/replicate; reference and initialization fits have seen them.",
            "Cached-reference subset is a control for evaluation sampling, with no ICA refit.",
            "OOB distribution geometry is not the ordinary in-bag bootstrap of the complete estimator.",
            "Alignment to a reference relabels coordinates and does not establish one latent mode."],
        "published_inspiration": {"ICASSO": "https://doi.org/10.1016/j.neuroimage.2004.03.027"},
        "output_sha256": {name: geometry._sha256(out_dir / name) for name in
            ("launch_contract.json", "replicate_pair_evidence.tsv", "fit_diagnostics.tsv", "representation_pair_evidence.tsv")},
    }
    manifest["manifest_payload_sha256"] = geometry._canonical_sha256(manifest)
    (out_dir / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def score(*, output_root: Path, formal_root: Path, evidence_dir: Path) -> dict[str, Any]:
    if (evidence_dir / "diagnostic_summary.json").exists() or list(evidence_dir.glob("scored_*.tsv")):
        raise FileExistsError("Refusing to overwrite existing diagnostic scores")
    manifest = geometry._verified_manifest(evidence_dir / "evidence_manifest.json", SCHEMA)
    for name, expected in manifest["sources"].items():
        if geometry._sha256(SCRIPT_DIR / name) != expected:
            raise ValueError(f"Changed evidence source: {name}")
    for name, expected in manifest["output_sha256"].items():
        if geometry._sha256(evidence_dir / name) != expected:
            raise ValueError(f"Changed evidence table: {name}")
    for inputs in manifest["input_sha256_by_world"].values():
        for name, expected in inputs.items():
            if geometry._sha256(Path(name)) != expected:
                raise ValueError(f"Changed evidence input: {name}")
    table = pd.read_csv(evidence_dir / "representation_pair_evidence.tsv", sep="\t")
    truth_hashes = {}
    for key in manifest["selected_world_keys"]:
        block, seed = key.split("__world_seed", 1)
        path = formal_root / block / f"world_seed{int(seed)}" / "scorer/truth_typed_edges.tsv"
        truth_hashes[str(path.resolve())] = geometry._sha256(path)
    truth = depth._load_truth(formal_root, manifest["selected_world_keys"])
    baseline, baseline_payloads = depth._v458_baseline(output_root)
    labels = geometry._truth_labels(table, truth)
    table["truth_pair"] = labels
    table["represented_by_v458_incremental_candidates"] = [
        geometry._pair(r.left_modality, r.right_modality) in baseline.get(str(r.world_key), set())
        for r in table.itertuples(index=False)]
    summaries = []
    scored_tables = {}
    for mode in MODES:
        for metric in ("local_mutual", "global_coassigned"):
            frequency = f"{metric}_all_context_frequency"
            current = table[table["mode"].eq(mode)].copy()
            current["selected_diagnostic_proposal"] = False
            for block in sorted(current["block"].unique()):
                null = current[~current["block"].eq(block) & ~current["truth_pair"]]
                if null.empty:
                    raise ValueError("No other-block nulls for diagnostic calibration")
                floor = float(null[frequency].max())
                target = current["block"].eq(block)
                current.loc[target, "null_frequency_upper_envelope"] = floor
                current.loc[target, "selected_diagnostic_proposal"] = (
                    current.loc[target, frequency].gt(0.5) & current.loc[target, frequency].gt(floor))
            selected = current["selected_diagnostic_proposal"]
            summaries.append({"mode": mode, "metric": metric,
                "diagnostic_true_assay_pairs": int((selected & current["truth_pair"]).sum()),
                "diagnostic_false_assay_pairs": int((selected & ~current["truth_pair"]).sum()),
                "truth_assay_pair_count": int(current["truth_pair"].sum()),
                "mean_true_frequency": float(current.loc[current["truth_pair"], frequency].mean()),
                "mean_null_frequency": float(current.loc[~current["truth_pair"], frequency].mean()),
                "maximum_null_frequency": float(current.loc[~current["truth_pair"], frequency].max())})
            scored_tables[f"scored_{mode}_{metric}.tsv"] = current
    for name, current in scored_tables.items():
        current.to_csv(evidence_dir / name, sep="\t", index=False)
    result = {"schema_version": SCHEMA, "evidence_payload_sha256": manifest["manifest_payload_sha256"],
        "world_count": manifest["world_count"], "replicates": manifest["replicates"],
        "new_worlds_created": False, "fresh_locked_certification": False,
        "interpretation": "representation sensitivity and assay-pair proposal diagnostic; not the four-family production benchmark",
        "v458_baseline_payloads": baseline_payloads, "results": summaries,
        "truth_input_sha256": truth_hashes,
        "source_sha256": _source_hashes(),
        "output_sha256": {name: geometry._sha256(evidence_dir / name) for name in scored_tables}}
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
    parser.add_argument("--random-seed", type=int, default=462000)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "freeze":
        result = freeze(output_root=args.output_root, formal_root=args.formal_root, out_dir=args.out_dir,
                        worlds_per_block=args.worlds_per_block, random_seed=args.random_seed,
                        replicates=args.replicates, max_workers=args.max_workers)
        print(json.dumps({k: result[k] for k in ("world_count", "replicates", "manifest_payload_sha256")}, indent=2))
    else:
        print(json.dumps(score(output_root=args.output_root, formal_root=args.formal_root,
                               evidence_dir=args.out_dir), indent=2))


if __name__ == "__main__":
    main()
