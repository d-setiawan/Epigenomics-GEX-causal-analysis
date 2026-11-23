#!/usr/bin/env python3
"""Freeze then score the v4.59 cached-world campaign in separate phases.

The complete prediction tables and source/input hashes are frozen before this
scorer opens target partitions. Reusing worlds makes this retrospective even
though fitting excludes each target block. Exact pair/direction stay stress
metrics. The common benchmark aggregation API supplies all four families.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import bootstrap_dual_null_disagreement_rescue_v4 as calibration
import summarize_cross_assay_benchmark_panel_v4 as aggregation

core = calibration.parent_scorer.core_scorer
combined = calibration.parent_scorer.combined_scorer
geometry = calibration.geometry
PRETRUTH_SCHEMA = "ca_boss_v4.dual_null_disagreement_campaign_pretruth.v1"
SCORE_SCHEMA = "ca_boss_v4.bootstrap_dual_null_disagreement_rescue.scorer.v1"
CAMPAIGN_SCHEMA = "ca_boss_v4.dual_null_disagreement_campaign_summary.v1"
BRIDGE_PANELS = ("panel09", "panel10", "panel11", "panel12", "remaining57")
TARGET_SUFFIX = {
    "development_34001_34016": "dev340", "development_47001_47016": "dev470",
    "train_41001_41024": "train410", "train_46001_46024": "train460",
    "train_49001_49024": "train490",
}


def normalize_predictions(core_frame: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    core_pairs = set()
    for row in core_frame.itertuples(index=False):
        if calibration.as_bool(row.hypothesis_selected):
            core_pairs.update(calibration.parser._parse_pair_payload(row.candidate_assay_pairs_json))
    incremental_pairs = set()
    for row in candidates.itertuples(index=False):
        incremental_pairs.update(calibration.parser._parse_pair_payload(row.candidate_assay_pairs_json))
    if core_pairs & incremental_pairs:
        raise ValueError("Incremental assay alternatives duplicate the retained core")
    incremental = candidates.rename(columns={"dual_null_hyperedge_id": "synchronized_hyperedge_id"}).copy()
    predictions = combined.normalize_predictions(core_frame, incremental)
    ids = set(candidates["dual_null_hyperedge_id"].astype(str))
    predictions.loc[predictions["hyperedge_id"].astype(str).isin(ids), "prediction_source"] = "v459_dual_null_disagreement_rescue"
    return predictions


def prepare_campaign(*, outputs_root: Path, worlds_root: Path, calibration_root: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    outputs_root = outputs_root.resolve(strict=True)
    worlds_root = worlds_root.resolve(strict=True)
    calibration_root = calibration_root.resolve(strict=True)
    calibrations = {}
    for block, suffix in TARGET_SUFFIX.items():
        directory = calibration_root / f"target_{suffix}"
        manifest, candidates, _ = calibration.verify_calibration(directory)
        if manifest["target_block"] != block:
            raise ValueError("Child calibration block mismatch")
        calibrations[block] = (directory, manifest, candidates)
    bridge_dirs = {}
    for panel in BRIDGE_PANELS:
        root = outputs_root / (
            "component_conditioned_bridge_v444_source_current_remaining57_retrospective"
            if panel == "remaining57" else
            f"component_conditioned_bridge_v444_source_current_dev_replay_{panel}"
        )
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if path.name in bridge_dirs:
                raise ValueError(f"Duplicate bridge world: {path.name}")
            bridge_dirs[path.name] = path
    expected = {key for _, manifest, _ in calibrations.values() for key in manifest["target_world_keys"]}
    if set(bridge_dirs) != expected:
        raise ValueError("Bridge worlds and frozen calibration worlds differ")
    predictions = {}
    worlds = []
    inputs = {}
    for key, directory in sorted(bridge_dirs.items()):
        bridge, provenance = core._verify_bridge(directory)
        block, seed = str(provenance["block"]), int(provenance["seed"])
        if key != f"{block}__world_seed{seed}":
            raise ValueError("Bridge world identity mismatch")
        cp, cm, candidates = calibrations[block]
        selected = candidates[candidates["world_key"].astype(str).eq(key)]
        core_frame = pd.read_csv(bridge / "stable_cross_assay_hyperedges.tsv", sep="\t", keep_default_na=False)
        predictions[key] = normalize_predictions(core_frame, selected)
        inputs[str(bridge / "provenance.json")] = geometry._sha256(bridge / "provenance.json")
        inputs[str(bridge / "stable_cross_assay_hyperedges.tsv")] = geometry._sha256(bridge / "stable_cross_assay_hyperedges.tsv")
        inputs[str(cp / "calibration_manifest.json")] = geometry._sha256(cp / "calibration_manifest.json")
        worlds.append({"world_key": key, "block": block, "seed": seed, "bridge_dir": str(bridge),
                       "calibration_dir": str(cp.resolve()), "calibration_payload_sha256": cm["manifest_payload_sha256"],
                       "scorer_dir": str((worlds_root / block / f"world_seed{seed}" / "scorer").resolve())})
    out_dir.mkdir(parents=True)
    output = {}
    for key, frame in predictions.items():
        path = out_dir / f"{key}.predictions.tsv"
        frame.to_csv(path, sep="\t", index=False)
        output[path.name] = geometry._sha256(path)
    modules = (Path(__file__).resolve(), Path(calibration.__file__).resolve(), Path(core.__file__).resolve(),
               Path(combined.__file__).resolve(), Path(aggregation.__file__).resolve(),
               Path(calibration.parent_scorer.benchmark.__file__).resolve())
    manifest = {"schema_version": PRETRUTH_SCHEMA, "algorithm": calibration.ALGORITHM,
                "world_count": len(worlds), "worlds": worlds, "all_predictions_frozen_before_target_truth": True,
                "target_truth_opened": False, "target_block_truth_used_for_fit_or_selection": False,
                "retrospective_cached_world_exploration": True, "fresh_locked_certification": False,
                "new_worlds_or_datasets_created": False,
                "source_sha256": {p.name: geometry._sha256(p) for p in modules},
                "input_sha256": inputs, "output_sha256": output}
    manifest["manifest_payload_sha256"] = geometry._canonical_sha256(manifest)
    (out_dir / "pretruth_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_pretruth(directory: Path) -> dict[str, Any]:
    manifest = geometry._verified_manifest(directory / "pretruth_manifest.json", PRETRUTH_SCHEMA)
    calibration.verify_files(directory, manifest)
    if manifest.get("target_truth_opened") is not False or manifest.get("all_predictions_frozen_before_target_truth") is not True:
        raise ValueError("Invalid campaign pretruth boundary")
    for name, expected in manifest["input_sha256"].items():
        path = Path(name)
        if path.is_symlink() or geometry._sha256(path) != expected:
            raise ValueError(f"Changed pretruth input: {name}")
    for world in manifest["worlds"]:
        core._verify_bridge(Path(world["bridge_dir"]))
    for directory_name in sorted({world["calibration_dir"] for world in manifest["worlds"]}):
        calibration.verify_calibration(Path(directory_name))
    return manifest


def score_campaign(*, pretruth_dir: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    manifest = verify_pretruth(pretruth_dir)
    # All source, prediction, bridge, and calibration hashes verified above.
    out_dir.mkdir(parents=True)
    summaries, scores, rows = [], [], []
    truth_inputs, outputs = {}, {}
    for world in manifest["worlds"]:
        key = world["world_key"]
        prediction = pd.read_csv(pretruth_dir / f"{key}.predictions.tsv", sep="\t", keep_default_na=False)
        truth_path = Path(world["scorer_dir"]) / "truth_typed_edges.tsv"
        roles_path = Path(world["scorer_dir"]) / "role_assignments.tsv"
        if any(p.is_symlink() or not p.is_file() for p in (truth_path, roles_path)):
            raise ValueError("Invalid scorer partition")
        truths = core._truth_hyperedges(truth_path)
        roles = pd.read_csv(roles_path, sep="\t", keep_default_na=False)
        sentinels = set(roles.loc[roles["role"].isin({"N1", "N2"}), "node"].astype(str))
        scored, metrics = combined.score_predictions(prediction, truths, sentinels)
        path = out_dir / f"{key}.scores.tsv"
        scored.to_csv(path, sep="\t", index=False)
        outputs[path.name] = geometry._sha256(path)
        truth_inputs.update({str(p): geometry._sha256(p) for p in (truth_path, roles_path)})
        summary = {"schema_version": SCORE_SCHEMA, "block": world["block"], "seed": world["seed"],
                   "benchmark_metrics": metrics, "truth_opened_after_all_campaign_predictions_frozen": True,
                   "target_block_truth_used_for_fit_or_selection": False, "fresh_locked_certification": False,
                   "pretruth_manifest_payload_sha256": manifest["manifest_payload_sha256"],
                   "output_sha256": {path.name: outputs[path.name]}}
        summary["summary_payload_sha256"] = geometry._canonical_sha256(summary)
        summary_path = out_dir / f"{key}.score_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        outputs[summary_path.name] = geometry._sha256(summary_path)
        summaries.append(summary)
        scores.append(scored)
        rows.append({"world_key": key, "block": world["block"], "seed": world["seed"],
                     **metrics["shared_state_hyperedge"],
                     "assay_participation_precision": metrics["assay_participation_and_sentinel_false_positive"]["assay_participation_precision"],
                     "assay_participation_recall": metrics["assay_participation_and_sentinel_false_positive"]["assay_participation_recall"],
                     "sentinel_false_positive_rate": metrics["assay_participation_and_sentinel_false_positive"]["sentinel_false_positive_rate"],
                     "equivalence_class_coverage": metrics["equivalence_class"]["equivalence_class_coverage"],
                     "average_ambiguity_set_size": metrics["equivalence_class"]["average_ambiguity_set_size"],
                     "exact_pair_true_positives": metrics["identifiability_stress_test"]["exact_cross_assay_pair_true_positives"]})
    metrics = aggregation.aggregate_panel_metrics(summaries, scores)
    table = out_dir / "campaign_world_metrics.tsv"
    pd.DataFrame(rows).to_csv(table, sep="\t", index=False)
    outputs[table.name] = geometry._sha256(table)
    campaign = {"schema_version": CAMPAIGN_SCHEMA, "algorithm": calibration.ALGORITHM,
                "campaign_world_count": len(rows), "benchmark_metrics": metrics,
                "all_predictions_frozen_before_target_truth": True,
                "target_block_truth_used_for_fit_or_selection": False,
                "other_block_simulation_truth_used_for_null_calibration": True,
                "other_block_simulation_truth_used_for_supervised_training": False,
                "retrospective_cached_world_exploration": True, "fresh_locked_certification": False,
                "new_worlds_or_datasets_created": False, "candidate_alternatives_expanded_as_edges": False,
                "exact_pair_and_direction_interpretation": "identifiability stress test, not an expected success criterion",
                "goal_completion_claimed": False,
                "pretruth_manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "pretruth_manifest_file_sha256": geometry._sha256(pretruth_dir / "pretruth_manifest.json"),
                "source_sha256": manifest["source_sha256"], "truth_input_sha256": truth_inputs,
                "output_sha256": outputs}
    campaign["summary_payload_sha256"] = geometry._canonical_sha256(campaign)
    (out_dir / "campaign_summary.json").write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    return campaign


def main() -> None:
    arg = argparse.ArgumentParser(description=__doc__)
    sub = arg.add_subparsers(dest="phase", required=True)
    prepare = sub.add_parser("prepare")
    for name in ("outputs-root", "worlds-root", "calibration-root", "out-dir"):
        prepare.add_argument("--" + name, type=Path, required=True)
    score = sub.add_parser("score")
    for name in ("pretruth-dir", "out-dir"):
        score.add_argument("--" + name, type=Path, required=True)
    args = vars(arg.parse_args())
    phase = args.pop("phase")
    result = prepare_campaign(**args) if phase == "prepare" else score_campaign(**args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
