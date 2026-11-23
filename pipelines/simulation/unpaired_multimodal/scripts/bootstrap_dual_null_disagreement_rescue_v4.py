#!/usr/bin/env python3
"""Truth-blind v4.59 child of v4.58, using cached dual-null evidence.

The two-context stratum must beat both its own empirical null and the global
worst-null bootstrap count. All connected alternatives stay one ambiguity
hypothesis. The retrospective rule does not claim fresh certification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import diagnose_dual_null_disagreement_stability_v4 as diagnostic
import score_bootstrap_joint_context_stratified_stability_rescue_v4 as parent_scorer

parent = diagnostic.v458
geometry = parent.diagnostic.v457.joint.geometry
parser = parent.diagnostic.v457.v455.v454.v453.v449
pair_tools = parent.diagnostic.v457.v455.v454
as_bool = parent_scorer.combined_scorer._as_bool
SCHEMA = "ca_boss_v4.bootstrap_dual_null_disagreement_rescue.v1"
ALGORITHM = "CA-BOSS v4.59 dual-null disagreement stability rescue"
FIXED_SELECTION_RULE = diagnostic.FIXED_SELECTION_RULE + (
    "_union_with_verified_v458_merge_connected_conflicts_as_single_unexpanded_"
    "shared_state_equivalence_hypotheses"
)
CANDIDATE_COLUMNS = [
    "dual_null_hyperedge_id", "world_key", "block", "seed", "modalities",
    "n_modalities", "candidate_assay_pairs_json", "candidate_assay_pair_count",
    "candidate_cross_assay_feature_pairs_json", "candidate_cross_assay_feature_pair_count",
    "candidate_pair_selection_audit_json", "selection_sources", "selection_status", "hypothesis_selected",
    "hypothesis_type", "assay_participation_is_set_valued", "exact_assay_participation_claimed",
    "candidate_pairs_may_be_expanded_as_edges", "within_assay_direct_structure_claimed",
    "exact_cross_assay_pair_claimed", "cross_assay_direction_claimed", "fixed_selection_rule",
]


def verify_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    for key, root in (("source_sha256", SCRIPT_DIR), ("output_sha256", directory)):
        if not manifest.get(key):
            raise ValueError(f"Missing {key}")
        for name, expected in manifest[key].items():
            path = root / name
            if path.is_symlink() or not path.is_file() or geometry._sha256(path) != expected:
                raise ValueError(f"Changed bound artifact: {path}")


def _verify_diagnostic(directory: Path, *, block: str, evidence_path: Path,
                       evidence_manifest: Mapping[str, Any], v457_path: Path,
                       v457_manifest: Mapping[str, Any], v458_path: Path,
                       v458_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = geometry._verified_manifest(directory / "diagnostic_manifest.json", diagnostic.SCHEMA)
    verify_files(directory, manifest)
    if manifest.get("target_block") != block or manifest.get("fixed_selection_rule") != diagnostic.FIXED_SELECTION_RULE:
        raise ValueError("Dual-null diagnostic target or rule changed")
    inputs = {
        "joint_evidence_manifest.json": evidence_path / "evidence_manifest.json",
        "v457_diagnostic_manifest.json": v457_path / "diagnostic_manifest.json",
        "v458_diagnostic_manifest.json": v458_path / "diagnostic_manifest.json",
        "v458_pair_audit.tsv": v458_path / "joint_context_stratified_pair_audit.tsv",
        "v458_null_audit.tsv": v458_path / "joint_context_stratified_null_audit.tsv",
    }
    if manifest["input_sha256"] != {k: geometry._sha256(p) for k, p in inputs.items()}:
        raise ValueError("Dual-null input bindings changed")
    parents = {"joint_evidence": evidence_manifest, "v457_diagnostic": v457_manifest,
               "v458_diagnostic": v458_manifest}
    if manifest["parent_manifest_payload_sha256"] != {
        key: value["manifest_payload_sha256"] for key, value in parents.items()
    }:
        raise ValueError("Dual-null parent bindings changed")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True or manifest.get("fresh_locked_certification") is not False:
        raise ValueError("Dual-null pretruth or retrospective contract changed")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("target_block_truth_opened") is not False or boundary.get("target_world_truth_used_for_fit_or_selection") is not False:
        raise ValueError("Dual-null target truth boundary changed")
    null = pd.read_csv(directory / "dual_null_disagreement_null_audit.tsv", sep="\t", keep_default_na=False)
    parent_null = pd.read_csv(v458_path / "joint_context_stratified_null_audit.tsv", sep="\t", keep_default_na=False)
    try:
        pd.testing.assert_frame_equal(null[parent_null.columns], parent_null, check_dtype=False, rtol=1e-12, atol=1e-12)
    except (AssertionError, KeyError) as error:
        raise ValueError("Dual-null null rows differ from verified parent") from error
    if null["calibration_candidate_block"].astype(str).eq(block).any() or null["world_key"].astype(str).str.startswith(block + "__").any():
        raise ValueError("Target block occurs in null calibration")
    counts = pd.to_numeric(null[parent.diagnostic.LOCAL_CONTEXT_COLUMN], errors="raise")
    if (~counts.between(0, 3) | counts.mod(1).ne(0)).any():
        raise ValueError("Invalid null context counts")
    comparable = null[counts.eq(2)]
    if comparable.empty:
        raise ValueError("No two-context null")
    worst = int(comparable["bootstrap_selection_count"].astype(int).max())
    global_worst = int(v457_manifest["empirical_null_maximum_selection_count"])
    if worst != manifest["two_context_stratum_null_maximum_selection_count"] or global_worst != manifest["unstratified_global_null_maximum_selection_count"]:
        raise ValueError("Dual-null ceilings changed")
    pair = pd.read_csv(directory / "dual_null_disagreement_pair_audit.tsv", sep="\t", keep_default_na=False)
    parent_pair = pd.read_csv(v458_path / "joint_context_stratified_pair_audit.tsv", sep="\t", keep_default_na=False)
    try:
        pd.testing.assert_frame_equal(pair[parent_pair.columns], parent_pair, check_dtype=False, rtol=1e-12, atol=1e-12)
    except (AssertionError, KeyError) as error:
        raise ValueError("Dual-null candidate rows differ from verified parent") from error
    if set(pair["block"].astype(str)) != {block} or set(pair["world_key"].astype(str)) != set(manifest["target_world_keys"]):
        raise ValueError("Dual-null target identities changed")
    for row in pair.to_dict("records"):
        count = float(row[parent.diagnostic.LOCAL_CONTEXT_COLUMN])
        bootstrap = float(row["bootstrap_selection_count"])
        if not count.is_integer() or not 0 <= count <= 3 or not bootstrap.is_integer() or not 0 <= bootstrap <= 100:
            raise ValueError("Invalid candidate context or bootstrap count")
        selected, probability = diagnostic._selection_decision(
            nominal_new_proposal=as_bool(row["nominal_joint_proposal_new_relative_to_v455"]),
            local_context_count=int(count), candidate_count=int(bootstrap),
            stratum_null_count=worst, global_null_count=global_worst,
        )
        if selected != as_bool(row["selected_by_dual_null_disagreement_stability"]) or abs(probability - float(row["posterior_probability_exceeds_two_context_stratum_null"])) > 1e-12:
            raise ValueError("Dual-null selection does not recompute")
    return manifest, pair


def union_candidates(baseline: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline.to_dict("records"):
        sources[str(row["world_key"])].append({
            "pairs": parser._parse_pair_payload(row["candidate_assay_pairs_json"]),
            "features": parser._parse_feature_pair_payload(row["candidate_cross_assay_feature_pairs_json"]),
            "audit": json.loads(row["candidate_pair_selection_audit_json"]),
            "source": "target_safe_v458", "block": row["block"], "seed": int(row["seed"]),
        })
    for row in selected.to_dict("records"):
        sources[str(row["world_key"])].append({
            "pairs": {pair_tools._pair(row["left_modality"], row["right_modality"])},
            "features": parser._parse_feature_pair_payload(row["candidate_cross_assay_feature_pairs_json"]),
            "audit": [{k: row[k] for k in (
                "left_modality", "right_modality", "bootstrap_selection_count",
                parent.diagnostic.LOCAL_CONTEXT_COLUMN,
                "two_context_stratum_null_maximum_selection_count",
                "unstratified_global_null_maximum_selection_count",
                "posterior_probability_exceeds_two_context_stratum_null") }],
            "source": "v459_dual_null_disagreement", "block": row["block"], "seed": int(row["seed"]),
        })
    output = []
    for world_key, world_sources in sorted(sources.items()):
        pairs = set().union(*(source["pairs"] for source in world_sources))
        for index, component in enumerate(pair_tools.v448.v447._connected_pair_components(pairs), 1):
            members = [source for source in world_sources if source["pairs"] & component]
            features = set().union(*(source["features"] for source in members))
            modalities = sorted(set().union(*(set(pair) for pair in component)))
            output.append({
                "dual_null_hyperedge_id": f"dual_null_shared_state_{index:03d}",
                "world_key": world_key, "block": members[0]["block"], "seed": members[0]["seed"],
                "modalities": ";".join(modalities), "n_modalities": len(modalities),
                "candidate_assay_pairs_json": json.dumps([list(pair_tools._pair_tuple(p)) for p in sorted(component, key=pair_tools._pair_tuple)]),
                "candidate_assay_pair_count": len(component),
                "candidate_cross_assay_feature_pairs_json": json.dumps(sorted(features)),
                "candidate_cross_assay_feature_pair_count": len(features),
                "candidate_pair_selection_audit_json": json.dumps([{"source": m["source"], "evidence": m["audit"]} for m in members], sort_keys=True),
                "selection_sources": ";".join(sorted({m["source"] for m in members})),
                "selection_status": "selected_as_dual_null_ambiguity" if len(component) != 1 else "selected_as_dual_null_point_hypothesis",
                "hypothesis_selected": True, "hypothesis_type": "cross_assay_shared_state_equivalence",
                "assay_participation_is_set_valued": len(component) != 1,
                "exact_assay_participation_claimed": len(component) == 1,
                "candidate_pairs_may_be_expanded_as_edges": False,
                "within_assay_direct_structure_claimed": False,
                "exact_cross_assay_pair_claimed": False, "cross_assay_direction_claimed": False,
                "fixed_selection_rule": FIXED_SELECTION_RULE,
            })
    return pd.DataFrame(output, columns=CANDIDATE_COLUMNS)


def calibrate_target_block(*, evidence_dir: Path, v455_calibration_dir: Path,
                           v457_diagnostic_dir: Path, v458_diagnostic_dir: Path,
                           v458_calibration_dir: Path, diagnostic_dir: Path,
                           target_block: str, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    ep, em, evidence = parent.diagnostic.v457._verified_joint_evidence(evidence_dir)
    target = evidence[evidence["block"].astype(str).eq(target_block)]
    keys = sorted(target["world_key"].astype(str).unique())
    if not keys:
        raise ValueError("No target worlds")
    bp, bm, _ = parent.diagnostic.v457._verified_v455_baseline(v455_calibration_dir, target_block=target_block, target_world_keys=keys)
    p7, m7, _, _ = parent.diagnostic._verify_v457_diagnostic(v457_diagnostic_dir, target_block=target_block, evidence_path=ep, evidence_manifest=em)
    p8, m8, _ = parent._verify_diagnostic(v458_diagnostic_dir, target_block=target_block, evidence_path=ep, evidence_manifest=em, v457_path=p7, v457_manifest=m7)
    cp, cm, _, _ = parent_scorer._verify_calibration(
        v458_calibration_dir, bridge_provenance={"block": target_block, "seed": int(target["seed"].iloc[0])},
        evidence_path=ep, evidence_manifest=em, v455_path=bp, v455_manifest=bm,
        v457_path=p7, v457_manifest=m7, diagnostic_path=p8, diagnostic_manifest=m8)
    dp = diagnostic_dir.resolve(strict=True)
    dm, pair = _verify_diagnostic(dp, block=target_block, evidence_path=ep, evidence_manifest=em,
        v457_path=p7, v457_manifest=m7, v458_path=p8, v458_manifest=m8)
    baseline = pd.read_csv(cp / "bootstrap_joint_context_stability_candidates.tsv", sep="\t", keep_default_na=False)
    if set(baseline["world_key"].astype(str)) - set(keys) or set(pair["world_key"].astype(str)) != set(keys):
        raise ValueError("Baseline or diagnostic crosses target world set")
    selected = pair[pair["selected_by_dual_null_disagreement_stability"].map(as_bool)].copy()
    baseline_pairs = {(str(r.world_key), p) for r in baseline.itertuples(index=False)
                      for p in parser._parse_pair_payload(r.candidate_assay_pairs_json)}
    if any((str(r.world_key), pair_tools._pair(r.left_modality, r.right_modality)) in baseline_pairs for r in selected.itertuples(index=False)):
        raise ValueError("Dual-null proposal already present in v4.58")
    candidates = union_candidates(baseline, selected)
    out_dir.mkdir(parents=True)
    candidates.to_csv(out_dir / "dual_null_candidates.tsv", sep="\t", index=False)
    pair.to_csv(out_dir / "dual_null_pair_audit.tsv", sep="\t", index=False)
    inputs = {"joint_evidence": ep / "evidence_manifest.json", "v455": bp / "calibration_manifest.json",
              "v457_diagnostic": p7 / "diagnostic_manifest.json", "v458_diagnostic": p8 / "diagnostic_manifest.json",
              "v458_calibration": cp / "calibration_manifest.json", "dual_null_diagnostic": dp / "diagnostic_manifest.json"}
    manifest = {
        "schema_version": SCHEMA, "algorithm": ALGORITHM, "fixed_selection_rule": FIXED_SELECTION_RULE,
        "target_block": target_block, "target_world_keys": keys, "target_world_count": len(keys),
        "newly_selected_v459_assay_pairs": len(selected), "selected_union_hyperedges": len(candidates),
        "required_new_pair_local_context_count": 2, "bootstrap_replicates": 100,
        "minimum_selection_frequency": 0.5, "minimum_selection_frequency_comparison": "strictly_greater_than",
        "posterior_dominance_threshold": 0.975,
        "posterior_interpretation": "working_beta_comparison_not_an_independent_trial_posterior",
        "same_stratum_null_ceiling": dm["two_context_stratum_null_maximum_selection_count"],
        "global_null_ceiling": dm["unstratified_global_null_maximum_selection_count"],
        "target_block_truth_used_for_fit_or_selection": False,
        "other_block_simulation_truth_used_for_null_calibration": True,
        "other_block_simulation_truth_used_for_supervised_training": False,
        "formal_pretruth_barrier_satisfied": True, "fresh_locked_certification": False,
        "retrospective_cached_world_exploration": True, "new_worlds_or_datasets_created": False,
        "perfect_matching_assumed": False, "fixed_shared_relationship_count_assumed": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "within_assay_direct_structure_claimed": False, "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "source_sha256": {p.name: geometry._sha256(p) for p in (Path(__file__).resolve(), Path(diagnostic.__file__).resolve(), Path(parent_scorer.__file__).resolve())},
        "input_paths": {k: str(p) for k, p in inputs.items()},
        "input_sha256": {k: geometry._sha256(p) for k, p in inputs.items()},
        "parent_manifest_payload_sha256": {k: m["manifest_payload_sha256"] for k, m in zip(inputs, (em, bm, m7, m8, cm, dm))},
        "output_sha256": {name: geometry._sha256(out_dir / name) for name in ("dual_null_candidates.tsv", "dual_null_pair_audit.tsv")},
    }
    manifest["manifest_payload_sha256"] = geometry._canonical_sha256(manifest)
    (out_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_calibration(directory: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = geometry._verified_manifest(directory / "calibration_manifest.json", SCHEMA)
    verify_files(directory, manifest)
    if manifest.get("fixed_selection_rule") != FIXED_SELECTION_RULE or manifest.get("target_block_truth_used_for_fit_or_selection") is not False:
        raise ValueError("Child selection rule or boundary changed")
    for key, path in manifest["input_paths"].items():
        if geometry._sha256(Path(path)) != manifest["input_sha256"][key]:
            raise ValueError(f"Changed parent input: {key}")
    candidates = pd.read_csv(directory / "dual_null_candidates.tsv", sep="\t", keep_default_na=False)
    if list(candidates.columns) != CANDIDATE_COLUMNS or set(candidates["world_key"].astype(str)) - set(manifest["target_world_keys"]):
        raise ValueError("Invalid child candidate schema or world identities")
    audit = pd.read_csv(directory / "dual_null_pair_audit.tsv", sep="\t", keep_default_na=False)
    return manifest, candidates, audit


def main() -> None:
    arg = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence-dir", "v455-calibration-dir", "v457-diagnostic-dir", "v458-diagnostic-dir", "v458-calibration-dir", "diagnostic-dir", "out-dir"):
        arg.add_argument("--" + name, type=Path, required=True)
    arg.add_argument("--target-block", required=True)
    print(json.dumps(calibrate_target_block(**vars(arg.parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
