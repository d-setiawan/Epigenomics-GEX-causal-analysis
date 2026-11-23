#!/usr/bin/env python3
"""Two-phase cached-world audit of CA-BOSS v4.27 topology consensus.

Phase A combines the already frozen feature evidence and within-assay direct
candidate table, freezes the fixed state/topology conjunction, and selects
balanced random panels without opening scorer files.  Phase B verifies that
closure, opens existing labels and the hash-bound v4.26 outer predictions,
then performs leave-one-source-block-out sentinel certification.

This is retrospective development on existing worlds.  The utility has no
simulation or missing-data generation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import topology_consensus_state_calibration_v4 as calibration  # noqa: E402


SCHEMA = "ca_boss_v4.nested_topology_consensus_state.v1"
KEY_COLUMNS = list(calibration.KEY_COLUMNS)
PHASE_A_TABLES = (
    "phase_a_observable_topology_evidence.tsv",
    "phase_a_random_panels.tsv",
)
PHASE_B_TABLES = (
    "phase_b_outer_feature_predictions.tsv",
    "phase_b_outer_block_metrics.tsv",
    "phase_b_world_metrics.tsv",
    "phase_b_random_panel_metrics.tsv",
)
SOURCE_FILES = (
    "audit_nested_topology_consensus_state_v4.py",
    "topology_consensus_state_calibration_v4.py",
)
DEFAULT_CACHE_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_layered_contract_v423_full_pool"
)
DEFAULT_MEMBERSHIP = DEFAULT_CACHE_DIR / "phase_a_feature_membership.tsv"
DEFAULT_DIRECT = DEFAULT_CACHE_DIR / "phase_a_direct_predictions.tsv"
DEFAULT_CACHE_MANIFEST = DEFAULT_CACHE_DIR / "phase_a_manifest.json"
DEFAULT_SCORES = DEFAULT_CACHE_DIR / "phase_b_feature_scores.tsv"
DEFAULT_V426_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_multievidence_v426_full_pool"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def _regular_file(path: Path, *, phase: str) -> Path:
    lexical = path.absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if phase.casefold() not in resolved.name.casefold():
        raise ValueError(f"Expected an explicitly {phase} file: {path}")
    if phase.casefold() == "phase_a":
        forbidden = {"truth", "scorer"}
        if {part.casefold() for part in resolved.parts} & forbidden:
            raise ValueError(f"Phase A refuses scorer/truth path: {path}")
    return resolved


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    observed = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} payload hash mismatch")
    return observed


def _balanced_random_panels(
    evidence: pd.DataFrame,
    *,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> pd.DataFrame:
    worlds = evidence.loc[:, ["block", "seed"]].drop_duplicates().copy()
    blocks = sorted(set(worlds["block"].astype(str)))
    if panel_size < len(blocks):
        raise ValueError("panel_size must include every source block")
    minimum, remainder = divmod(panel_size, len(blocks))
    rows: list[dict[str, Any]] = []
    for panel_index in range(panels):
        seed = int(selection_seed + panel_index)
        rng = np.random.default_rng(seed)
        extra = set(rng.choice(blocks, size=remainder, replace=False).tolist())
        selected: list[tuple[str, int]] = []
        for block in blocks:
            candidates = (
                worlds.loc[worlds["block"].astype(str).eq(block), "seed"]
                .astype(int)
                .sort_values()
                .to_numpy()
            )
            count = minimum + int(block in extra)
            if len(candidates) < count:
                raise ValueError(f"Source block is too small: {block}")
            selected.extend(
                (block, int(value))
                for value in rng.choice(candidates, size=count, replace=False)
            )
        if len(selected) != panel_size or len(set(selected)) != panel_size:
            raise RuntimeError("Balanced panel selection was not exact")
        for block, world_seed in sorted(selected):
            rows.append(
                {
                    "panel_id": f"panel_{panel_index + 1:02d}",
                    "selection_seed": seed,
                    "block": block,
                    "seed": world_seed,
                }
            )
    return pd.DataFrame(rows)


def _direct_degrees(
    membership: pd.DataFrame, direct: pd.DataFrame
) -> pd.DataFrame:
    required = {"block", "seed", "source_assay", "parent", "child"}
    missing = sorted(required - set(direct.columns))
    if missing:
        raise ValueError(f"Direct cache is missing columns {missing}")
    if direct.duplicated(["block", "seed", "parent", "child"]).any():
        raise ValueError("Direct cache contains duplicate directed arms")
    feature_modalities = membership.loc[
        :, ["block", "seed", "feature", "modality"]
    ].drop_duplicates()
    if feature_modalities.duplicated(KEY_COLUMNS).any():
        raise ValueError("Membership contains duplicate feature keys")
    endpoints = pd.concat(
        [
            direct.loc[:, ["block", "seed", "source_assay", "parent"]].rename(
                columns={"parent": "feature"}
            ),
            direct.loc[:, ["block", "seed", "source_assay", "child"]].rename(
                columns={"child": "feature"}
            ),
        ],
        ignore_index=True,
    )
    checked = endpoints.merge(
        feature_modalities,
        on=["block", "seed", "feature"],
        how="left",
        validate="many_to_one",
    )
    if checked["modality"].isna().any():
        raise ValueError("Direct cache contains an unknown feature endpoint")
    if not checked["source_assay"].astype(str).equals(
        checked["modality"].astype(str)
    ):
        raise ValueError("Direct cache endpoint modality mismatch")
    return (
        checked.groupby(["block", "seed", "feature"], sort=True)
        .size()
        .rename("within_assay_direct_candidate_degree")
        .reset_index()
    )


def run_phase_a(
    *,
    out_dir: Path,
    membership_path: Path,
    direct_path: Path,
    cache_manifest_path: Path,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Phase-A output already exists: {out_dir}")
    membership_source = _regular_file(membership_path, phase="phase_a")
    direct_source = _regular_file(direct_path, phase="phase_a")
    manifest_source = _regular_file(cache_manifest_path, phase="phase_a")
    cache_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    cache_payload_hash = _validate_self_hash(
        cache_manifest,
        field="manifest_payload_sha256",
        label="upstream cache Phase-A manifest",
    )
    expected_outputs = cache_manifest.get("output_sha256", {})
    if (
        expected_outputs.get(membership_source.name) != _sha256(membership_source)
        or expected_outputs.get(direct_source.name) != _sha256(direct_source)
        or cache_manifest.get("scorer_or_truth_opened") is not False
        or cache_manifest.get("new_worlds_or_datasets_created") is not False
    ):
        raise ValueError("Upstream cached evidence does not preserve Phase A")
    membership = pd.read_csv(membership_source, sep="\t")
    direct = pd.read_csv(direct_source, sep="\t")
    degrees = _direct_degrees(membership, direct)
    evidence = membership.merge(
        degrees,
        on=["block", "seed", "feature"],
        how="left",
        validate="one_to_one",
    )
    evidence["within_assay_direct_candidate_degree"] = (
        evidence["within_assay_direct_candidate_degree"].fillna(0).astype(int)
    )
    observable = calibration.observable_topology_evidence(evidence)
    random_panels = _balanced_random_panels(
        observable,
        selection_seed=selection_seed,
        panels=panels,
        panel_size=panel_size,
    )
    out_dir.mkdir(parents=True)
    observable.to_csv(out_dir / PHASE_A_TABLES[0], sep="\t", index=False)
    random_panels.to_csv(out_dir / PHASE_A_TABLES[1], sep="\t", index=False)
    manifest = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": calibration.SCHEMA,
        "phase": "a",
        "observable_inputs": {
            "membership": str(membership_source),
            "direct_candidates": str(direct_source),
            "upstream_manifest": str(manifest_source),
        },
        "observable_input_sha256": {
            "membership": _sha256(membership_source),
            "direct_candidates": _sha256(direct_source),
            "upstream_manifest": _sha256(manifest_source),
            "upstream_manifest_payload_sha256": cache_payload_hash,
        },
        "source_sha256": _source_hashes(),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_A_TABLES
        },
        "fixed_selection_rule": (
            "formal_test_eligible_and_positive_marginal_and_partial_state_fit_"
            "and_within_assay_direct_candidate_degree_at_least_one"
        ),
        "minimum_direct_candidate_degree": (
            calibration.MINIMUM_DIRECT_CANDIDATE_DEGREE
        ),
        "state_specific_gate": calibration.STATE_SPECIFIC_GATE,
        "null_control_method": calibration.NULL_CONTROL_METHOD,
        "selection_seed": int(selection_seed),
        "random_panels": int(panels),
        "panel_size": int(panel_size),
        "worlds": int(
            observable.loc[:, ["block", "seed"]].drop_duplicates().shape[0]
        ),
        "source_blocks": int(observable["block"].nunique()),
        "observable_feature_rows": int(len(observable)),
        "direct_endpoint_rows": int(
            observable["topology_consensus_residual_structure_support"].sum()
        ),
        "scorer_or_truth_opened": False,
        "upstream_v426_phase_b_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "retrospective_development_only": True,
        "design_was_informed_by_cached_development_results": True,
        "fresh_locked_claim": False,
        "direct_structure_alone_implies_state_membership": False,
        "exact_cross_assay_pair_claimed": False,
        "direct_causal_claimed": False,
        "pairwise_expansion_permitted": False,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest_path = out_dir / "phase_a_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="Phase-A manifest"
    )
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("source_sha256") != _source_hashes()
        or manifest.get("scorer_or_truth_opened") is not False
        or manifest.get("upstream_v426_phase_b_opened") is not False
        or manifest.get("new_simulation_worlds_or_datasets_created") is not False
        or manifest.get("direct_structure_alone_implies_state_membership")
        is not False
    ):
        raise ValueError("Phase-A topology freeze is not current or truth closed")
    for label, value in manifest["observable_inputs"].items():
        source = Path(str(value))
        expected = manifest["observable_input_sha256"].get(label)
        if _sha256(source) != expected:
            raise ValueError(f"Observable input changed after Phase A: {label}")
    for name in PHASE_A_TABLES:
        if _sha256(out_dir / name) != manifest["output_sha256"].get(name):
            raise ValueError(f"Phase-A output changed after freeze: {name}")
    observable = calibration.observable_topology_evidence(
        pd.read_csv(out_dir / PHASE_A_TABLES[0], sep="\t")
    )
    panels = pd.read_csv(out_dir / PHASE_A_TABLES[1], sep="\t")
    return manifest, observable, panels


def _label_table(path: Path) -> tuple[Path, pd.DataFrame]:
    source = _regular_file(path, phase="phase_b")
    frame = pd.read_csv(source, sep="\t")
    columns = [*KEY_COLUMNS, "truth_role", "truth_state_member"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature-score table is missing columns {missing}")
    output = frame.loc[:, columns].copy()
    if output.duplicated(KEY_COLUMNS).any():
        raise ValueError("Feature-score table contains duplicate keys")
    output["truth_state_member"] = calibration._boolean(
        output["truth_state_member"], name="truth_state_member"
    )
    return source, output


def _verified_v426_predictions(
    v426_dir: Path, *, labels: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    summary_path = _regular_file(
        v426_dir / "phase_b_summary.json", phase="phase_b"
    )
    prediction_path = _regular_file(
        v426_dir / "phase_b_outer_feature_predictions.tsv", phase="phase_b"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload_hash = _validate_self_hash(
        summary, field="summary_payload_sha256", label="upstream v4.26 summary"
    )
    if summary.get("schema_version") != (
        "ca_boss_v4.nested_multievidence_state_calibration.v1"
    ):
        raise ValueError("Unsupported upstream v4.26 audit schema")
    if (
        summary.get("output_sha256", {}).get(prediction_path.name)
        != _sha256(prediction_path)
    ):
        raise ValueError("Upstream v4.26 prediction hash mismatch")
    frame = pd.read_csv(prediction_path, sep="\t")
    columns = [
        *KEY_COLUMNS,
        "state_v426_stacked_supported_member",
        "state_v426_stacked_membership_tier",
        "outer_truth_used_for_fit_or_threshold",
        "truth_state_member",
    ]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Upstream v4.26 predictions are missing columns {missing}")
    output = frame.loc[:, columns].copy()
    if output.duplicated(KEY_COLUMNS).any():
        raise ValueError("Upstream v4.26 predictions contain duplicate keys")
    for column in (
        "state_v426_stacked_supported_member",
        "outer_truth_used_for_fit_or_threshold",
        "truth_state_member",
    ):
        output[column] = calibration._boolean(output[column], name=column)
    if output["outer_truth_used_for_fit_or_threshold"].any():
        raise ValueError("Upstream v4.26 outer prediction used held-block truth")
    check = output.merge(
        labels.loc[:, [*KEY_COLUMNS, "truth_state_member"]],
        on=KEY_COLUMNS,
        suffixes=("_upstream", "_label"),
        validate="one_to_one",
    )
    if len(check) != len(labels) or not check[
        "truth_state_member_upstream"
    ].equals(check["truth_state_member_label"]):
        raise ValueError("Upstream v4.26 truth keys disagree with scorer labels")
    output = output.drop(columns=["truth_state_member"])
    return output, {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "summary_payload_sha256": payload_hash,
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
    }


def _metrics(frame: pd.DataFrame, selected_column: str) -> dict[str, Any]:
    selected = frame[selected_column].astype(bool)
    truth = frame["truth_state_member"].astype(bool)
    tp = int((selected & truth).sum())
    fp = int((selected & ~truth).sum())
    fn = int((~selected & truth).sum())
    return {
        "true_positive_members": tp,
        "false_positive_members": fp,
        "false_negative_members": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def _apply_outer_block(
    merged: pd.DataFrame, *, held_block: str
) -> tuple[pd.DataFrame, calibration.FrozenTopologyConsensusStateCalibrator]:
    fit = ~merged["block"].astype(str).eq(held_block)
    held = merged["block"].astype(str).eq(held_block)
    calibrator = calibration.fit_topology_consensus_state_calibrator(
        merged.loc[fit, list(calibration.OBSERVABLE_COLUMNS)],
        null_feature=~merged.loc[fit, "truth_state_member"].astype(bool),
    )
    annotated = calibration.annotate_topology_consensus_state_membership(
        merged.loc[held, list(calibration.OBSERVABLE_COLUMNS)],
        upstream_supported=merged.loc[
            held, "state_v426_stacked_supported_member"
        ],
        upstream_tier=merged.loc[
            held, "state_v426_stacked_membership_tier"
        ],
        calibrator=calibrator,
    )
    annotated["state_v426_stacked_supported_member"] = merged.loc[
        held, "state_v426_stacked_supported_member"
    ].to_numpy(copy=True)
    annotated["state_v426_stacked_membership_tier"] = merged.loc[
        held, "state_v426_stacked_membership_tier"
    ].to_numpy(copy=True)
    held_labels = merged.loc[
        held, [*KEY_COLUMNS, "truth_role", "truth_state_member"]
    ]
    annotated = annotated.merge(
        held_labels, on=KEY_COLUMNS, validate="one_to_one"
    )
    annotated["outer_held_block"] = held_block
    annotated["outer_truth_used_for_fit_or_certification"] = False
    return annotated, calibrator


def run_phase_b(
    *,
    out_dir: Path,
    feature_scores_path: Path,
    v426_dir: Path,
) -> None:
    for name in (
        *PHASE_B_TABLES,
        "phase_b_full_calibrator.json",
        "phase_b_summary.json",
    ):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")
    manifest, observable, panels = _load_phase_a(out_dir)
    label_path, labels = _label_table(feature_scores_path)
    merged = observable.merge(labels, on=KEY_COLUMNS, validate="one_to_one")
    upstream, upstream_provenance = _verified_v426_predictions(
        v426_dir, labels=labels
    )
    merged = merged.merge(upstream, on=KEY_COLUMNS, validate="one_to_one")
    if len(merged) != len(observable):
        raise ValueError("Phase-B inputs do not cover every observable row")
    blocks = sorted(set(merged["block"].astype(str)))
    if len(blocks) < 4:
        raise ValueError("Nested evaluation requires at least four source blocks")

    predictions_list: list[pd.DataFrame] = []
    block_rows: list[dict[str, Any]] = []
    for held_block in blocks:
        annotated, calibrator = _apply_outer_block(
            merged, held_block=held_block
        )
        predictions_list.append(annotated)
        upstream_metrics = _metrics(
            annotated, "state_v426_stacked_supported_member"
        )
        final_metrics = _metrics(annotated, "state_v427_supported_member")
        addition = annotated["topology_consensus_state_member"].astype(bool)
        truth = annotated["truth_state_member"].astype(bool)
        block_rows.append(
            {
                "outer_held_block": held_block,
                "sentinel_certified": calibrator.sentinel_certified,
                "training_null_rows": calibrator.training_null_rows,
                "training_rule_eligible_null_rows": (
                    calibrator.training_rule_eligible_null_rows
                ),
                "worlds": int(
                    annotated.loc[:, ["block", "seed"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "v426_true_positive_members": upstream_metrics[
                    "true_positive_members"
                ],
                "v426_false_positive_members": upstream_metrics[
                    "false_positive_members"
                ],
                "v426_recall": upstream_metrics["recall"],
                "topology_true_additions": int((addition & truth).sum()),
                "topology_false_additions": int((addition & ~truth).sum()),
                "v427_true_positive_members": final_metrics[
                    "true_positive_members"
                ],
                "v427_false_positive_members": final_metrics[
                    "false_positive_members"
                ],
                "v427_false_negative_members": final_metrics[
                    "false_negative_members"
                ],
                "v427_precision": final_metrics["precision"],
                "v427_recall": final_metrics["recall"],
            }
        )
    predictions = pd.concat(predictions_list, ignore_index=True)
    if predictions.duplicated(KEY_COLUMNS).any() or len(predictions) != len(merged):
        raise RuntimeError("Outer predictions are not one complete held-block table")
    block_metrics = pd.DataFrame(block_rows)

    world_rows: list[dict[str, Any]] = []
    for (block, seed), world in predictions.groupby(
        ["block", "seed"], sort=True
    ):
        upstream_metrics = _metrics(
            world, "state_v426_stacked_supported_member"
        )
        final_metrics = _metrics(world, "state_v427_supported_member")
        addition = world["topology_consensus_state_member"].astype(bool)
        truth = world["truth_state_member"].astype(bool)
        world_rows.append(
            {
                "block": block,
                "seed": int(seed),
                "v426_precision": upstream_metrics["precision"],
                "v426_recall": upstream_metrics["recall"],
                "topology_true_additions": int((addition & truth).sum()),
                "topology_false_additions": int((addition & ~truth).sum()),
                "v427_precision": final_metrics["precision"],
                "v427_recall": final_metrics["recall"],
                "v427_world_reliability_success": bool(
                    final_metrics["precision"] == 1.0
                    and final_metrics["recall"] >= 0.80
                ),
            }
        )
    world_metrics = pd.DataFrame(world_rows)

    panel_rows: list[dict[str, Any]] = []
    for panel_id, selection in panels.groupby("panel_id", sort=True):
        current = predictions.merge(
            selection.loc[:, ["block", "seed"]],
            on=["block", "seed"],
            validate="many_to_one",
        )
        upstream_metrics = _metrics(
            current, "state_v426_stacked_supported_member"
        )
        final_metrics = _metrics(current, "state_v427_supported_member")
        addition = current["topology_consensus_state_member"].astype(bool)
        truth = current["truth_state_member"].astype(bool)
        panel_rows.append(
            {
                "panel_id": panel_id,
                "selection_seed": int(selection["selection_seed"].iloc[0]),
                "worlds": int(
                    selection.loc[:, ["block", "seed"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "v426_precision": upstream_metrics["precision"],
                "v426_recall": upstream_metrics["recall"],
                "topology_true_additions": int((addition & truth).sum()),
                "topology_false_additions": int((addition & ~truth).sum()),
                "v427_precision": final_metrics["precision"],
                "v427_recall": final_metrics["recall"],
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)

    full_calibrator = calibration.fit_topology_consensus_state_calibrator(
        merged.loc[:, list(calibration.OBSERVABLE_COLUMNS)],
        null_feature=~merged["truth_state_member"].astype(bool),
    )
    artifact = {
        **full_calibrator.to_payload(),
        "artifact_role": "future_application_only",
        "training_label_input_sha256": _sha256(label_path),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "fixed_state_topology_conjunction_selected_without_application_labels": True,
        "retrospective_design_informed_by_cached_development_results": True,
        "training_rows_are_not_validation_rows": True,
        "application_requires_truth": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    _write_json(out_dir / "phase_b_full_calibrator.json", artifact)

    predictions.to_csv(out_dir / PHASE_B_TABLES[0], sep="\t", index=False)
    block_metrics.to_csv(out_dir / PHASE_B_TABLES[1], sep="\t", index=False)
    world_metrics.to_csv(out_dir / PHASE_B_TABLES[2], sep="\t", index=False)
    panel_metrics.to_csv(out_dir / PHASE_B_TABLES[3], sep="\t", index=False)

    upstream_metrics = _metrics(
        predictions, "state_v426_stacked_supported_member"
    )
    final_metrics = _metrics(predictions, "state_v427_supported_member")
    addition = predictions["topology_consensus_state_member"].astype(bool)
    truth = predictions["truth_state_member"].astype(bool)
    role_recall = {
        role: float(
            predictions.loc[
                predictions["truth_role"].astype(str).eq(role),
                "state_v427_supported_member",
            ]
            .astype(bool)
            .mean()
        )
        for role in ("P", "Q", "C", "S", "RNA")
    }
    worlds = len(world_metrics)
    false_worlds = int(
        predictions.loc[addition & ~truth, ["block", "seed"]]
        .drop_duplicates()
        .shape[0]
    )
    summary = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": calibration.SCHEMA,
        "phase": "b",
        "retrospective_development_only": True,
        "design_was_informed_by_cached_development_results": True,
        "fresh_locked_claim": False,
        "new_simulation_worlds_or_datasets_created": False,
        "worlds": worlds,
        "source_blocks": len(blocks),
        "evaluation_protocol": (
            "leave_one_source_block_out_sentinel_certification_of_a_fixed_"
            "positive_state_fit_and_within_assay_direct_endpoint_conjunction"
        ),
        "upstream_v426_state_layer": upstream_metrics,
        "nested_v427_topology_consensus_route": {
            "route_true_positive_members": int((addition & truth).sum()),
            "route_false_positive_members": int((addition & ~truth).sum()),
            "incremental_true_positive_members_beyond_v426": int(
                (addition & truth).sum()
            ),
            "incremental_false_positive_members_beyond_v426": int(
                (addition & ~truth).sum()
            ),
            "false_addition_worlds": false_worlds,
            "world_familywise_false_addition_rate": float(false_worlds / worlds),
            "outer_blocks_with_sentinel_certification": int(
                block_metrics["sentinel_certified"].astype(bool).sum()
            ),
            "outer_blocks_with_zero_false_additions": int(
                block_metrics["topology_false_additions"].astype(int).eq(0).sum()
            ),
            "outer_blocks": len(block_metrics),
        },
        "state_v427_stacked_union": {
            **final_metrics,
            "role_recall": role_recall,
            "minimum_world_recall": float(world_metrics["v427_recall"].min()),
            "worlds_meeting_precision_one_recall_0.80": int(
                world_metrics["v427_world_reliability_success"]
                .astype(bool)
                .sum()
            ),
            "worlds": worlds,
            "v426_membership_removed": 0,
            "direct_edges_created_or_changed": 0,
            "pairwise_cross_assay_edges_created": 0,
        },
        "random_cached_panels": {
            "panels": len(panel_metrics),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "all_panels_zero_false_additions": bool(
                panel_metrics["topology_false_additions"].astype(int).eq(0).all()
            ),
            "mean_v426_recall": float(panel_metrics["v426_recall"].mean()),
            "mean_v427_recall": float(panel_metrics["v427_recall"].mean()),
            "minimum_v427_recall": float(panel_metrics["v427_recall"].min()),
            "maximum_v427_recall": float(panel_metrics["v427_recall"].max()),
        },
        "future_application_calibrator": {
            "artifact": "phase_b_full_calibrator.json",
            "artifact_sha256": _sha256(out_dir / "phase_b_full_calibrator.json"),
            "scored_as_training_validation": False,
            "application_requires_truth": False,
        },
        "upstream_v426": upstream_provenance,
        "gates": {
            "all_outer_training_folds_sentinel_certified": bool(
                block_metrics["sentinel_certified"].astype(bool).all()
            ),
            "all_outer_blocks_zero_false_additions": bool(
                block_metrics["topology_false_additions"].astype(int).eq(0).all()
            ),
            "zero_outer_world_fwer": false_worlds == 0,
            "precision_preserved_at_one": final_metrics["precision"] == 1.0,
            "recall_improved_beyond_v426": (
                final_metrics["recall"] > upstream_metrics["recall"]
            ),
            "all_worlds_meet_precision_one_recall_0.80": bool(
                world_metrics["v427_world_reliability_success"].astype(bool).all()
            ),
            "direct_structure_alone_is_insufficient": True,
            "positive_marginal_and_partial_state_gate": True,
            "no_pairwise_or_direct_claims": True,
        },
        "identifiability_boundary": {
            "estimand": (
                "within_assay weak state participation supported by separately "
                "typed residual topology"
            ),
            "direct_edge_alone_implies_state_membership": False,
            "exact_cross_assay_feature_pairing": "not claimed",
            "cell_specific_cross_assay_direction": "not identifiable",
        },
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "truth_label_input_sha256": _sha256(label_path),
        "source_sha256": _source_hashes(),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_B_TABLES
        },
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    _write_json(out_dir / "phase_b_summary.json", summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--direct-predictions", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--feature-scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--v426-dir", type=Path, default=DEFAULT_V426_DIR)
    parser.add_argument("--selection-seed", type=int, default=20261004)
    parser.add_argument("--panels", type=int, default=5)
    parser.add_argument("--panel-size", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.panels <= 0 or args.panel_size <= 0:
        raise ValueError("panels and panel-size must be positive")
    if args.phase == "a":
        run_phase_a(
            out_dir=args.out_dir,
            membership_path=args.feature_membership,
            direct_path=args.direct_predictions,
            cache_manifest_path=args.cache_manifest,
            selection_seed=args.selection_seed,
            panels=args.panels,
            panel_size=args.panel_size,
        )
    else:
        run_phase_b(
            out_dir=args.out_dir,
            feature_scores_path=args.feature_scores,
            v426_dir=args.v426_dir,
        )


if __name__ == "__main__":
    main()
