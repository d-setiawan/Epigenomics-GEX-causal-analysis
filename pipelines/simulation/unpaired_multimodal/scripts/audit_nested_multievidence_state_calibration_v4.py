#!/usr/bin/env python3
"""Audit CA-BOSS v4.26 multievidence state calibration on cached worlds.

Phase A freezes the truth-free observable evidence, fixed three-route score
family, random cached replay panels, and executable source hashes.  Phase B
then performs leave-one-source-block-out null fitting and threshold calibration.
The held block is never used to estimate component/route null distributions or
the familywise threshold.

The audit also stacks each held-block v4.26 prediction on the already-audited
v4.25 outer prediction.  That upstream table is opened only in Phase B and is
verified against its self-hashed summary.  This is retrospective development
on existing cached worlds, not fresh locked validation.  No command in this
utility can generate or modify a simulation world.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import multievidence_empirical_null_state_calibration_v4 as calibration  # noqa: E402


SCHEMA = "ca_boss_v4.nested_multievidence_state_calibration.v1"
PHASE_A_TABLES = (
    "phase_a_observable_feature_evidence.tsv",
    "phase_a_random_panels.tsv",
)
PHASE_B_TABLES = (
    "phase_b_outer_feature_predictions.tsv",
    "phase_b_outer_block_metrics.tsv",
    "phase_b_random_panel_metrics.tsv",
)
SOURCE_FILES = (
    "audit_nested_multievidence_state_calibration_v4.py",
    "multievidence_empirical_null_state_calibration_v4.py",
)
KEY_COLUMNS = list(calibration.KEY_COLUMNS)
DEFAULT_MEMBERSHIP = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_layered_contract_v423_full_pool/phase_a_feature_membership.tsv"
)
DEFAULT_SCORES = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_layered_contract_v423_full_pool/phase_b_feature_scores.tsv"
)
DEFAULT_V425_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_empirical_null_v425_full_pool"
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def _regular_file(path: Path, *, observable: bool) -> Path:
    lexical = path.absolute()
    if not lexical.is_file() or lexical.is_symlink():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if observable:
        forbidden_parts = {"scorer", "truth"}
        if {part.casefold() for part in resolved.parts} & forbidden_parts:
            raise ValueError(f"Phase A refuses scorer/truth path: {path}")
        if "phase_a" not in resolved.name.casefold():
            raise ValueError("Phase-A membership input must be explicitly phase_a")
    return resolved


def _balanced_random_panels(
    worlds: pd.DataFrame,
    *,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> pd.DataFrame:
    universe = worlds.loc[:, ["block", "seed"]].drop_duplicates().copy()
    if universe.duplicated(["block", "seed"]).any() or universe.empty:
        raise ValueError("World universe must contain unique nonempty keys")
    blocks = sorted(set(universe["block"].astype(str)))
    if panel_size < len(blocks):
        raise ValueError("panel_size must include at least one world per block")
    minimum = panel_size // len(blocks)
    remainder = panel_size % len(blocks)
    if any(
        len(universe.loc[universe["block"].astype(str).eq(block)])
        < minimum + int(remainder > 0)
        for block in blocks
    ):
        raise ValueError("A source block is too small for balanced selection")
    rows: list[dict[str, Any]] = []
    for panel_index in range(panels):
        panel_seed = int(selection_seed + panel_index)
        rng = np.random.default_rng(panel_seed)
        extra_blocks = set(
            rng.choice(blocks, size=remainder, replace=False).tolist()
        )
        selected: list[tuple[str, int]] = []
        for block in blocks:
            candidates = (
                universe.loc[universe["block"].astype(str).eq(block), "seed"]
                .astype(int)
                .sort_values()
                .to_numpy()
            )
            count = minimum + int(block in extra_blocks)
            seeds = rng.choice(candidates, size=count, replace=False)
            selected.extend((block, int(seed)) for seed in seeds)
        if len(selected) != panel_size or len(set(selected)) != panel_size:
            raise RuntimeError("Balanced panel selection produced invalid keys")
        for block, seed in sorted(selected):
            rows.append(
                {
                    "panel_id": f"panel_{panel_index + 1:02d}",
                    "selection_seed": panel_seed,
                    "block": block,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def run_phase_a(
    *,
    out_dir: Path,
    membership_path: Path,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Phase-A output already exists: {out_dir}")
    source = _regular_file(membership_path, observable=True)
    membership = pd.read_csv(source, sep="\t")
    observable = calibration.observable_feature_evidence(membership)
    panels_frame = _balanced_random_panels(
        observable,
        selection_seed=selection_seed,
        panels=panels,
        panel_size=panel_size,
    )
    out_dir.mkdir(parents=True)
    observable.to_csv(out_dir / PHASE_A_TABLES[0], sep="\t", index=False)
    panels_frame.to_csv(out_dir / PHASE_A_TABLES[1], sep="\t", index=False)
    manifest = {
        "schema_version": SCHEMA,
        "phase": "a",
        "algorithm_schema_version": calibration.SCHEMA,
        "observable_input": str(source),
        "observable_input_sha256": _sha256(source),
        "source_sha256": _source_hashes(),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_A_TABLES
        },
        "score_component_columns": list(calibration.SCORE_COMPONENT_COLUMNS),
        "route_names": list(calibration.ROUTE_NAMES),
        "route_weights": [list(values) for values in calibration.ROUTE_WEIGHTS],
        "route_selection": "none_fixed_familywise_maximum",
        "state_specific_gate": "positive_marginal_and_partial_state_fit",
        "threshold_method": (
            "max over fixed routes followed by one source-block-maximum "
            "spacing beyond the training-null family maximum"
        ),
        "selection_seed": int(selection_seed),
        "random_panels": int(panels),
        "panel_size": int(panel_size),
        "worlds": int(observable[["block", "seed"]].drop_duplicates().shape[0]),
        "source_blocks": int(observable["block"].nunique()),
        "observable_feature_rows": int(len(observable)),
        "calibration_candidate_rows": int(
            observable["multievidence_calibration_candidate"].sum()
        ),
        "scorer_or_truth_opened": False,
        "upstream_v425_phase_b_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "retrospective_development_only": True,
        "design_was_informed_by_cached_development_results": True,
        "fresh_locked_claim": False,
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
    expected = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if _canonical_sha256(unsigned) != expected:
        raise ValueError("Phase-A manifest payload hash mismatch")
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported Phase-A manifest schema")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("Executable source changed after Phase A")
    if manifest.get("scorer_or_truth_opened") is not False:
        raise ValueError("Phase A did not preserve the truth barrier")
    if manifest.get("upstream_v425_phase_b_opened") is not False:
        raise ValueError("Phase A opened the upstream Phase-B table")
    if manifest.get("new_simulation_worlds_or_datasets_created") is not False:
        raise ValueError("Phase A is not cached-world-only")
    if manifest.get("route_names") != list(calibration.ROUTE_NAMES):
        raise ValueError("Frozen route family does not match executable source")
    if manifest.get("route_weights") != [
        list(values) for values in calibration.ROUTE_WEIGHTS
    ]:
        raise ValueError("Frozen route weights do not match executable source")
    source = Path(str(manifest["observable_input"]))
    if _sha256(source) != manifest["observable_input_sha256"]:
        raise ValueError("Observable membership input changed after Phase A")
    for name in PHASE_A_TABLES:
        if _sha256(out_dir / name) != manifest["output_sha256"].get(name):
            raise ValueError(f"Phase-A output changed after freeze: {name}")
    observable = pd.read_csv(out_dir / PHASE_A_TABLES[0], sep="\t")
    observable = calibration.observable_feature_evidence(observable)
    panels = pd.read_csv(out_dir / PHASE_A_TABLES[1], sep="\t")
    return manifest, observable, panels


def _label_table(path: Path) -> tuple[Path, pd.DataFrame]:
    source = _regular_file(path, observable=False)
    if "phase_b" not in source.name.casefold():
        raise ValueError("Truth labels must be an explicitly phase_b table")
    frame = pd.read_csv(source, sep="\t")
    columns = [*KEY_COLUMNS, "truth_role", "truth_state_member"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature score table is missing columns {missing}")
    output = frame.loc[:, columns].copy()
    if output.duplicated(KEY_COLUMNS).any():
        raise ValueError("Feature score table contains duplicate keys")
    output["truth_state_member"] = calibration._boolean(
        output["truth_state_member"], name="truth_state_member"
    )
    return source, output


def _verified_v425_predictions(
    v425_dir: Path,
    *,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    summary_path = _regular_file(v425_dir / "phase_b_summary.json", observable=False)
    prediction_path = _regular_file(
        v425_dir / "phase_b_outer_feature_predictions.tsv", observable=False
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = str(summary.get("summary_payload_sha256", ""))
    unsigned = dict(summary)
    unsigned.pop("summary_payload_sha256", None)
    if _canonical_sha256(unsigned) != expected:
        raise ValueError("Upstream v4.25 summary payload hash mismatch")
    if summary.get("schema_version") != (
        "ca_boss_v4.nested_empirical_null_state_calibration.v1"
    ):
        raise ValueError("Unsupported upstream v4.25 audit schema")
    expected_prediction_hash = (
        summary.get("output_sha256", {}).get(prediction_path.name)
    )
    if _sha256(prediction_path) != expected_prediction_hash:
        raise ValueError("Upstream v4.25 prediction hash mismatch")
    frame = pd.read_csv(prediction_path, sep="\t")
    columns = [
        *KEY_COLUMNS,
        "state_calibrated_supported_member",
        "state_calibrated_membership_tier",
        "empirical_null_state_member",
        "outer_truth_used_for_fit_or_threshold",
        "truth_state_member",
    ]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Upstream v4.25 predictions are missing columns {missing}")
    output = frame.loc[:, columns].copy()
    if output.duplicated(KEY_COLUMNS).any():
        raise ValueError("Upstream v4.25 predictions contain duplicate keys")
    for column in (
        "state_calibrated_supported_member",
        "empirical_null_state_member",
        "outer_truth_used_for_fit_or_threshold",
        "truth_state_member",
    ):
        output[column] = calibration._boolean(output[column], name=column)
    if output["outer_truth_used_for_fit_or_threshold"].any():
        raise ValueError("Upstream v4.25 outer prediction used held-block truth")
    check = output.merge(
        labels.loc[:, [*KEY_COLUMNS, "truth_state_member"]],
        on=KEY_COLUMNS,
        suffixes=("_upstream", "_label"),
        validate="one_to_one",
    )
    if len(check) != len(labels) or not check[
        "truth_state_member_upstream"
    ].equals(check["truth_state_member_label"]):
        raise ValueError("Upstream v4.25 truth keys disagree with scorer labels")
    output = output.drop(columns=["truth_state_member"])
    return output, {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "summary_payload_sha256": expected,
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
    }


def _metrics(frame: pd.DataFrame, *, selected_column: str) -> dict[str, Any]:
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


def _fit_apply(
    merged: pd.DataFrame,
    *,
    fit_blocks: Sequence[str],
    apply_block: str,
) -> tuple[pd.DataFrame, calibration.FrozenMultievidenceStateCalibrator]:
    fit = merged["block"].astype(str).isin(fit_blocks)
    apply = merged["block"].astype(str).eq(str(apply_block))
    observable_columns = list(calibration.OBSERVABLE_COLUMNS)
    fitted = calibration.fit_multievidence_state_calibrator(
        merged.loc[fit, observable_columns],
        null_feature=~merged.loc[fit, "truth_state_member"].astype(bool),
    )
    annotated = calibration.annotate_multievidence_state_membership(
        merged.loc[apply, observable_columns], calibrator=fitted
    )
    return annotated, fitted


def run_phase_b(
    *,
    out_dir: Path,
    feature_scores_path: Path,
    v425_dir: Path,
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_full_calibrator.json", "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")
    manifest, observable, panels = _load_phase_a(out_dir)
    label_path, labels = _label_table(feature_scores_path)
    merged = observable.merge(labels, on=KEY_COLUMNS, validate="one_to_one")
    if len(merged) != len(observable):
        raise ValueError("Truth label table does not cover every observable row")
    v425, v425_provenance = _verified_v425_predictions(
        v425_dir, labels=labels
    )
    merged = merged.merge(v425, on=KEY_COLUMNS, validate="one_to_one")
    if len(merged) != len(observable):
        raise ValueError("Upstream v4.25 predictions do not cover every row")
    blocks = sorted(set(merged["block"].astype(str)))
    if len(blocks) < 4:
        raise ValueError("Nested source-block evaluation requires at least four blocks")

    outer_predictions: list[pd.DataFrame] = []
    outer_metric_rows: list[dict[str, Any]] = []
    for outer_block in blocks:
        fit_blocks = [block for block in blocks if block != outer_block]
        annotated, fitted = _fit_apply(
            merged, fit_blocks=fit_blocks, apply_block=outer_block
        )
        held = merged.loc[
            merged["block"].astype(str).eq(outer_block),
            [
                *KEY_COLUMNS,
                "truth_role",
                "truth_state_member",
                "state_calibrated_supported_member",
                "state_calibrated_membership_tier",
                "empirical_null_state_member",
            ],
        ]
        annotated = annotated.merge(held, on=KEY_COLUMNS, validate="one_to_one")
        addition = annotated["multievidence_empirical_null_state_member"].astype(bool)
        upstream = annotated["state_calibrated_supported_member"].astype(bool)
        annotated["state_v426_stacked_supported_member"] = upstream | addition
        annotated["state_v426_stacked_membership_tier"] = annotated[
            "state_calibrated_membership_tier"
        ].astype(str)
        annotated.loc[
            addition & ~upstream, "state_v426_stacked_membership_tier"
        ] = calibration.TIER
        annotated["outer_held_block"] = outer_block
        annotated["outer_truth_used_for_fit_or_threshold"] = False
        outer_predictions.append(annotated)

        truth = annotated["truth_state_member"].astype(bool)
        incremental = addition & ~upstream
        base = _metrics(annotated, selected_column="state_supported_member")
        previous = _metrics(
            annotated, selected_column="state_calibrated_supported_member"
        )
        stacked = _metrics(
            annotated, selected_column="state_v426_stacked_supported_member"
        )
        false_worlds = int(
            annotated.loc[addition & ~truth, ["block", "seed"]]
            .drop_duplicates()
            .shape[0]
        )
        outer_metric_rows.append(
            {
                "outer_held_block": outer_block,
                "selection_threshold": fitted.selection_threshold,
                "maximum_training_null_family_score": (
                    fitted.maximum_null_family_score
                ),
                "block_maximum_extrapolation": fitted.block_maximum_extrapolation,
                "worlds": int(
                    annotated[["block", "seed"]].drop_duplicates().shape[0]
                ),
                "base_v424_true_positive_members": base["true_positive_members"],
                "base_v424_false_positive_members": base["false_positive_members"],
                "base_v424_recall": base["recall"],
                "base_v425_true_positive_members": previous["true_positive_members"],
                "base_v425_false_positive_members": previous["false_positive_members"],
                "base_v425_recall": previous["recall"],
                "v426_route_true_members": int((addition & truth).sum()),
                "v426_route_false_members": int((addition & ~truth).sum()),
                "v426_incremental_true_members": int((incremental & truth).sum()),
                "v426_incremental_false_members": int((incremental & ~truth).sum()),
                "false_addition_worlds": false_worlds,
                "stacked_true_positive_members": stacked["true_positive_members"],
                "stacked_false_positive_members": stacked["false_positive_members"],
                "stacked_false_negative_members": stacked["false_negative_members"],
                "stacked_precision": stacked["precision"],
                "stacked_recall": stacked["recall"],
            }
        )

    predictions = pd.concat(outer_predictions, ignore_index=True)
    block_metrics = pd.DataFrame(outer_metric_rows)
    if predictions.duplicated(KEY_COLUMNS).any() or len(predictions) != len(merged):
        raise RuntimeError("Outer predictions do not form one complete held-block table")

    panel_metric_rows: list[dict[str, Any]] = []
    for panel_id, selection in panels.groupby("panel_id", sort=True):
        current = predictions.merge(
            selection.loc[:, ["block", "seed"]],
            on=["block", "seed"],
            validate="many_to_one",
        )
        previous = _metrics(
            current, selected_column="state_calibrated_supported_member"
        )
        stacked = _metrics(
            current, selected_column="state_v426_stacked_supported_member"
        )
        addition = current["multievidence_empirical_null_state_member"].astype(bool)
        upstream = current["state_calibrated_supported_member"].astype(bool)
        truth = current["truth_state_member"].astype(bool)
        incremental = addition & ~upstream
        panel_metric_rows.append(
            {
                "panel_id": panel_id,
                "selection_seed": int(
                    selection["selection_seed"].astype(int).iloc[0]
                ),
                "worlds": int(
                    selection[["block", "seed"]].drop_duplicates().shape[0]
                ),
                "base_v425_precision": previous["precision"],
                "base_v425_recall": previous["recall"],
                "v426_incremental_true_members": int(
                    (incremental & truth).sum()
                ),
                "v426_incremental_false_members": int(
                    (incremental & ~truth).sum()
                ),
                "stacked_precision": stacked["precision"],
                "stacked_recall": stacked["recall"],
            }
        )
    panel_metrics = pd.DataFrame(panel_metric_rows)

    full_calibrator = calibration.fit_multievidence_state_calibrator(
        merged.loc[:, list(calibration.OBSERVABLE_COLUMNS)],
        null_feature=~merged["truth_state_member"].astype(bool),
    )
    calibrator_payload = {
        **full_calibrator.to_payload(),
        "artifact_role": "future_application_only",
        "training_label_input_sha256": _sha256(label_path),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "fixed_route_family_selected_without_application_labels": True,
        "retrospective_design_informed_by_cached_development_results": True,
        "training_rows_are_not_validation_rows": True,
        "application_requires_truth": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    calibrator_payload["artifact_payload_sha256"] = _canonical_sha256(
        calibrator_payload
    )
    _write_json(out_dir / "phase_b_full_calibrator.json", calibrator_payload)

    predictions.to_csv(out_dir / PHASE_B_TABLES[0], sep="\t", index=False)
    block_metrics.to_csv(out_dir / PHASE_B_TABLES[1], sep="\t", index=False)
    panel_metrics.to_csv(out_dir / PHASE_B_TABLES[2], sep="\t", index=False)

    base = _metrics(predictions, selected_column="state_supported_member")
    previous = _metrics(
        predictions, selected_column="state_calibrated_supported_member"
    )
    stacked = _metrics(
        predictions, selected_column="state_v426_stacked_supported_member"
    )
    addition = predictions["multievidence_empirical_null_state_member"].astype(bool)
    upstream = predictions["state_calibrated_supported_member"].astype(bool)
    truth = predictions["truth_state_member"].astype(bool)
    incremental = addition & ~upstream
    false_worlds = int(
        predictions.loc[addition & ~truth, ["block", "seed"]]
        .drop_duplicates()
        .shape[0]
    )
    role_recall = {
        role: float(
            predictions.loc[
                predictions["truth_role"].astype(str).eq(role),
                "state_v426_stacked_supported_member",
            ].astype(bool).mean()
        )
        for role in ("P", "Q", "C", "S", "RNA")
    }
    winning_route_counts = {
        route: int(
            (
                addition
                & predictions["multievidence_winning_route"].astype(str).eq(route)
            ).sum()
        )
        for route in calibration.ROUTE_NAMES
    }
    worlds = int(predictions[["block", "seed"]].drop_duplicates().shape[0])
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
            "leave-one-source-block-out component null fitting, fixed-route "
            "family normalization, threshold calibration, and outer prediction"
        ),
        "route_selection": "none_fixed_familywise_maximum",
        "base_v424_state_supported_layer": base,
        "upstream_v425_state_calibrated_layer": previous,
        "nested_v426_multievidence_route": {
            "route_true_positive_members": int((addition & truth).sum()),
            "route_false_positive_members": int((addition & ~truth).sum()),
            "incremental_true_positive_members_beyond_v425": int(
                (incremental & truth).sum()
            ),
            "incremental_false_positive_members_beyond_v425": int(
                (incremental & ~truth).sum()
            ),
            "false_addition_worlds": false_worlds,
            "world_familywise_false_addition_rate": float(false_worlds / worlds),
            "outer_blocks_with_zero_false_additions": int(
                block_metrics["v426_route_false_members"].astype(int).eq(0).sum()
            ),
            "outer_blocks": len(block_metrics),
            "winning_route_counts": winning_route_counts,
            "minimum_block_stacked_recall": float(
                block_metrics["stacked_recall"].min()
            ),
            "maximum_block_stacked_recall": float(
                block_metrics["stacked_recall"].max()
            ),
        },
        "state_v426_stacked_union": {
            **stacked,
            "role_recall": role_recall,
            "strict_or_equivalence_membership_removed": 0,
            "v425_membership_removed": 0,
            "direct_edges_created_or_changed": 0,
            "pairwise_cross_assay_edges_created": 0,
        },
        "random_cached_panels": {
            "panels": len(panel_metrics),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "all_panels_zero_false_additions": bool(
                panel_metrics["v426_incremental_false_members"].astype(int).eq(0).all()
            ),
            "mean_base_v425_recall": float(
                panel_metrics["base_v425_recall"].mean()
            ),
            "mean_stacked_recall": float(panel_metrics["stacked_recall"].mean()),
            "minimum_stacked_recall": float(panel_metrics["stacked_recall"].min()),
            "maximum_stacked_recall": float(panel_metrics["stacked_recall"].max()),
        },
        "future_application_calibrator": {
            "artifact": "phase_b_full_calibrator.json",
            "artifact_sha256": _sha256(out_dir / "phase_b_full_calibrator.json"),
            "scored_as_training_validation": False,
            "application_requires_truth": False,
        },
        "upstream_v425": v425_provenance,
        "gates": {
            "all_outer_blocks_zero_false_additions": bool(
                block_metrics["v426_route_false_members"].astype(int).eq(0).all()
            ),
            "zero_outer_world_fwer": false_worlds == 0,
            "precision_preserved_at_one": stacked["precision"] == 1.0,
            "recall_improved_beyond_v425": stacked["recall"] > previous["recall"],
            "all_random_panels_zero_false_additions": bool(
                panel_metrics["v426_incremental_false_members"].astype(int).eq(0).all()
            ),
            "positive_marginal_and_partial_state_gate": True,
            "no_pairwise_or_direct_claims": True,
        },
        "rejected_development_candidate": {
            "candidate": "adaptive_eight_score_inner_selector",
            "reason": (
                "one held source block admitted one sentinel null; fixed route "
                "family retained instead"
            ),
            "promoted_to_production": False,
        },
        "identifiability_boundary": {
            "estimand": "within-assay evidence of participation in shared latent state",
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
    parser.add_argument(
        "--feature-membership", type=Path, default=DEFAULT_MEMBERSHIP
    )
    parser.add_argument("--feature-scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--v425-dir", type=Path, default=DEFAULT_V425_DIR)
    parser.add_argument("--selection-seed", type=int, default=20260924)
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
            selection_seed=args.selection_seed,
            panels=args.panels,
            panel_size=args.panel_size,
        )
    else:
        run_phase_b(
            out_dir=args.out_dir,
            feature_scores_path=args.feature_scores,
            v425_dir=args.v425_dir,
        )


if __name__ == "__main__":
    main()
