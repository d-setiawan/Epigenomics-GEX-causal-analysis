#!/usr/bin/env python3
"""Audit CA-BOSS v4 empirical-null state calibration on cached worlds only.

Phase A freezes the observable feature table, a block-balanced random replay
panel, the candidate weight grid, and all executable source hashes before any
truth table is opened.  Phase B verifies that closure, then evaluates the
entire training procedure with nested leave-one-source-block-out prediction:
the outer block is never used to choose the score weight, fit the empirical
null, or set its threshold.

The resulting tier is an additive within-assay shared-state annotation.  It
does not alter existing strict/equivalence membership, direct edges, assay
participation, or any cross-assay pair/direction claim.  The full-data frozen
calibrator emitted by Phase B is a future-application artifact, not a score on
its own training rows.

No command in this utility can generate or modify a simulation world.
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

import empirical_null_state_calibration_v4 as calibration  # noqa: E402


SCHEMA = "ca_boss_v4.nested_empirical_null_state_calibration.v1"
PHASE_A_TABLES = (
    "phase_a_observable_feature_evidence.tsv",
    "phase_a_random_panels.tsv",
)
PHASE_B_TABLES = (
    "phase_b_outer_feature_predictions.tsv",
    "phase_b_outer_block_metrics.tsv",
    "phase_b_inner_weight_selection.tsv",
    "phase_b_random_panel_metrics.tsv",
)
SOURCE_FILES = (
    "audit_nested_empirical_null_state_calibration_v4.py",
    "empirical_null_state_calibration_v4.py",
    "state_equivalence_membership_v4.py",
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
    """Select deterministic block-balanced panels without outcome columns."""

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
    observable.to_csv(
        out_dir / PHASE_A_TABLES[0], sep="\t", index=False
    )
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
        "conditional_weight_grid": list(
            calibration.DEFAULT_CONDITIONAL_WEIGHT_GRID
        ),
        "weight_tie_target": calibration.DEFAULT_CONDITIONAL_WEIGHT,
        "weight_selection": (
            "maximum inner true additions subject to zero inner false additions; "
            "ties nearest 1.5 then lower declared grid position"
        ),
        "threshold_method": (
            "robust-null composite with one source-block-maximum spacing "
            "extrapolated beyond the training-null maximum"
        ),
        "selection_seed": int(selection_seed),
        "random_panels": int(panels),
        "panel_size": int(panel_size),
        "worlds": int(observable[["block", "seed"]].drop_duplicates().shape[0]),
        "source_blocks": int(observable["block"].nunique()),
        "observable_feature_rows": int(len(observable)),
        "calibration_candidate_rows": int(
            observable["empirical_null_calibration_candidate"].sum()
        ),
        "scorer_or_truth_opened": False,
        "new_worlds_or_datasets_created": False,
        "retrospective_development_only": True,
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
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Phase A is not cached-world-only")
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


def _label_table(path: Path) -> pd.DataFrame:
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
    return output


def _fit_apply(
    merged: pd.DataFrame,
    *,
    fit_blocks: Sequence[str],
    apply_block: str,
    conditional_weight: float,
) -> tuple[pd.DataFrame, calibration.FrozenEmpiricalNullStateCalibrator]:
    fit = merged["block"].astype(str).isin(fit_blocks)
    apply = merged["block"].astype(str).eq(str(apply_block))
    train_observable = merged.loc[fit, list(calibration.OBSERVABLE_COLUMNS)]
    train_null = ~merged.loc[fit, "truth_state_member"].astype(bool)
    calibrator = calibration.fit_empirical_null_state_calibrator(
        train_observable,
        null_feature=train_null,
        conditional_weight=conditional_weight,
    )
    annotated = calibration.annotate_empirical_null_state_membership(
        merged.loc[apply, list(calibration.OBSERVABLE_COLUMNS)],
        calibrator=calibrator,
    )
    return annotated, calibrator


def _inner_weight_metrics(
    merged: pd.DataFrame,
    *,
    outer_block: str,
    weight: float,
) -> tuple[int, int, int]:
    training_blocks = sorted(
        set(merged["block"].astype(str)) - {str(outer_block)}
    )
    true_additions = 0
    false_additions = 0
    selected_rows = 0
    for inner_block in training_blocks:
        fit_blocks = [block for block in training_blocks if block != inner_block]
        annotated, _ = _fit_apply(
            merged,
            fit_blocks=fit_blocks,
            apply_block=inner_block,
            conditional_weight=weight,
        )
        labels = merged.loc[
            merged["block"].astype(str).eq(inner_block),
            [*KEY_COLUMNS, "truth_state_member"],
        ]
        scored = annotated.merge(labels, on=KEY_COLUMNS, validate="one_to_one")
        addition = scored["empirical_null_state_member"].astype(bool)
        truth = scored["truth_state_member"].astype(bool)
        true_additions += int((addition & truth).sum())
        false_additions += int((addition & ~truth).sum())
        selected_rows += int(addition.sum())
    return true_additions, false_additions, selected_rows


def _choose_weight(
    metrics: Sequence[dict[str, Any]],
    *,
    tie_target: float,
) -> float | None:
    eligible = [row for row in metrics if int(row["false_additions"]) == 0]
    if not eligible:
        return None
    best = max(
        eligible,
        key=lambda row: (
            int(row["true_additions"]),
            -abs(float(row["conditional_weight"]) - tie_target),
            -int(row["grid_index"]),
        ),
    )
    return float(best["conditional_weight"])


def _metrics(
    frame: pd.DataFrame,
    *,
    selected_column: str,
) -> dict[str, Any]:
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


def run_phase_b(
    *,
    out_dir: Path,
    feature_scores_path: Path,
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_full_calibrator.json", "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")
    manifest, observable, panels = _load_phase_a(out_dir)
    label_path = _regular_file(feature_scores_path, observable=False)
    labels = _label_table(label_path)
    merged = observable.merge(labels, on=KEY_COLUMNS, validate="one_to_one")
    if len(merged) != len(observable):
        raise ValueError("Truth label table does not cover every observable row")
    blocks = sorted(set(merged["block"].astype(str)))
    if len(blocks) < 4:
        raise ValueError("Nested source-block evaluation requires at least four blocks")
    weights = [float(value) for value in manifest["conditional_weight_grid"]]
    tie_target = float(manifest["weight_tie_target"])

    weight_rows: list[dict[str, Any]] = []
    outer_predictions: list[pd.DataFrame] = []
    outer_metric_rows: list[dict[str, Any]] = []
    for outer_block in blocks:
        current_metrics: list[dict[str, Any]] = []
        for grid_index, weight in enumerate(weights):
            tp, fp, selected = _inner_weight_metrics(
                merged, outer_block=outer_block, weight=weight
            )
            row = {
                "outer_held_block": outer_block,
                "grid_index": grid_index,
                "conditional_weight": weight,
                "true_additions": tp,
                "false_additions": fp,
                "selected_additions": selected,
                "zero_false_gate": fp == 0,
                "selected_weight": False,
            }
            current_metrics.append(row)
            weight_rows.append(row)
        chosen_weight = _choose_weight(current_metrics, tie_target=tie_target)
        held = merged.loc[merged["block"].astype(str).eq(outer_block)].copy()
        if chosen_weight is None:
            annotated = calibration.annotate_empirical_null_state_membership(
                held.loc[:, list(calibration.OBSERVABLE_COLUMNS)],
                calibrator=calibration.FrozenEmpiricalNullStateCalibrator(
                    schema_version=calibration.SCHEMA,
                    conditional_weight=tie_target,
                    marginal_null_median=0.0,
                    marginal_null_iqr=1.0,
                    conditional_excess_null_median=0.0,
                    conditional_excess_null_iqr=1.0,
                    maximum_null_score=0.0,
                    second_largest_block_maximum_null_score=0.0,
                    block_maximum_extrapolation=0.0,
                    selection_threshold=float(np.nextafter(0.0, np.inf)),
                    training_source_blocks=("abstain_a", "abstain_b"),
                    training_candidate_rows=1,
                    training_null_rows=1,
                ),
            )
            annotated["empirical_null_state_member"] = False
            annotated["state_calibrated_supported_member"] = annotated[
                "state_supported_member"
            ]
            annotated["state_calibrated_membership_tier"] = annotated[
                "state_membership_tier"
            ]
            threshold = float("inf")
            correction = float("nan")
        else:
            for row in weight_rows:
                if (
                    row["outer_held_block"] == outer_block
                    and float(row["conditional_weight"]) == chosen_weight
                ):
                    row["selected_weight"] = True
            fit_blocks = [block for block in blocks if block != outer_block]
            annotated, fitted = _fit_apply(
                merged,
                fit_blocks=fit_blocks,
                apply_block=outer_block,
                conditional_weight=chosen_weight,
            )
            threshold = fitted.selection_threshold
            correction = fitted.block_maximum_extrapolation
        annotated = annotated.merge(
            held.loc[:, [*KEY_COLUMNS, "truth_role", "truth_state_member"]],
            on=KEY_COLUMNS,
            validate="one_to_one",
        )
        annotated["outer_held_block"] = outer_block
        annotated["outer_chosen_conditional_weight"] = (
            chosen_weight if chosen_weight is not None else np.nan
        )
        annotated["outer_truth_used_for_fit_or_threshold"] = False
        outer_predictions.append(annotated)
        addition = annotated["empirical_null_state_member"].astype(bool)
        truth = annotated["truth_state_member"].astype(bool)
        base = _metrics(annotated, selected_column="state_supported_member")
        union = _metrics(
            annotated, selected_column="state_calibrated_supported_member"
        )
        false_worlds = int(
            annotated.loc[addition & ~truth, ["block", "seed"]]
            .drop_duplicates()
            .shape[0]
        )
        outer_metric_rows.append(
            {
                "outer_held_block": outer_block,
                "chosen_conditional_weight": chosen_weight,
                "selection_threshold": threshold,
                "block_maximum_extrapolation": correction,
                "worlds": int(annotated[["block", "seed"]].drop_duplicates().shape[0]),
                "base_true_positive_members": base["true_positive_members"],
                "base_false_positive_members": base["false_positive_members"],
                "base_false_negative_members": base["false_negative_members"],
                "base_recall": base["recall"],
                "added_true_members": int((addition & truth).sum()),
                "added_false_members": int((addition & ~truth).sum()),
                "false_addition_worlds": false_worlds,
                "calibrated_true_positive_members": union["true_positive_members"],
                "calibrated_false_positive_members": union["false_positive_members"],
                "calibrated_false_negative_members": union["false_negative_members"],
                "calibrated_precision": union["precision"],
                "calibrated_recall": union["recall"],
            }
        )

    predictions = pd.concat(outer_predictions, ignore_index=True)
    block_metrics = pd.DataFrame(outer_metric_rows)
    weight_selection = pd.DataFrame(weight_rows)
    if predictions.duplicated(KEY_COLUMNS).any() or len(predictions) != len(merged):
        raise RuntimeError("Outer predictions do not form one complete held-block table")

    panel_metrics_rows: list[dict[str, Any]] = []
    for panel_id, selection in panels.groupby("panel_id", sort=True):
        current = predictions.merge(
            selection.loc[:, ["block", "seed"]],
            on=["block", "seed"],
            validate="many_to_one",
        )
        base = _metrics(current, selected_column="state_supported_member")
        union = _metrics(
            current, selected_column="state_calibrated_supported_member"
        )
        addition = current["empirical_null_state_member"].astype(bool)
        truth = current["truth_state_member"].astype(bool)
        panel_metrics_rows.append(
            {
                "panel_id": panel_id,
                "selection_seed": int(
                    selection["selection_seed"].astype(int).iloc[0]
                ),
                "worlds": int(selection[["block", "seed"]].drop_duplicates().shape[0]),
                "base_precision": base["precision"],
                "base_recall": base["recall"],
                "added_true_members": int((addition & truth).sum()),
                "added_false_members": int((addition & ~truth).sum()),
                "calibrated_precision": union["precision"],
                "calibrated_recall": union["recall"],
            }
        )
    panel_metrics = pd.DataFrame(panel_metrics_rows)

    full_weight_rows: list[dict[str, Any]] = []
    for grid_index, weight in enumerate(weights):
        tp = fp = selected = 0
        for held_block in blocks:
            annotated, _ = _fit_apply(
                merged,
                fit_blocks=[block for block in blocks if block != held_block],
                apply_block=held_block,
                conditional_weight=weight,
            )
            held_truth = merged.loc[
                merged["block"].astype(str).eq(held_block),
                "truth_state_member",
            ].astype(bool).to_numpy()
            addition = annotated["empirical_null_state_member"].astype(bool).to_numpy()
            tp += int((addition & held_truth).sum())
            fp += int((addition & ~held_truth).sum())
            selected += int(addition.sum())
        full_weight_rows.append(
            {
                "grid_index": grid_index,
                "conditional_weight": weight,
                "true_additions": tp,
                "false_additions": fp,
                "selected_additions": selected,
            }
        )
    full_weight = _choose_weight(full_weight_rows, tie_target=tie_target)
    if full_weight is None:
        raise RuntimeError("No full-data weight passes zero-false crossfit")
    full_calibrator = calibration.fit_empirical_null_state_calibrator(
        merged.loc[:, list(calibration.OBSERVABLE_COLUMNS)],
        null_feature=~merged["truth_state_member"].astype(bool),
        conditional_weight=full_weight,
    )
    calibrator_payload = {
        **full_calibrator.to_payload(),
        "artifact_role": "future_application_only",
        "training_label_input_sha256": _sha256(label_path),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "weight_selection_crossfit": full_weight_rows,
        "training_rows_are_not_validation_rows": True,
        "application_requires_truth": False,
        "new_worlds_or_datasets_created": False,
    }
    calibrator_payload["artifact_payload_sha256"] = _canonical_sha256(
        calibrator_payload
    )
    _write_json(out_dir / "phase_b_full_calibrator.json", calibrator_payload)

    predictions.to_csv(out_dir / PHASE_B_TABLES[0], sep="\t", index=False)
    block_metrics.to_csv(out_dir / PHASE_B_TABLES[1], sep="\t", index=False)
    weight_selection.to_csv(out_dir / PHASE_B_TABLES[2], sep="\t", index=False)
    panel_metrics.to_csv(out_dir / PHASE_B_TABLES[3], sep="\t", index=False)

    base = _metrics(predictions, selected_column="state_supported_member")
    union = _metrics(
        predictions, selected_column="state_calibrated_supported_member"
    )
    addition = predictions["empirical_null_state_member"].astype(bool)
    truth = predictions["truth_state_member"].astype(bool)
    added_tp = int((addition & truth).sum())
    added_fp = int((addition & ~truth).sum())
    false_worlds = int(
        predictions.loc[addition & ~truth, ["block", "seed"]]
        .drop_duplicates()
        .shape[0]
    )
    worlds = int(predictions[["block", "seed"]].drop_duplicates().shape[0])
    role_recall = {
        role: float(
            predictions.loc[
                predictions["truth_role"].astype(str).eq(role),
                "state_calibrated_supported_member",
            ].astype(bool).mean()
        )
        for role in ("P", "Q", "C", "S", "RNA")
    }
    summary = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": calibration.SCHEMA,
        "phase": "b",
        "retrospective_development_only": True,
        "fresh_locked_claim": False,
        "new_worlds_or_datasets_created": False,
        "worlds": worlds,
        "source_blocks": len(blocks),
        "evaluation_protocol": (
            "nested leave-one-source-block-out weight selection, null fitting, "
            "threshold calibration, and outer prediction"
        ),
        "base_v424_state_supported_layer": base,
        "nested_empirical_null_addition": {
            "true_positive_additions": added_tp,
            "false_positive_additions": added_fp,
            "false_addition_worlds": false_worlds,
            "world_familywise_false_addition_rate": float(false_worlds / worlds),
            "outer_blocks_with_zero_false_additions": int(
                block_metrics["added_false_members"].astype(int).eq(0).sum()
            ),
            "outer_blocks": len(block_metrics),
            "conditional_weights_selected": sorted(
                set(
                    float(value)
                    for value in block_metrics[
                        "chosen_conditional_weight"
                    ].dropna()
                )
            ),
            "minimum_block_recall": float(block_metrics["calibrated_recall"].min()),
            "maximum_block_recall": float(block_metrics["calibrated_recall"].max()),
        },
        "state_calibrated_union": {
            **union,
            "role_recall": role_recall,
            "strict_or_equivalence_membership_removed": 0,
            "direct_edges_created_or_changed": 0,
            "pairwise_cross_assay_edges_created": 0,
        },
        "random_cached_panels": {
            "panels": len(panel_metrics),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "all_panels_zero_false_additions": bool(
                panel_metrics["added_false_members"].astype(int).eq(0).all()
            ),
            "mean_base_recall": float(panel_metrics["base_recall"].mean()),
            "mean_calibrated_recall": float(
                panel_metrics["calibrated_recall"].mean()
            ),
            "minimum_calibrated_recall": float(
                panel_metrics["calibrated_recall"].min()
            ),
            "maximum_calibrated_recall": float(
                panel_metrics["calibrated_recall"].max()
            ),
        },
        "future_application_calibrator": {
            "conditional_weight": full_weight,
            "artifact": "phase_b_full_calibrator.json",
            "artifact_sha256": _sha256(out_dir / "phase_b_full_calibrator.json"),
            "scored_as_training_validation": False,
            "application_requires_truth": False,
        },
        "gates": {
            "all_outer_blocks_zero_false_additions": bool(
                block_metrics["added_false_members"].astype(int).eq(0).all()
            ),
            "zero_outer_world_fwer": false_worlds == 0,
            "precision_preserved_at_one": union["precision"] == 1.0,
            "recall_improved": union["recall"] > base["recall"],
            "all_random_panels_zero_false_additions": bool(
                panel_metrics["added_false_members"].astype(int).eq(0).all()
            ),
            "no_pairwise_or_direct_claims": True,
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
    parser.add_argument("--selection-seed", type=int, default=20260914)
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
        )


if __name__ == "__main__":
    main()
