#!/usr/bin/env python3
"""Audit a conservative LOO cross-fit disagreement gate on retired TRAIN.

The expensive observed-only candidate universe is reused byte-for-byte from
``retired_crossfit_hierarchical_collider_audit_v1``.  This script first
reproduces the C=0.03 calibrated star selected for every held-out block using
truth labels from the other three blocks only.  It then evaluates a posthoc
two-tier disagreement gate:

Tier A
    calibrated probability >= 0.99 and cross-fitted minimum drop-one gain > 0.
Tier B
    calibrated probability >= 0.70, joint gain >= 0.009, minimum parent
    positive posterior >= 0.94, and both proposed parents pass the positive
    state/conditional-Wald consistency check.

An assay is replaced only when the selected star differs from production.
The thresholds were assembled after retired labels were available.  Thus this
is development evidence for a future fresh TRAIN block, never certification.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

import audit_retired_crossfit_hierarchical_collider_v1 as crossfit
from rescore_retired_crossfit_hierarchical_collider_v1 import (
    _aggregate,
    _edges_from_stars,
    _score_edges,
    _truth_edges,
)


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
TARGET_SEEDS = (36023, 38010, 38011, 38012, 38014)
SCHEMA = "ca_boss_v4.retired_crossfit_disagreement_gate.v1"
BASELINE_THRESHOLDS = {
    "tier_a_minimum_calibrated_score": 0.99,
    "tier_a_minimum_drop_one_gain": 0.0,
    "tier_b_minimum_calibrated_score": 0.70,
    "tier_b_minimum_joint_gain": 0.009,
    "tier_b_minimum_parent_positive_posterior": 0.94,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _verify_inputs(input_dir: Path) -> dict[str, Any]:
    phase_a_path = input_dir / "phase_a_manifest.json"
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    if phase_a.get("phase") != "observed_only_before_truth":
        raise ValueError("Observed-only Phase-A barrier is absent")
    if phase_a.get("truth_or_scorer_opened") is not False:
        raise ValueError("Phase A reports truth/scorer access")
    if phase_a.get("dev_or_locked_opened") is not False:
        raise ValueError("Phase A reports DEV/LOCKED access")
    for name, expected in phase_a["output_sha256"].items():
        if _sha256(input_dir / name) != expected:
            raise ValueError(f"Frozen Phase-A drift: {name}")
    if _sha256(Path(crossfit.__file__)) != phase_a["source_sha256"]:
        raise ValueError("Cross-fit source no longer matches frozen Phase A")

    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase_a_manifest_sha256") != _sha256(phase_a_path):
        raise ValueError("Phase-B outputs are not bound to frozen Phase A")
    for name in (
        "phase_b_labeled_crossfit_star_candidates.tsv",
        "phase_b_selected_stars.tsv",
    ):
        if _sha256(input_dir / name) != manifest["output_sha256"][name]:
            raise ValueError(f"Frozen Phase-B drift: {name}")
    return {
        "phase_a_manifest_sha256": _sha256(phase_a_path),
        "phase_b_manifest_sha256": _sha256(manifest_path),
        "candidate_sha256": phase_a["output_sha256"][
            "phase_a_crossfit_star_candidates.tsv"
        ],
        "crossfit_source_sha256": phase_a["source_sha256"],
    }


def _reproduce_held_block_scores(
    labels: pd.DataFrame, saved: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    matrix, numeric, categorical = crossfit._candidate_matrix(labels)
    reproduced: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for held_block in BLOCKS:
        train = labels["block"].ne(held_block)
        test = ~train
        model = crossfit._models(numeric, categorical)["calibrated_logit_C0.03"]
        model.fit(matrix.loc[train], labels.loc[train, "truth_exact_star"])
        scored = labels.loc[test].copy()
        scored["calibrated_score"] = model.predict_proba(matrix.loc[test])[:, 1]
        selected = crossfit._top_predictions(scored, "calibrated_score")
        selected.insert(0, "method", "calibrated_logit_C0.03")
        reproduced.append(selected)

        expected = saved.loc[saved["block"].astype(str).eq(held_block)].copy()
        keys = ["block", "seed", "source_assay", "center", "outer1", "outer2"]
        left = selected.sort_values(keys).reset_index(drop=True)
        right = expected.sort_values(keys).reset_index(drop=True)
        identities_match = bool(left[keys].equals(right[keys]))
        score_difference = float(
            np.max(np.abs(left["calibrated_score"] - right["calibrated_score"]))
        )
        margin_difference = float(
            np.max(np.abs(left["candidate_margin"] - right["candidate_margin"]))
        )
        audit_rows.append(
            {
                "held_out_block": held_block,
                "training_blocks": ";".join(
                    block for block in BLOCKS if block != held_block
                ),
                "held_out_truth_rows_used_in_fit": 0,
                "n_selected_stars": len(left),
                "selected_identities_match_frozen_output": identities_match,
                "maximum_calibrated_score_absolute_difference": score_difference,
                "maximum_candidate_margin_absolute_difference": margin_difference,
                "exact_numeric_reproduction": bool(
                    identities_match
                    and score_difference <= 1e-12
                    and margin_difference <= 1e-12
                ),
            }
        )
    reproduction = pd.concat(reproduced, ignore_index=True, sort=False)
    audit = pd.DataFrame(audit_rows)
    if not audit["exact_numeric_reproduction"].all():
        raise ValueError("Held-block-out calibration did not reproduce exactly")
    return reproduction, audit


def _edge_set(row: Any) -> set[tuple[str, str]]:
    return {(str(row.outer1), str(row.center)), (str(row.outer2), str(row.center))}


def _select_replacements(
    selected: pd.DataFrame,
    production: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    production_by_assay = {
        (str(block), int(seed), str(assay)): {
            (str(row.parent), str(row.child)) for row in frame.itertuples(index=False)
        }
        for (block, seed, assay), frame in production.groupby(
            ["block", "seed", "source_assay"], sort=True
        )
    }
    differs = [
        _edge_set(row)
        != production_by_assay.get(
            (str(row.block), int(row.seed), str(row.source_assay)), set()
        )
        for row in selected.itertuples(index=False)
    ]
    tier_a = (
        selected["calibrated_score"].ge(
            thresholds["tier_a_minimum_calibrated_score"]
        )
        & selected["crossfit_min_drop_one_gain"].gt(
            thresholds["tier_a_minimum_drop_one_gain"]
        )
    )
    tier_b = (
        selected["calibrated_score"].ge(
            thresholds["tier_b_minimum_calibrated_score"]
        )
        & selected["crossfit_joint_gain_over_baseline"].ge(
            thresholds["tier_b_minimum_joint_gain"]
        )
        & selected["laplace_min_parent_positive_posterior"].ge(
            thresholds["tier_b_minimum_parent_positive_posterior"]
        )
        & selected["two_parent_positive_state_wald_consistency"].astype(bool)
    )
    output = selected.loc[pd.Series(differs, index=selected.index) & (tier_a | tier_b)].copy()
    output["tier_a_pass"] = tier_a.loc[output.index].astype(bool)
    output["tier_b_pass"] = tier_b.loc[output.index].astype(bool)
    output["selection_tier"] = [
        "A_and_B" if row.tier_a_pass and row.tier_b_pass else "A" if row.tier_a_pass else "B"
        for row in output.itertuples(index=False)
    ]
    return output


def _apply_replacements(
    production: pd.DataFrame, replacements: pd.DataFrame
) -> pd.DataFrame:
    replacement_edges = _edges_from_stars(replacements)
    keys = set(
        map(
            tuple,
            replacement_edges[["block", "seed", "source_assay"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    retained = production.loc[
        [
            (str(row.block), int(row.seed), str(row.source_assay)) not in keys
            for row in production.itertuples(index=False)
        ]
    ]
    return pd.concat([retained, replacement_edges], ignore_index=True, sort=False)


def _sensitivity_grid(
    selected: pd.DataFrame,
    production: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    values = {
        "tier_a_minimum_calibrated_score": (0.985, 0.990, 0.995),
        "tier_a_minimum_drop_one_gain": (0.0, 0.00025, 0.00050),
        "tier_b_minimum_calibrated_score": (0.65, 0.70, 0.75),
        "tier_b_minimum_joint_gain": (0.0085, 0.0090, 0.0095),
        "tier_b_minimum_parent_positive_posterior": (0.93, 0.94, 0.95),
    }
    names = list(values)
    rows: list[dict[str, Any]] = []
    for index, combination in enumerate(itertools.product(*(values[name] for name in names))):
        thresholds = dict(zip(names, combination, strict=True))
        replacements = _select_replacements(selected, production, thresholds)
        edges = _apply_replacements(production, replacements)
        metrics = _score_edges(f"grid_{index:03d}", edges, truth)
        by_block = metrics.groupby("block")["direct_reliable_precision99"].sum()
        rows.append(
            {
                "grid_id": f"grid_{index:03d}",
                **thresholds,
                "n_replacements": len(replacements),
                "n_exact_replacements": int(replacements["truth_exact_star"].sum()),
                "n_false_replacements": int(
                    (~replacements["truth_exact_star"].astype(bool)).sum()
                ),
                "replacement_precision": float(
                    replacements["truth_exact_star"].mean()
                )
                if len(replacements)
                else math.nan,
                **{
                    f"{block}_reliable_worlds": int(by_block.get(block, 0))
                    for block in BLOCKS
                },
                "all_blocks_meet_22_of_24": bool(
                    all(int(by_block.get(block, 0)) >= 22 for block in BLOCKS)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_crossfit_hierarchical_collider_audit_v1"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "retired_crossfit_disagreement_gate_audit_v1"
        ),
    )
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    input_identity = _verify_inputs(args.input_dir)

    labels = pd.read_csv(
        args.input_dir / "phase_b_labeled_crossfit_star_candidates.tsv", sep="\t"
    )
    frozen_selected = pd.read_csv(
        args.input_dir / "phase_b_selected_stars.tsv", sep="\t"
    )
    frozen_selected = frozen_selected.loc[
        frozen_selected["method"].astype(str).eq("calibrated_logit_C0.03")
    ].copy()
    production = pd.read_csv(
        args.input_dir / "phase_a_production_direct_edges.tsv", sep="\t"
    )
    truth = _truth_edges(labels)
    selected, reproduction = _reproduce_held_block_scores(labels, frozen_selected)

    replacements = _select_replacements(
        selected, production, BASELINE_THRESHOLDS
    )
    predicted = _apply_replacements(production, replacements)
    world = _score_edges("posthoc_crossfit_disagreement_gate", predicted, truth)
    aggregate = _aggregate(world)
    sensitivity = _sensitivity_grid(selected, production, truth)

    replacement_path = args.out_dir / "selected_replacements.tsv"
    reproduction_path = args.out_dir / "held_block_score_reproduction.tsv"
    world_path = args.out_dir / "world_direct_metrics.tsv"
    aggregate_path = args.out_dir / "decoder_metrics.tsv"
    sensitivity_path = args.out_dir / "threshold_sensitivity_grid.tsv"
    replacements.to_csv(replacement_path, sep="\t", index=False)
    reproduction.to_csv(reproduction_path, sep="\t", index=False)
    world.to_csv(world_path, sep="\t", index=False)
    aggregate.to_csv(aggregate_path, sep="\t", index=False)
    sensitivity.to_csv(sensitivity_path, sep="\t", index=False)

    by_block = aggregate.loc[aggregate["scope"].isin(BLOCKS)].set_index("scope")
    summary = {
        "schema_version": SCHEMA,
        "status": "posthoc_retired_train_development_not_production_certification",
        "thresholds_assembled_after_retired_labels": True,
        "frozen_observed_phase_a_recomputed_or_modified": False,
        "held_block_calibration_exactly_reproduced": bool(
            reproduction["exact_numeric_reproduction"].all()
        ),
        "held_out_block_truth_rows_used_in_each_fit": 0,
        "baseline_thresholds": BASELINE_THRESHOLDS,
        "n_replacements": int(len(replacements)),
        "n_exact_replacements": int(replacements["truth_exact_star"].sum()),
        "n_false_replacements": int(
            (~replacements["truth_exact_star"].astype(bool)).sum()
        ),
        "block_reliable_worlds_precision99": {
            block: int(by_block.loc[block, "reliable_worlds_precision99"])
            for block in BLOCKS
        },
        "all_blocks_meet_22_of_24": bool(
            by_block["meets_22_of_24"].astype(bool).all()
        ),
        "sensitivity_grid": {
            "n_configurations": int(len(sensitivity)),
            "n_all_block_qualifying": int(
                sensitivity["all_blocks_meet_22_of_24"].sum()
            ),
            "n_zero_false_replacement": int(
                sensitivity["n_false_replacements"].eq(0).sum()
            ),
            "n_qualifying_and_zero_false_replacement": int(
                (
                    sensitivity["all_blocks_meet_22_of_24"]
                    & sensitivity["n_false_replacements"].eq(0)
                ).sum()
            ),
        },
        "input_identity": input_identity,
        "dev_or_locked_opened": False,
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    outputs = (
        replacement_path,
        reproduction_path,
        world_path,
        aggregate_path,
        sensitivity_path,
        summary_path,
    )
    manifest = {
        "source_sha256": _sha256(Path(__file__)),
        "input_identity": input_identity,
        "output_sha256": {path.name: _sha256(path) for path in outputs},
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
