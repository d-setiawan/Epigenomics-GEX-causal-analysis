#!/usr/bin/env python3
"""Correct Phase-B rescore of the frozen cross-fit collider Phase-A audit.

This script never recomputes or modifies Phase A.  It verifies every frozen
Phase-A hash, reconstructs each prediction set once, and applies the formal
per-world direct gate: precision >= 0.99, recall >= 0.90, and zero strict
reversals.  It also reports a clearly posthoc non-vacuous outer-null variant
(``p >= 2/32``); the preregistered ``p >= 1/32`` condition is explicitly
identified as vacuous because 31 permutations cannot produce a smaller p.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
TARGET_SEEDS = (38011, 38012, 38014, 38020)
SCHEMA = "ca_boss_v4.retired_crossfit_hierarchical_collider.precision99_rescore.v1"


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


def _verify_phase_a(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "phase_a_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "observed_only_before_truth":
        raise ValueError("Input does not have an observed-only Phase-A barrier")
    if manifest.get("truth_or_scorer_opened") is not False:
        raise ValueError("Input Phase A reports truth/scorer access")
    if manifest.get("dev_or_locked_opened") is not False:
        raise ValueError("Input Phase A reports DEV/LOCKED access")
    for name, expected in manifest["output_sha256"].items():
        actual = _sha256(input_dir / name)
        if actual != expected:
            raise ValueError(f"Frozen Phase-A drift: {name}")
    return manifest


def _edges_from_stars(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        for parent in (str(row.outer1), str(row.outer2)):
            rows.append(
                {
                    "block": str(row.block),
                    "seed": int(row.seed),
                    "source_assay": str(row.source_assay),
                    "parent": parent,
                    "child": str(row.center),
                }
            )
    columns = ["block", "seed", "source_assay", "parent", "child"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)


def _truth_edges(labels: pd.DataFrame) -> pd.DataFrame:
    selected = labels.loc[labels["truth_exact_star"].astype(bool)].copy()
    truth = _edges_from_stars(selected)
    expected = 12 * labels[["block", "seed"]].drop_duplicates().shape[0]
    if len(truth) != expected:
        raise ValueError("Truth-labeled candidate universe is incomplete")
    return truth


def _score_edges(method: str, predicted: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if predicted.duplicated(["block", "seed", "parent", "child"]).any():
        raise ValueError(f"Duplicate predicted edges for {method}")
    rows: list[dict[str, Any]] = []
    for block, seed in truth[["block", "seed"]].drop_duplicates().itertuples(index=False):
        observed = predicted.loc[
            predicted["block"].eq(block) & predicted["seed"].eq(seed)
        ]
        target = truth.loc[truth["block"].eq(block) & truth["seed"].eq(seed)]
        prediction = {
            (str(row.parent), str(row.child)) for row in observed.itertuples(index=False)
        }
        target_set = {
            (str(row.parent), str(row.child)) for row in target.itertuples(index=False)
        }
        tp = len(prediction & target_set)
        fp = len(prediction - target_set)
        fn = len(target_set - prediction)
        reverse = sum(
            (child, parent) in target_set for parent, child in prediction - target_set
        )
        precision = tp / len(prediction) if prediction else math.nan
        recall = tp / len(target_set)
        rows.append(
            {
                "method": method,
                "block": str(block),
                "seed": int(seed),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "strict_reverse": reverse,
                "precision": precision,
                "recall": recall,
                "direct_reliable_precision99": bool(
                    precision >= 0.99 and recall >= 0.90 and reverse == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(world: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, method_frame in world.groupby("method", sort=True):
        scopes = [("ALL_RETIRED", method_frame)] + [
            (block, method_frame.loc[method_frame["block"].eq(block)])
            for block in BLOCKS
        ]
        for scope, frame in scopes:
            tp = int(frame["tp"].sum())
            fp = int(frame["fp"].sum())
            fn = int(frame["fn"].sum())
            reliable = int(frame["direct_reliable_precision99"].sum())
            rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "strict_reverse": int(frame["strict_reverse"].sum()),
                    "precision": tp / (tp + fp) if tp + fp else math.nan,
                    "recall": tp / (tp + fn) if tp + fn else math.nan,
                    "reliable_worlds_precision99": reliable,
                    "n_worlds": int(len(frame)),
                    "meets_22_of_24": bool(
                        scope in BLOCKS and len(frame) == 24 and reliable >= 22
                    ),
                }
            )
    return pd.DataFrame(rows)


def _top_by_group(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in frame.groupby(["block", "seed", "source_assay"], sort=True):
        ordered = group.sort_values(
            [score, "center", "outer1", "outer2"],
            ascending=[False, True, True, True],
        )
        winner = ordered.iloc[0].copy()
        runner = float(ordered.iloc[1][score]) if len(ordered) > 1 else -math.inf
        winner["candidate_margin"] = float(
            winner[score] - runner
        )
        rows.append(winner)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else frame.iloc[0:0].copy()


def _hybrid(
    production: pd.DataFrame,
    replacements: pd.DataFrame,
    risk_status: pd.DataFrame,
    *,
    abstain_when_missing: bool,
) -> pd.DataFrame:
    replacement_keys = set(
        map(
            tuple,
            replacements[["block", "seed", "source_assay"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )
    risky_keys = set(
        map(
            tuple,
            risk_status.loc[
                risk_status["risk_trigger"].astype(bool),
                ["block", "seed", "source_assay"],
            ].itertuples(index=False, name=None),
        )
    )
    removed = risky_keys if abstain_when_missing else replacement_keys
    retained = production.loc[
        [
            (str(row.block), int(row.seed), str(row.source_assay)) not in removed
            for row in production.itertuples(index=False)
        ]
    ]
    return pd.concat([retained, replacements], ignore_index=True, sort=False).drop_duplicates()


def _posthoc_null2_two_tier(
    labels: pd.DataFrame, status: pd.DataFrame
) -> pd.DataFrame:
    risk = labels.merge(
        status.loc[status["risk_trigger"].astype(bool)],
        on=["block", "seed", "source_assay"],
        how="inner",
        validate="many_to_one",
    )
    strong = (
        risk["laplace_min_parent_positive_posterior"].ge(0.98)
        & risk["crossfit_joint_gain_over_baseline"].gt(0.005)
        & risk["crossfit_min_drop_one_gain"].gt(0.0)
        & risk["laplace_both_positive_fold_support"].ge(0.80)
        & risk["crossfit_joint_gain_over_baseline_reverse_center_margin"].gt(0.0)
        & risk["maximum_arm_empirical_positive_p"].le(0.125)
        & risk["outer_empirical_null_compatibility_p"].ge(0.0625)
    )
    incomplete = risk["risk_reasons"].fillna("").str.contains(
        "production_not_exact_two_arm_star", regex=False
    )
    state_wald = risk["two_parent_positive_state_wald_consistency"].astype(bool)
    stable_incomplete = risk["minimum_collider_separation"].gt(0.0)
    eligible = risk.loc[
        strong
        & ((~incomplete & state_wald) | (incomplete & (state_wald | stable_incomplete)))
    ].copy()
    eligible["posthoc_guard_score"] = (
        eligible["crossfit_joint_gain_over_baseline"]
        + eligible["crossfit_min_drop_one_gain"]
        + 0.02 * eligible["laplace_min_parent_positive_posterior"]
        + 0.25 * eligible["residual_star_separation"]
    )
    return _top_by_group(eligible, "posthoc_guard_score")


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
            "retired_crossfit_hierarchical_collider_precision99_rescore_v1"
        ),
    )
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    phase_a = _verify_phase_a(args.input_dir)

    base_manifest_path = args.input_dir / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("phase_a_manifest_sha256") != _sha256(
        args.input_dir / "phase_a_manifest.json"
    ):
        raise ValueError("Base Phase-B manifest is not bound to frozen Phase A")
    for name in (
        "phase_b_labeled_crossfit_star_candidates.tsv",
        "phase_b_selected_stars.tsv",
    ):
        expected = base_manifest["output_sha256"][name]
        if _sha256(args.input_dir / name) != expected:
            raise ValueError(f"Base Phase-B input drift: {name}")

    labels = pd.read_csv(
        args.input_dir / "phase_b_labeled_crossfit_star_candidates.tsv", sep="\t"
    )
    selected = pd.read_csv(args.input_dir / "phase_b_selected_stars.tsv", sep="\t")
    production = pd.read_csv(
        args.input_dir / "phase_a_production_direct_edges.tsv", sep="\t"
    )
    status = pd.read_csv(
        args.input_dir / "phase_a_production_assay_status.tsv", sep="\t"
    )
    truth = _truth_edges(labels)

    edge_sets: dict[str, pd.DataFrame] = {"production": production.drop_duplicates()}
    standalone_methods = (
        "calibrated_logit_C0.03",
        "calibrated_logit_C0.3",
        "calibrated_rf_leaf8",
        "laplace_empirical_null_guarded",
        "two_tier_state_wald_guard",
    )
    for method in standalone_methods:
        stars = selected.loc[selected["method"].astype(str).eq(method)]
        edge_sets[method] = _edges_from_stars(stars)

    risk = status.loc[status["risk_trigger"].astype(bool)]
    for method in (
        "calibrated_logit_C0.03",
        "calibrated_logit_C0.3",
        "calibrated_rf_leaf8",
    ):
        replacement = edge_sets[method]
        replacement_keys = set(
            map(
                tuple,
                risk[["block", "seed", "source_assay"]].itertuples(
                    index=False, name=None
                ),
            )
        )
        replacement = replacement.loc[
            [
                (str(row.block), int(row.seed), str(row.source_assay))
                in replacement_keys
                for row in replacement.itertuples(index=False)
            ]
        ]
        edge_sets[f"risk_trigger_hybrid__{method}"] = _hybrid(
            production, replacement, status, abstain_when_missing=False
        )

    two_tier_replacement = edge_sets["two_tier_state_wald_guard"]
    edge_sets["two_tier_risk_hybrid_retain"] = _hybrid(
        production, two_tier_replacement, status, abstain_when_missing=False
    )
    edge_sets["two_tier_risk_hybrid_abstain"] = _hybrid(
        production, two_tier_replacement, status, abstain_when_missing=True
    )

    posthoc_stars = _posthoc_null2_two_tier(labels, status)
    posthoc_star_path = args.out_dir / "posthoc_null2_two_tier_selected_stars.tsv"
    posthoc_stars.to_csv(posthoc_star_path, sep="\t", index=False)
    posthoc_replacement = _edges_from_stars(posthoc_stars)
    edge_sets["posthoc_null2_two_tier_retain"] = _hybrid(
        production, posthoc_replacement, status, abstain_when_missing=False
    )
    edge_sets["posthoc_null2_two_tier_abstain"] = _hybrid(
        production, posthoc_replacement, status, abstain_when_missing=True
    )

    world = pd.concat(
        [_score_edges(method, edges, truth) for method, edges in edge_sets.items()],
        ignore_index=True,
        sort=False,
    )
    if world.duplicated(["method", "block", "seed"]).any():
        raise ValueError("Corrected rescore contains duplicate method/world rows")
    world_path = args.out_dir / "corrected_world_direct_metrics.tsv"
    world.to_csv(world_path, sep="\t", index=False)
    aggregate = _aggregate(world)
    aggregate_path = args.out_dir / "corrected_decoder_metrics.tsv"
    aggregate.to_csv(aggregate_path, sep="\t", index=False)
    target_path = args.out_dir / "corrected_train380_target_metrics.tsv"
    world.loc[world["seed"].isin(TARGET_SEEDS)].sort_values(
        ["method", "seed"]
    ).to_csv(target_path, sep="\t", index=False)

    summary = {
        "schema_version": SCHEMA,
        "status": "corrected_posthoc_retired_train_diagnostic_not_production_certification",
        "frozen_phase_a_verified": True,
        "frozen_phase_a_manifest_sha256": _sha256(
            args.input_dir / "phase_a_manifest.json"
        ),
        "phase_a_candidate_sha256": phase_a["output_sha256"][
            "phase_a_crossfit_star_candidates.tsv"
        ],
        "phase_a_recomputed_or_modified": False,
        "formal_world_gate": {
            "minimum_precision": 0.99,
            "minimum_recall": 0.90,
            "maximum_strict_reverse": 0,
        },
        "corrections": [
            "precision gate corrected from 0.80 to 0.99",
            "each method/world scored exactly once",
            "generic hybrid loop restricted to calibrated standalone methods",
            "p>=1/32 outer-null condition declared vacuous at 31 permutations",
        ],
        "posthoc_development_only": {
            "nonvacuous_outer_null_minimum": 0.0625,
            "semantics": "two of 32 permutation-tail mass; not part of frozen Phase-A formula family",
        },
        "dev_or_locked_opened": False,
        "methods": sorted(edge_sets),
    }
    summary_path = args.out_dir / "summary.json"
    _write_json(summary_path, summary)
    manifest = {
        "source_sha256": _sha256(Path(__file__)),
        "input_sha256": {
            "phase_a_manifest.json": _sha256(
                args.input_dir / "phase_a_manifest.json"
            ),
            "manifest.json": _sha256(base_manifest_path),
            "phase_b_labeled_crossfit_star_candidates.tsv": _sha256(
                args.input_dir / "phase_b_labeled_crossfit_star_candidates.tsv"
            ),
            "phase_b_selected_stars.tsv": _sha256(
                args.input_dir / "phase_b_selected_stars.tsv"
            ),
        },
        "output_sha256": {
            path.name: _sha256(path)
            for path in (
                posthoc_star_path,
                world_path,
                aggregate_path,
                target_path,
                summary_path,
            )
        },
    }
    _write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
