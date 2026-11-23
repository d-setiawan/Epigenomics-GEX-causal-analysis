#!/usr/bin/env python3
"""Posthoc TRAIN-only sentinel-null audit for assay participation.

The audit treats each opened chromatin assay's two simulator sentinel features
as a pseudo-assay, adds one inert RNA pseudo-assay, and applies the exact
predeclared two-route omnibus followed by Holm across seven pseudo-assays.  It
reads only the already-opened canonical TRAIN330/360/370/380/400 replay tables;
it is a diagnostic of null behavior, never validation or certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binomtest


SCHEMA_VERSION = "assay_participation_omnibus_sentinel_nulls.v1"
CANONICAL_REPLAY = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/current_repaired_train_replay_v2_final"
)
EXPECTED_BLOCKS = {
    "TRAIN330": tuple(range(33_001, 33_025)),
    "TRAIN360": tuple(range(36_001, 36_025)),
    "TRAIN370": tuple(range(37_001, 37_025)),
    "TRAIN380": tuple(range(38_001, 38_025)),
    "TRAIN400": tuple(range(40_001, 40_025)),
}
EXPECTED_CHROMATIN_ASSAYS = 6
EXPECTED_ASSAYS_PER_WORLD = 7
ALPHA = 0.10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} is not a strict Boolean: {value!r}")


def _holm(values: Sequence[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm inputs must be finite probabilities")
    order = np.argsort(p, kind="mergesort")
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _exact_interval(successes: int, trials: int) -> dict[str, float]:
    interval = binomtest(successes, trials).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    return {"low": float(interval.low), "high": float(interval.high)}


def _one_pseudo_assay(group: pd.DataFrame) -> dict[str, Any]:
    if len(group) != 2:
        raise ValueError("Each chromatin pseudo-assay must contain N1 and N2")
    n_views_values = pd.to_numeric(group["n_views"], errors="raise")
    if n_views_values.nunique() != 1 or int(n_views_values.iloc[0]) != 6:
        raise ValueError("Sentinel pseudo-assays require the frozen six views")
    eligible = group.loc[group["formal_test_eligible"]].sort_values(
        ["empirical_null_p", "reference_conditional_wald", "feature"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    family_size = len(eligible)
    if family_size:
        single_order_p = float(eligible.iloc[0]["empirical_null_p"])
        single_p = min(1.0, family_size * single_order_p)
        single_views = int(
            np.rint(
                float(eligible.iloc[0]["conditional_wald_view_support"]) * 6
            )
        )
        reporter = str(eligible.iloc[0]["feature"])
    else:
        single_order_p = single_p = 1.0
        single_views = 0
        reporter = ""
    single_gate = bool(family_size >= 1 and single_views == 6)

    partial_witnesses = eligible.head(2) if family_size >= 2 else eligible.iloc[0:0]
    if len(partial_witnesses) == 2:
        partial_p = min(
            1.0,
            (family_size - 2 + 1)
            * float(partial_witnesses.iloc[-1]["empirical_null_p"]),
        )
        partial_views = int(
            np.rint(
                partial_witnesses["conditional_wald_view_support"].to_numpy(
                    dtype=float
                )
                * 6
            ).min()
        )
    else:
        partial_p = 1.0
        partial_views = 0
    partial_gate = bool(len(partial_witnesses) == 2 and partial_views == 6)
    live = {
        "single_feature": single_p if single_gate else 1.0,
        "partial_conjunction": partial_p if partial_gate else 1.0,
    }
    if any(value < 1.0 for value in live.values()):
        winning_route = min(
            live,
            key=lambda route: (
                live[route],
                0 if route == "partial_conjunction" else 1,
            ),
        )
    else:
        winning_route = "none"
    return {
        "family_size": family_size,
        "single_feature_order_statistic_p": single_order_p,
        "single_feature_p": single_p,
        "single_feature_witness_view_count": single_views,
        "single_feature_replication_gate": single_gate,
        "partial_conjunction_p": partial_p,
        "partial_conjunction_minimum_witness_view_count": partial_views,
        "partial_conjunction_replication_gate": partial_gate,
        "omnibus_route_count": 2,
        "omnibus_p": min(1.0, 2.0 * min(live.values())),
        "winning_route": winning_route,
        "reporter": reporter,
    }


def run_audit(replay_root: Path, out_dir: Path) -> dict[str, Any]:
    replay = replay_root.resolve(strict=True)
    expected = CANONICAL_REPLAY.resolve(strict=True)
    if replay != expected:
        raise ValueError("Audit accepts only the canonical opened TRAIN replay")
    state_path = replay / "phase_a_state_membership.tsv"
    outcome_path = replay / "phase_b_state_feature_outcomes.tsv"
    for path in (state_path, outcome_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    state = pd.read_csv(state_path, sep="\t", keep_default_na=False)
    outcomes = pd.read_csv(outcome_path, sep="\t", keep_default_na=False)
    required_state = {
        "block",
        "seed",
        "feature",
        "modality",
        "n_views",
        "formal_test_eligible",
        "empirical_null_p",
        "reference_conditional_wald",
        "conditional_wald_view_support",
    }
    required_outcomes = {"block", "seed", "feature", "truth_sentinel"}
    if missing := sorted(required_state.difference(state.columns)):
        raise ValueError(f"State replay is missing columns: {missing}")
    if missing := sorted(required_outcomes.difference(outcomes.columns)):
        raise ValueError(f"Outcome replay is missing columns: {missing}")
    for frame, label in ((state, "state"), (outcomes, "outcome")):
        if frame.duplicated(["block", "seed", "feature"]).any():
            raise ValueError(f"{label} replay contains duplicate feature keys")
    state["formal_test_eligible"] = state["formal_test_eligible"].map(
        lambda value: _strict_bool(value, label="formal_test_eligible")
    )
    outcomes["truth_sentinel"] = outcomes["truth_sentinel"].map(
        lambda value: _strict_bool(value, label="truth_sentinel")
    )
    merged = state.merge(
        outcomes[["block", "seed", "feature", "truth_sentinel"]],
        on=["block", "seed", "feature"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(state) or len(merged) != len(outcomes):
        raise ValueError("Canonical state and outcome feature universes differ")
    observed_blocks = {
        str(block): tuple(sorted(group["seed"].astype(int).unique()))
        for block, group in merged.groupby("block", sort=True)
    }
    if observed_blocks != EXPECTED_BLOCKS:
        raise ValueError("Canonical replay does not contain the exact opened blocks")
    sentinels = merged.loc[merged["truth_sentinel"]].copy()
    if len(sentinels) != 1_440:
        raise ValueError("Expected exactly 1,440 opened sentinel feature rows")

    rows: list[dict[str, Any]] = []
    for (block, seed), world in sentinels.groupby(["block", "seed"], sort=True):
        chrom = world.loc[world["modality"].astype(str).ne("rna")]
        if chrom["modality"].nunique() != EXPECTED_CHROMATIN_ASSAYS:
            raise ValueError("Each world must contain six chromatin pseudo-assays")
        world_rows: list[dict[str, Any]] = []
        for modality, group in chrom.groupby("modality", sort=True):
            world_rows.append(
                {
                    "block": str(block),
                    "seed": int(seed),
                    "modality": str(modality),
                    **_one_pseudo_assay(group),
                }
            )
        world_rows.append(
            {
                "block": str(block),
                "seed": int(seed),
                "modality": "rna_inert_pseudo_assay",
                "family_size": 1,
                "single_feature_order_statistic_p": 1.0,
                "single_feature_p": 1.0,
                "single_feature_witness_view_count": 0,
                "single_feature_replication_gate": False,
                "partial_conjunction_p": 1.0,
                "partial_conjunction_minimum_witness_view_count": 0,
                "partial_conjunction_replication_gate": False,
                "omnibus_route_count": 1,
                "omnibus_p": 1.0,
                "winning_route": "none",
                "reporter": "",
            }
        )
        adjusted = _holm([float(row["omnibus_p"]) for row in world_rows])
        if len(adjusted) != EXPECTED_ASSAYS_PER_WORLD:
            raise AssertionError("Internal pseudo-assay count error")
        for row, holm_p in zip(world_rows, adjusted, strict=True):
            row["omnibus_holm_p"] = holm_p
            row["selected"] = bool(
                holm_p <= ALPHA and row["winning_route"] != "none"
            )
            row["selection_route"] = (
                row["winning_route"] if row["selected"] else ""
            )
            rows.append(row)

    calls = pd.DataFrame(rows).sort_values(["block", "seed", "modality"])
    chrom_calls = calls.loc[calls["modality"].ne("rna_inert_pseudo_assay")]
    selected = chrom_calls.loc[chrom_calls["selected"]]
    world_calls = selected[["block", "seed"]].drop_duplicates()
    n_assay_calls = len(selected)
    n_assay_tests = len(chrom_calls)
    n_world_calls = len(world_calls)
    n_worlds = calls[["block", "seed"]].drop_duplicates().shape[0]
    route_counts = {
        route: int(count)
        for route, count in selected["selection_route"].value_counts().items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "posthoc_opened_TRAIN_diagnostic_not_certification",
        "estimand": (
            "sentinel_only_pseudo_assay_false_participation_under_the_frozen_"
            "two_route_omnibus_and_Holm_rule"
        ),
        "data_boundary": {
            "opened_blocks": sorted(EXPECTED_BLOCKS),
            "n_opened_worlds": n_worlds,
            "dev340_opened": False,
            "locked450_opened": False,
            "train410_or_new_world_opened": False,
            "paired_cells_external_data_or_interventions_used": False,
        },
        "rule": {
            "single_route": "m_times_min_empirical_p_if_leading_witness_is_6_of_6",
            "partial_conjunction_route": (
                "m_minus_r_plus_1_times_order_r_empirical_p_if_all_r_"
                "witnesses_are_6_of_6"
            ),
            "route_combination": "Bonferroni_over_K_predeclared_live_routes",
            "route_count": "K_1_for_structural_singleton_else_K_2",
            "across_assay_control": "Holm_across_7_pseudo_assays",
            "alpha": ALPHA,
        },
        "sentinel_feature_calibration": {
            "n": len(sentinels),
            "p_at_most_0_05": int((sentinels["empirical_null_p"] <= 0.05).sum()),
            "p_at_most_0_10": int((sentinels["empirical_null_p"] <= 0.10).sum()),
        },
        "assay_level": {
            "false_calls": n_assay_calls,
            "tests": n_assay_tests,
            "rate": n_assay_calls / n_assay_tests,
            "exact_95pct_CI": _exact_interval(n_assay_calls, n_assay_tests),
            "selected_route_counts": route_counts,
        },
        "world_level": {
            "worlds_with_at_least_one_false_call": n_world_calls,
            "worlds": n_worlds,
            "familywise_rate": n_world_calls / n_worlds,
            "exact_95pct_CI": _exact_interval(n_world_calls, n_worlds),
        },
        "selected_call_identities": selected[
            [
                "block",
                "seed",
                "modality",
                "reporter",
                "selection_route",
                "omnibus_p",
                "omnibus_holm_p",
            ]
        ].to_dict(orient="records"),
        "input_sha256": {
            state_path.name: _sha256(state_path),
            outcome_path.name: _sha256(outcome_path),
        },
    }
    if (n_assay_calls, n_assay_tests, n_world_calls, n_worlds) != (6, 720, 5, 120):
        raise ValueError(
            "Sentinel-null audit changed from the reviewed 6/720 calls in "
            "5/120 worlds; inspect before accepting this rule"
        )
    output = out_dir.resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calls_path = output / "sentinel_pseudo_assay_calls.tsv"
    calls.to_csv(calls_path, sep="\t", index=False)
    summary["output_sha256"] = {calls_path.name: _sha256(calls_path)}
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, default=CANONICAL_REPLAY)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(json.dumps(run_audit(args.replay_root, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
