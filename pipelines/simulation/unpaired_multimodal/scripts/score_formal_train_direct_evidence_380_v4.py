#!/usr/bin/env python3
"""Label the immutable TRAIN380 direct-evidence Phase-A freeze.

This is a retired-TRAIN, post-score diagnostic.  It refuses to proceed unless
every frozen Phase-A file and every frozen source dependency still matches the
hash recorded before scorer access.  It then opens only TRAIN380 scorer files
and registered score records.  DEV and LOCKED paths are neither parameters nor
inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import diagnose_formal_train_direct_failures_v4 as diagnostic
from freeze_formal_train_direct_evidence_380_v4 import PHASE_A_KEYS, SEEDS


WORLD_ROOT = Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_38001_38024")
RUN_ROOT = Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_38001_38024")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _verify_phase_a(phase_a: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = phase_a / "phase_a_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("seeds") != list(SEEDS):
        raise ValueError("Phase A does not cover exactly TRAIN380")
    if not manifest.get("all_observed_candidates_loaded_before_truth_access"):
        raise ValueError("Phase-A information barrier was not recorded")
    if manifest.get("formal_world_root_or_scorer_opened") is not False:
        raise ValueError("Phase A reports scorer access")
    for name, expected in manifest["phase_a_output_sha256"].items():
        path = phase_a / name
        actual = diagnostic._sha256(path)
        if actual != expected:
            raise ValueError(f"Phase-A output drift: {name}")
    for name, expected in manifest["source_sha256"].items():
        path = Path(name)
        actual = diagnostic._sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen source drift: {name}")

    frames: dict[str, pd.DataFrame] = {}
    for key in PHASE_A_KEYS:
        frames[key] = diagnostic._read_tsv(phase_a / f"phase_a_{key}.tsv")
    return manifest, frames


def _worlds_from_frozen(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    worlds: list[dict[str, Any]] = []
    for seed in SEEDS:
        world: dict[str, Any] = {"seed": seed}
        for key, frame in frames.items():
            if "seed" not in frame.columns:
                world[key] = frame.iloc[0:0].copy()
            else:
                world[key] = frame.loc[
                    pd.to_numeric(frame["seed"], errors="coerce").eq(seed)
                ].copy()
        worlds.append(world)
    return worlds


def _roles_on_edges(edges: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    role_map = {
        (int(row.seed), str(row.node)): str(row.role)
        for row in roles.itertuples(index=False)
    }
    result = edges.copy()
    result["parent_role"] = [
        role_map.get((int(seed), str(node)), "")
        for seed, node in zip(result["seed"], result["parent"])
    ]
    result["child_role"] = [
        role_map.get((int(seed), str(node)), "")
        for seed, node in zip(result["seed"], result["child"])
    ]
    return result


def _failed_gate_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, int]]:
    validation = _read_json(RUN_ROOT / "block_validation.json")
    doe_by_seed = {
        int(row["world_seed"]): int(row["DOE_cell"])
        for row in validation["result_records"]
    }
    rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        metrics = _read_json(RUN_ROOT / "scores" / f"world_seed{seed}" / "metrics.json")
        gates = metrics["preregistered_validation"]["gate_checks"]
        failed = sorted(key for key, value in gates.items() if not bool(value))
        direct = metrics["direct_score"]
        loading = metrics["assay_local_loading_pair_score"]
        state = metrics["set_scores"]["confirmed_state_response"]
        world_success = bool(
            metrics["preregistered_validation"]["world_reliability_success"]
        )
        world_rows.append(
            {
                "seed": seed,
                "DOE_cell": doe_by_seed[seed],
                "world_reliability_success": world_success,
                "failed_gate_count": len(failed),
                "failed_gates": ";".join(failed),
                "direct_tp": int(direct["n_true_predicted"]),
                "direct_fp": int(direct["n_false_predicted"]),
                "direct_fn": int(12 - direct["n_true_predicted"]),
                "direct_strict_reverse": int(direct["n_strict_reverse"]),
                "direct_precision": float(direct["skeleton_precision"]),
                "direct_recall": float(direct["skeleton_recall"]),
                "loading_tp": int(loading["n_true_expanded_pairs"]),
                "loading_fp": int(loading["n_false_expanded_pairs"]),
                "loading_recall": float(loading["pair_recall"]),
                "state_tp": int(state["n_true_selected"]),
                "state_sentinel": int(state["n_sentinel_selected"]),
                "state_recall": float(state["recall"]),
            }
        )
        for gate in failed:
            rows.append(
                {"seed": seed, "DOE_cell": doe_by_seed[seed], "failed_gate": gate}
            )
    return pd.DataFrame(rows), pd.DataFrame(world_rows), doe_by_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", required=True, type=Path)
    args = parser.parse_args()

    manifest, frames = _verify_phase_a(args.phase_a)
    worlds = _worlds_from_frozen(frames)

    # The immutable manifest and all its hashes have been checked.  Only now
    # are retired TRAIN380 scorer roles/truth and formal score records opened.
    truth_direct, truth_sets = diagnostic._load_direct_truth_after_barrier(
        worlds, WORLD_ROOT
    )
    roles = diagnostic._load_roles_after_barrier(worlds, WORLD_ROOT)
    edge_outcomes, assay_outcomes = diagnostic._direct_edge_outcomes(
        worlds, truth_sets
    )
    edge_outcomes = _roles_on_edges(edge_outcomes, roles)
    labeled_decisions = diagnostic._label_wedges(
        frames["production_repair_decisions"], truth_direct
    )
    labeled_completions = diagnostic._label_wedges(
        frames["production_completion_evidence"], truth_direct
    )
    failed_gates, world_summary, doe_by_seed = _failed_gate_rows()

    assay_outcomes.insert(
        1,
        "DOE_cell",
        assay_outcomes["seed"].astype(int).map(doe_by_seed),
    )
    decision_actions = (
        labeled_decisions.groupby(["seed", "source_assay"], sort=True)["action"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .rename("production_repair_actions")
        .reset_index()
    )
    assay_outcomes = assay_outcomes.merge(
        decision_actions, on=["seed", "source_assay"], how="left"
    )
    assay_outcomes["production_repair_actions"] = assay_outcomes[
        "production_repair_actions"
    ].fillna("none")

    repair_action_audit = (
        labeled_decisions.groupby("action", sort=True)
        .agg(
            n_decisions=("action", "size"),
            n_exact=("candidate_is_exact_true_wedge", "sum"),
            n_false=(
                "candidate_is_exact_true_wedge",
                lambda values: int((~values.astype(bool)).sum()),
            ),
        )
        .reset_index()
    )
    repair_action_audit["precision"] = (
        repair_action_audit["n_exact"] / repair_action_audit["n_decisions"]
    )

    doe_direct = (
        world_summary.groupby("DOE_cell", sort=True)
        .agg(
            n_worlds=("seed", "size"),
            n_reliable=("world_reliability_success", "sum"),
            direct_tp=("direct_tp", "sum"),
            direct_fp=("direct_fp", "sum"),
            direct_fn=("direct_fn", "sum"),
            direct_strict_reverse=("direct_strict_reverse", "sum"),
        )
        .reset_index()
    )

    outputs = {
        "phase_b_truth_direct.tsv": truth_direct,
        "phase_b_role_assignments.tsv": roles,
        "phase_b_direct_edge_outcomes.tsv": edge_outcomes,
        "phase_b_direct_assay_outcomes.tsv": assay_outcomes,
        "phase_b_labeled_production_repair_decisions.tsv": labeled_decisions,
        "phase_b_labeled_production_completion_evidence.tsv": labeled_completions,
        "phase_b_failed_gates.tsv": failed_gates,
        "phase_b_world_summary.tsv": world_summary,
        "phase_b_repair_action_audit.tsv": repair_action_audit,
        "phase_b_doe_direct_summary.tsv": doe_direct,
    }
    output_paths: list[Path] = []
    for name, frame in outputs.items():
        path = args.phase_a / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        frame.to_csv(path, sep="\t", index=False)
        output_paths.append(path)

    summary = {
        "schema_version": "ca_boss_v4.train380_direct_evidence.phase_b.v1",
        "block": "retired_formal_train_38001_38024_development_history",
        "phase_a_manifest_sha256": diagnostic._sha256(
            args.phase_a / "phase_a_manifest.json"
        ),
        "phase_a_verified_before_truth_access": True,
        "truth_opened_only_after_phase_a_verification": True,
        "dev_or_locked_opened": False,
        "n_worlds": len(SEEDS),
        "n_worlds_with_exact_direct_structure": int(
            assay_outcomes.groupby("seed")["exact"].all().sum()
        ),
        "failed_direct_worlds": sorted(
            assay_outcomes.loc[~assay_outcomes["exact"], "seed"]
            .astype(int)
            .unique()
            .tolist()
        ),
        "total_direct_false_predictions": int(
            assay_outcomes["other_false_predictions"].sum()
        ),
        "total_direct_strict_reverse_predictions": int(
            assay_outcomes["strict_reverse_predictions"].sum()
        ),
        "total_direct_false_negatives": int(assay_outcomes["fn"].sum()),
        "production_repair_actions": repair_action_audit.to_dict(orient="records"),
        "diagnostic_status": "posthoc_retired_train_not_certification",
    }
    summary_path = args.phase_a / "phase_b_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    _write_json(summary_path, summary)
    output_paths.append(summary_path)

    phase_b_manifest = {
        "schema_version": "ca_boss_v4.train380_direct_evidence.phase_b_manifest.v1",
        "phase_a_manifest_sha256": summary["phase_a_manifest_sha256"],
        "truth_opened_only_after_phase_a_verification": True,
        "phase_b_output_sha256": {
            path.name: diagnostic._sha256(path) for path in output_paths
        },
        "source_sha256": diagnostic._sha256(Path(__file__)),
    }
    manifest_path = args.phase_a / "phase_b_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    _write_json(manifest_path, phase_b_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
