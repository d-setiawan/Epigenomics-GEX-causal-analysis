#!/usr/bin/env python3
"""Freeze observable loading/state candidates for formal-TRAIN diagnosis.

This is a one-way Phase-A collector for the completed 36001--36024 formal
TRAIN block.  It never imports or opens scorer artifacts, role assignments,
world parameters, or score tables.  Every world and every candidate is loaded
before one global pre-role freeze is written.  Seeds 37001 and above, as well
as all DEV/LOCKED roots, are rejected explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd


FORMAL_TRAIN_SEEDS = tuple(range(36_001, 36_025))
FORBIDDEN_PATH_TOKENS = ("score", "scorer", "role", "truth", "locked", "dev")
FORBIDDEN_COLUMN_TOKENS = (
    "truth",
    "role",
    "scorer",
    "oracle",
    "generative",
    "counterfactual",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_observable_tsv(path: Path) -> pd.DataFrame:
    lower_parts = {part.casefold() for part in path.parts}
    if any(token in lower_parts for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"Refusing scorer-domain or non-TRAIN path: {path}")
    frame = pd.read_csv(path, sep="\t")
    forbidden = sorted(
        str(column)
        for column in frame.columns
        if any(token in str(column).casefold() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"Observable input contains privileged columns: {forbidden}")
    return frame


def _single_candidate_root(world: Path) -> Path:
    roots = sorted(
        path
        for path in world.glob("v4_*")
        if path.is_dir() and (path / "structure_anchored_loading_groups").is_dir()
    )
    if len(roots) != 1:
        raise ValueError(f"Expected one formal local candidate under {world}, found {roots}")
    return roots[0]


def _pair_lookup(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "pair_key",
        "absolute_loading_cosine",
        "absolute_raw_coefficient_loading_cosine",
        "absolute_state_response_correlation",
        "loading_stability_support",
        "loading_empirical_null_p",
        "loading_empirical_null_q",
        "sign_invariant_unit_loading_distance",
        "minimum_loading_norm",
        "geometric_mean_marginal_state_fit",
        "geometric_mean_partial_state_fit",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} pair evidence lacks {missing}")
    if frame["pair_key"].astype(str).duplicated().any():
        raise ValueError(f"{label} pair evidence duplicates pair keys")
    return frame.assign(pair_key=frame["pair_key"].astype(str)).set_index("pair_key")


def _feature_lookup(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {
        "feature",
        "relative_loading_uncertainty",
        "loading_uncertainty_norm",
        "conditional_loading_norm",
        "oof_marginal_state_fit",
        "oof_partial_state_fit",
        "loading_direction_stability",
        "bootstrap_direction_stability",
        "fold_direction_stability",
        "mean_integration_uncertainty",
        "event_fraction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} feature models lack {missing}")
    if frame["feature"].astype(str).duplicated().any():
        raise ValueError(f"{label} feature models duplicate features")
    return frame.assign(feature=frame["feature"].astype(str)).set_index("feature")


def _finite_float(value: Any, label: str) -> float:
    output = float(value)
    if not np.isfinite(output):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return output


def _endpoint_values(
    lookup: pd.DataFrame,
    feature: str,
    prefix: str,
) -> dict[str, float]:
    if feature not in lookup.index:
        raise ValueError(f"Feature model missing {feature}")
    row = lookup.loc[feature]
    return {
        f"{prefix}_relative_loading_uncertainty": _finite_float(
            row["relative_loading_uncertainty"], "relative_loading_uncertainty"
        ),
        f"{prefix}_loading_uncertainty_norm": _finite_float(
            row["loading_uncertainty_norm"], "loading_uncertainty_norm"
        ),
        f"{prefix}_conditional_loading_norm": _finite_float(
            row["conditional_loading_norm"], "conditional_loading_norm"
        ),
        f"{prefix}_oof_marginal_state_fit": _finite_float(
            row["oof_marginal_state_fit"], "oof_marginal_state_fit"
        ),
        f"{prefix}_oof_partial_state_fit": _finite_float(
            row["oof_partial_state_fit"], "oof_partial_state_fit"
        ),
        f"{prefix}_loading_direction_stability": _finite_float(
            row["loading_direction_stability"], "loading_direction_stability"
        ),
        f"{prefix}_bootstrap_direction_stability": _finite_float(
            row["bootstrap_direction_stability"], "bootstrap_direction_stability"
        ),
        f"{prefix}_fold_direction_stability": _finite_float(
            row["fold_direction_stability"], "fold_direction_stability"
        ),
        f"{prefix}_mean_integration_uncertainty": _finite_float(
            row["mean_integration_uncertainty"], "mean_integration_uncertainty"
        ),
        f"{prefix}_event_fraction": _finite_float(
            row["event_fraction"], "event_fraction"
        ),
    }


def _reliability(row: dict[str, Any], first: str, second: str) -> float:
    stability = np.sqrt(
        max(float(row[f"{first}_loading_direction_stability"]), 0.0)
        * max(float(row[f"{second}_loading_direction_stability"]), 0.0)
    )
    uncertainty = np.sqrt(
        (1.0 + float(row[f"{first}_relative_loading_uncertainty"]) ** 2)
        * (1.0 + float(row[f"{second}_relative_loading_uncertainty"]) ** 2)
    )
    return float(stability / max(uncertainty, 1e-12))


def _quantile(values: pd.Series, probability: float) -> float:
    return float(np.quantile(values.to_numpy(dtype=np.float64), probability))


def _phase_a_world(world: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    candidate_root = _single_candidate_root(world)
    anchor_root = candidate_root / "structure_anchored_loading_groups"
    anchor_path = anchor_root / "structure_anchor_pair_evidence.tsv"
    decisions_path = anchor_root / "assay_decisions.tsv"
    membership_path = world / "shared_state_subspace_v4" / "subspace_feature_membership.tsv"
    anchors = _read_observable_tsv(anchor_path)
    decisions = _read_observable_tsv(decisions_path)
    membership = _read_observable_tsv(membership_path)
    if len(decisions) != 6 or decisions["modality"].nunique() != 6:
        raise ValueError(f"World {seed} does not have six assay decisions")
    if not anchors.groupby("modality", sort=True).size().isin([6]).all():
        raise ValueError(f"World {seed} has a partial structural candidate set")
    if len(anchors) != int(decisions["n_structural_candidates"].sum()):
        raise ValueError(f"World {seed} decisions disagree with structural candidates")
    if anchors["pair_key"].astype(str).duplicated().any():
        raise ValueError(f"World {seed} duplicates structural pair keys")
    if anchors["parent_membership_used_as_gate"].astype(bool).any():
        raise ValueError("Formal candidate unexpectedly gates direct-parent membership")
    membership = membership.assign(feature=membership["feature"].astype(str))
    if membership["feature"].duplicated().any():
        raise ValueError("Formal state membership duplicates features")
    membership_lookup = membership.set_index("feature")
    state_fields = (
        "selection_status",
        "empirical_null_p",
        "empirical_null_fdr",
        "oof_marginal_state_fit",
        "oof_partial_state_fit",
        "median_conditional_wald",
        "lower_quartile_conditional_wald",
        "conditional_wald_null_95pct",
        "conditional_wald_above_null_95pct",
        "conditional_wald_view_support",
        "feature_global_maxT_p",
        "feature_global_maxT_confirmation_gate",
        "assay_global_maxT_p",
        "assay_global_maxT_admission_gate",
        "median_conditional_score_direction_stability",
        "minimum_conditional_score_direction_stability",
        "direction_stability_diagnostic",
        "feature_direction_stability_confirmation_gate",
    )
    missing_state = sorted(set(state_fields) - set(membership.columns))
    if missing_state:
        raise ValueError(f"Formal state membership misses {missing_state}")

    views = sorted(
        path for path in world.glob("cells*_seed*") if path.is_dir()
    )
    if len(views) != 6:
        raise ValueError(f"World {seed} must contain six loading views")
    pair_lookups: dict[str, pd.DataFrame] = {}
    feature_lookups: dict[str, pd.DataFrame] = {}
    input_hashes = {
        str(anchor_path.relative_to(world)): _sha256(anchor_path),
        str(decisions_path.relative_to(world)): _sha256(decisions_path),
        str(membership_path.relative_to(world)): _sha256(membership_path),
    }
    for view in views:
        pair_path = view / "cell_loading_pair_evidence.tsv"
        feature_path = view / "cell_loading_feature_models.tsv"
        pair_lookups[view.name] = _pair_lookup(
            _read_observable_tsv(pair_path), view.name
        )
        feature_lookups[view.name] = _feature_lookup(
            _read_observable_tsv(feature_path), view.name
        )
        input_hashes[str(pair_path.relative_to(world))] = _sha256(pair_path)
        input_hashes[str(feature_path.relative_to(world))] = _sha256(feature_path)

    pair_view_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict(orient="records"):
        pair_key = str(anchor["pair_key"])
        opposite_key = str(anchor["opposite_parent_pair_key"])
        parent = str(anchor["parent_candidate"])
        sibling = str(anchor["noncollider_sibling_candidate"])
        other_parent = str(anchor["other_parent_candidate"])
        if sibling not in membership_lookup.index:
            raise ValueError(f"State membership lacks structural sibling {sibling}")
        state = membership_lookup.loc[sibling]
        view_rows: list[dict[str, Any]] = []
        for view_id in sorted(pair_lookups):
            pairs = pair_lookups[view_id]
            features = feature_lookups[view_id]
            if pair_key not in pairs.index or opposite_key not in pairs.index:
                raise ValueError(f"Loading view {view_id} lacks a parent comparison")
            candidate = pairs.loc[pair_key]
            opposite = pairs.loc[opposite_key]
            row: dict[str, Any] = {
                "seed": seed,
                "view_id": view_id,
                "pair_key": pair_key,
                "opposite_parent_pair_key": opposite_key,
                "modality": str(anchor["modality"]),
                "parent_candidate": parent,
                "noncollider_sibling_candidate": sibling,
                "other_parent_candidate": other_parent,
            }
            for feature, prefix in (
                (parent, "parent"),
                (sibling, "sibling"),
                (other_parent, "other_parent"),
            ):
                row.update(_endpoint_values(features, feature, prefix))
            pair_metrics = (
                "absolute_loading_cosine",
                "absolute_raw_coefficient_loading_cosine",
                "absolute_state_response_correlation",
                "loading_stability_support",
                "loading_empirical_null_p",
                "loading_empirical_null_q",
                "sign_invariant_unit_loading_distance",
                "minimum_loading_norm",
                "geometric_mean_marginal_state_fit",
                "geometric_mean_partial_state_fit",
            )
            for metric in pair_metrics:
                row[f"candidate_{metric}"] = _finite_float(
                    candidate[metric], f"candidate_{metric}"
                )
                row[f"opposite_{metric}"] = _finite_float(
                    opposite[metric], f"opposite_{metric}"
                )
            row["loading_cosine_advantage"] = (
                row["candidate_absolute_loading_cosine"]
                - row["opposite_absolute_loading_cosine"]
            )
            row["raw_coefficient_cosine_advantage"] = (
                row["candidate_absolute_raw_coefficient_loading_cosine"]
                - row["opposite_absolute_raw_coefficient_loading_cosine"]
            )
            row["state_response_correlation_advantage"] = (
                row["candidate_absolute_state_response_correlation"]
                - row["opposite_absolute_state_response_correlation"]
            )
            row["candidate_pair_max_relative_uncertainty"] = max(
                row["parent_relative_loading_uncertainty"],
                row["sibling_relative_loading_uncertainty"],
            )
            row["opposite_pair_max_relative_uncertainty"] = max(
                row["other_parent_relative_loading_uncertainty"],
                row["sibling_relative_loading_uncertainty"],
            )
            candidate_reliability = _reliability(row, "parent", "sibling")
            opposite_reliability = _reliability(row, "other_parent", "sibling")
            row["candidate_endpoint_reliability"] = candidate_reliability
            row["opposite_endpoint_reliability"] = opposite_reliability
            row["candidate_reliability_shrunk_cosine"] = (
                row["candidate_absolute_loading_cosine"] * candidate_reliability
            )
            row["opposite_reliability_shrunk_cosine"] = (
                row["opposite_absolute_loading_cosine"] * opposite_reliability
            )
            row["reliability_shrunk_cosine_advantage"] = (
                row["candidate_reliability_shrunk_cosine"]
                - row["opposite_reliability_shrunk_cosine"]
            )
            view_rows.append(row)
            pair_view_rows.append(row)
        view_frame = pd.DataFrame(view_rows)
        enriched = {"seed": seed, **anchor}
        for field in state_fields:
            enriched[f"sibling_state__{field}"] = state[field]
        aggregations = {
            "candidate_absolute_loading_cosine": ("median", "q10", "min", "sd"),
            "opposite_absolute_loading_cosine": ("median", "q10", "min", "sd"),
            "loading_cosine_advantage": ("median", "q10", "min"),
            "raw_coefficient_cosine_advantage": ("median", "q10", "min"),
            "state_response_correlation_advantage": ("median", "q10", "min"),
            "candidate_reliability_shrunk_cosine": ("median", "q10", "min"),
            "opposite_reliability_shrunk_cosine": ("median", "q10", "min"),
            "reliability_shrunk_cosine_advantage": ("median", "q10", "min"),
            "candidate_pair_max_relative_uncertainty": ("median", "q90", "max"),
            "opposite_pair_max_relative_uncertainty": ("median", "q90", "max"),
            "candidate_endpoint_reliability": ("median", "q10", "min"),
            "opposite_endpoint_reliability": ("median", "q10", "min"),
            "candidate_loading_stability_support": ("median", "q10", "min"),
            "candidate_loading_empirical_null_p": ("median", "q90", "max"),
            "candidate_loading_empirical_null_q": ("median", "q90", "max"),
            "candidate_geometric_mean_marginal_state_fit": ("median", "q10", "min"),
            "candidate_geometric_mean_partial_state_fit": ("median", "q10", "min"),
            "parent_loading_direction_stability": ("median", "q10", "min"),
            "sibling_loading_direction_stability": ("median", "q10", "min"),
            "parent_relative_loading_uncertainty": ("median", "q90", "max"),
            "sibling_relative_loading_uncertainty": ("median", "q90", "max"),
        }
        for metric, summaries in aggregations.items():
            values = view_frame[metric]
            for summary in summaries:
                if summary == "median":
                    value = float(values.median())
                elif summary == "q10":
                    value = _quantile(values, 0.10)
                elif summary == "q90":
                    value = _quantile(values, 0.90)
                elif summary == "min":
                    value = float(values.min())
                elif summary == "max":
                    value = float(values.max())
                elif summary == "sd":
                    value = float(values.std(ddof=1))
                else:
                    raise AssertionError(summary)
                enriched[f"view_{summary}__{metric}"] = value
        for metric in (
            "loading_cosine_advantage",
            "raw_coefficient_cosine_advantage",
            "state_response_correlation_advantage",
            "reliability_shrunk_cosine_advantage",
        ):
            enriched[f"view_positive_support__{metric}"] = float(
                np.mean(view_frame[metric].to_numpy(dtype=np.float64) > 0.0)
            )
        for threshold in (0.125, 0.25, 0.50):
            enriched[
                f"view_support_candidate_null_p_le_{str(threshold).replace('.', '_')}"
            ] = float(
                np.mean(
                    view_frame["candidate_loading_empirical_null_p"].to_numpy(
                        dtype=np.float64
                    )
                    <= threshold
                )
            )
        enriched_rows.append(enriched)
    enriched_frame = pd.DataFrame(enriched_rows)
    decisions = decisions.copy()
    decisions.insert(0, "seed", seed)
    return enriched_frame, pd.DataFrame(pair_view_rows), decisions, input_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_36001_36024/artifacts",
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "formal_train_36001_36024_loading_recall_diagnostic_v1/phase_a"
        ),
    )
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    if artifact_root.name != "artifacts" or artifact_root.parent.name != "train_36001_36024":
        raise ValueError("Artifact root must be the completed formal TRAIN 36001--36024 block")
    if any(token in str(artifact_root).casefold() for token in ("locked", "dev")):
        raise ValueError("DEV/LOCKED roots are forbidden")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{out_dir.name}__", dir=out_dir.parent))
    try:
        candidate_frames: list[pd.DataFrame] = []
        view_frames: list[pd.DataFrame] = []
        decision_frames: list[pd.DataFrame] = []
        input_hashes: dict[str, dict[str, str]] = {}
        for seed in FORMAL_TRAIN_SEEDS:
            if seed >= 37_001:
                raise AssertionError("Refusing future formal seeds")
            world = artifact_root / f"world_seed{seed}"
            candidates, views, decisions, hashes = _phase_a_world(world, seed)
            candidate_frames.append(candidates)
            view_frames.append(views)
            decision_frames.append(decisions)
            input_hashes[str(seed)] = hashes
        candidates = pd.concat(candidate_frames, ignore_index=True)
        views = pd.concat(view_frames, ignore_index=True)
        decisions = pd.concat(decision_frames, ignore_index=True)
        if len(views) != 6 * len(candidates) or len(decisions) != 144:
            raise ValueError("Global formal TRAIN candidate counts are inconsistent")
        candidates.to_csv(temporary / "observable_candidates.tsv", sep="\t", index=False)
        views.to_csv(temporary / "observable_pair_views.tsv", sep="\t", index=False)
        decisions.to_csv(temporary / "formal_assay_decisions.tsv", sep="\t", index=False)
        freeze = {
            "schema_version": "ca_boss_v4.formal_train_loading_recall.phase_a.v1",
            "formal_train_seeds": list(FORMAL_TRAIN_SEEDS),
            "all_24_worlds_loaded_before_any_role_or_scorer_open": True,
            "role_or_scorer_domain_read": False,
            "future_seeds_37001_or_above_read": False,
            "dev_or_locked_read": False,
            "n_candidates": len(candidates),
            "n_pair_views": len(views),
            "n_assay_decisions": len(decisions),
            "input_sha256": input_hashes,
            "output_sha256": {
                "observable_candidates.tsv": _sha256(temporary / "observable_candidates.tsv"),
                "observable_pair_views.tsv": _sha256(temporary / "observable_pair_views.tsv"),
                "formal_assay_decisions.tsv": _sha256(temporary / "formal_assay_decisions.tsv"),
            },
            "information_boundary": {
                "formal_artifacts_only": True,
                "state_membership_observable_only": True,
                "assay_local_loading_views_only": True,
                "direct_parent_membership_used_as_gate": False,
                "cross_assay_matching_used": False,
            },
        }
        _write_json(temporary / "global_phase_a_freeze.json", freeze)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema_version": "ca_boss_v4.formal_train_loading_recall.phase_a_run.v1",
                "script_sha256": _sha256(Path(__file__).resolve()),
                "global_phase_a_freeze_sha256": _sha256(
                    temporary / "global_phase_a_freeze.json"
                ),
            },
        )
        os.replace(temporary, out_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
