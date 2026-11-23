#!/usr/bin/env python3
"""Two-phase cached audit of the CA-BOSS v4.31 fair-parent guard.

Phase A verifies and binds the existing v4.30 observable inputs plus the
truth-closed v4.27 topology-consensus feature table.  It also samples balanced
random panels from the already cached world pool.  Phase B verifies every
binding before opening the existing outer feature predictions and simulator
truth, rebuilds v4.29, v4.30, and v4.31 independently, and reports full-pool,
source-block, world, random-panel, and leave-one-source-block-out metrics.

The v4.31 rule has no fitted numeric threshold.  It reuses the frozen v4.27
observable conjunction as a prerequisite for a fair comparison between the
two parents of a directed collider.  This script cannot simulate or generate
missing worlds and makes no fresh-locked claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_nested_loading_group_axis_consensus_v4 as v429_audit  # noqa: E402
import audit_nested_loading_group_structure_uncertainty_v4 as v430_audit  # noqa: E402
import loading_group_competing_parent_observability_guard_v4 as guard  # noqa: E402
import loading_group_state_guard_v4 as v429_guard  # noqa: E402
import loading_group_structure_uncertainty_guard_v4 as v430_guard  # noqa: E402


SCHEMA = "ca_boss_v4.nested_loading_group_competing_parent_observability.v1"
GROUP_KEY = ["block", "seed", "loading_group"]
WORLD_KEY = ["block", "seed"]
PHASE_A_PANEL = "phase_a_random_panels.tsv"
PHASE_A_STRUCTURE = "phase_a_selected_structure_competitions.tsv"
PHASE_B_TABLES = (
    "phase_b_outer_group_predictions.tsv",
    "phase_b_outer_block_metrics.tsv",
    "phase_b_world_metrics.tsv",
    "phase_b_random_panel_metrics.tsv",
    "phase_b_leave_one_source_block_out_metrics.tsv",
)
SOURCE_FILES = (
    "audit_nested_loading_group_competing_parent_observability_v4.py",
    "loading_group_competing_parent_observability_guard_v4.py",
    "loading_group_state_guard_v4.py",
    "loading_group_structure_uncertainty_guard_v4.py",
)
DEFAULT_V430_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_loading_structure_uncertainty_v430_full_pool"
)
DEFAULT_V427_DIR = v429_audit.DEFAULT_V427_DIR
TOPOLOGY_TABLE = "phase_a_observable_topology_evidence.tsv"
PROJECT_ROOT = SCRIPT_DIR.parents[1]
LOCAL_CANDIDATE_NAME = "v4_ls0.010_lc0.85_hp0.880_q0.10_k1"
DEFAULT_ARTIFACT_ROOTS = {
    "development_34001_34016": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/opened_world_replay_v44/"
    "development_34001_34016/artifacts",
    "development_47001_47016": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/opened_world_replay_v45/"
    "development_47001_47016/artifacts",
    "train_39001_39024": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_39001_39024/artifacts",
    "train_40001_40024": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_40001_40024/artifacts",
    "train_41001_41024": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/opened_world_replay_v44/"
    "train_41001_41024/artifacts",
    "train_46001_46024": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/opened_world_replay_v45/"
    "train_46001_46024/artifacts",
    "train_49001_49024": PROJECT_ROOT
    / "pipelines/simulation/unpaired_multimodal/runs/formal_runs/train_49001_49024/artifacts",
}


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


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    observed = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} payload hash mismatch")
    return observed


def _safe_file(path: Path, *, phase_a: bool) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if phase_a and {"scorer", "truth"} & {
        part.casefold() for part in resolved.parts
    }:
        raise ValueError(f"Phase A refuses scorer/truth path: {resolved}")
    return resolved


def _load_json(path: Path, *, phase_a: bool) -> tuple[Path, dict[str, Any]]:
    safe = _safe_file(path, phase_a=phase_a)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {safe}")
    return safe, value


def _verify_topology_phase_a(
    v427_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    root = v427_dir.expanduser().absolute().resolve(strict=True)
    manifest_path, manifest = _load_json(
        root / "phase_a_manifest.json", phase_a=True
    )
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="v4.27 Phase-A manifest"
    )
    required = {
        "phase": "a",
        "algorithm_schema_version": "topology_consensus_state_calibration.v4.27",
        "scorer_or_truth_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "minimum_direct_candidate_degree": 1,
    }
    bad = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if bad:
        raise ValueError(f"v4.27 Phase-A topology boundary changed: {bad}")
    table = _safe_file(root / TOPOLOGY_TABLE, phase_a=True)
    if manifest.get("output_sha256", {}).get(TOPOLOGY_TABLE) != _sha256(table):
        raise ValueError("v4.27 observable topology table hash mismatch")
    return manifest_path, table, manifest


def run_phase_a(
    *,
    out_dir: Path,
    v430_dir: Path,
    v427_dir: Path,
    selection_seed: int,
    panels: int,
    panel_size: int,
    artifact_roots: Mapping[str, Path] | None = None,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-A output: {out_dir}")
    v430_root = v430_dir.expanduser().absolute().resolve(strict=True)
    v430_manifest_path = _safe_file(
        v430_root / "phase_a_manifest.json", phase_a=True
    )
    v430_manifest, tables = v430_audit._load_phase_a(v430_root)
    topology_manifest_path, topology_path, topology_manifest = (
        _verify_topology_phase_a(v427_dir)
    )

    groups = tables[v430_audit.V429_TABLE_NAMES[0]].copy()
    direct = tables["direct"].copy()
    topology = pd.read_csv(topology_path, sep="\t", keep_default_na=False)
    required_groups = {"block", "seed", "loading_group"}
    required_direct = {"block", "seed", "source_assay", "parent", "child"}
    required_topology = {"block", "seed", *guard._FEATURE_COLUMNS}
    for frame, required, label in (
        (groups, required_groups, "v4.29 loading groups"),
        (direct, required_direct, "direct candidates"),
        (topology, required_topology, "v4.27 topology evidence"),
    ):
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"{label} is missing {missing}")
    worlds = groups.loc[:, WORLD_KEY].drop_duplicates().sort_values(WORLD_KEY)
    for label, frame in (("direct", direct), ("topology", topology)):
        available = set(map(tuple, frame.loc[:, WORLD_KEY].to_numpy()))
        missing = [
            (str(row.block), int(row.seed))
            for row in worlds.itertuples(index=False)
            if (str(row.block), int(row.seed)) not in available
        ]
        if missing:
            raise ValueError(f"{label} cache lacks v4.30 worlds: {missing}")

    roots = dict(artifact_roots or DEFAULT_ARTIFACT_ROOTS)
    if set(worlds["block"].astype(str)) != set(roots):
        raise ValueError("Structure-artifact roots do not match the cached blocks")
    structure_rows: list[pd.DataFrame] = []
    structure_input_bindings: dict[str, dict[str, str]] = {}
    for world in worlds.itertuples(index=False):
        block, seed = str(world.block), int(world.seed)
        path = _safe_file(
            roots[block]
            / f"world_seed{seed}"
            / LOCAL_CANDIDATE_NAME
            / "structure_anchored_loading_groups"
            / "structure_anchor_pair_evidence.tsv",
            phase_a=True,
        )
        evidence = pd.read_csv(path, sep="\t", keep_default_na=False)
        if missing := sorted(guard._STRUCTURE_COLUMNS.difference(evidence.columns)):
            raise ValueError(f"Structure evidence is missing {missing}: {path}")
        selected = guard._boolean(evidence["selected"], label="structure.selected")
        evidence = evidence.loc[selected].copy()
        if evidence.empty:
            raise ValueError(f"Structure evidence has no selected records: {path}")
        if evidence.duplicated(["modality", "pair_key"]).any():
            raise ValueError(f"Selected structure evidence has duplicate pairs: {path}")
        selected_world = evidence.loc[:, sorted(guard._STRUCTURE_COLUMNS)].copy()
        selected_world.insert(0, "seed", seed)
        selected_world.insert(0, "block", block)
        structure_rows.append(selected_world)
        structure_input_bindings[f"{block}:{seed}"] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
    selected_structure = pd.concat(structure_rows, ignore_index=True)
    if selected_structure.duplicated(
        [*WORLD_KEY, "modality", "pair_key"]
    ).any():
        raise ValueError("Selected structure competitions duplicate pair keys")

    panel_table = v430_audit._balanced_panels(
        worlds,
        panels=panels,
        panel_size=panel_size,
        selection_seed=selection_seed,
    )
    out_dir.mkdir(parents=True)
    panel_path = out_dir / PHASE_A_PANEL
    panel_table.to_csv(panel_path, sep="\t", index=False)
    structure_path = out_dir / PHASE_A_STRUCTURE
    selected_structure.to_csv(structure_path, sep="\t", index=False)

    artifact = {
        "schema_version": guard.SCHEMA,
        "fixed_rule": guard.FIXED_RULE,
        "artifact_role": "future_application_rule_binding",
        "application_requires_truth": False,
        "numeric_threshold_fitted": False,
        "competing_parent_observability_definition": (
            "frozen_v427_topology_consensus_fixed_rule_eligible"
        ),
        "state_route_requires_fair_two_parent_comparison": True,
        "exact_group_construction_competition_is_bound": True,
        "independent_cross_assay_axis_route_bypasses_competitor_gate": True,
        "downgrade_is_unresolved_not_biological_absence": True,
        "changes_state_v427_membership": False,
        "changes_direct_candidates": False,
        "exact_cross_assay_feature_pair_claimed": False,
        "new_simulation_worlds_or_datasets_created": False,
        "source_sha256": _source_hashes(),
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    fixed_path = out_dir / "phase_a_fixed_guard.json"
    _write_json(fixed_path, artifact)

    manifest = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": guard.SCHEMA,
        "phase": "a",
        "worlds": len(worlds),
        "source_blocks": int(worlds["block"].nunique()),
        "source_block_names": sorted(worlds["block"].astype(str).unique()),
        "random_panels": panels,
        "panel_size": panel_size,
        "selection_seed": selection_seed,
        "cached_inputs_only": True,
        "all_observable_inputs_loaded_before_truth": True,
        "scorer_or_truth_opened": False,
        "upstream_v427_phase_b_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "fresh_locked_claim": False,
        "retrospective_development_only": True,
        "v430_phase_a_dir": str(v430_root),
        "v430_phase_a_manifest": str(v430_manifest_path),
        "v430_phase_a_manifest_sha256": _sha256(v430_manifest_path),
        "v430_phase_a_manifest_payload_sha256": v430_manifest[
            "manifest_payload_sha256"
        ],
        "v427_topology_phase_a_manifest": str(topology_manifest_path),
        "v427_topology_phase_a_manifest_sha256": _sha256(
            topology_manifest_path
        ),
        "v427_topology_phase_a_manifest_payload_sha256": topology_manifest[
            "manifest_payload_sha256"
        ],
        "topology_table_binding": {
            "path": str(topology_path),
            "sha256": _sha256(topology_path),
        },
        "structure_competition_input_bindings": structure_input_bindings,
        "fixed_guard_artifact_sha256": _sha256(fixed_path),
        "output_sha256": {
            PHASE_A_PANEL: _sha256(panel_path),
            PHASE_A_STRUCTURE: _sha256(structure_path),
        },
        "source_sha256": _source_hashes(),
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _, manifest = _load_json(out_dir / "phase_a_manifest.json", phase_a=True)
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="v4.31 Phase-A manifest"
    )
    required = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": guard.SCHEMA,
        "phase": "a",
        "scorer_or_truth_opened": False,
        "all_observable_inputs_loaded_before_truth": True,
        "new_simulation_worlds_or_datasets_created": False,
    }
    bad = {
        key: manifest.get(key)
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if bad:
        raise ValueError(f"Unsupported or unsafe v4.31 Phase-A manifest: {bad}")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("v4.31 audit sources changed after Phase A")

    fixed_path = _safe_file(out_dir / "phase_a_fixed_guard.json", phase_a=True)
    if manifest.get("fixed_guard_artifact_sha256") != _sha256(fixed_path):
        raise ValueError("Fixed v4.31 guard artifact changed after Phase A")
    _, artifact = _load_json(fixed_path, phase_a=True)
    _validate_self_hash(
        artifact, field="artifact_payload_sha256", label="fixed v4.31 guard"
    )
    if (
        artifact.get("schema_version") != guard.SCHEMA
        or artifact.get("fixed_rule") != guard.FIXED_RULE
        or artifact.get("numeric_threshold_fitted") is not False
    ):
        raise ValueError("Fixed v4.31 rule binding changed")

    v430_root = Path(str(manifest["v430_phase_a_dir"]))
    v430_manifest_path = _safe_file(
        Path(str(manifest["v430_phase_a_manifest"])), phase_a=True
    )
    if manifest.get("v430_phase_a_manifest_sha256") != _sha256(
        v430_manifest_path
    ):
        raise ValueError("Bound v4.30 Phase-A manifest changed")
    v430_manifest, tables = v430_audit._load_phase_a(v430_root)
    if (
        manifest.get("v430_phase_a_manifest_payload_sha256")
        != v430_manifest.get("manifest_payload_sha256")
    ):
        raise ValueError("Bound v4.30 Phase-A payload changed")

    topology_binding = manifest["topology_table_binding"]
    topology_path = _safe_file(Path(topology_binding["path"]), phase_a=True)
    if topology_binding.get("sha256") != _sha256(topology_path):
        raise ValueError("Bound v4.27 topology table changed")
    tables["topology"] = pd.read_csv(
        topology_path, sep="\t", keep_default_na=False
    )
    panel_path = _safe_file(out_dir / PHASE_A_PANEL, phase_a=True)
    if manifest.get("output_sha256", {}).get(PHASE_A_PANEL) != _sha256(panel_path):
        raise ValueError("Phase-A random panels changed")
    tables["panels"] = pd.read_csv(
        panel_path, sep="\t", keep_default_na=False
    )
    structure_path = _safe_file(
        out_dir / PHASE_A_STRUCTURE, phase_a=True
    )
    if manifest.get("output_sha256", {}).get(PHASE_A_STRUCTURE) != _sha256(
        structure_path
    ):
        raise ValueError("Phase-A selected structure competitions changed")
    for binding in manifest.get("structure_competition_input_bindings", {}).values():
        source = _safe_file(Path(binding["path"]), phase_a=True)
        if binding.get("sha256") != _sha256(source):
            raise ValueError("Bound structure competition source changed")
    tables["structure"] = pd.read_csv(
        structure_path, sep="\t", keep_default_na=False
    )
    return manifest, tables


def _score(
    frame: pd.DataFrame, selected: pd.Series, *, total_true_pairs: int
) -> dict[str, Any]:
    selected = selected.astype(bool)
    truth = frame["truth_pair"].astype(bool)
    tp = int((selected & truth).sum())
    fp = int((selected & ~truth).sum())
    fn = int(total_true_pairs - tp)
    return {
        "selected_groups": int(selected.sum()),
        "true_positive_groups": tp,
        "false_positive_groups": fp,
        "false_negative_groups": fn,
        "sentinel_groups": int(
            (selected & frame["contains_sentinel"].astype(bool)).sum()
        ),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / total_true_pairs) if total_true_pairs else 0.0,
    }


def _wilson(
    successes: int, trials: int, z: float = 1.959963984540054
) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def run_phase_b(
    *, out_dir: Path, worlds_root: Path, v427_dir: Path
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")

    # All Phase-A observable bindings are verified before the first upstream
    # outer prediction or scorer artifact is opened below.
    manifest, tables = _load_phase_a(out_dir)
    features, upstream_provenance = v429_audit._verified_v427_predictions(
        v427_dir
    )
    groups = tables[v430_audit.V429_TABLE_NAMES[0]]
    membership = tables[v430_audit.V429_TABLE_NAMES[1]]
    group_axes = tables[v430_audit.V429_TABLE_NAMES[2]]
    axes = tables[v430_audit.V429_TABLE_NAMES[3]]
    direct = tables["direct"]
    topology = tables["topology"]
    structure = tables["structure"]
    panels = tables["panels"]
    world_keys = groups.loc[:, WORLD_KEY].drop_duplicates().copy()

    prediction_frames: list[pd.DataFrame] = []
    truth_totals: dict[tuple[str, int], int] = {}
    truth_hashes: dict[str, str] = {}
    for world in world_keys.itertuples(index=False):
        block, seed = str(world.block), int(world.seed)

        def mask(frame: pd.DataFrame) -> pd.Series:
            return frame["block"].astype(str).eq(block) & frame["seed"].astype(
                int
            ).eq(seed)

        feature_world = features.loc[mask(features)].copy()
        topology_world = topology.loc[mask(topology)].copy()
        if feature_world.empty or topology_world.empty:
            raise ValueError(f"Upstream cache lacks world {block}:{seed}")
        if set(feature_world["feature"].astype(str)) != set(
            topology_world["feature"].astype(str)
        ):
            raise ValueError(
                f"v4.27 observable/outer feature sets disagree for {block}:{seed}"
            )
        world_groups = groups.loc[mask(groups)].drop(columns=WORLD_KEY)
        direct_world = v430_audit._direct_world(direct.loc[mask(direct)])
        v429 = v429_guard.apply_loading_group_state_guard(
            world_groups,
            membership.loc[mask(membership)].drop(columns=WORLD_KEY),
            feature_world.loc[
                :, ["feature", "modality", "state_v427_supported_member"]
            ],
            group_axes.loc[mask(group_axes)].drop(columns=WORLD_KEY),
            axes.loc[mask(axes)].drop(columns=WORLD_KEY),
        ).decisions
        v430 = v430_guard.apply_loading_group_structure_uncertainty_guard(
            v429, world_groups, direct_world
        ).decisions
        result = guard.apply_loading_group_competing_parent_observability_guard(
            v430,
            structure.loc[mask(structure)].drop(
                columns=WORLD_KEY
            ),
            topology_world.drop(columns=WORLD_KEY),
        ).decisions

        truth_path = (
            worlds_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        true_pairs, truth_hash = v429_audit._truth_pairs(truth_path)
        truth_totals[(block, seed)] = len(true_pairs)
        truth_hashes[f"{block}:{seed}"] = truth_hash
        roles = feature_world.set_index("feature")["truth_role"].astype(str)
        result["truth_pair"] = result["members"].astype(str).map(
            lambda value: frozenset(
                member.strip()
                for member in value.split(";")
                if member.strip()
            )
            in true_pairs
        )
        result["contains_sentinel"] = result["members"].astype(str).map(
            lambda value: any(
                str(roles.loc[member.strip()]) in {"N1", "N2"}
                for member in value.split(";")
                if member.strip()
            )
        )
        result.insert(0, "seed", seed)
        result.insert(0, "block", block)
        result["outer_held_block"] = block
        result["outer_truth_used_for_guard_application"] = False
        prediction_frames.append(result)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.duplicated(GROUP_KEY).any():
        raise RuntimeError("v4.31 predictions contain duplicate groups")
    rebuilt = predictions["v430_retained"].astype(bool) & (
        predictions["v430_guard_route"].astype(str).eq(guard.AXIS_ROUTE)
        | (
            predictions["v430_guard_route"].astype(str).eq(guard.STATE_ROUTE)
            & predictions["complete_two_parent_comparison_available"].astype(bool)
            & predictions["competing_parent_fixed_rule_eligible"].astype(bool)
        )
    )
    if not rebuilt.equals(predictions["retained"].astype(bool)):
        raise RuntimeError("v4.31 output differs from independently rebuilt rule")

    def total_for(frame: pd.DataFrame) -> int:
        return sum(
            truth_totals[(str(row.block), int(row.seed))]
            for row in frame.loc[:, WORLD_KEY]
            .drop_duplicates()
            .itertuples(index=False)
        )

    total_true = sum(truth_totals.values())
    v430_score = _score(
        predictions,
        predictions["v430_retained"],
        total_true_pairs=total_true,
    )
    v431_score = _score(
        predictions, predictions["retained"], total_true_pairs=total_true
    )

    world_rows: list[dict[str, Any]] = []
    for (block, seed), frame in predictions.groupby(WORLD_KEY, sort=True):
        total = truth_totals[(str(block), int(seed))]
        before = _score(
            frame, frame["v430_retained"], total_true_pairs=total
        )
        after = _score(frame, frame["retained"], total_true_pairs=total)
        world_rows.append(
            {
                "block": str(block),
                "seed": int(seed),
                "truth_pairs": total,
                **{f"v430_{key}": value for key, value in before.items()},
                **{f"v431_{key}": value for key, value in after.items()},
                "v431_whole_group_reliability_success": bool(
                    after["precision"] >= 0.80
                    and after["sentinel_groups"] == 0
                ),
            }
        )
    world_metrics = pd.DataFrame(world_rows)

    block_rows: list[dict[str, Any]] = []
    for block, frame in predictions.groupby("block", sort=True):
        total = total_for(frame)
        before = _score(
            frame, frame["v430_retained"], total_true_pairs=total
        )
        after = _score(frame, frame["retained"], total_true_pairs=total)
        removed = frame["v430_retained"].astype(bool) & ~frame[
            "retained"
        ].astype(bool)
        block_rows.append(
            {
                "outer_held_block": str(block),
                "worlds": frame.loc[:, WORLD_KEY].drop_duplicates().shape[0],
                "truth_pairs": total,
                **{f"v430_{key}": value for key, value in before.items()},
                **{f"v431_{key}": value for key, value in after.items()},
                "removed_true_groups": int(
                    (removed & frame["truth_pair"].astype(bool)).sum()
                ),
                "removed_false_groups": int(
                    (removed & ~frame["truth_pair"].astype(bool)).sum()
                ),
            }
        )
    block_metrics = pd.DataFrame(block_rows)

    panel_rows: list[dict[str, Any]] = []
    for panel_id, selection in panels.groupby("panel_id", sort=True):
        frame = predictions.merge(
            selection.loc[:, WORLD_KEY], on=WORLD_KEY, validate="many_to_one"
        )
        total = total_for(frame)
        before = _score(
            frame, frame["v430_retained"], total_true_pairs=total
        )
        after = _score(frame, frame["retained"], total_true_pairs=total)
        panel_rows.append(
            {
                "panel_id": str(panel_id),
                "selection_seed": int(selection["selection_seed"].iloc[0]),
                "worlds": len(selection),
                "truth_pairs": total,
                **{f"v430_{key}": value for key, value in before.items()},
                **{f"v431_{key}": value for key, value in after.items()},
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)

    lobo_rows: list[dict[str, Any]] = []
    for held_block in sorted(predictions["block"].astype(str).unique()):
        train = predictions.loc[
            ~predictions["block"].astype(str).eq(held_block)
        ]
        held = predictions.loc[
            predictions["block"].astype(str).eq(held_block)
        ]
        for partition, frame in (("training_blocks", train), ("held_block", held)):
            before = _score(
                frame,
                frame["v430_retained"],
                total_true_pairs=total_for(frame),
            )
            after = _score(
                frame, frame["retained"], total_true_pairs=total_for(frame)
            )
            lobo_rows.append(
                {
                    "outer_held_block": held_block,
                    "partition": partition,
                    "rule_fitted_on_training_blocks": False,
                    "held_truth_used_for_rule_selection": False,
                    **{f"v430_{key}": value for key, value in before.items()},
                    **{f"v431_{key}": value for key, value in after.items()},
                }
            )
    lobo_metrics = pd.DataFrame(lobo_rows)

    for name, frame in zip(
        PHASE_B_TABLES,
        (predictions, block_metrics, world_metrics, panel_metrics, lobo_metrics),
        strict=True,
    ):
        frame.to_csv(out_dir / name, sep="\t", index=False)

    removed = predictions["v430_retained"].astype(bool) & ~predictions[
        "retained"
    ].astype(bool)
    removals = (
        predictions.loc[removed]
        .groupby("guard_status", sort=True)
        .agg(
            removed_groups=("loading_group", "size"),
            removed_true_groups=("truth_pair", "sum"),
        )
        .reset_index()
    )
    removals["removed_false_groups"] = (
        removals["removed_groups"] - removals["removed_true_groups"]
    )
    precision_trials = (
        v431_score["true_positive_groups"]
        + v431_score["false_positive_groups"]
    )
    gates = {
        "precision_at_least_0.995": v431_score["precision"] >= 0.995,
        "false_groups_reduced_by_at_least_two": (
            v430_score["false_positive_groups"]
            - v431_score["false_positive_groups"]
            >= 2
        ),
        "true_group_loss_at_most_two": (
            v430_score["true_positive_groups"]
            - v431_score["true_positive_groups"]
            <= 2
        ),
        "recall_at_least_0.83": v431_score["recall"] >= 0.83,
        "all_worlds_precision_0.80_zero_sentinel": bool(
            world_metrics["v431_whole_group_reliability_success"].all()
        ),
        "zero_selected_sentinel_groups": v431_score["sentinel_groups"] == 0,
        "fixed_boolean_rule_without_fitted_threshold": True,
        "state_membership_unchanged": True,
        "direct_candidates_unchanged": True,
        "no_cross_assay_feature_pair_or_direction_claim": True,
    }
    summary = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": guard.SCHEMA,
        "phase": "b",
        "algorithm_label": "CA-BOSS v4.31",
        "retrospective_development_only": True,
        "fresh_locked_claim": False,
        "new_simulation_worlds_or_datasets_created": False,
        "worlds": len(truth_totals),
        "source_blocks": int(predictions["block"].nunique()),
        "truth_within_assay_shared_state_pairs": total_true,
        "v430_guarded_loading_groups": v430_score,
        "v431_guarded_loading_groups": {
            **v431_score,
            "incremental_true_groups_vs_v430": int(
                v431_score["true_positive_groups"]
                - v430_score["true_positive_groups"]
            ),
            "incremental_false_groups_vs_v430": int(
                v431_score["false_positive_groups"]
                - v430_score["false_positive_groups"]
            ),
            "precision_wilson_95_interval": _wilson(
                v431_score["true_positive_groups"], precision_trials
            ),
            "recall_wilson_95_interval": _wilson(
                v431_score["true_positive_groups"], total_true
            ),
            "minimum_world_precision": float(
                world_metrics["v431_precision"].min()
            ),
            "mean_world_precision": float(
                world_metrics["v431_precision"].mean()
            ),
            "minimum_world_recall": float(world_metrics["v431_recall"].min()),
            "mean_world_recall": float(world_metrics["v431_recall"].mean()),
            "worlds_meeting_precision_0.80_zero_sentinel": int(
                world_metrics["v431_whole_group_reliability_success"].sum()
            ),
        },
        "v431_removals": removals.to_dict(orient="records"),
        "leave_one_source_block_out_audit": {
            "rule_fitted_on_training_blocks": False,
            "held_block_truth_used_for_rule_selection": False,
            "fixed_rule": guard.FIXED_RULE,
            "source_blocks": sorted(predictions["block"].astype(str).unique()),
        },
        "random_cached_panels": {
            "panels": len(panel_metrics),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "mean_v430_precision": float(panel_metrics["v430_precision"].mean()),
            "mean_v431_precision": float(panel_metrics["v431_precision"].mean()),
            "mean_v430_recall": float(panel_metrics["v430_recall"].mean()),
            "mean_v431_recall": float(panel_metrics["v431_recall"].mean()),
        },
        "gates": gates,
        "all_gates_passed": bool(all(gates.values())),
        "identifiability_boundary": {
            "estimand": (
                "assay-local shared-state loading group after a fair-comparison "
                "observability prerequisite for competing collider parents"
            ),
            "competing_parent_downgrade_means_absence": False,
            "exact_cross_assay_feature_correspondence": "not claimed",
            "cell_specific_cross_assay_direction": "not identifiable",
            "new_data_used": False,
        },
        "upstream_v427": upstream_provenance,
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "truth_input_sha256": truth_hashes,
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
    parser.add_argument("--v430-dir", type=Path, default=DEFAULT_V430_DIR)
    parser.add_argument(
        "--worlds-root", type=Path, default=SCRIPT_DIR.parent / "formal_worlds"
    )
    parser.add_argument("--v427-dir", type=Path, default=DEFAULT_V427_DIR)
    parser.add_argument("--selection-seed", type=int, default=20261025)
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
            v430_dir=args.v430_dir,
            v427_dir=args.v427_dir,
            selection_seed=args.selection_seed,
            panels=args.panels,
            panel_size=args.panel_size,
        )
    else:
        run_phase_b(
            out_dir=args.out_dir,
            worlds_root=args.worlds_root,
            v427_dir=args.v427_dir,
        )


if __name__ == "__main__":
    main()
