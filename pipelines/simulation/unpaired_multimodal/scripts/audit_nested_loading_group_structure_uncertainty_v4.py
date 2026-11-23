#!/usr/bin/env python3
"""Two-phase cached audit of the CA-BOSS v4.30 loading-group guard.

Phase A binds the already frozen v4.29 observable loading/axis tables and the
truth-closed cached direct-candidate table, then selects random cached-world
panels.  Phase B verifies every binding before opening the existing v4.27
outer predictions and simulator truth.  It compares v4.29 with the fixed
v4.30 structure/uncertainty guard and reports leave-one-source-block-out
threshold sensitivity.

No simulation, missing-world generation, paired data, or external data is
available through this script.  This is retrospective evidence on reused
cached worlds, not a fresh locked-world result.
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
import loading_group_state_guard_v4 as v429_guard  # noqa: E402
import loading_group_structure_uncertainty_guard_v4 as guard  # noqa: E402


SCHEMA = "ca_boss_v4.nested_loading_group_structure_uncertainty.v1"
GROUP_KEY = ["block", "seed", "loading_group"]
WORLD_KEY = ["block", "seed"]
PHASE_A_PANEL = "phase_a_random_panels.tsv"
PHASE_B_TABLES = (
    "phase_b_outer_group_predictions.tsv",
    "phase_b_outer_block_metrics.tsv",
    "phase_b_world_metrics.tsv",
    "phase_b_random_panel_metrics.tsv",
    "phase_b_threshold_sensitivity.tsv",
    "phase_b_nested_threshold_metrics.tsv",
)
SOURCE_FILES = (
    "audit_nested_loading_group_structure_uncertainty_v4.py",
    "loading_group_state_guard_v4.py",
    "loading_group_structure_uncertainty_guard_v4.py",
)
DEFAULT_V429_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_loading_axis_consensus_v429_full_pool_final"
)
DEFAULT_DIRECT_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_layered_contract_v423_full_pool"
)
DEFAULT_V427_DIR = v429_audit.DEFAULT_V427_DIR
V429_TABLE_NAMES = (
    "phase_a_loading_groups.tsv",
    "phase_a_loading_group_membership.tsv",
    "phase_a_group_axis_evidence.tsv",
    "phase_a_axis_candidates.tsv",
)
TARGET_TRAIN_PRECISION = 0.995
THRESHOLD_GRID = tuple(float(value) for value in np.round(
    np.arange(0.80, 0.951, 0.005), 3
))


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


def _verify_v429_phase_a(root: Path) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    resolved = root.expanduser().absolute().resolve(strict=True)
    manifest_path, manifest = _load_json(
        resolved / "phase_a_manifest.json", phase_a=True
    )
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="v4.29 Phase-A manifest"
    )
    required = {
        "schema_version": v429_audit.SCHEMA,
        "algorithm_schema_version": v429_guard.SCHEMA,
        "phase": "a",
        "cached_inputs_only": True,
        "scorer_or_truth_opened": False,
        "upstream_v427_phase_b_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
    }
    bad = {key: manifest.get(key) for key, expected in required.items()
           if manifest.get(key) != expected}
    if bad:
        raise ValueError(f"v4.29 Phase-A boundary changed: {bad}")
    tables: dict[str, Path] = {}
    for name in V429_TABLE_NAMES:
        path = _safe_file(resolved / name, phase_a=True)
        if manifest.get("output_sha256", {}).get(name) != _sha256(path):
            raise ValueError(f"v4.29 Phase-A table hash mismatch: {name}")
        tables[name] = path
    return manifest_path, tables, manifest


def _verify_direct_phase_a(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    resolved = root.expanduser().absolute().resolve(strict=True)
    manifest_path, manifest = _load_json(
        resolved / "phase_a_manifest.json", phase_a=True
    )
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="direct Phase-A manifest"
    )
    required = {
        "all_observable_inputs_loaded_before_truth": True,
        "new_worlds_or_datasets_created": False,
    }
    bad = {key: manifest.get(key) for key, expected in required.items()
           if manifest.get(key) != expected}
    if bad:
        raise ValueError(f"Direct Phase-A boundary changed: {bad}")
    table = _safe_file(resolved / "phase_a_direct_predictions.tsv", phase_a=True)
    if manifest.get("output_sha256", {}).get(table.name) != _sha256(table):
        raise ValueError("Direct Phase-A table hash mismatch")
    return manifest_path, table, manifest


def _balanced_panels(
    worlds: pd.DataFrame,
    *,
    panels: int,
    panel_size: int,
    selection_seed: int,
) -> pd.DataFrame:
    unique = worlds.loc[:, WORLD_KEY].drop_duplicates().sort_values(WORLD_KEY)
    if panel_size > len(unique):
        raise ValueError("panel_size exceeds the cached world pool")
    blocks = {
        str(block): frame.reset_index(drop=True)
        for block, frame in unique.groupby("block", sort=True)
    }
    rows: list[dict[str, Any]] = []
    for panel_index in range(panels):
        seed = int(selection_seed + panel_index)
        rng = np.random.default_rng(seed)
        shuffled = {
            block: frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
            for block, frame in blocks.items()
        }
        offsets = {block: 0 for block in shuffled}
        chosen: list[tuple[str, int]] = []
        order = list(shuffled)
        rng.shuffle(order)
        while len(chosen) < panel_size:
            progressed = False
            for block in order:
                offset = offsets[block]
                if offset >= len(shuffled[block]):
                    continue
                row = shuffled[block].iloc[offset]
                chosen.append((str(row["block"]), int(row["seed"])))
                offsets[block] += 1
                progressed = True
                if len(chosen) == panel_size:
                    break
            if not progressed:
                raise RuntimeError("Could not fill a random cached panel")
        for rank, (block, world_seed) in enumerate(chosen, start=1):
            rows.append(
                {
                    "panel_id": f"panel_{panel_index + 1:03d}",
                    "selection_seed": seed,
                    "selection_rank": rank,
                    "block": block,
                    "seed": world_seed,
                }
            )
    return pd.DataFrame(rows)


def run_phase_a(
    *,
    out_dir: Path,
    v429_dir: Path,
    direct_dir: Path,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase-A output: {out_dir}")
    v429_manifest_path, tables, v429_manifest = _verify_v429_phase_a(v429_dir)
    direct_manifest_path, direct_table, direct_manifest = _verify_direct_phase_a(
        direct_dir
    )

    groups = pd.read_csv(tables[V429_TABLE_NAMES[0]], sep="\t", keep_default_na=False)
    direct = pd.read_csv(direct_table, sep="\t", keep_default_na=False)
    required_groups = {
        "block", "seed", "loading_group", "minimum_member_centroid_cosine"
    }
    required_direct = {"block", "seed", "source_assay", "parent", "child"}
    if missing := sorted(required_groups.difference(groups.columns)):
        raise ValueError(f"v4.29 groups are missing {missing}")
    if missing := sorted(required_direct.difference(direct.columns)):
        raise ValueError(f"direct candidates are missing {missing}")
    worlds = groups.loc[:, WORLD_KEY].drop_duplicates()
    direct_worlds = set(map(tuple, direct.loc[:, WORLD_KEY].to_numpy()))
    missing_direct = [
        (str(row.block), int(row.seed))
        for row in worlds.itertuples(index=False)
        if (str(row.block), int(row.seed)) not in direct_worlds
    ]
    if missing_direct:
        raise ValueError(f"Direct cache lacks v4.29 worlds: {missing_direct}")

    panel_table = _balanced_panels(
        worlds,
        panels=panels,
        panel_size=panel_size,
        selection_seed=selection_seed,
    )
    out_dir.mkdir(parents=True)
    panel_path = out_dir / PHASE_A_PANEL
    panel_table.to_csv(panel_path, sep="\t", index=False)

    artifact = {
        "schema_version": guard.SCHEMA,
        "fixed_rule": guard.FIXED_RULE,
        "minimum_state_route_member_centroid_cosine": (
            guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
        ),
        "artifact_role": "future_application_rule_binding",
        "threshold_origin": (
            "retrospective_cached_world_development_calibration; not fresh locked"
        ),
        "application_requires_truth": False,
        "changes_state_v427_membership": False,
        "changes_direct_candidates": False,
        "direct_overlap_is_unresolved_not_refuted": True,
        "exact_cross_assay_feature_pair_claimed": False,
        "new_simulation_worlds_or_datasets_created": False,
        "source_sha256": _source_hashes(),
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    _write_json(out_dir / "phase_a_fixed_guard.json", artifact)

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
        "v429_phase_a_manifest": str(v429_manifest_path),
        "v429_phase_a_manifest_sha256": _sha256(v429_manifest_path),
        "v429_phase_a_manifest_payload_sha256": v429_manifest[
            "manifest_payload_sha256"
        ],
        "v429_table_bindings": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in tables.items()
        },
        "direct_phase_a_manifest": str(direct_manifest_path),
        "direct_phase_a_manifest_sha256": _sha256(direct_manifest_path),
        "direct_phase_a_manifest_payload_sha256": direct_manifest[
            "manifest_payload_sha256"
        ],
        "direct_table_binding": {
            "path": str(direct_table),
            "sha256": _sha256(direct_table),
        },
        "fixed_guard_artifact_sha256": _sha256(
            out_dir / "phase_a_fixed_guard.json"
        ),
        "output_sha256": {PHASE_A_PANEL: _sha256(panel_path)},
        "source_sha256": _source_hashes(),
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(out_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path, manifest = _load_json(
        out_dir / "phase_a_manifest.json", phase_a=True
    )
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="v4.30 Phase-A manifest"
    )
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("algorithm_schema_version") != guard.SCHEMA
        or manifest.get("phase") != "a"
        or manifest.get("scorer_or_truth_opened") is not False
        or manifest.get("all_observable_inputs_loaded_before_truth") is not True
    ):
        raise ValueError("Unsupported or unsafe v4.30 Phase-A manifest")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("v4.30 audit sources changed after Phase A")

    fixed_path = _safe_file(out_dir / "phase_a_fixed_guard.json", phase_a=True)
    if manifest.get("fixed_guard_artifact_sha256") != _sha256(fixed_path):
        raise ValueError("Fixed guard artifact changed after Phase A")
    _, artifact = _load_json(fixed_path, phase_a=True)
    _validate_self_hash(
        artifact, field="artifact_payload_sha256", label="fixed v4.30 guard"
    )
    if (
        artifact.get("schema_version") != guard.SCHEMA
        or artifact.get("fixed_rule") != guard.FIXED_RULE
        or artifact.get("minimum_state_route_member_centroid_cosine")
        != guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
    ):
        raise ValueError("Fixed v4.30 rule binding changed")

    panel_path = _safe_file(out_dir / PHASE_A_PANEL, phase_a=True)
    if manifest.get("output_sha256", {}).get(PHASE_A_PANEL) != _sha256(panel_path):
        raise ValueError("Phase-A random panels changed")
    tables = {"panels": pd.read_csv(panel_path, sep="\t", keep_default_na=False)}
    for name, binding in manifest["v429_table_bindings"].items():
        path = _safe_file(Path(binding["path"]), phase_a=True)
        if binding.get("sha256") != _sha256(path):
            raise ValueError(f"Bound v4.29 table changed: {name}")
        tables[name] = pd.read_csv(path, sep="\t", keep_default_na=False)
    direct_binding = manifest["direct_table_binding"]
    direct_path = _safe_file(Path(direct_binding["path"]), phase_a=True)
    if direct_binding.get("sha256") != _sha256(direct_path):
        raise ValueError("Bound direct table changed")
    tables["direct"] = pd.read_csv(direct_path, sep="\t", keep_default_na=False)
    return manifest, tables


def _direct_world(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node1": frame["parent"].astype(str),
            "endpoint1": "TAIL",
            "node2": frame["child"].astype(str),
            "endpoint2": "ARROW",
            "edge_type": "within_assay_direct_candidate",
            "source_assay": frame["source_assay"].astype(str),
        }
    ).reset_index(drop=True)


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


def _threshold_selection(frame: pd.DataFrame, threshold: float) -> pd.Series:
    return (
        frame["v429_retained"].astype(bool)
        & ~frame["complete_direct_collider_overlap"].astype(bool)
        & (
            frame["v429_guard_route"].astype(str).eq(guard.AXIS_ROUTE)
            | (
                frame["v429_guard_route"].astype(str).eq(guard.STATE_ROUTE)
                & frame["minimum_member_centroid_cosine"].astype(float).ge(
                    float(threshold)
                )
            )
        )
    )


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
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
    *,
    out_dir: Path,
    worlds_root: Path,
    v427_dir: Path,
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")

    # Every observable artifact is read and hash-verified before this call opens
    # the first truth-bearing v4.27/scorer artifact.
    manifest, tables = _load_phase_a(out_dir)
    features, upstream_provenance = v429_audit._verified_v427_predictions(v427_dir)
    groups = tables[V429_TABLE_NAMES[0]]
    membership = tables[V429_TABLE_NAMES[1]]
    group_axes = tables[V429_TABLE_NAMES[2]]
    axes = tables[V429_TABLE_NAMES[3]]
    direct = tables["direct"]
    panels = tables["panels"]
    world_keys = groups.loc[:, WORLD_KEY].drop_duplicates().copy()

    predictions_frames: list[pd.DataFrame] = []
    truth_totals: dict[tuple[str, int], int] = {}
    truth_hashes: dict[str, str] = {}
    for world in world_keys.itertuples(index=False):
        block, seed = str(world.block), int(world.seed)
        masks = {
            "groups": groups["block"].astype(str).eq(block)
            & groups["seed"].astype(int).eq(seed),
            "membership": membership["block"].astype(str).eq(block)
            & membership["seed"].astype(int).eq(seed),
            "group_axes": group_axes["block"].astype(str).eq(block)
            & group_axes["seed"].astype(int).eq(seed),
            "axes": axes["block"].astype(str).eq(block)
            & axes["seed"].astype(int).eq(seed),
            "direct": direct["block"].astype(str).eq(block)
            & direct["seed"].astype(int).eq(seed),
            "features": features["block"].astype(str).eq(block)
            & features["seed"].astype(int).eq(seed),
        }
        feature_world = features.loc[masks["features"]].copy()
        if feature_world.empty:
            raise ValueError(f"Upstream v4.27 lacks world {block}:{seed}")
        world_groups = groups.loc[masks["groups"]].drop(columns=WORLD_KEY)
        v429 = v429_guard.apply_loading_group_state_guard(
            world_groups,
            membership.loc[masks["membership"]].drop(columns=WORLD_KEY),
            feature_world.loc[
                :, ["feature", "modality", "state_v427_supported_member"]
            ],
            group_axes.loc[masks["group_axes"]].drop(columns=WORLD_KEY),
            axes.loc[masks["axes"]].drop(columns=WORLD_KEY),
        ).decisions
        result = guard.apply_loading_group_structure_uncertainty_guard(
            v429,
            world_groups,
            _direct_world(direct.loc[masks["direct"]]),
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
                member.strip() for member in value.split(";") if member.strip()
            ) in true_pairs
        )
        result["contains_sentinel"] = result["members"].astype(str).map(
            lambda value: any(
                str(roles.loc[member.strip()]) in {"N1", "N2"}
                for member in value.split(";") if member.strip()
            )
        )
        result.insert(0, "seed", seed)
        result.insert(0, "block", block)
        result["outer_held_block"] = block
        result["outer_truth_used_for_guard_application"] = False
        predictions_frames.append(result)

    predictions = pd.concat(predictions_frames, ignore_index=True)
    if predictions.duplicated(GROUP_KEY).any():
        raise RuntimeError("v4.30 predictions contain duplicate groups")
    fixed_rebuilt = _threshold_selection(
        predictions, guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
    )
    if not fixed_rebuilt.equals(predictions["retained"].astype(bool)):
        raise RuntimeError("v4.30 output differs from independently rebuilt rule")

    def total_for(frame: pd.DataFrame) -> int:
        return sum(
            truth_totals[(str(row.block), int(row.seed))]
            for row in frame.loc[:, WORLD_KEY].drop_duplicates().itertuples(index=False)
        )

    total_true = sum(truth_totals.values())
    v429_score = _score(
        predictions,
        predictions["v429_retained"],
        total_true_pairs=total_true,
    )
    v430_score = _score(
        predictions,
        predictions["retained"],
        total_true_pairs=total_true,
    )

    world_rows: list[dict[str, Any]] = []
    for (block, seed), frame in predictions.groupby(WORLD_KEY, sort=True):
        total = truth_totals[(str(block), int(seed))]
        before = _score(frame, frame["v429_retained"], total_true_pairs=total)
        after = _score(frame, frame["retained"], total_true_pairs=total)
        world_rows.append({
            "block": str(block),
            "seed": int(seed),
            "truth_pairs": total,
            **{f"v429_{key}": value for key, value in before.items()},
            **{f"v430_{key}": value for key, value in after.items()},
            "v430_whole_group_reliability_success": bool(
                after["precision"] >= 0.80 and after["sentinel_groups"] == 0
            ),
        })
    world_metrics = pd.DataFrame(world_rows)

    block_rows: list[dict[str, Any]] = []
    for block, frame in predictions.groupby("block", sort=True):
        total = total_for(frame)
        before = _score(frame, frame["v429_retained"], total_true_pairs=total)
        after = _score(frame, frame["retained"], total_true_pairs=total)
        removed = frame["v429_retained"].astype(bool) & ~frame["retained"].astype(bool)
        block_rows.append({
            "outer_held_block": str(block),
            "worlds": frame.loc[:, WORLD_KEY].drop_duplicates().shape[0],
            "truth_pairs": total,
            **{f"v429_{key}": value for key, value in before.items()},
            **{f"v430_{key}": value for key, value in after.items()},
            "removed_true_groups": int((removed & frame["truth_pair"]).sum()),
            "removed_false_groups": int((removed & ~frame["truth_pair"]).sum()),
        })
    block_metrics = pd.DataFrame(block_rows)

    panel_rows: list[dict[str, Any]] = []
    for panel_id, selection in panels.groupby("panel_id", sort=True):
        frame = predictions.merge(
            selection.loc[:, WORLD_KEY], on=WORLD_KEY, validate="many_to_one"
        )
        total = total_for(frame)
        before = _score(frame, frame["v429_retained"], total_true_pairs=total)
        after = _score(frame, frame["retained"], total_true_pairs=total)
        panel_rows.append({
            "panel_id": str(panel_id),
            "selection_seed": int(selection["selection_seed"].iloc[0]),
            "worlds": len(selection),
            "truth_pairs": total,
            **{f"v429_{key}": value for key, value in before.items()},
            **{f"v430_{key}": value for key, value in after.items()},
        })
    panel_metrics = pd.DataFrame(panel_rows)

    sensitivity_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_GRID:
        score = _score(
            predictions,
            _threshold_selection(predictions, threshold),
            total_true_pairs=total_true,
        )
        sensitivity_rows.append({"threshold": threshold, **score})
    sensitivity = pd.DataFrame(sensitivity_rows)

    nested_rows: list[dict[str, Any]] = []
    nested_selection = pd.Series(False, index=predictions.index)
    for held_block in sorted(predictions["block"].astype(str).unique()):
        train = predictions.loc[
            ~predictions["block"].astype(str).eq(held_block)
        ].copy()
        held = predictions.loc[
            predictions["block"].astype(str).eq(held_block)
        ].copy()
        eligible: list[tuple[float, dict[str, Any]]] = []
        for threshold in THRESHOLD_GRID:
            score = _score(
                train,
                _threshold_selection(train, threshold),
                total_true_pairs=total_for(train),
            )
            if score["precision"] >= TARGET_TRAIN_PRECISION:
                eligible.append((threshold, score))
        if eligible:
            threshold, train_score = min(eligible, key=lambda item: item[0])
        else:
            threshold = THRESHOLD_GRID[-1]
            train_score = _score(
                train,
                _threshold_selection(train, threshold),
                total_true_pairs=total_for(train),
            )
        held_keep = _threshold_selection(held, threshold)
        held_score = _score(held, held_keep, total_true_pairs=total_for(held))
        nested_selection.loc[held.index] = held_keep.to_numpy()
        nested_rows.append({
            "outer_held_block": held_block,
            "selected_threshold": threshold,
            **{f"train_{key}": value for key, value in train_score.items()},
            **{f"held_{key}": value for key, value in held_score.items()},
            "held_truth_used_for_threshold_selection": False,
        })
    nested_metrics = pd.DataFrame(nested_rows)
    nested_score = _score(
        predictions, nested_selection, total_true_pairs=total_true
    )

    predictions.to_csv(out_dir / PHASE_B_TABLES[0], sep="\t", index=False)
    block_metrics.to_csv(out_dir / PHASE_B_TABLES[1], sep="\t", index=False)
    world_metrics.to_csv(out_dir / PHASE_B_TABLES[2], sep="\t", index=False)
    panel_metrics.to_csv(out_dir / PHASE_B_TABLES[3], sep="\t", index=False)
    sensitivity.to_csv(out_dir / PHASE_B_TABLES[4], sep="\t", index=False)
    nested_metrics.to_csv(out_dir / PHASE_B_TABLES[5], sep="\t", index=False)

    removed = predictions["v429_retained"].astype(bool) & ~predictions["retained"].astype(bool)
    removed_by_status = (
        predictions.loc[removed].groupby("guard_status", sort=True)
        .agg(
            removed_groups=("loading_group", "size"),
            removed_true_groups=("truth_pair", "sum"),
        )
        .reset_index()
    )
    removed_by_status["removed_false_groups"] = (
        removed_by_status["removed_groups"]
        - removed_by_status["removed_true_groups"]
    )
    precision_trials = (
        v430_score["true_positive_groups"] + v430_score["false_positive_groups"]
    )
    summary = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": guard.SCHEMA,
        "phase": "b",
        "retrospective_development_only": True,
        "fresh_locked_claim": False,
        "new_simulation_worlds_or_datasets_created": False,
        "worlds": len(truth_totals),
        "source_blocks": int(predictions["block"].nunique()),
        "truth_within_assay_shared_state_pairs": total_true,
        "v429_guarded_loading_groups": v429_score,
        "v430_guarded_loading_groups": {
            **v430_score,
            "incremental_true_groups_vs_v429": int(
                v430_score["true_positive_groups"]
                - v429_score["true_positive_groups"]
            ),
            "incremental_false_groups_vs_v429": int(
                v430_score["false_positive_groups"]
                - v429_score["false_positive_groups"]
            ),
            "precision_wilson_95_interval": _wilson(
                v430_score["true_positive_groups"], precision_trials
            ),
            "recall_wilson_95_interval": _wilson(
                v430_score["true_positive_groups"], total_true
            ),
            "minimum_world_precision": float(world_metrics["v430_precision"].min()),
            "mean_world_precision": float(world_metrics["v430_precision"].mean()),
            "minimum_world_recall": float(world_metrics["v430_recall"].min()),
            "mean_world_recall": float(world_metrics["v430_recall"].mean()),
            "worlds_meeting_precision_0.80_zero_sentinel": int(
                world_metrics["v430_whole_group_reliability_success"].sum()
            ),
        },
        "v430_removals": removed_by_status.to_dict(orient="records"),
        "leave_one_source_block_out_threshold_audit": {
            "target_training_precision": TARGET_TRAIN_PRECISION,
            "threshold_grid": list(THRESHOLD_GRID),
            "selected_thresholds": nested_metrics[
                "selected_threshold"
            ].astype(float).tolist(),
            "held_block_truth_used_for_threshold_selection": False,
            "pooled_outer_score": nested_score,
            "fixed_production_threshold": (
                guard.MINIMUM_STATE_ROUTE_MEMBER_CENTROID_COSINE
            ),
            "fixed_threshold_chosen_for_zero_cached_true_group_loss": True,
            "calibration_is_retrospective_not_fresh_locked": True,
        },
        "random_cached_panels": {
            "panels": len(panel_metrics),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "mean_v429_precision": float(panel_metrics["v429_precision"].mean()),
            "mean_v430_precision": float(panel_metrics["v430_precision"].mean()),
            "mean_v429_recall": float(panel_metrics["v429_recall"].mean()),
            "mean_v430_recall": float(panel_metrics["v430_recall"].mean()),
        },
        "gates": {
            "fewer_false_groups_than_v429": (
                v430_score["false_positive_groups"]
                < v429_score["false_positive_groups"]
            ),
            "no_true_group_loss_at_fixed_threshold": (
                v430_score["true_positive_groups"]
                == v429_score["true_positive_groups"]
            ),
            "precision_improved": v430_score["precision"] > v429_score["precision"],
            "recall_not_decreased": v430_score["recall"] >= v429_score["recall"],
            "zero_selected_sentinel_groups": v430_score["sentinel_groups"] == 0,
            "state_membership_unchanged": True,
            "direct_candidates_unchanged": True,
            "no_cross_assay_feature_pair_or_direction_claim": True,
        },
        "identifiability_boundary": {
            "estimand": (
                "assay-local shared-state loading group after uncertainty and "
                "direct-layer-overlap downgrades"
            ),
            "direct_overlap_downgrade_means_absence": False,
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
    parser.add_argument("--v429-dir", type=Path, default=DEFAULT_V429_DIR)
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT_DIR)
    parser.add_argument(
        "--worlds-root", type=Path, default=SCRIPT_DIR.parent / "formal_worlds"
    )
    parser.add_argument("--v427-dir", type=Path, default=DEFAULT_V427_DIR)
    parser.add_argument("--selection-seed", type=int, default=20261015)
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
            v429_dir=args.v429_dir,
            direct_dir=args.direct_dir,
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
