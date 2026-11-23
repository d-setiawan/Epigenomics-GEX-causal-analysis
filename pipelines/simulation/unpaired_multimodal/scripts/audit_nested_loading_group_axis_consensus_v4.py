#!/usr/bin/env python3
"""Two-phase cached-world audit of the CA-BOSS v4.29 loading guard.

Phase A inventories only already materialized observable loading-group and
latent-axis artifacts, freezes their hashes, and chooses block-balanced random
panels.  Phase B verifies that closure, opens the existing v4.27 outer
state-membership predictions and frozen scorer snapshots, then compares the
v4.28 all-members-state rule with the fixed v4.29 leave-one-group-out
cross-assay-axis consensus route.

The script has no simulation, missing-world generation, paired-data, or
external-data path.  Results are retrospective development evidence on the
existing cache, not a fresh locked-world claim.
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

import loading_group_state_guard_v4 as guard  # noqa: E402
from replay_existing_world_panel_v4 import (  # noqa: E402
    ExistingWorld,
    inventory_existing_worlds,
)


SCHEMA = "ca_boss_v4.nested_loading_group_axis_consensus.v1"
LOCAL_CANDIDATE_NAME = "v4_ls0.010_lc0.85_hp0.880_q0.10_k1"
GROUP_KEY = ["block", "seed", "loading_group"]
WORLD_KEY = ["block", "seed"]
PHASE_A_TABLES = (
    "phase_a_loading_groups.tsv",
    "phase_a_loading_group_membership.tsv",
    "phase_a_group_axis_evidence.tsv",
    "phase_a_axis_candidates.tsv",
    "phase_a_random_panels.tsv",
)
PHASE_B_TABLES = (
    "phase_b_outer_group_predictions.tsv",
    "phase_b_outer_block_metrics.tsv",
    "phase_b_world_metrics.tsv",
    "phase_b_random_panel_metrics.tsv",
)
SOURCE_FILES = (
    "audit_nested_loading_group_axis_consensus_v4.py",
    "loading_group_state_guard_v4.py",
)
DEFAULT_V427_DIR = Path(
    "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
    "cached_nested_topology_consensus_v427_full_pool"
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


def _validate_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    observed = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if observed != _canonical_sha256(unsigned):
        raise ValueError(f"{label} payload hash mismatch")
    return observed


def _observable_file(path: Path) -> Path:
    lexical = path.absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    forbidden = {"scorer", "truth"}
    if forbidden & {part.casefold() for part in resolved.parts}:
        raise ValueError(f"Phase A refuses scorer/truth path: {path}")
    return resolved


def _phase_b_file(path: Path) -> Path:
    lexical = path.absolute()
    if lexical.is_symlink() or not lexical.is_file():
        raise FileNotFoundError(path)
    resolved = lexical.resolve(strict=True)
    if "phase_b" not in resolved.name.casefold():
        raise ValueError(f"Expected an explicitly Phase-B file: {path}")
    return resolved


def _artifact_paths(world: ExistingWorld) -> dict[str, Path]:
    root = world.artifact_root / LOCAL_CANDIDATE_NAME
    return {
        "loading_groups": _observable_file(
            root / "loading_group_projection" / "loading_groups.tsv"
        ),
        "loading_group_membership": _observable_file(
            root
            / "loading_group_projection"
            / "loading_group_membership.tsv"
        ),
        "group_axis_evidence": _observable_file(
            root
            / "latent_axis_projection"
            / "group_axis_evidence.tsv"
        ),
        "axis_candidates": _observable_file(
            root / "latent_axis_projection" / "axis_candidates.tsv"
        ),
    }


def _eligible_worlds(
    runs_root: Path, worlds_root: Path
) -> tuple[tuple[ExistingWorld, dict[str, Path]], ...]:
    rows: list[tuple[ExistingWorld, dict[str, Path]]] = []
    for world in inventory_existing_worlds(runs_root, worlds_root):
        try:
            paths = _artifact_paths(world)
            groups = pd.read_csv(paths["loading_groups"], sep="\t", nrows=0)
            membership = pd.read_csv(
                paths["loading_group_membership"], sep="\t", nrows=0
            )
            group_axes = pd.read_csv(
                paths["group_axis_evidence"], sep="\t", nrows=0
            )
            axes = pd.read_csv(paths["axis_candidates"], sep="\t", nrows=0)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if not (
            guard._GROUP_COLUMNS.issubset(groups.columns)
            and guard._MEMBERSHIP_COLUMNS.issubset(membership.columns)
            and guard._GROUP_AXIS_COLUMNS.issubset(group_axes.columns)
            and guard._AXIS_COLUMNS.issubset(axes.columns)
        ):
            continue
        rows.append((world, paths))
    return tuple(sorted(rows, key=lambda item: (item[0].block, item[0].seed)))


def _tag(frame: pd.DataFrame, world: ExistingWorld) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "seed", int(world.seed))
    output.insert(0, "block", str(world.block))
    return output


def _balanced_random_panels(
    worlds: pd.DataFrame,
    *,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> pd.DataFrame:
    worlds = worlds.loc[:, WORLD_KEY].drop_duplicates().copy()
    blocks = sorted(set(worlds["block"].astype(str)))
    if panel_size < len(blocks):
        raise ValueError("panel-size must include every source block")
    minimum, remainder = divmod(panel_size, len(blocks))
    rows: list[dict[str, Any]] = []
    for panel_index in range(panels):
        rng_seed = int(selection_seed + panel_index)
        rng = np.random.default_rng(rng_seed)
        extra = set(rng.choice(blocks, size=remainder, replace=False).tolist())
        chosen: list[tuple[str, int]] = []
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
            chosen.extend(
                (block, int(seed))
                for seed in rng.choice(candidates, size=count, replace=False)
            )
        if len(chosen) != panel_size or len(set(chosen)) != panel_size:
            raise RuntimeError("Balanced random panel selection was not exact")
        for block, seed in sorted(chosen):
            rows.append(
                {
                    "panel_id": f"panel_{panel_index + 1:02d}",
                    "selection_seed": rng_seed,
                    "block": block,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def run_phase_a(
    *,
    out_dir: Path,
    runs_root: Path,
    worlds_root: Path,
    selection_seed: int,
    panels: int,
    panel_size: int,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Phase-A output already exists: {out_dir}")
    eligible = _eligible_worlds(runs_root, worlds_root)
    if not eligible:
        raise ValueError("No existing world has the frozen loading/axis inputs")

    group_frames: list[pd.DataFrame] = []
    membership_frames: list[pd.DataFrame] = []
    group_axis_frames: list[pd.DataFrame] = []
    axis_frames: list[pd.DataFrame] = []
    input_hashes: dict[str, dict[str, str]] = {}
    world_rows: list[dict[str, Any]] = []
    for world, paths in eligible:
        groups = pd.read_csv(paths["loading_groups"], sep="\t", keep_default_na=False)
        membership = pd.read_csv(
            paths["loading_group_membership"], sep="\t", keep_default_na=False
        )
        group_axes = pd.read_csv(
            paths["group_axis_evidence"], sep="\t", keep_default_na=False
        )
        axes = pd.read_csv(
            paths["axis_candidates"], sep="\t", keep_default_na=False
        )
        group_frames.append(_tag(groups, world))
        membership_frames.append(_tag(membership, world))
        group_axis_frames.append(_tag(group_axes, world))
        axis_frames.append(_tag(axes, world))
        key = f"{world.block}:{world.seed}"
        input_hashes[key] = {
            label: _sha256(path) for label, path in sorted(paths.items())
        }
        world_rows.append({"block": world.block, "seed": int(world.seed)})

    outputs = {
        PHASE_A_TABLES[0]: pd.concat(group_frames, ignore_index=True),
        PHASE_A_TABLES[1]: pd.concat(membership_frames, ignore_index=True),
        PHASE_A_TABLES[2]: pd.concat(group_axis_frames, ignore_index=True),
        PHASE_A_TABLES[3]: pd.concat(axis_frames, ignore_index=True),
        PHASE_A_TABLES[4]: _balanced_random_panels(
            pd.DataFrame(world_rows),
            selection_seed=selection_seed,
            panels=panels,
            panel_size=panel_size,
        ),
    }
    if outputs[PHASE_A_TABLES[0]].duplicated(GROUP_KEY).any():
        raise ValueError("Loading groups contain duplicate world/group keys")
    if outputs[PHASE_A_TABLES[1]].duplicated(
        [*GROUP_KEY, "feature"]
    ).any():
        raise ValueError("Loading-group membership contains duplicate keys")
    if outputs[PHASE_A_TABLES[2]].duplicated(
        [*GROUP_KEY, "axis_id"]
    ).any():
        raise ValueError("Group-axis evidence contains duplicate keys")
    if outputs[PHASE_A_TABLES[3]].duplicated(
        ["block", "seed", "axis_id"]
    ).any():
        raise ValueError("Axis candidates contain duplicate world/axis keys")

    out_dir.mkdir(parents=True)
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    blocks = sorted({world.block for world, _ in eligible})
    manifest = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": guard.SCHEMA,
        "phase": "a",
        "fixed_rule": guard.FIXED_RULE,
        "state_route": guard.STATE_ROUTE,
        "axis_route": guard.AXIS_ROUTE,
        "worlds": len(eligible),
        "source_blocks": len(blocks),
        "source_block_names": blocks,
        "selection_seed": int(selection_seed),
        "random_panels": int(panels),
        "panel_size": int(panel_size),
        "observable_input_sha256": input_hashes,
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_A_TABLES
        },
        "source_sha256": _source_hashes(),
        "scorer_or_truth_opened": False,
        "upstream_v427_phase_b_opened": False,
        "new_simulation_worlds_or_datasets_created": False,
        "cached_inputs_only": True,
        "retrospective_development_only": True,
        "fresh_locked_claim": False,
        "exact_cross_assay_feature_pair_claimed": False,
        "direct_causal_claimed": False,
        "changes_state_v427_membership": False,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = out_dir / "phase_a_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_self_hash(
        manifest, field="manifest_payload_sha256", label="Phase-A manifest"
    )
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("algorithm_schema_version") != guard.SCHEMA
        or manifest.get("fixed_rule") != guard.FIXED_RULE
        or manifest.get("source_sha256") != _source_hashes()
        or manifest.get("scorer_or_truth_opened") is not False
        or manifest.get("upstream_v427_phase_b_opened") is not False
        or manifest.get("new_simulation_worlds_or_datasets_created") is not False
    ):
        raise ValueError("Phase-A loading-axis freeze is not current or truth closed")
    tables: dict[str, pd.DataFrame] = {}
    for name in PHASE_A_TABLES:
        path = out_dir / name
        if _sha256(path) != manifest["output_sha256"].get(name):
            raise ValueError(f"Phase-A output changed after freeze: {name}")
        tables[name] = pd.read_csv(path, sep="\t", keep_default_na=False)
    return manifest, tables


def _verified_v427_predictions(
    v427_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    summary_path = _phase_b_file(v427_dir / "phase_b_summary.json")
    prediction_path = _phase_b_file(
        v427_dir / "phase_b_outer_feature_predictions.tsv"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload_hash = _validate_self_hash(
        summary, field="summary_payload_sha256", label="upstream v4.27 summary"
    )
    if summary.get("schema_version") != (
        "ca_boss_v4.nested_topology_consensus_state.v1"
    ):
        raise ValueError("Unsupported upstream v4.27 audit schema")
    if (
        summary.get("output_sha256", {}).get(prediction_path.name)
        != _sha256(prediction_path)
    ):
        raise ValueError("Upstream v4.27 prediction hash mismatch")
    frame = pd.read_csv(prediction_path, sep="\t", keep_default_na=False)
    columns = [
        "block",
        "seed",
        "feature",
        "modality",
        "state_v427_supported_member",
        "truth_role",
        "truth_state_member",
        "outer_held_block",
        "outer_truth_used_for_fit_or_certification",
    ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Upstream v4.27 predictions are missing {missing}")
    output = frame.loc[:, columns].copy()
    if output.duplicated(["block", "seed", "feature"]).any():
        raise ValueError("Upstream v4.27 predictions contain duplicate keys")
    if output["outer_truth_used_for_fit_or_certification"].astype(bool).any():
        raise ValueError("Upstream v4.27 prediction used held-block truth")
    if not output["block"].astype(str).equals(
        output["outer_held_block"].astype(str)
    ):
        raise ValueError("Upstream v4.27 rows are not outer-block predictions")
    return output, {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "summary_payload_sha256": payload_hash,
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
    }


def _truth_pairs(path: Path) -> tuple[set[frozenset[str]], str]:
    resolved = path.absolute().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(path)
    truth = pd.read_csv(resolved, sep="\t", keep_default_na=False)
    required = {"node1", "node2", "truth_type"}
    missing = sorted(required.difference(truth.columns))
    if missing:
        raise ValueError(f"Truth edge table is missing {missing}")
    pairs = {
        frozenset((str(row.node1), str(row.node2)))
        for row in truth.loc[
            truth["truth_type"].astype(str).eq("within_assay_shared_state")
        ].itertuples(index=False)
    }
    return pairs, _sha256(resolved)


def _score_counts(
    frame: pd.DataFrame,
    selected_column: str,
    *,
    total_true_pairs: int,
) -> dict[str, Any]:
    selected = frame[selected_column].astype(bool)
    truth = frame["truth_pair"].astype(bool)
    tp = int((selected & truth).sum())
    fp = int((selected & ~truth).sum())
    fn = int(total_true_pairs - tp)
    if fn < 0:
        raise ValueError("Selected true pairs exceed the truth denominator")
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


def _world_metrics(
    predictions: pd.DataFrame,
    truth_totals: Mapping[tuple[str, int], int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (block, seed), world in predictions.groupby(WORLD_KEY, sort=True):
        key = (str(block), int(seed))
        total = int(truth_totals[key])
        base = _score_counts(
            world, "all_members_state_v427_supported", total_true_pairs=total
        )
        final = _score_counts(world, "retained", total_true_pairs=total)
        axis = world["cross_assay_axis_consensus"].astype(bool)
        truth = world["truth_pair"].astype(bool)
        rows.append(
            {
                "block": str(block),
                "seed": int(seed),
                "truth_pairs": total,
                "v428_true_positive_groups": base["true_positive_groups"],
                "v428_false_positive_groups": base["false_positive_groups"],
                "v428_precision": base["precision"],
                "v428_recall": base["recall"],
                "axis_consensus_true_additions": int((axis & truth).sum()),
                "axis_consensus_false_additions": int((axis & ~truth).sum()),
                "axis_consensus_sentinel_additions": int(
                    (axis & world["contains_sentinel"].astype(bool)).sum()
                ),
                "v429_true_positive_groups": final["true_positive_groups"],
                "v429_false_positive_groups": final["false_positive_groups"],
                "v429_sentinel_groups": final["sentinel_groups"],
                "v429_precision": final["precision"],
                "v429_recall": final["recall"],
                "v429_whole_group_reliability_success": bool(
                    final["precision"] >= 0.80
                    and final["sentinel_groups"] == 0
                ),
            }
        )
    return pd.DataFrame(rows)


def run_phase_b(
    *,
    out_dir: Path,
    worlds_root: Path,
    v427_dir: Path,
) -> None:
    for name in (*PHASE_B_TABLES, "phase_b_fixed_guard.json", "phase_b_summary.json"):
        if (out_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite Phase-B output: {name}")
    manifest, tables = _load_phase_a(out_dir)
    features, upstream_provenance = _verified_v427_predictions(v427_dir)
    groups = tables[PHASE_A_TABLES[0]]
    membership = tables[PHASE_A_TABLES[1]]
    group_axes = tables[PHASE_A_TABLES[2]]
    axes = tables[PHASE_A_TABLES[3]]
    panels = tables[PHASE_A_TABLES[4]]
    world_keys = groups.loc[:, WORLD_KEY].drop_duplicates().copy()

    prediction_frames: list[pd.DataFrame] = []
    truth_totals: dict[tuple[str, int], int] = {}
    truth_hashes: dict[str, str] = {}
    for world in world_keys.itertuples(index=False):
        block, seed = str(world.block), int(world.seed)
        mask = groups["block"].astype(str).eq(block) & groups["seed"].astype(int).eq(seed)
        membership_mask = membership["block"].astype(str).eq(block) & membership[
            "seed"
        ].astype(int).eq(seed)
        group_axis_mask = group_axes["block"].astype(str).eq(block) & group_axes[
            "seed"
        ].astype(int).eq(seed)
        axis_mask = axes["block"].astype(str).eq(block) & axes["seed"].astype(int).eq(seed)
        feature_mask = features["block"].astype(str).eq(block) & features[
            "seed"
        ].astype(int).eq(seed)
        feature_world = features.loc[feature_mask].copy()
        if feature_world.empty:
            raise ValueError(f"Upstream v4.27 lacks world {block}:{seed}")
        result = guard.apply_loading_group_state_guard(
            groups.loc[mask].drop(columns=WORLD_KEY),
            membership.loc[membership_mask].drop(columns=WORLD_KEY),
            feature_world.loc[
                :, ["feature", "modality", "state_v427_supported_member"]
            ],
            group_axes.loc[group_axis_mask].drop(columns=WORLD_KEY),
            axes.loc[axis_mask].drop(columns=WORLD_KEY),
        ).decisions
        feature_roles = feature_world.set_index("feature")["truth_role"].astype(str)
        truth_path = (
            worlds_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        true_pairs, truth_hash = _truth_pairs(truth_path)
        truth_totals[(block, seed)] = len(true_pairs)
        truth_hashes[f"{block}:{seed}"] = truth_hash
        result["truth_pair"] = result["members"].astype(str).map(
            lambda value: frozenset(
                member.strip() for member in value.split(";") if member.strip()
            )
            in true_pairs
        )
        result["contains_sentinel"] = result["members"].astype(str).map(
            lambda value: any(
                str(feature_roles.loc[member.strip()]) in {"N1", "N2"}
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
        raise RuntimeError("Phase-B group predictions contain duplicate keys")
    if len(predictions) != int(
        groups["selection_status"].astype(str).eq("selected").groupby(
            [groups["block"], groups["seed"]]
        ).sum().sum()
    ):
        # The guard intentionally excludes selected singletons.  Recompute the
        # exact expected multi-feature count before treating this as an error.
        member_counts = (
            membership.groupby(GROUP_KEY, sort=False)["feature"].nunique()
        )
        selected = groups.set_index(GROUP_KEY)["selection_status"].astype(str).eq(
            "selected"
        )
        expected = int((selected & member_counts.reindex(selected.index).ge(2)).sum())
        if len(predictions) != expected:
            raise RuntimeError("Guard predictions do not cover every selected group")

    world_metrics = _world_metrics(predictions, truth_totals)
    block_rows: list[dict[str, Any]] = []
    for block, current in predictions.groupby("block", sort=True):
        keys = current.loc[:, WORLD_KEY].drop_duplicates()
        total = sum(
            truth_totals[(str(row.block), int(row.seed))]
            for row in keys.itertuples(index=False)
        )
        base = _score_counts(
            current,
            "all_members_state_v427_supported",
            total_true_pairs=total,
        )
        final = _score_counts(current, "retained", total_true_pairs=total)
        route = current["cross_assay_axis_consensus"].astype(bool)
        truth = current["truth_pair"].astype(bool)
        block_rows.append(
            {
                "outer_held_block": str(block),
                "worlds": len(keys),
                "truth_pairs": total,
                "v428_true_positive_groups": base["true_positive_groups"],
                "v428_false_positive_groups": base["false_positive_groups"],
                "v428_precision": base["precision"],
                "v428_recall": base["recall"],
                "axis_consensus_true_additions": int((route & truth).sum()),
                "axis_consensus_false_additions": int((route & ~truth).sum()),
                "axis_consensus_sentinel_additions": int(
                    (route & current["contains_sentinel"].astype(bool)).sum()
                ),
                "v429_true_positive_groups": final["true_positive_groups"],
                "v429_false_positive_groups": final["false_positive_groups"],
                "v429_sentinel_groups": final["sentinel_groups"],
                "v429_precision": final["precision"],
                "v429_recall": final["recall"],
            }
        )
    block_metrics = pd.DataFrame(block_rows)

    panel_rows: list[dict[str, Any]] = []
    for panel_id, selection in panels.groupby("panel_id", sort=True):
        current = predictions.merge(
            selection.loc[:, WORLD_KEY], on=WORLD_KEY, validate="many_to_one"
        )
        total = sum(
            truth_totals[(str(row.block), int(row.seed))]
            for row in selection.loc[:, WORLD_KEY].itertuples(index=False)
        )
        base = _score_counts(
            current,
            "all_members_state_v427_supported",
            total_true_pairs=total,
        )
        final = _score_counts(current, "retained", total_true_pairs=total)
        route = current["cross_assay_axis_consensus"].astype(bool)
        truth = current["truth_pair"].astype(bool)
        panel_rows.append(
            {
                "panel_id": str(panel_id),
                "selection_seed": int(selection["selection_seed"].iloc[0]),
                "worlds": len(selection),
                "truth_pairs": total,
                "v428_precision": base["precision"],
                "v428_recall": base["recall"],
                "axis_consensus_true_additions": int((route & truth).sum()),
                "axis_consensus_false_additions": int((route & ~truth).sum()),
                "axis_consensus_sentinel_additions": int(
                    (route & current["contains_sentinel"].astype(bool)).sum()
                ),
                "v429_precision": final["precision"],
                "v429_recall": final["recall"],
            }
        )
    panel_metrics = pd.DataFrame(panel_rows)

    artifact = {
        "schema_version": guard.SCHEMA,
        "fixed_rule": guard.FIXED_RULE,
        "state_route": guard.STATE_ROUTE,
        "axis_route": guard.AXIS_ROUTE,
        "artifact_role": "future_application_rule_binding",
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "application_requires_truth": False,
        "changes_state_v427_membership": False,
        "direct_structure_alone_is_insufficient": True,
        "exact_cross_assay_feature_pair_claimed": False,
        "new_simulation_worlds_or_datasets_created": False,
        "source_sha256": _source_hashes(),
    }
    artifact["artifact_payload_sha256"] = _canonical_sha256(artifact)
    _write_json(out_dir / "phase_b_fixed_guard.json", artifact)

    predictions.to_csv(out_dir / PHASE_B_TABLES[0], sep="\t", index=False)
    block_metrics.to_csv(out_dir / PHASE_B_TABLES[1], sep="\t", index=False)
    world_metrics.to_csv(out_dir / PHASE_B_TABLES[2], sep="\t", index=False)
    panel_metrics.to_csv(out_dir / PHASE_B_TABLES[3], sep="\t", index=False)

    total_true = sum(truth_totals.values())
    base = _score_counts(
        predictions,
        "all_members_state_v427_supported",
        total_true_pairs=total_true,
    )
    final = _score_counts(predictions, "retained", total_true_pairs=total_true)
    route = predictions["cross_assay_axis_consensus"].astype(bool)
    truth = predictions["truth_pair"].astype(bool)
    false_addition_worlds = int(
        predictions.loc[route & ~truth, WORLD_KEY].drop_duplicates().shape[0]
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
        "pre_v428_selected_groups": len(predictions),
        "v428_all_members_state_guard": base,
        "v429_axis_consensus_route": {
            "route_selected_groups": int(route.sum()),
            "route_true_additions": int((route & truth).sum()),
            "route_false_additions": int((route & ~truth).sum()),
            "route_sentinel_additions": int(
                (route & predictions["contains_sentinel"].astype(bool)).sum()
            ),
            "false_addition_worlds": false_addition_worlds,
            "outer_blocks_with_zero_false_additions": int(
                block_metrics["axis_consensus_false_additions"].astype(int).eq(0).sum()
            ),
            "outer_blocks_with_zero_sentinel_additions": int(
                block_metrics["axis_consensus_sentinel_additions"].astype(int).eq(0).sum()
            ),
            "outer_blocks": len(block_metrics),
        },
        "v429_guarded_loading_groups": {
            **final,
            "incremental_true_groups_beyond_v428": int(
                final["true_positive_groups"] - base["true_positive_groups"]
            ),
            "incremental_false_groups_beyond_v428": int(
                final["false_positive_groups"] - base["false_positive_groups"]
            ),
            "minimum_world_precision": float(world_metrics["v429_precision"].min()),
            "mean_world_precision": float(world_metrics["v429_precision"].mean()),
            "minimum_world_recall": float(world_metrics["v429_recall"].min()),
            "mean_world_recall": float(world_metrics["v429_recall"].mean()),
            "worlds_meeting_precision_0.80_zero_sentinel": int(
                world_metrics["v429_whole_group_reliability_success"].astype(bool).sum()
            ),
        },
        "random_cached_panels": {
            "panels": len(panel_metrics),
            "world_evaluations": int(panel_metrics["worlds"].sum()),
            "selection_seeds": panel_metrics["selection_seed"].astype(int).tolist(),
            "all_panels_zero_false_additions": bool(
                panel_metrics["axis_consensus_false_additions"].astype(int).eq(0).all()
            ),
            "all_panels_zero_sentinel_additions": bool(
                panel_metrics["axis_consensus_sentinel_additions"].astype(int).eq(0).all()
            ),
            "mean_v428_recall": float(panel_metrics["v428_recall"].mean()),
            "mean_v429_recall": float(panel_metrics["v429_recall"].mean()),
            "mean_v428_precision": float(panel_metrics["v428_precision"].mean()),
            "mean_v429_precision": float(panel_metrics["v429_precision"].mean()),
        },
        "gates": {
            "all_outer_blocks_zero_false_axis_additions": bool(
                block_metrics["axis_consensus_false_additions"].astype(int).eq(0).all()
            ),
            "all_outer_blocks_zero_sentinel_axis_additions": bool(
                block_metrics["axis_consensus_sentinel_additions"].astype(int).eq(0).all()
            ),
            "zero_outer_world_false_addition_rate": false_addition_worlds == 0,
            "precision_not_decreased": final["precision"] >= base["precision"],
            "recall_improved": final["recall"] > base["recall"],
            "state_membership_unchanged": True,
            "direct_structure_unchanged": True,
            "no_cross_assay_feature_pair_or_direction_claim": True,
        },
        "identifiability_boundary": {
            "estimand": (
                "assay-local loading group supported by a cross-assay latent "
                "axis estimated without the candidate group"
            ),
            "exact_cross_assay_feature_correspondence": "not claimed",
            "cell_specific_cross_assay_direction": "not identifiable",
            "state_membership_added_by_loading_route": False,
        },
        "upstream_v427": upstream_provenance,
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "truth_input_sha256": truth_hashes,
        "fixed_guard_artifact_sha256": _sha256(
            out_dir / "phase_b_fixed_guard.json"
        ),
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
        "--runs-root", type=Path, default=SCRIPT_DIR.parent / "formal_runs"
    )
    parser.add_argument(
        "--worlds-root", type=Path, default=SCRIPT_DIR.parent / "formal_worlds"
    )
    parser.add_argument("--v427-dir", type=Path, default=DEFAULT_V427_DIR)
    parser.add_argument("--selection-seed", type=int, default=20261014)
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
            runs_root=args.runs_root,
            worlds_root=args.worlds_root,
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
