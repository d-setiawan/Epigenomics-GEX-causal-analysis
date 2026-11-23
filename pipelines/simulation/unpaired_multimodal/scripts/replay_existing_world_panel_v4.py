#!/usr/bin/env python3
"""Randomly replay CA-BOSS direct decisions on existing cached worlds only.

This development utility never simulates, copies, or modifies a world.  Phase A
inventories already completed TRAIN/DEVELOPMENT directories, selects a
reproducible block-balanced random panel, and reconstructs the inexpensive
threshold/collider decisions from frozen observable evidence.  It refuses every
scorer/truth path.  Phase B verifies the complete Phase-A closure before opening
the already existing simulator scorer snapshots.

The result is retrospective development evidence, not a fresh locked claim.
Use the same selection seed for fast algorithm A/B comparisons.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from pandas.errors import EmptyDataError


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_ca_boss_v4 import (  # noqa: E402
    FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG,
    _direct_consensus_for_motif,
)
from channel_aware_boss_v4 import (  # noqa: E402
    CABossV4Config,
    apply_evidence_thresholds,
    build_typed_graph,
)
from consensus_ca_boss_v4 import consensus_typed_graph  # noqa: E402
from crossfit_hierarchical_collider_v4 import (  # noqa: E402
    CrossfitColliderWorldEvidence,
    CrossfitHierarchicalColliderConfig,
    deterministic_frame_sha256,
    resolve_crossfit_hierarchical_colliders_v4,
)
from observed_collider_structure import (  # noqa: E402
    apply_observed_collider_refinements,
    select_observed_collider_refinements,
)
from score_shared_state_subspace_v4 import (  # noqa: E402
    FROZEN_LOCAL_CONFIG,
)
from structure_aware_collider_motif import (  # noqa: E402
    select_unique_collider_motifs,
)


SCHEMA = "ca_boss_v4.existing_world_random_replay.v3"
WORLD_PATTERN = re.compile(r"^world_seed(?P<seed>[0-9]{4,8})$")
ALLOWED_BLOCK_PATTERN = re.compile(r"^(?:train|development)_[0-9]+_[0-9]+$")
TRUTH_PARTS = frozenset({"scorer", "truth", "truth_typed_edges.tsv"})
PHASE_A_TABLES = (
    "phase_a_direct_edges.tsv",
    "phase_a_unresolved_edges.tsv",
    "phase_a_observed_decisions.tsv",
    "phase_a_crossfit_decisions.tsv",
)
SOURCE_FILES = (
    "benchmark_ca_boss_v4.py",
    "channel_aware_boss_v4.py",
    "consensus_ca_boss_v4.py",
    "crossfit_hierarchical_collider_v4.py",
    "observed_collider_structure.py",
    "structure_aware_collider_motif.py",
    "replay_existing_world_panel_v4.py",
)


@dataclass(frozen=True)
class ExistingWorld:
    block: str
    seed: int
    run_root: Path
    world_root: Path
    artifact_root: Path

    def payload(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "seed": self.seed,
            "run_root": str(self.run_root.resolve()),
            "world_root": str(self.world_root.resolve()),
            "artifact_root": str(self.artifact_root.resolve()),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _assert_observable(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    folded = {part.casefold() for part in resolved.parts}
    if folded & TRUTH_PARTS:
        raise ValueError(f"Phase A refuses scorer/truth path: {path}")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(path)
    return resolved


def _read_tsv(path: Path, hashes: dict[str, str]) -> pd.DataFrame:
    safe = _assert_observable(path)
    hashes[str(safe)] = _sha256(safe)
    try:
        return pd.read_csv(
            safe,
            sep="\t",
            keep_default_na=False,
            float_precision="round_trip",
        )
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path, hashes: dict[str, str]) -> dict[str, Any]:
    safe = _assert_observable(path)
    hashes[str(safe)] = _sha256(safe)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _view_directories(artifact_root: Path) -> tuple[Path, ...]:
    views = tuple(
        sorted(
            path
            for path in artifact_root.glob("cells*_seed*")
            if path.is_dir() and not path.is_symlink()
        )
    )
    if len(views) != 6:
        raise ValueError(
            f"Expected six cached evidence views in {artifact_root}; found {len(views)}"
        )
    return views


def _observable_cache_complete(artifact_root: Path) -> bool:
    required = (
        artifact_root
        / "shared_state_subspace_v4"
        / "subspace_feature_membership.tsv",
        artifact_root
        / "crossfit_hierarchical_collider"
        / "world_generative_candidates.tsv",
        artifact_root
        / "crossfit_hierarchical_collider"
        / "world_evidence_provenance.json",
    )
    if not all(path.is_file() and not path.is_symlink() for path in required):
        return False
    try:
        views = _view_directories(artifact_root)
    except ValueError:
        return False
    view_files = (
        "raw_pair_evidence.tsv",
        "hierarchical_sparse_edge_evidence.tsv",
        "cell_loading_feature_models.tsv",
        "cell_loading_pair_evidence.tsv",
    )
    if any(not (view / name).is_file() for view in views for name in view_files):
        return False
    return all(
        (
            artifact_root
            / "collider_motif_evidence"
            / f"beta_binomial_vote{vote}.tsv"
        ).is_file()
        for vote in range(6)
    )


def inventory_existing_worlds(
    runs_root: Path, worlds_root: Path
) -> tuple[ExistingWorld, ...]:
    """Return eligible cached worlds without inspecting a scorer directory."""

    worlds: list[ExistingWorld] = []
    for run_root in sorted(runs_root.iterdir()):
        if (
            not run_root.is_dir()
            or run_root.is_symlink()
            or not ALLOWED_BLOCK_PATTERN.fullmatch(run_root.name)
        ):
            continue
        world_root = worlds_root / run_root.name
        if not world_root.is_dir() or world_root.is_symlink():
            continue
        artifacts = run_root / "artifacts"
        if not artifacts.is_dir():
            continue
        for artifact_root in sorted(artifacts.iterdir()):
            match = WORLD_PATTERN.fullmatch(artifact_root.name)
            if (
                match is None
                or not artifact_root.is_dir()
                or artifact_root.is_symlink()
                or not _observable_cache_complete(artifact_root)
            ):
                continue
            seed = int(match.group("seed"))
            world_dir = world_root / artifact_root.name
            if not world_dir.is_dir() or world_dir.is_symlink():
                continue
            worlds.append(
                ExistingWorld(
                    block=run_root.name,
                    seed=seed,
                    run_root=run_root,
                    world_root=world_root,
                    artifact_root=artifact_root,
                )
            )
    keys = [(world.block, world.seed) for world in worlds]
    if len(keys) != len(set(keys)):
        raise ValueError("Existing-world inventory contains duplicate block/seed keys")
    return tuple(worlds)


def select_balanced_random_panel(
    worlds: Sequence[ExistingWorld], panel_size: int, selection_seed: int
) -> tuple[ExistingWorld, ...]:
    """Sample randomly while keeping completed source blocks represented."""

    if panel_size < 1 or panel_size > len(worlds):
        raise ValueError("panel_size must lie between one and the eligible pool size")
    grouped: dict[str, list[ExistingWorld]] = {}
    for world in worlds:
        grouped.setdefault(world.block, []).append(world)
    rng = random.Random(selection_seed)
    for block_worlds in grouped.values():
        rng.shuffle(block_worlds)
    block_order = sorted(grouped)
    rng.shuffle(block_order)
    selected: list[ExistingWorld] = []
    cursor = {block: 0 for block in block_order}
    while len(selected) < panel_size:
        progressed = False
        for block in block_order:
            index = cursor[block]
            if index >= len(grouped[block]):
                continue
            selected.append(grouped[block][index])
            cursor[block] += 1
            progressed = True
            if len(selected) == panel_size:
                break
        if not progressed:
            raise RuntimeError("Could not fill the requested random panel")
        rng.shuffle(block_order)
    return tuple(sorted(selected, key=lambda world: (world.block, world.seed)))


def load_excluded_world_keys(
    selection_paths: Sequence[Path],
) -> tuple[set[tuple[str, int]], dict[str, str]]:
    """Load self-hashed, pre-truth selection files as an exclusion ledger."""

    excluded: set[tuple[str, int]] = set()
    hashes: dict[str, str] = {}
    for path in selection_paths:
        safe = _assert_observable(path)
        payload = json.loads(safe.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected selection JSON object: {path}")
        observed = str(payload.get("selection_payload_sha256", ""))
        unsigned = dict(payload)
        unsigned.pop("selection_payload_sha256", None)
        if not observed or observed != _canonical_sha256(unsigned):
            raise ValueError(f"Selection payload hash mismatch: {path}")
        records = payload.get("selected_worlds")
        if not isinstance(records, list):
            raise ValueError(f"Selection lacks selected_worlds: {path}")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Malformed selected world in {path}")
            excluded.add((str(record["block"]), int(record["seed"])))
        hashes[str(safe)] = _sha256(safe)
    return excluded, hashes


def _tag(frame: pd.DataFrame, world: ExistingWorld) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "seed", world.seed)
    output.insert(0, "block", world.block)
    return output


def _replay_world(
    world: ExistingWorld,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], dict[str, Any]]:
    hashes: dict[str, str] = {}
    cfg = CABossV4Config(**dict(FROZEN_LOCAL_CONFIG))
    if asdict(cfg) != dict(FROZEN_LOCAL_CONFIG):
        raise RuntimeError("Frozen local candidate config no longer round-trips")

    graphs: dict[str, pd.DataFrame] = {}
    residual_views: dict[str, pd.DataFrame] = {}
    feature_views: dict[str, pd.DataFrame] = {}
    loading_pair_views: dict[str, pd.DataFrame] = {}
    for view in _view_directories(world.artifact_root):
        raw = _read_tsv(view / "raw_pair_evidence.tsv", hashes)
        hierarchy = _read_tsv(
            view / "hierarchical_sparse_edge_evidence.tsv", hashes
        )
        feature_views[view.name] = _read_tsv(
            view / "cell_loading_feature_models.tsv", hashes
        )
        loading_pair_views[view.name] = _read_tsv(
            view / "cell_loading_pair_evidence.tsv", hashes
        )
        residual_views[view.name] = raw
        graphs[view.name] = build_typed_graph(
            apply_evidence_thresholds(raw, cfg),
            cfg,
            hierarchical_evidence=hierarchy,
        )

    beta_votes = pd.concat(
        [
            _read_tsv(
                world.artifact_root
                / "collider_motif_evidence"
                / f"beta_binomial_vote{vote}.tsv",
                hashes,
            )
            for vote in range(6)
        ],
        ignore_index=True,
        sort=False,
    )
    state_membership = _read_tsv(
        world.artifact_root
        / "shared_state_subspace_v4"
        / "subspace_feature_membership.tsv",
        hashes,
    )
    preliminary = consensus_typed_graph(list(graphs.values()), minimum_count=4)
    motif = select_unique_collider_motifs(
        _direct_consensus_for_motif(preliminary), beta_votes
    )
    observed = select_observed_collider_refinements(
        motif.component_diagnostics,
        motif.wedge_diagnostics,
        motif.demoted_or_unresolved,
        beta_votes,
        graphs,
        residual_views,
        feature_views,
        FROZEN_OBSERVED_COLLIDER_STRUCTURE_CONFIG,
        loading_pair_views=loading_pair_views,
        state_membership=state_membership,
    )
    motif = apply_observed_collider_refinements(motif, observed)

    crossfit_root = world.artifact_root / "crossfit_hierarchical_collider"
    candidates = _read_tsv(crossfit_root / "world_generative_candidates.tsv", hashes)
    provenance = _read_json(
        crossfit_root / "world_evidence_provenance.json", hashes
    )
    expected_hash = str(provenance.get("generative_candidates_sha256", ""))
    if deterministic_frame_sha256(candidates) != expected_hash:
        raise ValueError("Cached crossfit candidate hash differs from provenance")
    stored_crossfit_config = provenance.get("config")
    if not isinstance(stored_crossfit_config, Mapping):
        raise ValueError("Cached crossfit provenance lacks its fitted config")
    replay_config_payload = asdict(CrossfitHierarchicalColliderConfig())
    unknown_fields = set(stored_crossfit_config) - set(replay_config_payload)
    if unknown_fields:
        raise ValueError(
            "Cached crossfit config contains unknown fields: "
            f"{sorted(unknown_fields)}"
        )
    replay_config_payload.update(stored_crossfit_config)
    replay_config_payload["microstate_resolutions"] = tuple(
        replay_config_payload["microstate_resolutions"]
    )
    replay_crossfit_config = CrossfitHierarchicalColliderConfig(
        **replay_config_payload
    )
    normalized_replay_payload = asdict(replay_crossfit_config)
    selection_only_upgrade_fields = sorted(
        set(normalized_replay_payload) - set(stored_crossfit_config)
    )
    # Older caches predate selection-only tiers added to the resolver.  Their
    # fitted candidate statistics remain sufficient: merge those new defaults
    # in memory while retaining and hashing the original provenance on disk.
    replay_provenance = {
        **provenance,
        "config": normalized_replay_payload,
        "retrospective_selector_replay": True,
        "original_fitted_config_sha256": _canonical_sha256(
            stored_crossfit_config
        ),
        "selection_only_config_upgrade_fields": selection_only_upgrade_fields,
    }
    crossfit = resolve_crossfit_hierarchical_colliders_v4(
        world_evidence=CrossfitColliderWorldEvidence(
            generative_candidates=candidates,
            provenance=replay_provenance,
        ),
        beta_vote_evidence=beta_votes,
        base_edge_views=graphs,
        residual_views=residual_views,
        feature_model_views=feature_views,
        loading_pair_views=loading_pair_views,
        state_membership=state_membership,
        current_direct_candidates=motif.direct_candidates,
        current_demoted_or_unresolved=motif.demoted_or_unresolved,
        config=replay_crossfit_config,
    )
    tables = {
        "direct": _tag(crossfit.direct_candidates, world),
        "unresolved": _tag(crossfit.demoted_or_unresolved, world),
        "observed_decisions": _tag(observed.decisions, world),
        "crossfit_decisions": _tag(crossfit.decisions, world),
    }
    diagnostics = {
        "n_preliminary_edges": int(len(preliminary)),
        "n_observed_refinement_decisions": int(len(observed.decisions)),
        "n_crossfit_decisions": int(len(crossfit.decisions)),
        "n_direct_adjacencies": int(len(crossfit.direct_candidates)),
        "n_unresolved_adjacencies": int(len(crossfit.demoted_or_unresolved)),
        "selection_only_config_upgrade_fields": selection_only_upgrade_fields,
    }
    return tables, hashes, diagnostics


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    values = list(frames)
    nonempty = [frame for frame in values if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True, sort=False)
    return pd.DataFrame(columns=["block", "seed"])


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def run_phase_a(
    *,
    out_dir: Path,
    runs_root: Path,
    worlds_root: Path,
    panel_size: int,
    selection_seed: int,
    exclude_selections: Sequence[Path] = (),
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Phase-A output already exists: {out_dir}")
    inventory = inventory_existing_worlds(runs_root, worlds_root)
    excluded, exclusion_hashes = load_excluded_world_keys(
        exclude_selections
    )
    pool = tuple(
        world
        for world in inventory
        if (world.block, world.seed) not in excluded
    )
    selected = select_balanced_random_panel(pool, panel_size, selection_seed)
    out_dir.mkdir(parents=True)

    selection = {
        "schema_version": SCHEMA,
        "selection_seed": selection_seed,
        "sampling_method": (
            "random_block_balanced_without_replacement_after_pretruth_"
            "selection_exclusion"
        ),
        "inventory_pool_size": len(inventory),
        "eligible_pool_size": len(pool),
        "excluded_world_count": len(inventory) - len(pool),
        "excluded_world_keys": [
            {"block": block, "seed": seed}
            for block, seed in sorted(excluded)
        ],
        "exclusion_selection_sha256": exclusion_hashes,
        "eligible_block_counts": {
            block: sum(world.block == block for world in pool)
            for block in sorted({world.block for world in pool})
        },
        "panel_size": len(selected),
        "selected_worlds": [world.payload() for world in selected],
        "new_worlds_or_datasets_created": False,
        "scorer_or_truth_opened": False,
        "retrospective_development_only": True,
    }
    selection["selection_payload_sha256"] = _canonical_sha256(selection)
    _write_json(out_dir / "phase_a_selection.json", selection)

    table_lists: dict[str, list[pd.DataFrame]] = {
        "direct": [],
        "unresolved": [],
        "observed_decisions": [],
        "crossfit_decisions": [],
    }
    attempts: list[dict[str, Any]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for world in selected:
        try:
            tables, hashes, diagnostics = _replay_world(world)
        except Exception as exc:  # keep random-panel failures in the denominator
            attempts.append(
                {
                    "block": world.block,
                    "seed": world.seed,
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        for key, frame in tables.items():
            table_lists[key].append(frame)
        input_hashes[f"{world.block}:{world.seed}"] = hashes
        attempts.append(
            {
                "block": world.block,
                "seed": world.seed,
                "status": "ok",
                **diagnostics,
            }
        )

    outputs = {
        "phase_a_direct_edges.tsv": _concat(table_lists["direct"]),
        "phase_a_unresolved_edges.tsv": _concat(table_lists["unresolved"]),
        "phase_a_observed_decisions.tsv": _concat(
            table_lists["observed_decisions"]
        ),
        "phase_a_crossfit_decisions.tsv": _concat(
            table_lists["crossfit_decisions"]
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    _write_json(out_dir / "phase_a_attempts.json", attempts)

    manifest = {
        "schema_version": SCHEMA,
        "phase": "a",
        "selection_sha256": _sha256(out_dir / "phase_a_selection.json"),
        "attempts_sha256": _sha256(out_dir / "phase_a_attempts.json"),
        "output_sha256": {
            name: _sha256(out_dir / name) for name in PHASE_A_TABLES
        },
        "source_sha256": _source_hashes(),
        "observable_input_sha256": input_hashes,
        "selected_world_count": len(selected),
        "successful_world_count": sum(row["status"] == "ok" for row in attempts),
        "all_selected_worlds_attempted": len(attempts) == len(selected),
        "all_observable_inputs_loaded_before_truth": True,
        "scorer_or_truth_opened": False,
        "new_worlds_or_datasets_created": False,
        "retrospective_development_only": True,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_phase_a(out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    manifest = json.loads((out_dir / "phase_a_manifest.json").read_text())
    expected_payload = str(manifest.pop("manifest_payload_sha256", ""))
    if _canonical_sha256(manifest) != expected_payload:
        raise ValueError("Phase-A manifest payload hash mismatch")
    manifest["manifest_payload_sha256"] = expected_payload
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("Algorithm source changed after Phase A")
    if manifest.get("selection_sha256") != _sha256(
        out_dir / "phase_a_selection.json"
    ):
        raise ValueError("Phase-A selection changed")
    if manifest.get("attempts_sha256") != _sha256(
        out_dir / "phase_a_attempts.json"
    ):
        raise ValueError("Phase-A attempts changed")
    for name in PHASE_A_TABLES:
        if manifest["output_sha256"].get(name) != _sha256(out_dir / name):
            raise ValueError(f"Phase-A output changed: {name}")
    attempts = json.loads((out_dir / "phase_a_attempts.json").read_text())
    if not isinstance(attempts, list):
        raise ValueError("Phase-A attempts must be a list")
    direct = pd.read_csv(
        out_dir / "phase_a_direct_edges.tsv", sep="\t", keep_default_na=False
    )
    return manifest, attempts, direct


def _directed_edges(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for row in frame.itertuples(index=False):
        assay = str(row.source_assay)
        endpoint1 = str(row.endpoint1).upper()
        endpoint2 = str(row.endpoint2).upper()
        if (endpoint1, endpoint2) == ("TAIL", "ARROW"):
            edges.add((assay, str(row.node1), str(row.node2)))
        elif (endpoint1, endpoint2) == ("ARROW", "TAIL"):
            edges.add((assay, str(row.node2), str(row.node1)))
    return edges


def _adjacencies(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(row.source_assay), *sorted((str(row.node1), str(row.node2))))
        for row in frame.itertuples(index=False)
    }


def _truth_edges(path: Path) -> set[tuple[str, str, str]]:
    truth = pd.read_csv(path, sep="\t", keep_default_na=False)
    direct = truth.loc[truth["truth_type"].astype(str).eq("within_assay_direct")]
    output: set[tuple[str, str, str]] = set()
    for row in direct.itertuples(index=False):
        parent = str(row.parent)
        child = str(row.child)
        assay = (
            str(getattr(row, "source_assay"))
            if hasattr(row, "source_assay") and str(getattr(row, "source_assay"))
            else "chrom_" + child.rsplit("__", 1)[-1]
        )
        output.add((assay, parent, child))
    return output


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def run_phase_b(*, out_dir: Path) -> None:
    manifest, attempts, direct = _load_phase_a(out_dir)
    selection = json.loads((out_dir / "phase_a_selection.json").read_text())
    selected = selection.get("selected_worlds", [])
    if len(selected) != int(manifest["selected_world_count"]):
        raise ValueError("Selected world count differs from Phase-A manifest")

    # The complete prediction table and attempt ledger are in memory before this
    # loop opens the first already-existing scorer file.
    attempt_map = {
        (str(row["block"]), int(row["seed"])): row for row in attempts
    }
    world_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = total_reversed = 0
    for record in selected:
        block = str(record["block"])
        seed = int(record["seed"])
        scorer = (
            Path(str(record["world_root"]))
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        ).resolve(strict=True)
        truth = _truth_edges(scorer)
        predicted_frame = direct.loc[
            direct["block"].astype(str).eq(block)
            & direct["seed"].astype(int).eq(seed)
        ]
        predicted = _directed_edges(predicted_frame)
        predicted_adjacencies = _adjacencies(predicted_frame)
        truth_adjacencies = {
            (assay, *sorted((parent, child)))
            for assay, parent, child in truth
        }
        true_positive = predicted & truth
        false_positive = predicted - truth
        false_negative = truth - predicted
        reversed_edges = {
            edge
            for edge in false_positive
            if (edge[0], edge[2], edge[1]) in truth
        }
        adjacency_tp = predicted_adjacencies & truth_adjacencies
        status = str(attempt_map[(block, seed)]["status"])
        row = {
            "block": block,
            "seed": seed,
            "phase_a_status": status,
            "truth_edges": len(truth),
            "predicted_directed_edges": len(predicted),
            "predicted_adjacencies": len(predicted_adjacencies),
            "true_positive": len(true_positive),
            "false_positive": len(false_positive),
            "false_negative": len(false_negative),
            "strict_reversals": len(reversed_edges),
            "direct_precision": _rate(len(true_positive), len(predicted)),
            "direct_recall": _rate(len(true_positive), len(truth)),
            "adjacency_precision": _rate(len(adjacency_tp), len(predicted_adjacencies)),
            "adjacency_recall": _rate(len(adjacency_tp), len(truth_adjacencies)),
            "exact_directed_recovery": predicted == truth and status == "ok",
        }
        world_rows.append(row)
        total_tp += len(true_positive)
        total_fp += len(false_positive)
        total_fn += len(false_negative)
        total_reversed += len(reversed_edges)
        for outcome, edges in (
            ("false_positive", false_positive),
            ("false_negative", false_negative),
        ):
            for assay, parent, child in sorted(edges):
                failure_rows.append(
                    {
                        "block": block,
                        "seed": seed,
                        "outcome": outcome,
                        "source_assay": assay,
                        "parent": parent,
                        "child": child,
                        "strict_reverse": (
                            outcome == "false_positive"
                            and (assay, child, parent) in truth
                        ),
                    }
                )

    worlds = pd.DataFrame(world_rows).sort_values(["block", "seed"])
    failures = pd.DataFrame(failure_rows)
    worlds.to_csv(out_dir / "phase_b_world_metrics.tsv", sep="\t", index=False)
    failures.to_csv(out_dir / "phase_b_edge_failures.tsv", sep="\t", index=False)
    summary = {
        "schema_version": SCHEMA,
        "phase": "b",
        "retrospective_development_only": True,
        "new_worlds_or_datasets_created": False,
        "selection_seed": int(selection["selection_seed"]),
        "eligible_pool_size": int(selection["eligible_pool_size"]),
        "panel_size": len(worlds),
        "successful_phase_a_worlds": int(worlds["phase_a_status"].eq("ok").sum()),
        "invalid_phase_a_worlds": int(worlds["phase_a_status"].ne("ok").sum()),
        "exact_directed_worlds": int(worlds["exact_directed_recovery"].sum()),
        "micro_direct_precision": _rate(total_tp, total_tp + total_fp),
        "micro_direct_recall": _rate(total_tp, total_tp + total_fn),
        "mean_world_direct_precision": float(worlds["direct_precision"].mean()),
        "mean_world_direct_recall": float(worlds["direct_recall"].mean()),
        "total_true_positive": total_tp,
        "total_false_positive": total_fp,
        "total_false_negative": total_fn,
        "total_strict_reversals": total_reversed,
        "mean_predicted_direct_adjacencies": float(
            worlds["predicted_adjacencies"].mean()
        ),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "all_predictions_loaded_before_first_scorer_open": True,
    }
    summary["summary_payload_sha256"] = _canonical_sha256(summary)
    _write_json(out_dir / "phase_b_summary.json", summary)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--runs-root", type=Path, default=Path("pipelines/simulation/unpaired_multimodal/runs/formal_runs")
    )
    parser.add_argument(
        "--worlds-root", type=Path, default=Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds")
    )
    parser.add_argument("--panel-size", type=int, default=24)
    parser.add_argument("--selection-seed", type=int, default=20260901)
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        action="append",
        default=[],
        help=(
            "Self-hashed phase_a_selection.json whose worlds are excluded; "
            "repeat for multiple prior panels"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.phase == "a":
        run_phase_a(
            out_dir=args.out_dir,
            runs_root=args.runs_root,
            worlds_root=args.worlds_root,
            panel_size=args.panel_size,
            selection_seed=args.selection_seed,
            exclude_selections=args.exclude_selection,
        )
    else:
        run_phase_b(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
