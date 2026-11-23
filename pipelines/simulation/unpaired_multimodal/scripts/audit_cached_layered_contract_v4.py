#!/usr/bin/env python3
"""Audit the current CA-BOSS v4 layered contract on cached worlds only.

This utility is deliberately retrospective.  It never generates or modifies a
simulation world.  Phase A verifies current-source random-panel manifests,
deduplicates their observable direct predictions, and replays the frozen assay
participation and strong-core membership rules from observable membership
tables.  Phase B verifies the complete Phase-A closure before opening the
already existing simulator scorer files.

The audit keeps the identifiable outputs separate:

* within-assay directed candidate edges;
* assay-level shared-state participation;
* conservative shared-state feature membership.

It does not score exact cross-assay feature pairing or a cell-specific
cross-assay causal direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from replay_existing_world_panel_v4 import (  # noqa: E402
    ExistingWorld,
    _canonical_sha256,
    _directed_edges,
    _load_phase_a,
    _sha256,
    _truth_edges,
    inventory_existing_worlds,
)
from score_shared_state_subspace_v4 import (  # noqa: E402
    FROZEN_DISCOVERY_CONFIG,
)
from shared_state_subspace_v4 import (  # noqa: E402
    SharedStateSubspaceConfig,
    _select_assay_state_participation,
)
from state_equivalence_membership_v4 import (  # noqa: E402
    EQUIVALENCE_TIER,
    SCHEMA as STATE_EQUIVALENCE_SCHEMA,
    StateEquivalenceConfig,
    annotate_state_equivalence_membership,
    build_state_equivalence_groups,
)


SCHEMA = "ca_boss_v4.cached_layered_contract_audit.v2"
PHASE_A_TABLES = (
    "phase_a_direct_predictions.tsv",
    "phase_a_assay_participation.tsv",
    "phase_a_feature_membership.tsv",
    "phase_a_state_equivalence_groups.tsv",
    "phase_a_world_coverage.tsv",
    "phase_a_panel_provenance.tsv",
)
PHASE_B_TABLES = (
    "phase_b_world_metrics.tsv",
    "phase_b_feature_scores.tsv",
    "phase_b_assay_scores.tsv",
    "phase_b_direct_failures.tsv",
)
TRUTH_PARTS = frozenset({"scorer", "truth", "truth_typed_edges.tsv"})
STATE_MEMBER_ROLES = frozenset({"P", "Q", "S", "C"})
SENTINEL_ROLES = frozenset({"N1", "N2"})
SOURCE_FILES = (
    "audit_cached_layered_contract_v4.py",
    "replay_existing_world_panel_v4.py",
    "shared_state_subspace_v4.py",
    "score_shared_state_subspace_v4.py",
    "state_equivalence_membership_v4.py",
    "crossfit_hierarchical_collider_v4.py",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(SCRIPT_DIR / name) for name in SOURCE_FILES}


def _assert_observable(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if {part.casefold() for part in resolved.parts} & TRUTH_PARTS:
        raise ValueError(f"Phase A refuses scorer/truth path: {path}")
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(path)
    return resolved


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Selection is not an object: {path}")
    expected = str(payload.get("selection_payload_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("selection_payload_sha256", None)
    if _canonical_sha256(unsigned) != expected:
        raise ValueError(f"Selection payload hash mismatch: {path}")
    if payload.get("new_worlds_or_datasets_created") is not False:
        raise ValueError(f"Panel is not cached-world-only: {path}")
    if payload.get("scorer_or_truth_opened") is not False:
        raise ValueError(f"Phase-A selection opened scorer/truth: {path}")
    return payload


def _panel_direct_predictions(
    panel_dirs: Sequence[Path],
) -> tuple[
    dict[tuple[str, int], set[tuple[str, str, str]]],
    dict[tuple[str, int], set[str]],
    pd.DataFrame,
]:
    predictions: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    appearances: dict[tuple[str, int], set[str]] = {}
    provenance_rows: list[dict[str, Any]] = []
    for panel_dir in panel_dirs:
        manifest, attempts, direct = _load_phase_a(panel_dir)
        selection_path = panel_dir / "phase_a_selection.json"
        selection = _selection_payload(selection_path)
        attempt_map = {
            (str(row["block"]), int(row["seed"])): str(row["status"])
            for row in attempts
        }
        selected_records = selection.get("selected_worlds", [])
        if len(selected_records) != int(manifest["selected_world_count"]):
            raise ValueError(f"Panel selection count mismatch: {panel_dir}")
        for record in selected_records:
            key = (str(record["block"]), int(record["seed"]))
            if attempt_map.get(key) != "ok":
                continue
            frame = direct.loc[
                direct["block"].astype(str).eq(key[0])
                & direct["seed"].astype(int).eq(key[1])
            ]
            current = _directed_edges(frame)
            if key in predictions and predictions[key] != current:
                raise ValueError(
                    "Repeated random panels disagree for "
                    f"{key[0]} world_seed{key[1]}"
                )
            predictions[key] = current
            appearances.setdefault(key, set()).add(panel_dir.name)
        provenance_rows.append(
            {
                "panel": panel_dir.name,
                "selection_seed": int(selection["selection_seed"]),
                "panel_size": int(selection["panel_size"]),
                "successful_worlds": int(manifest["successful_world_count"]),
                "phase_a_manifest_sha256": _sha256(
                    panel_dir / "phase_a_manifest.json"
                ),
                "phase_a_selection_sha256": _sha256(selection_path),
                "phase_a_direct_predictions_sha256": _sha256(
                    panel_dir / "phase_a_direct_edges.tsv"
                ),
            }
        )
    return predictions, appearances, pd.DataFrame(provenance_rows)


def replay_strong_core_guard(
    membership: pd.DataFrame,
    *,
    config: SharedStateSubspaceConfig | None = None,
) -> pd.DataFrame:
    """Replay the current subtractive weak-tail guard idempotently.

    Older cached blocks predate the guard, while newer blocks already contain
    it.  Reconstructing the immediately pre-guard selected set makes the audit
    produce the same result for either schema without changing a frozen file.
    """

    cfg = config or SharedStateSubspaceConfig(**FROZEN_DISCOVERY_CONFIG)
    required = {
        "feature",
        "modality",
        "selection_status",
        "abstention_reason",
        "feature_global_maxT_p",
    }
    missing = sorted(required - set(membership))
    if missing:
        raise ValueError(f"Membership table is missing columns {missing}")
    output = membership.copy()
    status = output["selection_status"].fillna("").astype(str)
    reason = output["abstention_reason"].fillna("").astype(str)
    pre_guard_selected = status.eq("selected")
    if "within_assay_weak_maxT_tail_demoted" in output:
        demoted = output["within_assay_weak_maxT_tail_demoted"].astype(bool)
        pre_guard_selected |= demoted & reason.eq(
            "within_assay_strong_maxT_core_weak_tail"
        )
    p_values = pd.to_numeric(
        output["feature_global_maxT_p"], errors="raise"
    )
    strong = pre_guard_selected & p_values.le(cfg.within_assay_strong_core_maxT_p)
    counts = (
        pd.DataFrame(
            {
                "modality": output["modality"].astype(str),
                "strong": strong.astype(int),
            }
        )
        .groupby("modality", sort=True)["strong"]
        .transform("sum")
        .astype(int)
    )
    weak_tail = (
        pre_guard_selected
        & counts.ge(cfg.within_assay_strong_core_minimum_members)
        & p_values.gt(cfg.within_assay_strong_core_maxT_p)
    )
    selected = pre_guard_selected & ~weak_tail
    before = set(output.loc[pre_guard_selected, "modality"].astype(str))
    after = set(output.loc[selected, "modality"].astype(str))
    if after != before:
        raise ValueError("Strong-core replay changed assay participation")
    output["audit_pre_guard_selected"] = pre_guard_selected
    output["audit_within_assay_strong_core_member"] = strong
    output["audit_within_assay_strong_core_count"] = counts
    output["audit_within_assay_weak_maxT_tail_demoted"] = weak_tail
    output["audit_selected"] = selected
    return output


def _load_observable_state_layers(
    worlds: Sequence[ExistingWorld],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    config = SharedStateSubspaceConfig(**FROZEN_DISCOVERY_CONFIG)
    memberships: list[pd.DataFrame] = []
    participation: list[pd.DataFrame] = []
    equivalence_groups: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for world in worlds:
        path = _assert_observable(
            world.artifact_root
            / "shared_state_subspace_v4"
            / "subspace_feature_membership.tsv"
        )
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        guarded = replay_strong_core_guard(frame, config=config)
        annotated = annotate_state_equivalence_membership(guarded)
        assays = _select_assay_state_participation(frame, config)
        groups = build_state_equivalence_groups(annotated)
        for output in (annotated, assays, groups):
            output.insert(0, "seed", world.seed)
            output.insert(0, "block", world.block)
        memberships.append(annotated)
        participation.append(assays)
        equivalence_groups.append(groups)
        hashes[f"{world.block}:{world.seed}"] = _sha256(path)
    return (
        pd.concat(memberships, ignore_index=True, sort=False),
        pd.concat(participation, ignore_index=True, sort=False),
        pd.concat(equivalence_groups, ignore_index=True, sort=False),
        hashes,
    )


def run_phase_a(
    *,
    out_dir: Path,
    runs_root: Path,
    worlds_root: Path,
    panel_root: Path,
    panel_pattern: str,
) -> None:
    if out_dir.exists():
        raise FileExistsError(f"Audit Phase-A output already exists: {out_dir}")
    worlds = inventory_existing_worlds(runs_root, worlds_root)
    world_map = {(world.block, world.seed): world for world in worlds}
    panel_dirs = tuple(
        path
        for path in sorted(panel_root.glob(panel_pattern))
        if path.is_dir() and not path.is_symlink()
    )
    if not panel_dirs:
        raise FileNotFoundError(
            f"No existing random panels match {panel_root / panel_pattern}"
        )
    predictions, appearances, panel_provenance = _panel_direct_predictions(panel_dirs)
    unknown = sorted(set(predictions) - set(world_map))
    if unknown:
        raise ValueError(f"Random panels contain worlds outside inventory: {unknown}")

    direct_rows = [
        {
            "block": block,
            "seed": seed,
            "source_assay": assay,
            "parent": parent,
            "child": child,
        }
        for (block, seed), edges in sorted(predictions.items())
        for assay, parent, child in sorted(edges)
    ]
    direct = pd.DataFrame(
        direct_rows,
        columns=("block", "seed", "source_assay", "parent", "child"),
    )
    (
        membership,
        participation,
        equivalence_groups,
        membership_hashes,
    ) = _load_observable_state_layers(worlds)
    coverage = pd.DataFrame(
        [
            {
                "block": world.block,
                "seed": world.seed,
                "direct_prediction_available": key in predictions,
                "random_panel_appearances": len(appearances.get(key, set())),
                "random_panels": ";".join(sorted(appearances.get(key, set()))),
                "state_membership_available": True,
            }
            for world in worlds
            for key in [(world.block, world.seed)]
        ]
    )

    out_dir.mkdir(parents=True)
    tables = {
        "phase_a_direct_predictions.tsv": direct,
        "phase_a_assay_participation.tsv": participation,
        "phase_a_feature_membership.tsv": membership,
        "phase_a_state_equivalence_groups.tsv": equivalence_groups,
        "phase_a_world_coverage.tsv": coverage,
        "phase_a_panel_provenance.tsv": panel_provenance,
    }
    for name, frame in tables.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)
    manifest = {
        "schema_version": SCHEMA,
        "phase": "a",
        "eligible_cached_worlds": len(worlds),
        "direct_prediction_worlds": len(predictions),
        "all_cached_worlds_have_direct_predictions": set(predictions) == set(world_map),
        "random_panels": len(panel_dirs),
        "panel_pattern": panel_pattern,
        "panel_root": str(panel_root.resolve()),
        "source_sha256": _source_hashes(),
        "membership_input_sha256": membership_hashes,
        "output_sha256": {name: _sha256(out_dir / name) for name in PHASE_A_TABLES},
        "all_observable_inputs_loaded_before_truth": True,
        "scorer_or_truth_opened": False,
        "new_worlds_or_datasets_created": False,
        "retrospective_development_only": True,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    _write_json(out_dir / "phase_a_manifest.json", manifest)


def _load_audit_phase_a(
    out_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = out_dir / "phase_a_manifest.json"
    manifest = _read_json(path)
    expected = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if _canonical_sha256(unsigned) != expected:
        raise ValueError("Audit Phase-A manifest payload hash mismatch")
    if manifest.get("source_sha256") != _source_hashes():
        raise ValueError("Audit source changed after Phase A")
    for name in PHASE_A_TABLES:
        if manifest["output_sha256"].get(name) != _sha256(out_dir / name):
            raise ValueError(f"Audit Phase-A output changed: {name}")
    direct = pd.read_csv(
        out_dir / "phase_a_direct_predictions.tsv",
        sep="\t",
        keep_default_na=False,
    )
    membership = pd.read_csv(
        out_dir / "phase_a_feature_membership.tsv",
        sep="\t",
        keep_default_na=False,
    )
    participation = pd.read_csv(
        out_dir / "phase_a_assay_participation.tsv",
        sep="\t",
        keep_default_na=False,
    )
    return manifest, direct, membership, participation


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _direct_score(
    worlds: Sequence[ExistingWorld], direct: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    world_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for world in worlds:
        key_mask = direct["block"].astype(str).eq(world.block) & direct[
            "seed"
        ].astype(int).eq(world.seed)
        predicted = {
            (str(row.source_assay), str(row.parent), str(row.child))
            for row in direct.loc[key_mask].itertuples(index=False)
        }
        truth_path = (
            world.world_root
            / f"world_seed{world.seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        ).resolve(strict=True)
        truth = _truth_edges(truth_path)
        tp = predicted & truth
        fp = predicted - truth
        fn = truth - predicted
        reversals = {edge for edge in fp if (edge[0], edge[2], edge[1]) in truth}
        world_rows.append(
            {
                "block": world.block,
                "seed": world.seed,
                "truth_edges": len(truth),
                "predicted_edges": len(predicted),
                "true_positive": len(tp),
                "false_positive": len(fp),
                "false_negative": len(fn),
                "strict_reversals": len(reversals),
                "precision": _rate(len(tp), len(predicted)),
                "recall": _rate(len(tp), len(truth)),
                "exact_directed_recovery": predicted == truth,
            }
        )
        for outcome, edges in (("false_positive", fp), ("false_negative", fn)):
            for assay, parent, child in sorted(edges):
                failure_rows.append(
                    {
                        "block": world.block,
                        "seed": world.seed,
                        "outcome": outcome,
                        "source_assay": assay,
                        "parent": parent,
                        "child": child,
                        "strict_reverse": bool(
                            outcome == "false_positive"
                            and (assay, child, parent) in truth
                        ),
                    }
                )
    return pd.DataFrame(world_rows), pd.DataFrame(failure_rows)


def _feature_and_assay_scores(
    worlds: Sequence[ExistingWorld],
    membership: pd.DataFrame,
    participation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_maps: dict[tuple[str, int], dict[str, str]] = {}
    for world in worlds:
        path = (
            world.world_root
            / f"world_seed{world.seed}"
            / "scorer"
            / "role_assignments.tsv"
        ).resolve(strict=True)
        roles = pd.read_csv(path, sep="\t", keep_default_na=False)
        role_maps[(world.block, world.seed)] = dict(
            zip(roles["node"].astype(str), roles["role"].astype(str), strict=True)
        )

    feature_rows: list[dict[str, Any]] = []
    for row in membership.itertuples(index=False):
        key = (str(row.block), int(row.seed))
        modality = str(row.modality)
        feature = str(row.feature)
        if modality == "rna":
            role = "RNA"
            truth_member = True
        else:
            role = role_maps[key].get(feature, "")
            if role not in STATE_MEMBER_ROLES | SENTINEL_ROLES:
                raise ValueError(f"No scorer role for {key} feature {feature}")
            truth_member = role in STATE_MEMBER_ROLES
        selected = bool(row.strict_state_member)
        supported = bool(row.state_supported_member)
        support_tier = str(row.state_membership_tier)
        feature_rows.append(
            {
                "block": key[0],
                "seed": key[1],
                "feature": feature,
                "modality": modality,
                "truth_role": role,
                "truth_state_member": truth_member,
                "selected": selected,
                "state_supported": supported,
                "state_membership_tier": support_tier,
                "state_equivalence_only_member": bool(
                    row.state_equivalence_only_member
                ),
                "outcome": (
                    "true_positive"
                    if selected and truth_member
                    else "false_positive"
                    if selected
                    else "false_negative"
                    if truth_member
                    else "true_negative"
                ),
                "supported_outcome": (
                    "true_positive"
                    if supported and truth_member
                    else "false_positive"
                    if supported
                    else "false_negative"
                    if truth_member
                    else "true_negative"
                ),
            }
        )
    feature_scores = pd.DataFrame(feature_rows)

    assay_rows: list[dict[str, Any]] = []
    for world in worlds:
        key = (world.block, world.seed)
        world_membership = membership.loc[
            membership["block"].astype(str).eq(key[0])
            & membership["seed"].astype(int).eq(key[1])
        ]
        expected = set(world_membership["modality"].astype(str))
        role_values = set(role_maps[key].values())
        if not STATE_MEMBER_ROLES.issubset(role_values):
            raise ValueError(f"World {key} lacks the planted state-member roles")
        selected = set(
            participation.loc[
                participation["block"].astype(str).eq(key[0])
                & participation["seed"].astype(int).eq(key[1])
                & participation["selection_status"].astype(str).eq("selected"),
                "modality",
            ].astype(str)
        )
        for modality in sorted(expected):
            assay_rows.append(
                {
                    "block": key[0],
                    "seed": key[1],
                    "modality": modality,
                    "truth_state_participant": True,
                    "selected": modality in selected,
                    "outcome": (
                        "true_positive" if modality in selected else "false_negative"
                    ),
                }
            )
    return feature_scores, pd.DataFrame(assay_rows)


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    return int(pd.to_numeric(frame[column], errors="raise").sum())


def _world_factor_summary(
    worlds: Sequence[ExistingWorld],
) -> tuple[dict[str, Any], str]:
    """Summarize randomized/stress factors already present in cached worlds."""

    design_counts: dict[tuple[int, int, int, int], int] = {}
    base_cells: set[int] = set()
    cell_counts: list[int] = []
    detection_probabilities: list[float] = []
    population_fractions: list[float] = []
    population_shift_means: list[float] = []
    background_proxy_noise: list[float] = []
    beta_p: list[float] = []
    beta_q: list[float] = []
    gamma_p: list[float] = []
    gamma_q: list[float] = []
    peak_gene_coefficients: list[float] = []
    feature_layouts: set[str] = set()
    null_layouts: set[str] = set()
    state_loading_layouts: set[str] = set()
    mark_rotations: set[tuple[str, ...]] = set()
    stress_mark_sets: set[tuple[str, ...]] = set()
    fast_marks: dict[str, int] = {}
    slow_marks: dict[str, int] = {}
    condition_counts = {
        "detection_stress": {"false": 0, "true": 0},
        "population_divergence": {"false": 0, "true": 0},
        "background_integration_stress": {"false": 0, "true": 0},
    }
    modalities_per_world: set[int] = set()
    marks_per_world: set[int] = set()
    input_hashes: list[dict[str, Any]] = []
    for world in worlds:
        scorer = world.world_root / f"world_seed{world.seed}" / "scorer"
        parameters_path = (scorer / "world_parameters.json").resolve(strict=True)
        roles_path = (scorer / "role_assignments.tsv").resolve(strict=True)
        parameters = _read_json(parameters_path)
        roles = pd.read_csv(roles_path, sep="\t", keep_default_na=False)
        design = parameters["design"]
        design_key = tuple(
            int(design[name])
            for name in (
                "A_cells",
                "B_detection",
                "C_population_divergence",
                "D_background_observability",
            )
        )
        design_counts[design_key] = design_counts.get(design_key, 0) + 1
        base_cells.add(int(design["base_cells"]))
        for name in condition_counts:
            label = str(bool(design[name])).lower()
            condition_counts[name][label] += 1
        cell_counts.extend(int(value) for value in parameters["cell_counts"].values())
        detection_probabilities.extend(
            pd.to_numeric(roles["detection_probability"], errors="raise").tolist()
        )
        population_fractions.extend(
            float(value) for value in parameters["population_fractions"].values()
        )
        population_shift_means.append(float(parameters["population_shift_mean"]))
        background_proxy_noise.append(
            float(
                parameters["background_observation_model"][
                    "background_state_proxy_noise_sd"
                ]
            )
        )
        beta_p.extend(float(value) for value in parameters["beta_p_to_c"].values())
        beta_q.extend(float(value) for value in parameters["beta_q_to_c"].values())
        gamma_p.extend(
            float(value)
            for value in parameters["gamma_p_to_c_detection_log_odds"].values()
        )
        gamma_q.extend(
            float(value)
            for value in parameters["gamma_q_to_c_detection_log_odds"].values()
        )
        peak_gene_coefficients.extend(
            (
                float(parameters["slow_peak_gene_coefficient"]),
                float(parameters["fast_peak_gene_coefficient"]),
            )
        )
        feature_layouts.add(_canonical_sha256(parameters["feature_nodes"]))
        null_layouts.add(
            _canonical_sha256(
                {
                    mark: {role: nodes[role] for role in ("N1", "N2")}
                    for mark, nodes in parameters["feature_nodes"].items()
                }
            )
        )
        state_loading_layouts.add(
            _canonical_sha256(
                {
                    "dimension": parameters["state_dimension"],
                    "sign": parameters["state_sign"],
                    "q_dimension": parameters["q_dimension"],
                    "q_sign": parameters["q_sign"],
                }
            )
        )
        mark_rotations.add(tuple(str(value) for value in parameters["mark_rotation"]))
        stress_mark_sets.add(tuple(sorted(str(value) for value in parameters["stress_marks"])))
        for destination, key in (
            (fast_marks, str(parameters["fast_mark"])),
            (slow_marks, str(parameters["slow_mark"])),
        ):
            destination[key] = destination.get(key, 0) + 1
        modalities_per_world.add(len(parameters["modalities"]))
        marks_per_world.add(len(parameters["marks"]))
        input_hashes.append(
            {
                "block": world.block,
                "seed": world.seed,
                "world_parameters_sha256": _sha256(parameters_path),
                "role_assignments_sha256": _sha256(roles_path),
            }
        )

    def numeric_range(values: Sequence[float | int]) -> dict[str, float]:
        return {"min": float(min(values)), "max": float(max(values))}

    summary = {
        "worlds": len(worlds),
        "factorial_design": {
            "name": "2^(4-1)_fractional_factorial",
            "unique_design_cells": len(design_counts),
            "cell_counts": [
                {
                    "A_cells": key[0],
                    "B_detection": key[1],
                    "C_population_divergence": key[2],
                    "D_background_observability": key[3],
                    "worlds": count,
                }
                for key, count in sorted(design_counts.items())
            ],
        },
        "modalities_per_world": sorted(modalities_per_world),
        "chromatin_marks_per_world": sorted(marks_per_world),
        "base_cells": sorted(base_cells),
        "assay_cell_count_range": numeric_range(cell_counts),
        "focal_detection_probability_range": numeric_range(detection_probabilities),
        "population_fraction_range": numeric_range(population_fractions),
        "population_shift_mean_range": numeric_range(population_shift_means),
        "background_state_proxy_noise_sd_range": numeric_range(
            background_proxy_noise
        ),
        "beta_p_to_c_range": numeric_range(beta_p),
        "beta_q_to_c_range": numeric_range(beta_q),
        "gamma_p_to_c_detection_log_odds_range": numeric_range(gamma_p),
        "gamma_q_to_c_detection_log_odds_range": numeric_range(gamma_q),
        "peak_gene_coefficient_range": numeric_range(peak_gene_coefficients),
        "unique_feature_layouts": len(feature_layouts),
        "unique_null_feature_layouts": len(null_layouts),
        "unique_state_loading_layouts": len(state_loading_layouts),
        "unique_mark_rotations": len(mark_rotations),
        "unique_stress_mark_sets": len(stress_mark_sets),
        "fast_mark_world_counts": dict(sorted(fast_marks.items())),
        "slow_mark_world_counts": dict(sorted(slow_marks.items())),
        "condition_world_counts": condition_counts,
        "detection_draws_precede_role_assignment": True,
        "truth_topology_invariant_across_conditions": True,
    }
    return summary, _canonical_sha256(input_hashes)


def run_phase_b(
    *,
    out_dir: Path,
    runs_root: Path,
    worlds_root: Path,
    sentinel_null_summary: Path,
    v3_summary: Path,
) -> None:
    manifest, direct, membership, participation = _load_audit_phase_a(out_dir)
    worlds = inventory_existing_worlds(runs_root, worlds_root)
    if len(worlds) != int(manifest["eligible_cached_worlds"]):
        raise ValueError("Eligible cached-world inventory changed after Phase A")

    # All predictions and state-layer decisions are in memory before the first
    # scorer-domain path is constructed below.
    world_metrics, direct_failures = _direct_score(worlds, direct)
    feature_scores, assay_scores = _feature_and_assay_scores(
        worlds, membership, participation
    )
    factor_summary, factor_input_hash_list_sha256 = _world_factor_summary(worlds)
    tables = {
        "phase_b_world_metrics.tsv": world_metrics,
        "phase_b_feature_scores.tsv": feature_scores,
        "phase_b_assay_scores.tsv": assay_scores,
        "phase_b_direct_failures.tsv": direct_failures,
    }
    for name, frame in tables.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    direct_tp = _sum_column(world_metrics, "true_positive")
    direct_fp = _sum_column(world_metrics, "false_positive")
    direct_fn = _sum_column(world_metrics, "false_negative")
    direct_reversals = _sum_column(world_metrics, "strict_reversals")
    feature_counts = feature_scores["outcome"].value_counts().to_dict()
    feature_tp = int(feature_counts.get("true_positive", 0))
    feature_fp = int(feature_counts.get("false_positive", 0))
    feature_fn = int(feature_counts.get("false_negative", 0))
    role_recall = {
        role: _rate(
            int(
                feature_scores.loc[
                    feature_scores["truth_role"].eq(role)
                    & feature_scores["selected"].astype(bool)
                ].shape[0]
            ),
            int(feature_scores["truth_role"].eq(role).sum()),
        )
        for role in ("P", "Q", "C", "S", "RNA")
    }
    supported_counts = (
        feature_scores["supported_outcome"].value_counts().to_dict()
    )
    supported_tp = int(supported_counts.get("true_positive", 0))
    supported_fp = int(supported_counts.get("false_positive", 0))
    supported_fn = int(supported_counts.get("false_negative", 0))
    equivalence_only = feature_scores["state_membership_tier"].eq(
        EQUIVALENCE_TIER
    )
    truth_members = feature_scores["truth_state_member"].astype(bool)
    equivalence_tp = int((equivalence_only & truth_members).sum())
    equivalence_fp = int((equivalence_only & ~truth_members).sum())
    supported_role_recall = {
        role: _rate(
            int(
                feature_scores.loc[
                    feature_scores["truth_role"].eq(role)
                    & feature_scores["state_supported"].astype(bool)
                ].shape[0]
            ),
            int(feature_scores["truth_role"].eq(role).sum()),
        )
        for role in ("P", "Q", "C", "S", "RNA")
    }
    leave_one_block_out: list[dict[str, Any]] = []
    logo = LeaveOneGroupOut()
    for _, test_indices in logo.split(
        feature_scores,
        truth_members,
        groups=feature_scores["block"].astype(str),
    ):
        held_out = feature_scores.iloc[test_indices]
        held_out_truth = held_out["truth_state_member"].astype(bool)
        held_out_supported = held_out["state_supported"].astype(bool)
        tp = int((held_out_supported & held_out_truth).sum())
        fp = int((held_out_supported & ~held_out_truth).sum())
        fn = int((~held_out_supported & held_out_truth).sum())
        blocks = sorted(set(held_out["block"].astype(str)))
        if len(blocks) != 1:
            raise ValueError("LeaveOneGroupOut produced a mixed source block")
        leave_one_block_out.append(
            {
                "held_out_block": blocks[0],
                "true_positive_members": tp,
                "false_positive_members": fp,
                "false_negative_members": fn,
                "precision": _rate(tp, tp + fp),
                "recall": _rate(tp, tp + fn),
            }
        )
    selected_assays = int(assay_scores["selected"].astype(bool).sum())
    worlds_all_assays = int(
        assay_scores.groupby(["block", "seed"], sort=True)["selected"]
        .all()
        .sum()
    )
    sentinel = _read_json(sentinel_null_summary)
    v3 = _read_json(v3_summary)
    summary = {
        "schema_version": SCHEMA,
        "algorithm_schema_version": "ca_boss_v4.layered_contract.v4.24",
        "direct_selector_schema_version": "crossfit_hierarchical_collider.v4.22",
        "phase": "b",
        "retrospective_development_only": True,
        "fresh_locked_claim": False,
        "new_worlds_or_datasets_created": False,
        "eligible_cached_worlds": len(worlds),
        "random_panel_union_worlds": int(
            pd.read_csv(
                out_dir / "phase_a_world_coverage.tsv", sep="\t"
            )["direct_prediction_available"].astype(bool).sum()
        ),
        "direct_layer": {
            "exact_worlds": int(world_metrics["exact_directed_recovery"].sum()),
            "true_positive_directed_arms": direct_tp,
            "false_positive_directed_arms": direct_fp,
            "false_negative_directed_arms": direct_fn,
            "strict_reversals": direct_reversals,
            "micro_precision": _rate(direct_tp, direct_tp + direct_fp),
            "micro_recall": _rate(direct_tp, direct_tp + direct_fn),
            "mean_direct_adjacencies_per_world": float(
                world_metrics["predicted_edges"].mean()
            ),
        },
        "shared_state_assay_layer": {
            "estimand": "multi_assay_state_responsive_set",
            "assay_tests": len(assay_scores),
            "selected_assays": selected_assays,
            "worlds_with_all_modalities": worlds_all_assays,
            "assay_participation_recall": _rate(selected_assays, len(assay_scores)),
            "assay_precision_identifiable_from_current_worlds": False,
            "sentinel_pseudo_assay_null_audit": sentinel,
            "sentinel_pseudo_assay_summary_sha256": _sha256(
                sentinel_null_summary
            ),
        },
        "strict_state_feature_layer": {
            "true_positive_members": feature_tp,
            "false_positive_members": feature_fp,
            "false_negative_members": feature_fn,
            "precision": _rate(feature_tp, feature_tp + feature_fp),
            "recall": _rate(feature_tp, feature_tp + feature_fn),
            "role_recall": role_recall,
            "interpretation": (
                "conservative member annotation only; assay participation does "
                "not promote reporters into feature membership"
            ),
        },
        "state_equivalence_feature_layer": {
            "schema_version": STATE_EQUIVALENCE_SCHEMA,
            "maximum_feature_global_maxT_p": (
                StateEquivalenceConfig().maximum_feature_global_maxT_p
            ),
            "equivalence_only_true_positive_members": equivalence_tp,
            "equivalence_only_false_positive_members": equivalence_fp,
            "supported_true_positive_members": supported_tp,
            "supported_false_positive_members": supported_fp,
            "supported_false_negative_members": supported_fn,
            "supported_precision": _rate(
                supported_tp, supported_tp + supported_fp
            ),
            "supported_recall": _rate(
                supported_tp, supported_tp + supported_fn
            ),
            "supported_role_recall": supported_role_recall,
            "leave_one_source_block_out_replay": leave_one_block_out,
            "strict_membership_changed": False,
            "assay_participation_changed": False,
            "pairwise_cross_assay_edges_created": 0,
            "direct_edges_created_or_changed": 0,
            "interpretation": (
                "strong redundancy-demoted reporters are retained only as "
                "within-assay state-equivalence members"
            ),
        },
        "integration_robustness": {
            "held_out_background_only": True,
            "cross_fitted_feature_models": True,
            "assay_local_embedding_feature_folds": 2,
            "aggregation_resolutions": 3,
            "integration_seeds_per_resolution": 2,
            "conditional_model_views": int(
                FROZEN_DISCOVERY_CONFIG["observed_model_views"]
            ),
            "assay_participation_requires_all_views": int(
                FROZEN_DISCOVERY_CONFIG[
                    "minimum_assay_state_participation_views"
                ]
            ),
            "common_coordinate_registration_used": False,
        },
        "existing_world_factor_coverage": factor_summary,
        "existing_world_factor_input_hash_list_sha256": (
            factor_input_hash_list_sha256
        ),
        "v3_baseline": {
            "candidate": v3.get("candidate"),
            "gapca_world_success_rate": v3["arms"]["gapca"][
                "world_success_rate"
            ],
            "gapca_mean_estimated_biological_edges": v3["arms"]["gapca"][
                "mean_estimated_biological_edges"
            ],
            "gapca_mean_matched_skeleton_f1": v3["arms"]["gapca"][
                "mean_matched_skeleton_f1"
            ],
            "validation_summary_sha256": _sha256(v3_summary),
        },
        "source_sha256": _source_hashes(),
        "phase_a_manifest_sha256": _sha256(out_dir / "phase_a_manifest.json"),
        "all_predictions_loaded_before_first_scorer_open": True,
        "interpretation": {
            "within_assay_direct_structure": (
                "retrospective recovery on the complete eligible cached pool"
            ),
            "cross_assay_structure": (
                "one hypothesis-only shared-state participation set; no pairwise "
                "expansion"
            ),
            "state_equivalence_structure": (
                "within-assay uncertainty sets for strong redundant reporters; "
                "not exact cross-assay pairing"
            ),
            "exact_cross_assay_feature_pairing": "not claimed",
            "cell_specific_cross_assay_direct_effects": (
                "not identifiable from the agreed disjoint-cell variables"
            ),
        },
    }
    summary["output_sha256"] = {
        name: _sha256(out_dir / name) for name in PHASE_B_TABLES
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
    parser.add_argument(
        "--panel-root",
        type=Path,
        default=Path("pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4"),
    )
    parser.add_argument(
        "--panel-pattern",
        default="existing_random_panel_v422_*",
    )
    parser.add_argument(
        "--sentinel-null-summary",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/"
            "assay_participation_omnibus_sentinel_nulls_v1/summary.json"
        ),
    )
    parser.add_argument(
        "--v3-summary",
        type=Path,
        default=Path(
            "pipelines/simulation/unpaired_multimodal/results/channel_aware_boss_validation_summary.json"
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
            panel_root=args.panel_root,
            panel_pattern=args.panel_pattern,
        )
    else:
        run_phase_b(
            out_dir=args.out_dir,
            runs_root=args.runs_root,
            worlds_root=args.worlds_root,
            sentinel_null_summary=args.sentinel_null_summary,
            v3_summary=args.v3_summary,
        )


if __name__ == "__main__":
    main()
