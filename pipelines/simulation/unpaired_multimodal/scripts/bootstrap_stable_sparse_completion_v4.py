#!/usr/bin/env python3
"""World-bootstrap stability guard for CA-BOSS v4.51.

This stage treats v4.49 sparse completions as proposals, not automatic gains.
For each target block with a proposal, it resamples whole training worlds
within every non-target block 100 times and reruns the nested dual-
regularization reporter selector.  A proposed reporter is retained only when
it remains the unique passing candidate in at least 80% of replicates.

The stable completions are unioned with the target-safe v4.47 baseline.  This
intentionally rolls back v4.48--v4.50 pairs that were true in the retrospective
campaign but unstable under world resampling.  No target truth, new world,
paired cell, simulator state, external biological prior, direct cross-assay
edge, or direction is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crossfit_multivariate_reporter_rescue_v4 as v448  # noqa: E402
import crossfit_sparse_reporter_completion_v4 as v449  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402


SCHEMA = "ca_boss_v4.bootstrap_stable_sparse_completion.v1"
ALGORITHM = "CA-BOSS v4.51 world-bootstrap-stable sparse completion"
BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_FREQUENCY_THRESHOLD = 0.80
BOOTSTRAP_BASE_SEED = 451000
FEATURE_COLUMNS = v449.FEATURE_COLUMNS
MODEL_C_VALUES = v449.MODEL_C_VALUES
FIXED_SELECTION_RULE = (
    "treat_only_target_safe_v449_new_sparse_completions_as_proposals_for_"
    "each_target_with_a_proposal_resample_whole_training_worlds_with_"
    "replacement_within_each_non_target_training_block_100_times_rerun_"
    "the_nested_C_0_1_and_0_3_polynomial_reporter_models_and_each_"
    "corresponding_full_calibration_block_sentinel_null_envelope_require_"
    "the_reporter_to_remain_the_unique_dual_regularization_pass_in_at_"
    "least_80_percent_of_replicates_retain_its_previously_unique_all_"
    "context_partner_union_only_with_target_safe_v447_merge_connected_"
    "pairs_as_unexpanded_ambiguity_hyperedges_and_never_claim_direct_"
    "structure_or_direction"
)

STABILITY_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "modality",
    "feature",
    "assay_upstream_selected",
    "proposed_by_target_safe_v449",
    "bootstrap_selection_count",
    "bootstrap_replicates",
    "bootstrap_selection_frequency",
    "bootstrap_frequency_threshold",
    "retained_by_bootstrap_stability_guard",
    "selection_status",
]

REPLICATE_COLUMNS = [
    "target_block",
    "bootstrap_seed",
    "bootstrap_replicate",
    "model_c",
    "calibration_blocks",
    "calibration_null_maxima_json",
    "null_probability_upper_envelope",
    "all_nested_models_converged",
    "dual_model_raw_pass_candidate_count",
    "unique_unresolved_reporter_count",
]

PAIR_AUDIT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "unresolved_modality",
    "candidate_reporter_feature",
    "partner_modality",
    "partner_reporter_feature",
    "candidate_assay_pair_json",
    "candidate_cross_assay_feature_pair_json",
    "proposed_by_target_safe_v449",
    "reporter_bootstrap_selection_frequency",
    "reporter_passes_bootstrap_stability_guard",
    "selected_by_v451_bootstrap_stability",
    "selected_by_target_safe_v447",
    "newly_selected_by_v451",
    "retained_in_union",
    "merged_hyperedge_id",
]

CANDIDATE_COLUMNS = [
    "stable_hyperedge_id",
    "world_key",
    "block",
    "seed",
    "modalities",
    "n_modalities",
    "candidate_assay_pairs_json",
    "candidate_assay_pair_count",
    "candidate_cross_assay_feature_pairs_json",
    "candidate_cross_assay_feature_pair_count",
    "candidate_pair_selection_audit_json",
    "selection_sources",
    "bootstrap_replicates",
    "bootstrap_frequency_threshold",
    "assay_participation_is_set_valued",
    "exact_assay_participation_claimed",
    "hypothesis_selected",
    "selection_status",
    "hypothesis_type",
    "within_assay_direct_structure_claimed",
    "exact_cross_assay_pair_claimed",
    "cross_assay_direction_claimed",
    "candidate_pairs_may_be_expanded_as_edges",
    "fixed_selection_rule",
]


def _as_bool(value: Any) -> bool:
    return synchronization._as_bool(value)


def _pair_key(left: str, right: str) -> frozenset[str]:
    return v448._pair_key(left, right)


def _pair_sort_key(pair: frozenset[str]) -> tuple[str, str]:
    return v448._pair_sort_key(pair)


def _verified_v449_calibration(
    directory: Path, *, target_block: str
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest = v449._verified_payload(
        resolved / "calibration_manifest.json", "manifest_payload_sha256"
    )
    if manifest.get("schema_version") != v449.SCHEMA:
        raise ValueError("Unexpected v4.49 calibration schema")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("target_block_scorer_truth_opened") is not False:
        raise ValueError("v4.49 opened target-block truth")
    if boundary.get("target_world_truth_used_for_fit_or_selection") is not False:
        raise ValueError("v4.49 used target-world truth")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.49 lacks a target-specific pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.49 created new data")
    if list(manifest.get("model_feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("v4.49 model feature set changed")
    if [float(value) for value in manifest.get("model_c_values", [])] != list(
        MODEL_C_VALUES
    ):
        raise ValueError("v4.49 regularization guard changed")
    if int(manifest.get("polynomial_degree", -1)) != v449.POLYNOMIAL_DEGREE:
        raise ValueError("v4.49 polynomial degree changed")
    if manifest.get("all_nested_models_converged") is not True:
        raise ValueError("v4.49 nested model did not converge")
    if str(manifest.get("target_block")) != str(target_block):
        raise ValueError("v4.49 calibration targets a different block")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"v4.49 output hash mismatch: {name}")
    reporter_audit = pd.read_csv(
        resolved / "nested_reporter_candidate_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    pair_audit = pd.read_csv(
        resolved / "reporter_partner_pair_audit.tsv",
        sep="\t",
        keep_default_na=False,
    )
    return resolved, manifest, reporter_audit, pair_audit


def _baseline_components(candidates: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        pairs = v449._parse_pair_payload(row.candidate_assay_pairs_json)
        features = v449._parse_feature_pair_payload(
            row.candidate_cross_assay_feature_pairs_json
        )
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.47 candidate assay-pair count mismatch")
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.47 candidate feature-pair count mismatch")
        components[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(str(row.candidate_pair_selection_audit_json)),
                "source": "target_safe_v447",
            }
        )
    return components


def _bootstrap_frequency_passes(
    selection_count: int,
    replicate_count: int,
    *,
    threshold: float = BOOTSTRAP_FREQUENCY_THRESHOLD,
) -> bool:
    """Return the inclusive, predeclared stability-selection decision."""

    if replicate_count <= 0:
        return False
    return (float(selection_count) / float(replicate_count)) >= float(threshold)


def _candidate_identity(row: Any) -> tuple[str, str, str]:
    return (str(row.world_key), str(row.modality), str(row.feature))


def _union_candidates(
    *,
    target_keys: Sequence[str],
    evidence_frames: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    stable_pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[pd.DataFrame, dict[tuple[str, frozenset[str]], str]]:
    """Merge stable proposals with v4.47 without expanding ambiguity sets."""

    rows: list[dict[str, Any]] = []
    merged_id_by_pair: dict[tuple[str, frozenset[str]], str] = {}
    for world_key in sorted(target_keys):
        frame = evidence_frames[world_key]
        block = str(frame["block"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        sources: list[dict[str, Any]] = [
            dict(value) for value in baseline.get(world_key, [])
        ]
        for selected in stable_pair_rows.get(world_key, []):
            pair = _pair_key(
                selected["unresolved_modality"], selected["partner_modality"]
            )
            feature_pair = tuple(
                sorted(
                    (
                        str(selected["candidate_reporter_feature"]),
                        str(selected["partner_reporter_feature"]),
                    )
                )
            )
            sources.append(
                {
                    "pairs": {pair},
                    "features": {feature_pair},
                    "audit": [
                        {
                            "assay_pair": list(_pair_sort_key(pair)),
                            "candidate_reporter": str(
                                selected["candidate_reporter_feature"]
                            ),
                            "partner_reporter": str(
                                selected["partner_reporter_feature"]
                            ),
                            "bootstrap_selection_frequency": float(
                                selected["reporter_bootstrap_selection_frequency"]
                            ),
                            "source": "v451_bootstrap_stable_sparse_completion",
                        }
                    ],
                    "source": "v451_bootstrap_stable_sparse_completion",
                }
            )
        all_pairs = (
            set().union(*(source["pairs"] for source in sources))
            if sources
            else set()
        )
        for component_index, component_pairs in enumerate(
            v448.v447._connected_pair_components(all_pairs), start=1
        ):
            hyperedge_id = f"bootstrap_stable_shared_state_{component_index:03d}"
            modalities = sorted(
                set().union(*(set(pair) for pair in component_pairs))
            )
            component_sources = [
                source
                for source in sources
                if set(source["pairs"]) & component_pairs
            ]
            features = set().union(
                *(source["features"] for source in component_sources)
            )
            audits: list[dict[str, Any]] = []
            source_names: set[str] = set()
            for source in component_sources:
                audit = source["audit"]
                if isinstance(audit, list):
                    audits.extend(audit)
                else:
                    audits.append(dict(audit))
                source_names.add(str(source["source"]))
            pair_payload = [
                list(_pair_sort_key(pair))
                for pair in sorted(component_pairs, key=_pair_sort_key)
            ]
            for pair in component_pairs:
                merged_id_by_pair[(world_key, pair)] = hyperedge_id
            rows.append(
                {
                    "stable_hyperedge_id": hyperedge_id,
                    "world_key": world_key,
                    "block": block,
                    "seed": seed,
                    "modalities": ";".join(modalities),
                    "n_modalities": len(modalities),
                    "candidate_assay_pairs_json": json.dumps(
                        pair_payload, separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": len(component_pairs),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        sorted(features), separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(features),
                    "candidate_pair_selection_audit_json": json.dumps(
                        audits, sort_keys=True, separators=(",", ":")
                    ),
                    "selection_sources": ";".join(sorted(source_names)),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_frequency_threshold": (
                        BOOTSTRAP_FREQUENCY_THRESHOLD
                    ),
                    "assay_participation_is_set_valued": (
                        len(component_pairs) != 1
                    ),
                    "exact_assay_participation_claimed": (
                        len(component_pairs) == 1
                    ),
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_bootstrap_stable_ambiguity"
                        if len(component_pairs) != 1
                        else "selected_as_bootstrap_stable_point_hypothesis"
                    ),
                    "hypothesis_type": "cross_assay_shared_state_equivalence",
                    "within_assay_direct_structure_claimed": False,
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                }
            )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS), merged_id_by_pair


def calibrate_target_block(
    *,
    evidence_roots: Sequence[Path],
    v447_calibration_dir: Path,
    v449_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Bootstrap v4.49 proposals and retain only stable target hypotheses."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    worlds = v449._discover_completion_worlds(evidence_roots)
    available_blocks = sorted(
        {str(manifest["block"]) for _, manifest in worlds.values()}
    )
    target_block = str(target_block)
    if target_block not in available_blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    calibration_blocks = [
        block for block in available_blocks if block != target_block
    ]
    if len(calibration_blocks) < 3:
        raise ValueError("At least three non-target blocks are required")

    v447_dir, v447_manifest, _ = v448._verified_v447_calibration(
        v447_calibration_dir, target_block=target_block
    )
    v447_candidates = pd.read_csv(
        v447_dir / "crossfit_empirical_null_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if len(v447_candidates) and not v447_candidates[
        "hypothesis_selected"
    ].map(_as_bool).all():
        raise ValueError("v4.47 contains an unselected candidate")
    (
        v449_dir,
        v449_manifest,
        v449_reporter_audit,
        v449_pair_audit,
    ) = _verified_v449_calibration(
        v449_calibration_dir, target_block=target_block
    )

    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)
    candidate_frames: dict[str, pd.DataFrame] = {}
    roles_by_world: dict[str, dict[str, str]] = {}
    evidence_input_hashes: dict[str, dict[str, str]] = {}
    training_role_hashes: dict[str, str] = {}
    world_block: dict[str, str] = {}
    target_keys: list[str] = []
    for key in sorted(worlds):
        directory, manifest = worlds[key]
        block = str(manifest["block"])
        seed = int(manifest["seed"])
        candidate_path = directory / "external_reporter_candidates.tsv"
        candidates = pd.read_csv(
            candidate_path, sep="\t", keep_default_na=False
        )
        if len(candidates) != int(manifest["external_candidate_count"]):
            raise ValueError(f"Candidate count mismatch: {key}")
        if len(candidates) and (
            ~candidates["world_key"].astype(str).eq(key)
            | ~candidates["block"].astype(str).eq(block)
            | ~pd.to_numeric(candidates["seed"], errors="raise").eq(seed)
        ).any():
            raise ValueError(f"Candidate identity mismatch: {key}")
        if candidates.loc[:, ["modality", "feature"]].duplicated().any():
            raise ValueError(f"Duplicate reporter candidate: {key}")
        candidate_frames[key] = candidates
        world_block[key] = block
        evidence_input_hashes[key] = {
            "reporter_completion_evidence_manifest.json": (
                synchronization._sha256(
                    directory / "reporter_completion_evidence_manifest.json"
                )
            ),
            "external_reporter_candidates.tsv": synchronization._sha256(
                candidate_path
            ),
            "unresolved_reporter_pair_evidence.tsv": synchronization._sha256(
                directory / "unresolved_reporter_pair_evidence.tsv"
            ),
            "all_assay_synchronization_contexts.tsv": synchronization._sha256(
                directory / "all_assay_synchronization_contexts.tsv"
            ),
        }
        if block == target_block:
            target_keys.append(key)
            continue

        role_path = (
            formal_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "role_assignments.tsv"
        )
        if role_path.is_symlink() or not role_path.is_file():
            raise FileNotFoundError(role_path)
        roles = pd.read_csv(role_path, sep="\t", keep_default_na=False)
        if roles["node"].astype(str).duplicated().any():
            raise ValueError(f"Role table contains duplicate nodes: {key}")
        roles_by_world[key] = dict(
            zip(roles["node"].astype(str), roles["role"].astype(str))
        )
        training_role_hashes[key] = synchronization._sha256(role_path)

    if not target_keys:
        raise ValueError("Target block has no cached evidence worlds")
    target_key_set = set(target_keys)
    for label, manifest in (("v4.47", v447_manifest), ("v4.49", v449_manifest)):
        if set(map(str, manifest.get("target_world_keys", []))) != target_key_set:
            raise ValueError(f"{label} and v4.51 target worlds differ")
    if len(v447_candidates) and not v447_candidates["world_key"].astype(
        str
    ).isin(target_key_set).all():
        raise ValueError("v4.47 candidates include a non-target world")
    expected_evidence_inputs = v449_manifest.get("input_sha256", {}).get(
        "completion_evidence_worlds", {}
    )
    if expected_evidence_inputs != evidence_input_hashes:
        raise ValueError("v4.49 and v4.51 evidence-world bindings differ")

    target_candidates = pd.concat(
        [candidate_frames[key] for key in sorted(target_keys)],
        ignore_index=True,
    )
    candidate_identities = [
        _candidate_identity(row)
        for row in target_candidates.itertuples(index=False)
    ]
    if len(set(candidate_identities)) != len(candidate_identities):
        raise ValueError("Target reporter candidate identities are not unique")

    reporter_unique = v449_reporter_audit[
        "uniquely_selected_reporter_within_assay"
    ].map(_as_bool)
    nominal_reporter_identities = {
        _candidate_identity(row)
        for row in v449_reporter_audit.loc[reporter_unique].itertuples(
            index=False
        )
    }
    proposal_pair_mask = v449_pair_audit[
        "newly_selected_by_v449_sparse_completion"
    ].map(_as_bool)
    proposal_pairs = v449_pair_audit.loc[proposal_pair_mask].copy()
    proposal_reporter_identities = {
        (
            str(row.world_key),
            str(row.unresolved_modality),
            str(row.candidate_reporter_feature),
        )
        for row in proposal_pairs.itertuples(index=False)
    }
    if not proposal_reporter_identities.issubset(nominal_reporter_identities):
        raise ValueError("A v4.49 proposal lacks its unique reporter decision")

    block_keys: dict[str, list[str]] = {
        block: sorted(
            key for key, value in world_block.items() if value == block
        )
        for block in calibration_blocks
    }
    if any(not keys for keys in block_keys.values()):
        raise ValueError("A calibration block has no evidence worlds")

    selection_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    replicate_audit_rows: list[dict[str, Any]] = []
    bootstrap_replicates_run = (
        BOOTSTRAP_REPLICATES if proposal_reporter_identities else 0
    )
    bootstrap_seed = BOOTSTRAP_BASE_SEED + available_blocks.index(target_block)
    for replicate in range(bootstrap_replicates_run):
        fold_samples: dict[str, list[str]] = {}
        for calibration_index, calibration_block in enumerate(calibration_blocks):
            sampled_keys: list[str] = []
            training_blocks = [
                block
                for block in calibration_blocks
                if block != calibration_block
            ]
            for training_index, training_block in enumerate(training_blocks):
                keys = block_keys[training_block]
                seed_sequence = np.random.SeedSequence(
                    [
                        bootstrap_seed,
                        replicate,
                        calibration_index,
                        training_index,
                    ]
                )
                rng = np.random.default_rng(seed_sequence)
                sampled_indices = rng.integers(0, len(keys), size=len(keys))
                sampled_keys.extend(keys[int(index)] for index in sampled_indices)
            fold_samples[calibration_block] = sampled_keys

        raw_pass_by_c: dict[float, np.ndarray] = {}
        convergence_by_c: dict[float, bool] = {}
        fold_records_by_c: dict[float, dict[str, Any]] = {}
        for model_c in MODEL_C_VALUES:
            target_fold_probabilities: list[np.ndarray] = []
            fold_null_maxima: dict[str, float] = {}
            fold_convergence: list[bool] = []
            for calibration_block in calibration_blocks:
                training = pd.concat(
                    [candidate_frames[key] for key in fold_samples[calibration_block]],
                    ignore_index=True,
                )
                calibration_frame = pd.concat(
                    [candidate_frames[key] for key in block_keys[calibration_block]],
                    ignore_index=True,
                )
                training_labels = v449._reporter_labels(
                    training, roles_by_world
                )
                calibration_labels = v449._reporter_labels(
                    calibration_frame, roles_by_world
                )
                _, target_probability, fold_audit = v449._fit_fold(
                    training,
                    training_labels,
                    calibration_frame,
                    calibration_labels,
                    target_candidates,
                    model_c=model_c,
                )
                target_fold_probabilities.append(target_probability)
                fold_null_maxima[calibration_block] = float(
                    fold_audit[
                        "maximum_calibration_sentinel_null_probability"
                    ]
                )
                fold_convergence.append(bool(fold_audit["model_converged"]))
            probabilities = np.vstack(target_fold_probabilities).min(axis=0)
            envelope = max(fold_null_maxima.values())
            converged = all(fold_convergence)
            raw_pass_by_c[model_c] = probabilities > envelope
            convergence_by_c[model_c] = converged
            fold_records_by_c[model_c] = {
                "fold_null_maxima": fold_null_maxima,
                "envelope": envelope,
            }

        all_converged = all(convergence_by_c.values())
        dual_pass = np.logical_and.reduce(
            [raw_pass_by_c[model_c] for model_c in MODEL_C_VALUES]
        )
        if not all_converged:
            dual_pass = np.zeros(len(target_candidates), dtype=bool)
        working = target_candidates.loc[
            :, ["world_key", "modality", "feature", "assay_upstream_selected"]
        ].copy()
        working["dual_pass"] = dual_pass
        working["assay_upstream_selected_bool"] = working[
            "assay_upstream_selected"
        ].map(_as_bool)
        passing_counts = (
            working.groupby(["world_key", "modality"], sort=False)["dual_pass"]
            .sum()
            .to_dict()
        )
        selected_identities: set[tuple[str, str, str]] = set()
        for row in working.itertuples(index=False):
            assay_key = (str(row.world_key), str(row.modality))
            if (
                not bool(row.assay_upstream_selected_bool)
                and bool(row.dual_pass)
                and int(passing_counts[assay_key]) == 1
            ):
                selected_identities.add(
                    (assay_key[0], assay_key[1], str(row.feature))
                )
        for identity in selected_identities:
            selection_counts[identity] += 1
        for model_c in MODEL_C_VALUES:
            record = fold_records_by_c[model_c]
            replicate_audit_rows.append(
                {
                    "target_block": target_block,
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_replicate": replicate + 1,
                    "model_c": model_c,
                    "calibration_blocks": ";".join(calibration_blocks),
                    "calibration_null_maxima_json": json.dumps(
                        record["fold_null_maxima"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "null_probability_upper_envelope": float(
                        record["envelope"]
                    ),
                    "all_nested_models_converged": bool(
                        convergence_by_c[model_c]
                    ),
                    "dual_model_raw_pass_candidate_count": int(
                        dual_pass.sum()
                    ),
                    "unique_unresolved_reporter_count": len(
                        selected_identities
                    ),
                }
            )

    retained_reporter_identities: set[tuple[str, str, str]] = set()
    stability_rows: list[dict[str, Any]] = []
    for row in target_candidates.itertuples(index=False):
        identity = _candidate_identity(row)
        proposed = identity in proposal_reporter_identities
        count = int(selection_counts.get(identity, 0))
        frequency = (
            float(count) / float(bootstrap_replicates_run)
            if bootstrap_replicates_run
            else np.nan
        )
        stable = bool(
            proposed
            and _bootstrap_frequency_passes(
                count, bootstrap_replicates_run
            )
        )
        if stable:
            retained_reporter_identities.add(identity)
        if _as_bool(row.assay_upstream_selected):
            status = "upstream_reporter_already_resolved"
        elif stable:
            status = "retained_stable_v449_proposal"
        elif proposed:
            status = "rejected_unstable_v449_proposal"
        elif bootstrap_replicates_run:
            status = "not_a_v449_proposal"
        else:
            status = "not_a_v449_proposal_bootstrap_not_required"
        stability_rows.append(
            {
                "world_key": identity[0],
                "block": str(row.block),
                "seed": int(row.seed),
                "modality": identity[1],
                "feature": identity[2],
                "assay_upstream_selected": _as_bool(
                    row.assay_upstream_selected
                ),
                "proposed_by_target_safe_v449": proposed,
                "bootstrap_selection_count": count,
                "bootstrap_replicates": bootstrap_replicates_run,
                "bootstrap_selection_frequency": frequency,
                "bootstrap_frequency_threshold": (
                    BOOTSTRAP_FREQUENCY_THRESHOLD
                ),
                "retained_by_bootstrap_stability_guard": stable,
                "selection_status": status,
            }
        )
    stability_audit = pd.DataFrame(stability_rows, columns=STABILITY_COLUMNS)
    replicate_audit = pd.DataFrame(
        replicate_audit_rows, columns=REPLICATE_COLUMNS
    )

    baseline = _baseline_components(v447_candidates)
    baseline_pairs_by_world: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs_by_world[key].update(component["pairs"])

    stable_pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_audit_rows: list[dict[str, Any]] = []
    for row in v449_pair_audit.itertuples(index=False):
        identity = (
            str(row.world_key),
            str(row.unresolved_modality),
            str(row.candidate_reporter_feature),
        )
        proposed = _as_bool(row.newly_selected_by_v449_sparse_completion)
        count = int(selection_counts.get(identity, 0))
        frequency = (
            float(count) / float(bootstrap_replicates_run)
            if bootstrap_replicates_run
            else np.nan
        )
        reporter_stable = _bootstrap_frequency_passes(
            count, bootstrap_replicates_run
        )
        selected = bool(proposed and identity in retained_reporter_identities)
        pair = _pair_key(row.unresolved_modality, row.partner_modality)
        baseline_selected = pair in baseline_pairs_by_world.get(
            str(row.world_key), set()
        )
        values = {
            "world_key": str(row.world_key),
            "block": str(row.block),
            "seed": int(row.seed),
            "unresolved_modality": str(row.unresolved_modality),
            "candidate_reporter_feature": str(
                row.candidate_reporter_feature
            ),
            "partner_modality": str(row.partner_modality),
            "partner_reporter_feature": str(row.partner_reporter_feature),
            "candidate_assay_pair_json": str(row.candidate_assay_pair_json),
            "candidate_cross_assay_feature_pair_json": str(
                row.candidate_cross_assay_feature_pair_json
            ),
            "proposed_by_target_safe_v449": proposed,
            "reporter_bootstrap_selection_frequency": frequency,
            "reporter_passes_bootstrap_stability_guard": reporter_stable,
            "selected_by_v451_bootstrap_stability": selected,
            "selected_by_target_safe_v447": baseline_selected,
            "newly_selected_by_v451": selected and not baseline_selected,
            "retained_in_union": selected or baseline_selected,
            "merged_hyperedge_id": "",
        }
        pair_audit_rows.append(values)
        if selected:
            stable_pair_rows[str(row.world_key)].append(values)

    candidates, merged_id_by_pair = _union_candidates(
        target_keys=target_keys,
        evidence_frames=candidate_frames,
        baseline=baseline,
        stable_pair_rows=stable_pair_rows,
    )
    for row in pair_audit_rows:
        pair = _pair_key(
            row["unresolved_modality"], row["partner_modality"]
        )
        row["merged_hyperedge_id"] = merged_id_by_pair.get(
            (str(row["world_key"]), pair), ""
        )
    pair_audit = pd.DataFrame(pair_audit_rows, columns=PAIR_AUDIT_COLUMNS)

    out_dir.mkdir(parents=True)
    outputs = {
        "bootstrap_stable_candidates.tsv": candidates,
        "bootstrap_reporter_stability_audit.tsv": stability_audit,
        "bootstrap_replicate_audit.tsv": replicate_audit,
        "bootstrap_stable_pair_audit.tsv": pair_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    panel_input_hashes: dict[str, str] = {}
    for root in evidence_roots:
        resolved_root = root.expanduser().absolute().resolve(strict=True)
        if resolved_root.name in panel_input_hashes:
            raise ValueError(
                f"Duplicate sparse reporter panel name: {resolved_root.name}"
            )
        panel_input_hashes[resolved_root.name] = synchronization._sha256(
            resolved_root / "panel_evidence_manifest.json"
        )
    v447_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v447_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v447_dir / str(name))
            for name in v447_manifest.get("output_sha256", {})
        },
    }
    v449_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v449_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v449_dir / str(name))
            for name in v449_manifest.get("output_sha256", {})
        },
    }
    retained_pair_count = int(
        pair_audit["newly_selected_by_v451"].map(_as_bool).sum()
    )
    manifest = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "fixed_selection_rule": FIXED_SELECTION_RULE,
        "target_block": target_block,
        "target_world_count": len(target_keys),
        "target_world_keys": sorted(target_keys),
        "available_blocks": available_blocks,
        "calibration_blocks": calibration_blocks,
        "baseline_algorithm": "CA-BOSS v4.47 target-safe empirical-null synchronization",
        "proposal_algorithm": "CA-BOSS v4.49 target-safe sparse reporter completion",
        "model_feature_columns": list(FEATURE_COLUMNS),
        "model_c_values": list(MODEL_C_VALUES),
        "polynomial_degree": v449.POLYNOMIAL_DEGREE,
        "bootstrap_unit": "whole_training_world_within_seed_block",
        "bootstrap_sampling": "with_replacement_same_block_world_count",
        "bootstrap_replicates_configured": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_run": bootstrap_replicates_run,
        "bootstrap_base_seed": BOOTSTRAP_BASE_SEED,
        "bootstrap_target_block_seed": bootstrap_seed,
        "bootstrap_frequency_threshold": BOOTSTRAP_FREQUENCY_THRESHOLD,
        "bootstrap_threshold_comparison": "inclusive_greater_than_or_equal",
        "nominal_v449_proposal_reporter_count": len(
            proposal_reporter_identities
        ),
        "bootstrap_stable_proposal_reporter_count": len(
            retained_reporter_identities
        ),
        "newly_selected_by_v451_assay_pair_count": retained_pair_count,
        "rejected_unstable_v449_assay_pair_count": (
            len(proposal_pairs) - retained_pair_count
        ),
        "selected_union_hyperedge_count": len(candidates),
        "selected_union_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "selected_exact_assay_hypothesis_count": int(
            candidates["exact_assay_participation_claimed"].map(_as_bool).sum()
        ),
        "selected_ambiguity_hypothesis_count": int(
            candidates["assay_participation_is_set_valued"].map(_as_bool).sum()
        ),
        "maximum_candidate_assay_pair_count": (
            int(candidates["candidate_assay_pair_count"].max())
            if len(candidates)
            else 0
        ),
        "rollback_policy": {
            "v448_supervised_multivariate_additions_carried_forward": False,
            "v450_interaction_additions_carried_forward": False,
            "reason": (
                "cached_true_additions_failed_predeclared_whole_world_"
                "bootstrap_stability_and_are_not_reliable_checkpoint_gains"
            ),
        },
        "information_boundary": {
            "other_block_simulation_roles_used_for_supervised_training": True,
            "other_block_sentinel_null_roles_used_for_null_calibration": True,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "target_safe_v447_used_only_as_target_baseline_union": True,
            "target_safe_v449_used_only_to_nominate_proposals": True,
            "non_target_v449_selections_used_for_training": False,
            "known_sentinel_null_peaks_used_for_empirical_null": True,
            "modality_identity_used_as_model_feature": False,
            "candidate_feature_identity_used_as_model_feature": False,
            "paired_cells_used": False,
            "cross_assay_cell_correspondence_used": False,
            "simulator_latent_state_used": False,
            "external_biological_features_or_priors_used": False,
        },
        "target_specific_pretruth_barrier_satisfied": True,
        "formal_pretruth_barrier_satisfied": True,
        "retrospective_rule_development_on_cached_worlds": True,
        "fresh_locked_certification": False,
        "new_worlds_or_datasets_created": False,
        "candidate_pairs_may_be_expanded_as_edges": False,
        "within_assay_direct_structure_claimed": False,
        "exact_cross_assay_pair_claimed": False,
        "cross_assay_direction_claimed": False,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "source_sha256": {
            Path(__file__).name: synchronization._sha256(
                Path(__file__).resolve()
            )
        },
        "input_sha256": {
            "completion_evidence_panels": panel_input_hashes,
            "completion_evidence_worlds": evidence_input_hashes,
            "target_safe_v447_calibration": v447_input_hashes,
            "target_safe_v449_calibration": v449_input_hashes,
            "calibration_role_assignments": training_role_hashes,
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
            "stability_selection": {
                "citation": "Meinshausen and Buehlmann, JRSS B 2010",
                "doi": "10.1111/j.1467-9868.2010.00740.x",
            },
            "permutation_synchronization": (
                "Pachauri, Kondor, and Singh, NeurIPS 2013"
            ),
            "multiway_matching": "Tang and Jebara, AISTATS 2017",
            "unpaired_multi_domain_causal_representation": (
                "Sturma, Squires, Drton, and Uhler, NeurIPS 2023"
            ),
        },
    }
    manifest["manifest_payload_sha256"] = synchronization._canonical_sha256(
        manifest
    )
    (out_dir / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", type=Path, action="append", required=True
    )
    parser.add_argument("--v447-calibration-dir", type=Path, required=True)
    parser.add_argument("--v449-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        evidence_roots=args.evidence_root,
        v447_calibration_dir=args.v447_calibration_dir,
        v449_calibration_dir=args.v449_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
