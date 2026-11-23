#!/usr/bin/env python3
"""Nested sparse-reporter completion for CA-BOSS v4.49.

This stage operates only on the frozen v4.49 evidence derived from the cached
worlds.  For one target seed block, it trains on reporter-role labels from the
other blocks, calibrates an out-of-fold sentinel-null envelope block by block,
and scores target candidates without opening target truth.  Two polynomial
logistic models with fixed regularization strengths must both clear their
independent null envelopes.  An unresolved assay is completed only when
exactly one of its three external candidates passes both models and exactly
one already-supported assay partner passes all three synchronization contexts.

The output is unioned with the target-safe v4.48 hypotheses.  Connected assay
pairs remain unexpanded ambiguity hyperedges; no direct or directional
cross-assay relation is claimed.  Other-block roles are disclosed supervised
calibration inputs, not model features.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crossfit_multivariate_reporter_rescue_v4 as v448  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import sparse_reporter_completion_evidence_v4 as completion  # noqa: E402


SCHEMA = "ca_boss_v4.crossfit_sparse_reporter_completion.v1"
ALGORITHM = "CA-BOSS v4.49 nested cross-fit sparse reporter completion"
MODEL_C_VALUES = (0.1, 0.3)
MODEL_RANDOM_STATE = 0
MODEL_MAX_ITER = 4000
MODEL_TOLERANCE = 1e-4
POLYNOMIAL_DEGREE = 2
FEATURE_COLUMNS = completion.FEATURE_COLUMNS
FIXED_SELECTION_RULE = (
    "for_each_target_seed_block_use_all_three_noncollider_focal_candidates_"
    "from_each_assay_fit_nested_standardized_degree_two_balanced_logistic_"
    "reporter_models_on_other_block_S_versus_N1_N2_roles_at_C_0_1_and_0_3_"
    "set_each_regularizations_threshold_to_its_maximum_out_of_fold_sentinel_"
    "null_probability_score_each_target_candidate_by_its_minimum_across_"
    "fold_probability_require_strict_exceedance_for_both_regularizations_"
    "require_exactly_one_passing_candidate_within_an_unresolved_assay_then_"
    "require_exactly_one_supported_assay_partner_globally_coassigned_in_all_"
    "three_contexts_with_all_positive_exclusion_margins_and_all_scaled_KS_"
    "ratios_at_most_one_union_with_target_safe_v448_merge_connected_pairs_"
    "as_unexpanded_ambiguity_hyperedges_and_never_claim_direct_structure_or_"
    "direction"
)


MODEL_AUDIT_COLUMNS = [
    "target_block",
    "calibration_block",
    "training_blocks",
    "model_c",
    "training_candidate_count",
    "training_reporter_count",
    "training_sentinel_null_count",
    "calibration_candidate_count",
    "calibration_reporter_count",
    "calibration_sentinel_null_count",
    "maximum_calibration_sentinel_null_probability",
    "feature_columns_json",
    "polynomial_feature_names_json",
    "standard_scaler_mean_json",
    "standard_scaler_scale_json",
    "logistic_coefficients_json",
    "logistic_intercept",
    "iterations",
    "model_converged",
    "class_weight",
    "solver",
    "max_iter",
    "tolerance",
]

REPORTER_AUDIT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "modality",
    "feature",
    "assay_upstream_selected",
    *FEATURE_COLUMNS,
    "c_0_1_nested_probabilities_json",
    "c_0_1_minimum_probability",
    "c_0_1_null_probability_upper_envelope",
    "c_0_1_strictly_exceeds_null_envelope",
    "c_0_3_nested_probabilities_json",
    "c_0_3_minimum_probability",
    "c_0_3_null_probability_upper_envelope",
    "c_0_3_strictly_exceeds_null_envelope",
    "all_nested_models_converged",
    "passes_dual_regularization_guard",
    "passing_candidate_count_within_assay",
    "uniquely_selected_reporter_within_assay",
    "selection_status",
]

PAIR_AUDIT_COLUMNS = [
    *completion.PAIR_COLUMNS[:-2],
    "candidate_reporter_passes_dual_regularization_guard",
    "reporter_uniquely_selected_within_assay",
    "all_pair_synchronization_guards_passed",
    "strict_partner_candidate_count_for_selected_reporter",
    "unique_partner_guard_passed",
    "selected_by_sparse_reporter_completion",
    "already_represented_by_target_safe_v448",
    "newly_selected_by_v449_sparse_completion",
    "retained_in_union",
    "merged_hyperedge_id",
]

CANDIDATE_COLUMNS = [
    "completion_hyperedge_id",
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
    "calibration_thresholds_json",
    "all_nested_models_converged",
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


def _world_key(block: str, seed: int) -> str:
    return f"{block}__world_seed{int(seed)}"


def _json_floats(values: Sequence[float]) -> str:
    return json.dumps(
        [float(value) for value in values], separators=(",", ":")
    )


def _c_prefix(model_c: float) -> str:
    return f"c_{str(model_c).replace('.', '_')}"


def _verified_payload(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = str(value.get(field, ""))
    unsigned = dict(value)
    unsigned.pop(field, None)
    if observed != synchronization._canonical_sha256(unsigned):
        raise ValueError(f"Payload hash mismatch: {path}")
    return value


def _verified_completion_world(
    directory: Path,
) -> tuple[Path, dict[str, Any]]:
    resolved = synchronization._forbid_truth_path(directory)
    manifest_path = resolved / "reporter_completion_evidence_manifest.json"
    manifest = _verified_payload(manifest_path, "manifest_payload_sha256")
    if manifest.get("schema_version") != completion.SCHEMA:
        raise ValueError("Unexpected sparse reporter evidence schema")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("scorer_truth_used") is not False:
        raise ValueError("Sparse reporter evidence opened scorer truth")
    if boundary.get("unresolved_assay_reporter_selected") is not False:
        raise ValueError("Evidence selected an unresolved reporter pre-calibration")
    if manifest.get("formal_pretruth_barrier_satisfied") is not True:
        raise ValueError("Sparse reporter evidence lacks a pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("Sparse reporter evidence created new data")
    if manifest.get("hypothesis_selected") is not False:
        raise ValueError("Sparse reporter evidence selected a hypothesis")
    if list(manifest.get("model_feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("Sparse reporter evidence feature set changed")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"Sparse reporter output hash mismatch: {name}")
    expected_key = _world_key(manifest["block"], int(manifest["seed"]))
    if resolved.name != expected_key or manifest.get("world_key") != expected_key:
        raise ValueError("Sparse reporter evidence world identity mismatch")
    return resolved, manifest


def _discover_completion_worlds(
    roots: Sequence[Path],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    worlds: dict[str, tuple[Path, dict[str, Any]]] = {}
    for root in roots:
        resolved = synchronization._forbid_truth_path(root)
        panel_path = resolved / "panel_evidence_manifest.json"
        panel = _verified_payload(panel_path, "manifest_payload_sha256")
        if panel.get("schema_version") != (
            "ca_boss_v4.sparse_reporter_completion_evidence_panel.v1"
        ):
            raise ValueError("Unexpected sparse reporter panel schema")
        if panel.get("scorer_truth_used") is not False:
            raise ValueError("Sparse reporter panel opened scorer truth")
        if panel.get("new_worlds_or_datasets_created") is not False:
            raise ValueError("Sparse reporter panel created new data")
        expected_keys = set(map(str, panel.get("world_keys", [])))
        observed_dirs = {
            path.name: path for path in resolved.iterdir() if path.is_dir()
        }
        if set(observed_dirs) != expected_keys:
            raise ValueError("Sparse reporter panel world set mismatch")
        if int(panel.get("world_count", -1)) != len(expected_keys):
            raise ValueError("Sparse reporter panel world count mismatch")
        for key in sorted(expected_keys):
            directory, manifest = _verified_completion_world(observed_dirs[key])
            expected_hash = panel.get("world_manifest_file_sha256", {}).get(key)
            if synchronization._sha256(
                directory / "reporter_completion_evidence_manifest.json"
            ) != str(expected_hash):
                raise ValueError(f"Sparse reporter panel binding mismatch: {key}")
            if key in worlds:
                raise ValueError(f"Duplicate sparse reporter world: {key}")
            worlds[key] = (directory, manifest)
    if not worlds:
        raise ValueError("No sparse reporter evidence worlds were discovered")
    return worlds


def _verified_v448_calibration(
    directory: Path, *, target_block: str
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest_path = resolved / "calibration_manifest.json"
    manifest = _verified_payload(manifest_path, "manifest_payload_sha256")
    if manifest.get("schema_version") != v448.SCHEMA:
        raise ValueError("Unexpected v4.48 calibration schema")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("target_block_scorer_truth_opened") is not False:
        raise ValueError("v4.48 opened target-block truth")
    if boundary.get("target_world_truth_used_for_fit_or_selection") is not False:
        raise ValueError("v4.48 used target-world truth")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.48 lacks a target-specific pre-truth barrier")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.48 created new data")
    if str(manifest.get("target_block")) != str(target_block):
        raise ValueError("v4.48 calibration targets a different block")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"v4.48 calibration output hash mismatch: {name}")
    candidates = pd.read_csv(
        resolved / "crossfit_multivariate_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if len(candidates) and not candidates["hypothesis_selected"].map(_as_bool).all():
        raise ValueError("v4.48 calibration contains an unselected candidate")
    return resolved, manifest, candidates


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Reporter candidates lack model features: {missing}")
    numeric = frame.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    return (
        numeric.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _reporter_labels(
    frame: pd.DataFrame, roles_by_world: Mapping[str, Mapping[str, str]]
) -> np.ndarray:
    labels: list[bool] = []
    for row in frame.itertuples(index=False):
        world_key = str(row.world_key)
        if world_key not in roles_by_world:
            raise ValueError(f"Roles were not loaded for training world {world_key}")
        role = roles_by_world[world_key].get(str(row.feature))
        if role not in {"S", "N1", "N2"}:
            raise ValueError(
                f"External candidate has unexpected or missing role: {role}"
            )
        labels.append(role == "S")
    return np.asarray(labels, dtype=bool)


def _fit_fold(
    training: pd.DataFrame,
    training_labels: np.ndarray,
    calibration: pd.DataFrame,
    calibration_labels: np.ndarray,
    target: pd.DataFrame,
    *,
    model_c: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one deterministic nested polynomial model."""

    if len(training) != len(training_labels):
        raise ValueError("Training candidates and labels differ in length")
    if len(calibration) != len(calibration_labels):
        raise ValueError("Calibration candidates and labels differ in length")
    if len(np.unique(training_labels.astype(int))) != 2:
        raise ValueError("Nested training fold must contain both reporter roles")
    if not (~calibration_labels).any():
        raise ValueError("Nested calibration fold lacks sentinel-null candidates")

    scaler = StandardScaler()
    polynomial = PolynomialFeatures(
        degree=POLYNOMIAL_DEGREE, include_bias=False
    )
    x_training = polynomial.fit_transform(
        scaler.fit_transform(_model_matrix(training))
    )
    model = LogisticRegression(
        C=float(model_c),
        class_weight="balanced",
        fit_intercept=True,
        max_iter=MODEL_MAX_ITER,
        random_state=MODEL_RANDOM_STATE,
        solver="lbfgs",
        tol=MODEL_TOLERANCE,
    )
    model.fit(x_training, training_labels.astype(int))
    calibration_probability = model.predict_proba(
        polynomial.transform(scaler.transform(_model_matrix(calibration)))
    )[:, 1]
    target_probability = model.predict_proba(
        polynomial.transform(scaler.transform(_model_matrix(target)))
    )[:, 1]
    polynomial_names = polynomial.get_feature_names_out(FEATURE_COLUMNS)
    iterations = int(model.n_iter_[0])
    audit = {
        "training_candidate_count": len(training),
        "training_reporter_count": int(training_labels.sum()),
        "training_sentinel_null_count": int((~training_labels).sum()),
        "calibration_candidate_count": len(calibration),
        "calibration_reporter_count": int(calibration_labels.sum()),
        "calibration_sentinel_null_count": int((~calibration_labels).sum()),
        "maximum_calibration_sentinel_null_probability": float(
            calibration_probability[~calibration_labels].max()
        ),
        "feature_columns_json": json.dumps(
            list(FEATURE_COLUMNS), separators=(",", ":")
        ),
        "polynomial_feature_names_json": json.dumps(
            polynomial_names.tolist(), separators=(",", ":")
        ),
        "standard_scaler_mean_json": _json_floats(scaler.mean_),
        "standard_scaler_scale_json": _json_floats(scaler.scale_),
        "logistic_coefficients_json": _json_floats(model.coef_[0]),
        "logistic_intercept": float(model.intercept_[0]),
        "iterations": iterations,
        "model_converged": iterations < MODEL_MAX_ITER,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": MODEL_MAX_ITER,
        "tolerance": MODEL_TOLERANCE,
    }
    return calibration_probability, target_probability, audit


def _parse_pair_payload(value: Any) -> set[frozenset[str]]:
    parsed = json.loads(str(value))
    pairs = {_pair_key(left, right) for left, right in parsed}
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Candidate assay-pair payload is malformed")
    return pairs


def _parse_feature_pair_payload(value: Any) -> set[tuple[str, str]]:
    parsed = json.loads(str(value))
    result = {
        tuple(sorted((str(left), str(right)))) for left, right in parsed
    }
    if any(len(pair) != 2 for pair in result):
        raise ValueError("Candidate feature-pair payload is malformed")
    return result


def _baseline_components(
    candidates: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates.itertuples(index=False):
        pairs = _parse_pair_payload(row.candidate_assay_pairs_json)
        if len(pairs) != int(row.candidate_assay_pair_count):
            raise ValueError("v4.48 candidate assay-pair count mismatch")
        features = _parse_feature_pair_payload(
            row.candidate_cross_assay_feature_pairs_json
        )
        if len(features) != int(row.candidate_cross_assay_feature_pair_count):
            raise ValueError("v4.48 candidate feature-pair count mismatch")
        components[str(row.world_key)].append(
            {
                "pairs": pairs,
                "features": features,
                "audit": json.loads(str(row.candidate_pair_selection_audit_json)),
                "source": "target_safe_v448",
            }
        )
    return components


def _union_candidates(
    *,
    target_keys: Sequence[str],
    evidence_frames: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[float, float],
    all_models_converged: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, frozenset[str]], str]]:
    rows: list[dict[str, Any]] = []
    merged_id_by_pair: dict[tuple[str, frozenset[str]], str] = {}
    threshold_json = json.dumps(
        {str(model_c): float(thresholds[model_c]) for model_c in MODEL_C_VALUES},
        sort_keys=True,
        separators=(",", ":"),
    )
    for world_key in sorted(target_keys):
        frame = evidence_frames[world_key]
        block = str(frame["block"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        sources: list[dict[str, Any]] = [
            dict(value) for value in baseline.get(world_key, [])
        ]
        for selected in selected_pair_rows.get(world_key, []):
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
                            "source": "v449_sparse_reporter_completion",
                        }
                    ],
                    "source": "v449_sparse_reporter_completion",
                }
            )
        all_pairs = set().union(
            *(source["pairs"] for source in sources)
        ) if sources else set()
        for component_index, component_pairs in enumerate(
            v448.v447._connected_pair_components(all_pairs), start=1
        ):
            hyperedge_id = f"sparse_completion_shared_state_{component_index:03d}"
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
                audits.extend(source["audit"])
                source_names.add(str(source["source"]))
            pair_payload = [
                list(_pair_sort_key(pair))
                for pair in sorted(component_pairs, key=_pair_sort_key)
            ]
            for pair in component_pairs:
                merged_id_by_pair[(world_key, pair)] = hyperedge_id
            rows.append(
                {
                    "completion_hyperedge_id": hyperedge_id,
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
                    "calibration_thresholds_json": threshold_json,
                    "all_nested_models_converged": all_models_converged,
                    "assay_participation_is_set_valued": (
                        len(component_pairs) != 1
                    ),
                    "exact_assay_participation_claimed": (
                        len(component_pairs) == 1
                    ),
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_sparse_completion_ambiguity"
                        if len(component_pairs) != 1
                        else "selected_as_sparse_completion_point_hypothesis"
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
    v448_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Calibrate on other blocks and select sparse completions in one target."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    worlds = _discover_completion_worlds(evidence_roots)
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

    v448_dir, v448_manifest, v448_candidates = _verified_v448_calibration(
        v448_calibration_dir, target_block=target_block
    )
    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)

    candidate_frames: dict[str, pd.DataFrame] = {}
    pair_frames: dict[str, pd.DataFrame] = {}
    roles_by_world: dict[str, dict[str, str]] = {}
    evidence_input_hashes: dict[str, dict[str, str]] = {}
    training_role_hashes: dict[str, str] = {}
    target_keys: list[str] = []
    for key in sorted(worlds):
        directory, manifest = worlds[key]
        block = str(manifest["block"])
        seed = int(manifest["seed"])
        candidate_path = directory / "external_reporter_candidates.tsv"
        pair_path = directory / "unresolved_reporter_pair_evidence.tsv"
        candidates = pd.read_csv(
            candidate_path, sep="\t", keep_default_na=False
        )
        pairs = pd.read_csv(pair_path, sep="\t", keep_default_na=False)
        if len(candidates) != int(manifest["external_candidate_count"]):
            raise ValueError(f"Candidate count mismatch: {key}")
        if len(pairs) != int(
            manifest["unresolved_reporter_pair_evidence_count"]
        ):
            raise ValueError(f"Pair-evidence count mismatch: {key}")
        if len(candidates) and (
            ~candidates["world_key"].astype(str).eq(key)
            | ~candidates["block"].astype(str).eq(block)
            | ~pd.to_numeric(candidates["seed"], errors="raise").eq(seed)
        ).any():
            raise ValueError(f"Candidate identity mismatch: {key}")
        if len(pairs) and (
            ~pairs["world_key"].astype(str).eq(key)
            | ~pairs["block"].astype(str).eq(block)
            | ~pd.to_numeric(pairs["seed"], errors="raise").eq(seed)
        ).any():
            raise ValueError(f"Pair-evidence identity mismatch: {key}")
        candidate_key_count = candidates.loc[
            :, ["modality", "feature"]
        ].drop_duplicates().shape[0]
        if candidate_key_count != len(candidates):
            raise ValueError(f"Duplicate reporter candidate: {key}")
        candidate_frames[key] = candidates
        pair_frames[key] = pairs
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
                pair_path
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
        if block == target_block:
            raise RuntimeError("Attempted to open target-block roles")
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
    if set(map(str, v448_manifest.get("target_world_keys", []))) != set(
        target_keys
    ):
        raise ValueError("v4.48 and sparse completion target worlds differ")
    if len(v448_candidates) and not v448_candidates["world_key"].astype(
        str
    ).isin(target_keys).all():
        raise ValueError("v4.48 candidates include a non-target world")

    target_candidates = pd.concat(
        [candidate_frames[key] for key in sorted(target_keys)],
        ignore_index=True,
    )
    if target_candidates.empty:
        raise ValueError("Target block has no reporter candidates")
    candidate_identity = list(
        zip(
            target_candidates["world_key"].astype(str),
            target_candidates["modality"].astype(str),
            target_candidates["feature"].astype(str),
        )
    )
    if len(set(candidate_identity)) != len(candidate_identity):
        raise ValueError("Target reporter candidate identities are not unique")

    probabilities_by_c: dict[float, np.ndarray] = {}
    fold_probabilities_by_c: dict[float, np.ndarray] = {}
    thresholds: dict[float, float] = {}
    model_audit_rows: list[dict[str, Any]] = []
    for model_c in MODEL_C_VALUES:
        target_fold_probabilities: list[np.ndarray] = []
        fold_null_maxima: list[float] = []
        for calibration_block in calibration_blocks:
            training_blocks = [
                block
                for block in calibration_blocks
                if block != calibration_block
            ]
            training = pd.concat(
                [
                    candidate_frames[key]
                    for key in sorted(candidate_frames)
                    if str(worlds[key][1]["block"]) in training_blocks
                ],
                ignore_index=True,
            )
            calibration = pd.concat(
                [
                    candidate_frames[key]
                    for key in sorted(candidate_frames)
                    if str(worlds[key][1]["block"]) == calibration_block
                ],
                ignore_index=True,
            )
            training_labels = _reporter_labels(training, roles_by_world)
            calibration_labels = _reporter_labels(
                calibration, roles_by_world
            )
            _, target_probability, fold_audit = _fit_fold(
                training,
                training_labels,
                calibration,
                calibration_labels,
                target_candidates,
                model_c=model_c,
            )
            fold_null_maxima.append(
                float(
                    fold_audit[
                        "maximum_calibration_sentinel_null_probability"
                    ]
                )
            )
            target_fold_probabilities.append(target_probability)
            model_audit_rows.append(
                {
                    "target_block": target_block,
                    "calibration_block": calibration_block,
                    "training_blocks": ";".join(training_blocks),
                    "model_c": model_c,
                    **fold_audit,
                }
            )
        stacked = np.vstack(target_fold_probabilities).T
        fold_probabilities_by_c[model_c] = stacked
        probabilities_by_c[model_c] = stacked.min(axis=1)
        thresholds[model_c] = max(fold_null_maxima)

    model_audit = pd.DataFrame(model_audit_rows, columns=MODEL_AUDIT_COLUMNS)
    all_models_converged = bool(model_audit["model_converged"].astype(bool).all())
    raw_pass_by_c = {
        model_c: probabilities_by_c[model_c] > thresholds[model_c]
        for model_c in MODEL_C_VALUES
    }
    dual_pass = np.logical_and.reduce(
        [raw_pass_by_c[model_c] for model_c in MODEL_C_VALUES]
    )
    if not all_models_converged:
        dual_pass = np.zeros(len(target_candidates), dtype=bool)

    working = target_candidates.copy()
    working["passes_dual_regularization_guard"] = dual_pass
    working["assay_upstream_selected_bool"] = working[
        "assay_upstream_selected"
    ].map(_as_bool)
    passing_counts = (
        working.groupby(["world_key", "modality"], sort=False)[
            "passes_dual_regularization_guard"
        ]
        .sum()
        .to_dict()
    )
    unique_reporter_mask = np.asarray(
        [
            (
                not bool(row.assay_upstream_selected_bool)
                and bool(row.passes_dual_regularization_guard)
                and int(passing_counts[(str(row.world_key), str(row.modality))])
                == 1
            )
            for row in working.itertuples(index=False)
        ],
        dtype=bool,
    )
    working["uniquely_selected_reporter_within_assay"] = unique_reporter_mask
    selected_reporter_by_assay = {
        (str(row.world_key), str(row.modality)): str(row.feature)
        for row in working.loc[unique_reporter_mask].itertuples(index=False)
    }

    reporter_audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(working.itertuples(index=False)):
        assay_key = (str(row.world_key), str(row.modality))
        count = int(passing_counts[assay_key])
        upstream_selected = bool(row.assay_upstream_selected_bool)
        selected = bool(row.uniquely_selected_reporter_within_assay)
        if upstream_selected:
            status = "upstream_reporter_already_resolved"
        elif not all_models_converged:
            status = "abstained_model_nonconvergence"
        elif count == 0:
            status = "abstained_no_candidate_above_both_null_envelopes"
        elif count > 1:
            status = "abstained_multiple_candidates_above_null_envelopes"
        elif selected:
            status = "selected_unique_sparse_reporter"
        else:
            status = "not_selected_within_unique_reporter_assay"
        values: dict[str, Any] = {
            "world_key": str(row.world_key),
            "block": str(row.block),
            "seed": int(row.seed),
            "modality": str(row.modality),
            "feature": str(row.feature),
            "assay_upstream_selected": upstream_selected,
        }
        for feature in FEATURE_COLUMNS:
            value = getattr(row, feature)
            values[feature] = (
                _as_bool(value) if feature == "component_stable" else float(value)
            )
        for model_c in MODEL_C_VALUES:
            prefix = _c_prefix(model_c)
            values[f"{prefix}_nested_probabilities_json"] = _json_floats(
                fold_probabilities_by_c[model_c][index]
            )
            values[f"{prefix}_minimum_probability"] = float(
                probabilities_by_c[model_c][index]
            )
            values[f"{prefix}_null_probability_upper_envelope"] = float(
                thresholds[model_c]
            )
            values[f"{prefix}_strictly_exceeds_null_envelope"] = bool(
                raw_pass_by_c[model_c][index]
            )
        values.update(
            {
                "all_nested_models_converged": all_models_converged,
                "passes_dual_regularization_guard": bool(dual_pass[index]),
                "passing_candidate_count_within_assay": count,
                "uniquely_selected_reporter_within_assay": selected,
                "selection_status": status,
            }
        )
        reporter_audit_rows.append(values)
    reporter_audit = pd.DataFrame(
        reporter_audit_rows, columns=REPORTER_AUDIT_COLUMNS
    )

    baseline = _baseline_components(v448_candidates)
    baseline_pairs_by_world: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs_by_world[key].update(component["pairs"])

    raw_pair_records: list[dict[str, Any]] = []
    strict_partner_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for key in sorted(target_keys):
        frame = pair_frames[key]
        for row in frame.itertuples(index=False):
            reporter_key = (
                str(row.world_key),
                str(row.unresolved_modality),
                str(row.candidate_reporter_feature),
            )
            unique_reporter = (
                selected_reporter_by_assay.get(reporter_key[:2])
                == reporter_key[2]
            )
            pair_guard = bool(
                int(row.global_coassignment_context_count) == 3
                and float(row.minimum_pair_exclusion_relative_margin) > 0.0
                and float(row.maximum_reporter_pair_scaled_ks_ratio) <= 1.0
            )
            if unique_reporter and pair_guard:
                strict_partner_counts[reporter_key] += 1
            raw_pair_records.append(
                {
                    "row": row,
                    "reporter_key": reporter_key,
                    "unique_reporter": unique_reporter,
                    "pair_guard": pair_guard,
                }
            )

    reporter_pass_lookup = {
        (str(row.world_key), str(row.modality), str(row.feature)): bool(
            row.passes_dual_regularization_guard
        )
        for row in reporter_audit.itertuples(index=False)
    }
    pair_audit_rows: list[dict[str, Any]] = []
    selected_pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_pair_records:
        row = record["row"]
        reporter_key = record["reporter_key"]
        unique_partner = bool(
            record["unique_reporter"]
            and strict_partner_counts[reporter_key] == 1
        )
        selected = bool(
            all_models_converged and record["pair_guard"] and unique_partner
        )
        pair = _pair_key(row.unresolved_modality, row.partner_modality)
        already = pair in baseline_pairs_by_world.get(str(row.world_key), set())
        values = {
            column: getattr(row, column)
            for column in completion.PAIR_COLUMNS[:-2]
        }
        values.update(
            {
                "candidate_reporter_passes_dual_regularization_guard": (
                    reporter_pass_lookup.get(reporter_key, False)
                ),
                "reporter_uniquely_selected_within_assay": bool(
                    record["unique_reporter"]
                ),
                "all_pair_synchronization_guards_passed": bool(
                    record["pair_guard"]
                ),
                "strict_partner_candidate_count_for_selected_reporter": int(
                    strict_partner_counts[reporter_key]
                ),
                "unique_partner_guard_passed": unique_partner,
                "selected_by_sparse_reporter_completion": selected,
                "already_represented_by_target_safe_v448": already,
                "newly_selected_by_v449_sparse_completion": (
                    selected and not already
                ),
                "retained_in_union": selected or already,
                "merged_hyperedge_id": "",
            }
        )
        pair_audit_rows.append(values)
        if selected:
            selected_pair_rows[str(row.world_key)].append(values)

    candidates, merged_id_by_pair = _union_candidates(
        target_keys=target_keys,
        evidence_frames=candidate_frames,
        baseline=baseline,
        selected_pair_rows=selected_pair_rows,
        thresholds=thresholds,
        all_models_converged=all_models_converged,
    )
    for row in pair_audit_rows:
        pair = _pair_key(
            row["unresolved_modality"], row["partner_modality"]
        )
        row["merged_hyperedge_id"] = merged_id_by_pair.get(
            (str(row["world_key"]), pair), ""
        )
    pair_audit = pd.DataFrame(pair_audit_rows, columns=PAIR_AUDIT_COLUMNS)

    null_summary = model_audit.loc[
        :,
        [
            "model_c",
            "calibration_block",
            "calibration_sentinel_null_count",
            "maximum_calibration_sentinel_null_probability",
        ],
    ].copy()
    null_summary["null_probability_upper_envelope"] = null_summary[
        "model_c"
    ].map(thresholds)

    out_dir.mkdir(parents=True)
    outputs = {
        "crossfit_sparse_completion_candidates.tsv": candidates,
        "nested_reporter_candidate_audit.tsv": reporter_audit,
        "nested_reporter_model_audit.tsv": model_audit,
        "nested_reporter_null_summary.tsv": null_summary,
        "reporter_partner_pair_audit.tsv": pair_audit,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    v448_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v448_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v448_dir / str(name))
            for name in v448_manifest.get("output_sha256", {})
        },
    }
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
    newly_selected = int(
        pair_audit["newly_selected_by_v449_sparse_completion"]
        .astype(bool)
        .sum()
    )
    selected_reporters = int(
        reporter_audit["uniquely_selected_reporter_within_assay"]
        .astype(bool)
        .sum()
    )
    ambiguous_assays = int(
        reporter_audit.loc[
            ~reporter_audit["assay_upstream_selected"].astype(bool),
            ["world_key", "modality", "passing_candidate_count_within_assay"],
        ]
        .drop_duplicates()
        ["passing_candidate_count_within_assay"]
        .gt(1)
        .sum()
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
        "model_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_imputation": "nonfinite_or_missing_to_zero",
        "model_c_values": list(MODEL_C_VALUES),
        "model_class_weight": "balanced",
        "model_solver": "lbfgs",
        "model_random_state": MODEL_RANDOM_STATE,
        "model_max_iter": MODEL_MAX_ITER,
        "model_tolerance": MODEL_TOLERANCE,
        "polynomial_degree": POLYNOMIAL_DEGREE,
        "polynomial_include_bias": False,
        "nested_model_count": len(model_audit),
        "all_nested_models_converged": all_models_converged,
        "null_probability_upper_envelope_by_c": {
            str(model_c): thresholds[model_c]
            for model_c in MODEL_C_VALUES
        },
        "selected_unique_sparse_reporter_count": selected_reporters,
        "ambiguous_multi_reporter_assay_abstention_count": ambiguous_assays,
        "selected_by_sparse_completion_assay_pair_count": int(
            pair_audit["selected_by_sparse_reporter_completion"]
            .astype(bool)
            .sum()
        ),
        "newly_selected_by_v449_assay_pair_count": newly_selected,
        "selected_union_hyperedge_count": len(candidates),
        "selected_union_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "selected_exact_assay_hypothesis_count": int(
            candidates["exact_assay_participation_claimed"].astype(bool).sum()
        ),
        "selected_ambiguity_hypothesis_count": int(
            candidates["assay_participation_is_set_valued"].astype(bool).sum()
        ),
        "maximum_candidate_assay_pair_count": (
            int(candidates["candidate_assay_pair_count"].max())
            if len(candidates)
            else 0
        ),
        "information_boundary": {
            "other_block_simulation_roles_used_for_supervised_training": True,
            "other_block_sentinel_null_roles_used_for_null_calibration": True,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "target_safe_v448_used_only_as_target_baseline_union": True,
            "non_target_v448_selections_used_for_training": False,
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
            "target_safe_v448_calibration": v448_input_hashes,
            "calibration_role_assignments": training_role_hashes,
        },
        "output_sha256": {
            name: synchronization._sha256(out_dir / name) for name in outputs
        },
        "published_method_inspiration": {
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
    parser.add_argument("--v448-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        evidence_roots=args.evidence_root,
        v448_calibration_dir=args.v448_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
