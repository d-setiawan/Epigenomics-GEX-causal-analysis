#!/usr/bin/env python3
"""Dual-regularization interaction rescue for CA-BOSS v4.50.

For one target seed block, this stage learns nonlinear interactions among the
four v4.48 reporter-pair evidence signals using only truth labels from the
other cached blocks.  For each of two fixed regularization strengths, four
nested models exclude both the target block and one calibration block.  A
target pair must exceed the maximum corresponding out-of-fold null score at
both regularizations.  The target-safe v4.49 hypotheses are then unioned and
connected pairs are represented as unexpanded ambiguity hyperedges.

Only truth-blind v4.44 and v4.46 selections are excluded from the model
universe.  Non-target v4.47--v4.49 selections are deliberately ignored to
avoid indirect target-block leakage.  No modality identity, paired cells,
simulator state, external biology, direct edge, or direction is used.
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
import crossfit_sparse_reporter_completion_v4 as v449  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402


SCHEMA = "ca_boss_v4.crossfit_interaction_reporter_rescue.v1"
ALGORITHM = "CA-BOSS v4.50 nested interaction-stability reporter rescue"
FEATURE_COLUMNS = v448.FEATURE_COLUMNS
MODEL_C_VALUES = (0.02, 0.03)
MODEL_RANDOM_STATE = 0
MODEL_MAX_ITER = 4000
MODEL_TOLERANCE = 1e-4
POLYNOMIAL_DEGREE = 2
FIXED_SELECTION_RULE = (
    "for_each_target_seed_block_use_reporter_pair_evidence_remaining_after_"
    "only_truth_blind_v444_and_v446_exclusions_fit_nested_standardized_"
    "degree_two_balanced_logistic_models_on_the_four_fixed_v448_signals_at_"
    "C_0_02_and_0_03_each_model_excluding_the_target_and_one_calibration_"
    "block_score_each_target_pair_by_the_minimum_across_fold_probability_"
    "set_each_regularizations_threshold_to_the_maximum_corresponding_out_of_"
    "fold_null_probability_require_strict_exceedance_at_both_"
    "regularizations_union_with_target_safe_v449_deduplicate_and_merge_"
    "connected_assay_pairs_as_unexpanded_ambiguity_hyperedges_without_"
    "claiming_direct_structure_or_direction"
)

MODEL_AUDIT_COLUMNS = [
    "target_block",
    "calibration_block",
    "training_blocks",
    "model_c",
    "training_pair_count",
    "training_positive_pair_count",
    "training_null_pair_count",
    "calibration_pair_count",
    "calibration_positive_pair_count",
    "calibration_null_pair_count",
    "maximum_calibration_null_probability",
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

PAIR_AUDIT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "left_modality",
    "right_modality",
    "v444_assay_pair_already_represented",
    "selected_by_v446_guarded_rescue",
    "eligible_for_nested_interaction_model",
    *FEATURE_COLUMNS,
    "c_0_02_nested_probabilities_json",
    "c_0_02_minimum_probability",
    "c_0_02_null_probability_upper_envelope",
    "c_0_02_strictly_exceeds_null_envelope",
    "c_0_03_nested_probabilities_json",
    "c_0_03_minimum_probability",
    "c_0_03_null_probability_upper_envelope",
    "c_0_03_strictly_exceeds_null_envelope",
    "all_nested_models_converged",
    "passes_dual_regularization_guard",
    "selected_by_target_safe_v449",
    "newly_selected_by_v450_interaction_rescue",
    "retained_in_union",
    "merged_hyperedge_id",
]

CANDIDATE_COLUMNS = [
    "interaction_hyperedge_id",
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


def _json_floats(values: Sequence[float]) -> str:
    return json.dumps(
        [float(value) for value in values], separators=(",", ":")
    )


def _c_prefix(model_c: float) -> str:
    return f"c_{str(model_c).replace('.', '_')}"


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Reporter evidence lacks model features: {missing}")
    numeric = frame.loc[:, FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    return (
        numeric.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def _fit_fold(
    training: pd.DataFrame,
    training_labels: np.ndarray,
    calibration: pd.DataFrame,
    calibration_labels: np.ndarray,
    target: pd.DataFrame,
    *,
    model_c: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(training) != len(training_labels):
        raise ValueError("Training pairs and labels differ in length")
    if len(calibration) != len(calibration_labels):
        raise ValueError("Calibration pairs and labels differ in length")
    if len(np.unique(training_labels.astype(int))) != 2:
        raise ValueError("Nested training fold must contain both classes")
    if not (~calibration_labels).any():
        raise ValueError("Nested calibration fold lacks a null pair")

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
    iterations = int(model.n_iter_[0])
    audit = {
        "training_pair_count": len(training),
        "training_positive_pair_count": int(training_labels.sum()),
        "training_null_pair_count": int((~training_labels).sum()),
        "calibration_pair_count": len(calibration),
        "calibration_positive_pair_count": int(calibration_labels.sum()),
        "calibration_null_pair_count": int((~calibration_labels).sum()),
        "maximum_calibration_null_probability": float(
            calibration_probability[~calibration_labels].max()
        ),
        "feature_columns_json": json.dumps(
            list(FEATURE_COLUMNS), separators=(",", ":")
        ),
        "polynomial_feature_names_json": json.dumps(
            polynomial.get_feature_names_out(FEATURE_COLUMNS).tolist(),
            separators=(",", ":"),
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


def _verified_v449_calibration(
    directory: Path, *, target_block: str
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
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
    if str(manifest.get("target_block")) != str(target_block):
        raise ValueError("v4.49 calibration targets a different block")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = resolved / str(name)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if synchronization._sha256(path) != str(expected):
            raise ValueError(f"v4.49 calibration output hash mismatch: {name}")
    candidates = pd.read_csv(
        resolved / "crossfit_sparse_completion_candidates.tsv",
        sep="\t",
        keep_default_na=False,
    )
    if len(candidates) and not candidates["hypothesis_selected"].map(_as_bool).all():
        raise ValueError("v4.49 calibration contains an unselected candidate")
    return resolved, manifest, candidates


def _baseline_components(
    candidates: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    components = v449._baseline_components(candidates)
    for world_components in components.values():
        for component in world_components:
            component["source"] = "target_safe_v449"
    return components


def _union_candidates(
    *,
    target_keys: Sequence[str],
    evidence_frames: Mapping[str, pd.DataFrame],
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_pairs: Mapping[str, set[frozenset[str]]],
    score_lookup: Mapping[
        tuple[str, frozenset[str]], Mapping[str, Any]
    ],
    thresholds: Mapping[float, float],
    all_models_converged: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, frozenset[str]], str]]:
    rows: list[dict[str, Any]] = []
    merged_id_by_pair: dict[tuple[str, frozenset[str]], str] = {}
    threshold_json = json.dumps(
        {str(model_c): thresholds[model_c] for model_c in MODEL_C_VALUES},
        sort_keys=True,
        separators=(",", ":"),
    )
    for world_key in sorted(target_keys):
        frame = evidence_frames[world_key]
        block = str(frame["block"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        row_by_pair = {
            _pair_key(row.left_modality, row.right_modality): row
            for row in frame.itertuples(index=False)
        }
        sources: list[dict[str, Any]] = [
            dict(value) for value in baseline.get(world_key, [])
        ]
        for pair in sorted(
            selected_pairs.get(world_key, set()), key=_pair_sort_key
        ):
            row = row_by_pair[pair]
            features = v449._parse_feature_pair_payload(
                row.candidate_cross_assay_feature_pairs_json
            )
            score = score_lookup[(world_key, pair)]
            sources.append(
                {
                    "pairs": {pair},
                    "features": features,
                    "audit": [
                        {
                            "assay_pair": list(_pair_sort_key(pair)),
                            "minimum_probability_by_c": {
                                str(model_c): float(
                                    score["minimum_probability_by_c"][model_c]
                                )
                                for model_c in MODEL_C_VALUES
                            },
                            "threshold_by_c": {
                                str(model_c): thresholds[model_c]
                                for model_c in MODEL_C_VALUES
                            },
                            "source": "v450_nested_interaction_stability",
                        }
                    ],
                    "source": "v450_nested_interaction_stability",
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
            hyperedge_id = f"interaction_stable_shared_state_{component_index:03d}"
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
                    "interaction_hyperedge_id": hyperedge_id,
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
                        "selected_as_interaction_stable_ambiguity"
                        if len(component_pairs) != 1
                        else "selected_as_interaction_stable_point_hypothesis"
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
    relaxed_roots: Sequence[Path],
    rescue_roots: Sequence[Path],
    v449_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    relaxed_worlds = v448.v447._discover_worlds(
        relaxed_roots,
        manifest_name="evidence_manifest.json",
        schema=v448.relaxed.SCHEMA,
    )
    rescue_worlds = v448.v447._discover_worlds(
        rescue_roots,
        manifest_name="rescue_manifest.json",
        schema=v448.guarded_rescue.SCHEMA,
    )
    if set(relaxed_worlds) != set(rescue_worlds):
        raise ValueError("Relaxed evidence and v4.46 world sets differ")
    available_blocks = sorted(
        {str(manifest["block"]) for _, manifest in relaxed_worlds.values()}
    )
    target_block = str(target_block)
    if target_block not in available_blocks:
        raise ValueError(f"Unknown target block: {target_block}")
    calibration_blocks = [
        block for block in available_blocks if block != target_block
    ]
    if len(calibration_blocks) < 3:
        raise ValueError("At least three non-target blocks are required")

    v449_dir, v449_manifest, v449_candidates = _verified_v449_calibration(
        v449_calibration_dir, target_block=target_block
    )
    baseline = _baseline_components(v449_candidates)
    baseline_pairs_by_world: dict[str, set[frozenset[str]]] = defaultdict(set)
    for key, components in baseline.items():
        for component in components:
            baseline_pairs_by_world[key].update(component["pairs"])

    formal_root = formal_worlds_root.expanduser().absolute().resolve(strict=True)
    evidence_frames: dict[str, pd.DataFrame] = {}
    model_frames: dict[str, pd.DataFrame] = {}
    truth_blind_v446_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    truth_pairs_by_world: dict[str, set[frozenset[str]]] = {}
    evidence_input_hashes: dict[str, dict[str, str]] = {}
    rescue_input_hashes: dict[str, dict[str, str]] = {}
    training_truth_hashes: dict[str, str] = {}
    target_keys: list[str] = []

    for key in sorted(relaxed_worlds):
        evidence_dir, evidence_manifest = relaxed_worlds[key]
        rescue_dir, rescue_manifest = rescue_worlds[key]
        block = str(evidence_manifest["block"])
        seed = int(evidence_manifest["seed"])
        if (
            block != str(rescue_manifest["block"])
            or seed != int(rescue_manifest["seed"])
        ):
            raise ValueError(f"Upstream world identities disagree: {key}")
        evidence_path = evidence_dir / "relaxed_reporter_synchronization_pairs.tsv"
        frame = pd.read_csv(evidence_path, sep="\t", keep_default_na=False)
        frame["world_key"] = key
        frame["block"] = block
        frame["seed"] = seed
        pair_keys = [
            _pair_key(left, right)
            for left, right in zip(
                frame["left_modality"], frame["right_modality"]
            )
        ]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError(f"Duplicate assay pair in reporter evidence: {key}")
        evidence_frames[key] = frame

        rescue_path = rescue_dir / "heldout_synchronized_majority_candidates.tsv"
        rescue_candidates = pd.read_csv(
            rescue_path, sep="\t", keep_default_na=False
        )
        evidence_pair_set = set(pair_keys)
        for row in rescue_candidates.itertuples(index=False):
            if not _as_bool(row.hypothesis_selected):
                continue
            modalities = [
                value for value in str(row.modalities).split(";") if value
            ]
            if len(modalities) != 2:
                raise ValueError("v4.46 guarded candidate is not an assay pair")
            pair = frozenset(modalities)
            if pair not in evidence_pair_set:
                raise ValueError("v4.46 pair lacks relaxed reporter evidence")
            truth_blind_v446_pairs[key].add(pair)

        v444_mask = frame["v444_assay_pair_already_represented"].map(_as_bool)
        v446_mask = np.asarray(
            [pair in truth_blind_v446_pairs[key] for pair in pair_keys],
            dtype=bool,
        )
        model_frames[key] = frame.loc[
            ~(v444_mask.to_numpy(dtype=bool) | v446_mask)
        ].copy()
        evidence_input_hashes[key] = {
            "evidence_manifest.json": synchronization._sha256(
                evidence_dir / "evidence_manifest.json"
            ),
            "relaxed_reporter_synchronization_pairs.tsv": (
                synchronization._sha256(evidence_path)
            ),
        }
        rescue_input_hashes[key] = {
            "rescue_manifest.json": synchronization._sha256(
                rescue_dir / "rescue_manifest.json"
            ),
            "heldout_synchronized_majority_candidates.tsv": (
                synchronization._sha256(rescue_path)
            ),
        }
        if block == target_block:
            target_keys.append(key)
            continue
        truth_path = (
            formal_root
            / block
            / f"world_seed{seed}"
            / "scorer"
            / "truth_typed_edges.tsv"
        )
        if block == target_block:
            raise RuntimeError("Attempted to open target-block truth")
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        truth_pairs_by_world[key] = v448.v447._truth_assay_pairs(truth_path)
        training_truth_hashes[key] = synchronization._sha256(truth_path)

    if not target_keys:
        raise ValueError("Target block has no cached worlds")
    if set(map(str, v449_manifest.get("target_world_keys", []))) != set(
        target_keys
    ):
        raise ValueError("v4.49 and current evidence target worlds differ")

    target_model = pd.concat(
        [model_frames[key] for key in sorted(target_keys)], ignore_index=True
    )
    if target_model.empty:
        raise ValueError("Target block has no model-eligible assay pairs")

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
                    model_frames[key]
                    for key in sorted(model_frames)
                    if str(relaxed_worlds[key][1]["block"]) in training_blocks
                ],
                ignore_index=True,
            )
            calibration_frame = pd.concat(
                [
                    model_frames[key]
                    for key in sorted(model_frames)
                    if str(relaxed_worlds[key][1]["block"])
                    == calibration_block
                ],
                ignore_index=True,
            )
            training_labels = v448._truth_labels(
                training, truth_pairs_by_world
            )
            calibration_labels = v448._truth_labels(
                calibration_frame, truth_pairs_by_world
            )
            _, target_probability, fold_audit = _fit_fold(
                training,
                training_labels,
                calibration_frame,
                calibration_labels,
                target_model,
                model_c=model_c,
            )
            target_fold_probabilities.append(target_probability)
            fold_null_maxima.append(
                float(fold_audit["maximum_calibration_null_probability"])
            )
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
    pass_by_c = {
        model_c: probabilities_by_c[model_c] > thresholds[model_c]
        for model_c in MODEL_C_VALUES
    }
    dual_pass = np.logical_and.reduce(
        [pass_by_c[model_c] for model_c in MODEL_C_VALUES]
    )
    if not all_models_converged:
        dual_pass = np.zeros(len(target_model), dtype=bool)

    target_model = target_model.copy()
    target_model["passes_dual_regularization_guard"] = dual_pass
    score_lookup: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
    model_pairs_by_world: dict[str, set[frozenset[str]]] = defaultdict(set)
    for index, row in enumerate(target_model.itertuples(index=False)):
        pair = _pair_key(row.left_modality, row.right_modality)
        lookup_key = (str(row.world_key), pair)
        if lookup_key in score_lookup:
            raise ValueError("Duplicate target model assay pair")
        score_lookup[lookup_key] = {
            "minimum_probability_by_c": {
                model_c: float(probabilities_by_c[model_c][index])
                for model_c in MODEL_C_VALUES
            },
            "probabilities_by_c": {
                model_c: fold_probabilities_by_c[model_c][index]
                for model_c in MODEL_C_VALUES
            },
            "passes_by_c": {
                model_c: bool(pass_by_c[model_c][index])
                for model_c in MODEL_C_VALUES
            },
            "dual_pass": bool(dual_pass[index]),
        }
        if dual_pass[index]:
            model_pairs_by_world[str(row.world_key)].add(pair)

    candidates, merged_id_by_pair = _union_candidates(
        target_keys=target_keys,
        evidence_frames=evidence_frames,
        baseline=baseline,
        selected_pairs=model_pairs_by_world,
        score_lookup=score_lookup,
        thresholds=thresholds,
        all_models_converged=all_models_converged,
    )

    pair_audit_rows: list[dict[str, Any]] = []
    total_model_pairs = 0
    total_new_pairs = 0
    for key in sorted(target_keys):
        baseline_pairs = baseline_pairs_by_world.get(key, set())
        model_pairs = model_pairs_by_world.get(key, set())
        union_pairs = baseline_pairs | model_pairs
        total_model_pairs += len(model_pairs)
        total_new_pairs += len(model_pairs - baseline_pairs)
        for row in evidence_frames[key].itertuples(index=False):
            pair = _pair_key(row.left_modality, row.right_modality)
            v444_selected = _as_bool(row.v444_assay_pair_already_represented)
            v446_selected = pair in truth_blind_v446_pairs.get(key, set())
            score = score_lookup.get((key, pair))
            values: dict[str, Any] = {
                "world_key": key,
                "block": str(row.block),
                "seed": int(row.seed),
                "left_modality": str(row.left_modality),
                "right_modality": str(row.right_modality),
                "v444_assay_pair_already_represented": v444_selected,
                "selected_by_v446_guarded_rescue": v446_selected,
                "eligible_for_nested_interaction_model": not (
                    v444_selected or v446_selected
                ),
            }
            for feature in FEATURE_COLUMNS:
                value = pd.to_numeric(
                    pd.Series([getattr(row, feature)]), errors="coerce"
                ).fillna(0.0).iloc[0]
                values[feature] = float(value)
            for model_c in MODEL_C_VALUES:
                prefix = _c_prefix(model_c)
                values[f"{prefix}_nested_probabilities_json"] = (
                    _json_floats(score["probabilities_by_c"][model_c])
                    if score
                    else "[]"
                )
                values[f"{prefix}_minimum_probability"] = (
                    float(score["minimum_probability_by_c"][model_c])
                    if score
                    else np.nan
                )
                values[f"{prefix}_null_probability_upper_envelope"] = float(
                    thresholds[model_c]
                )
                values[f"{prefix}_strictly_exceeds_null_envelope"] = bool(
                    score and score["passes_by_c"][model_c]
                )
            values.update(
                {
                    "all_nested_models_converged": all_models_converged,
                    "passes_dual_regularization_guard": bool(
                        score and score["dual_pass"]
                    ),
                    "selected_by_target_safe_v449": pair in baseline_pairs,
                    "newly_selected_by_v450_interaction_rescue": (
                        pair in model_pairs and pair not in baseline_pairs
                    ),
                    "retained_in_union": pair in union_pairs,
                    "merged_hyperedge_id": merged_id_by_pair.get(
                        (key, pair), ""
                    ),
                }
            )
            pair_audit_rows.append(values)
    pair_audit = pd.DataFrame(pair_audit_rows, columns=PAIR_AUDIT_COLUMNS)

    null_summary = model_audit.loc[
        :,
        [
            "model_c",
            "calibration_block",
            "calibration_null_pair_count",
            "maximum_calibration_null_probability",
        ],
    ].copy()
    null_summary["null_probability_upper_envelope"] = null_summary[
        "model_c"
    ].map(thresholds)

    out_dir.mkdir(parents=True)
    outputs = {
        "crossfit_interaction_candidates.tsv": candidates,
        "crossfit_interaction_pair_audit.tsv": pair_audit,
        "nested_interaction_model_audit.tsv": model_audit,
        "nested_interaction_null_summary.tsv": null_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    v449_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v449_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v449_dir / str(name))
            for name in v449_manifest.get("output_sha256", {})
        },
    }
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
        "polynomial_degree": POLYNOMIAL_DEGREE,
        "polynomial_include_bias": False,
        "model_class_weight": "balanced",
        "model_solver": "lbfgs",
        "model_random_state": MODEL_RANDOM_STATE,
        "model_max_iter": MODEL_MAX_ITER,
        "model_tolerance": MODEL_TOLERANCE,
        "nested_model_count": len(model_audit),
        "all_nested_models_converged": all_models_converged,
        "null_probability_upper_envelope_by_c": {
            str(model_c): thresholds[model_c]
            for model_c in MODEL_C_VALUES
        },
        "selected_by_interaction_model_assay_pair_count": total_model_pairs,
        "newly_selected_by_v450_assay_pair_count": total_new_pairs,
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
            "other_block_simulation_truth_used_for_supervised_training": True,
            "other_block_simulation_truth_used_for_null_calibration": True,
            "target_block_scorer_truth_opened": False,
            "target_world_truth_used_for_fit_or_selection": False,
            "non_target_v447_v448_v449_selections_used_in_model_universe": False,
            "truth_blind_v444_and_v446_exclusions_only_in_model_universe": True,
            "target_safe_v449_used_only_as_target_baseline_union": True,
            "modality_identity_used_as_model_feature": False,
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
            "relaxed_evidence_worlds": evidence_input_hashes,
            "v446_guarded_rescue_worlds": rescue_input_hashes,
            "target_safe_v449_calibration": v449_input_hashes,
            "calibration_truth_typed_edges": training_truth_hashes,
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
        "--relaxed-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--rescue-root", type=Path, action="append", required=True
    )
    parser.add_argument("--v449-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        relaxed_roots=args.relaxed_root,
        rescue_roots=args.rescue_root,
        v449_calibration_dir=args.v449_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
