#!/usr/bin/env python3
"""Nested block-cross-fit multivariate rescue for CA-BOSS v4.48.

For one target simulation block, this stage learns a four-signal reporter
evidence score from the other cached blocks.  Each target score is the minimum
probability from four models.  Model ``j`` is trained without both the target
block and calibration block ``j``; its null envelope is evaluated only on
calibration block ``j``.  The final threshold is the maximum out-of-fold null
score.  This gives a conservative, block-level empirical-null guard.

Only truth-blind v4.44 and v4.46 selections are removed from the model
universe.  In particular, non-target v4.47 selections are not used because
their calibration could depend indirectly on the current target block.  The
target block's own v4.47 output is safe to use as the baseline union because
that output was calibrated without target-block truth.

The target scorer partition is never opened.  Other-block simulation truth is
an explicit retrospective calibration input, not an observable feature.  No
modality identity, paired cells, simulator latent state, or external biology
is used.  Candidate assay-pair alternatives remain unexpanded and no direct
or directional cross-assay edge is claimed.
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
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import crossfit_empirical_null_reporter_synchronization_v4 as v447  # noqa: E402
import heldout_synchronized_majority_rescue_v4 as guarded_rescue  # noqa: E402
import multi_assay_permutation_synchronization_v4 as synchronization  # noqa: E402
import relaxed_reporter_synchronization_evidence_v4 as relaxed  # noqa: E402


SCHEMA = "ca_boss_v4.crossfit_multivariate_reporter_rescue.v1"
ALGORITHM = "CA-BOSS v4.48 nested block-cross-fit multivariate reporter rescue"
FEATURE_COLUMNS = (
    "statistical_context_support_count",
    "global_reporter_coassignment_context_count",
    "maximum_reporter_pair_scaled_ks_ratio",
    "mean_coassigned_pair_exclusion_relative_margin",
)
EXPECTED_COEFFICIENT_SIGNS = (1, 1, -1, 1)
MODEL_C = 1.0
MODEL_RANDOM_STATE = 0
MODEL_MAX_ITER = 2000
MODEL_TOLERANCE = 1e-4
SCORE_COLUMN = "minimum_nested_block_crossfit_probability"
THRESHOLD_COLUMN = "out_of_fold_null_probability_upper_envelope"
FIXED_SELECTION_RULE = (
    "for_each_target_seed_block_fit_four_standardized_balanced_ridge_"
    "logistic_models_each_excluding_the_target_and_one_calibration_block_"
    "using_only_statistical_context_count_global_reporter_coassignment_"
    "count_maximum_reporter_pair_scaled_ks_ratio_and_mean_coassigned_pair_"
    "exclusion_margin_take_each_target_pairs_minimum_probability_set_the_"
    "threshold_to_the_maximum_corresponding_out_of_fold_calibration_null_"
    "probability_require_all_coefficient_sign_guards_select_strictly_above_"
    "the_threshold_union_with_the_target_safe_v447_pairs_deduplicate_and_"
    "merge_pairs_sharing_an_assay_into_unexpanded_ambiguity_hyperedges_"
    "without_claiming_direct_structure_or_direction"
)
CANDIDATE_COLUMNS = [
    "multivariate_hyperedge_id",
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
    "calibration_score_name",
    "out_of_fold_null_probability_upper_envelope",
    "all_coefficient_sign_guards_passed",
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
PAIR_AUDIT_COLUMNS = [
    "world_key",
    "block",
    "seed",
    "left_modality",
    "right_modality",
    "v444_assay_pair_already_represented",
    "selected_by_v446_guarded_rescue",
    "eligible_for_nested_multivariate_model",
    *FEATURE_COLUMNS,
    "nested_model_probabilities_json",
    SCORE_COLUMN,
    THRESHOLD_COLUMN,
    "all_coefficient_sign_guards_passed",
    "strictly_exceeds_out_of_fold_null_envelope",
    "selected_by_target_safe_v447",
    "newly_selected_by_v448_multivariate_rescue",
    "retained_in_union",
    "merged_hyperedge_id",
]
MODEL_AUDIT_COLUMNS = [
    "target_block",
    "calibration_block",
    "training_blocks",
    "training_pair_count",
    "training_positive_pair_count",
    "training_null_pair_count",
    "calibration_pair_count",
    "calibration_positive_pair_count",
    "calibration_null_pair_count",
    "maximum_calibration_null_probability",
    "feature_columns_json",
    "standard_scaler_mean_json",
    "standard_scaler_scale_json",
    "standardized_logistic_coefficients_json",
    "logistic_intercept",
    "expected_coefficient_signs_json",
    "coefficient_sign_guard_passed",
    "model_c",
    "class_weight",
    "solver",
    "max_iter",
    "tolerance",
]


def _as_bool(value: Any) -> bool:
    return synchronization._as_bool(value)


def _pair_key(left: str, right: str) -> frozenset[str]:
    return v447._pair_key(left, right)


def _pair_sort_key(pair: frozenset[str]) -> tuple[str, str]:
    return v447._pair_sort_key(pair)


def _json_floats(values: Sequence[float]) -> str:
    return json.dumps(
        [float(value) for value in values], separators=(",", ":")
    )


def _verified_v447_calibration(
    directory: Path, *, target_block: str
) -> tuple[Path, dict[str, Any], pd.DataFrame]:
    """Verify one target-safe v4.47 calibration and return its pair audit."""

    resolved = directory.expanduser().absolute().resolve(strict=True)
    manifest_path = resolved / "calibration_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != v447.SCHEMA:
        raise ValueError("Unexpected v4.47 calibration schema")
    observed = str(manifest.get("manifest_payload_sha256", ""))
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if observed != synchronization._canonical_sha256(unsigned):
        raise ValueError("v4.47 calibration manifest payload hash mismatch")
    boundary = manifest.get("information_boundary", {})
    if boundary.get("target_block_scorer_truth_opened") is not False:
        raise ValueError("v4.47 calibration opened target-block truth")
    if boundary.get("target_world_truth_used_for_fit_or_selection") is not False:
        raise ValueError("v4.47 calibration used target-world truth")
    if (
        boundary.get("other_block_simulation_truth_used_for_null_calibration")
        is not True
    ):
        raise ValueError("v4.47 calibration did not disclose training truth")
    if manifest.get("target_specific_pretruth_barrier_satisfied") is not True:
        raise ValueError("v4.47 target-specific truth barrier was not satisfied")
    if manifest.get("new_worlds_or_datasets_created") is not False:
        raise ValueError("v4.47 calibration created new data")
    if str(manifest.get("target_block")) != str(target_block):
        raise ValueError("v4.47 calibration targets a different block")
    for name, expected in manifest.get("output_sha256", {}).items():
        output = resolved / str(name)
        if output.is_symlink() or not output.is_file():
            raise FileNotFoundError(output)
        if synchronization._sha256(output) != str(expected):
            raise ValueError(f"v4.47 calibration output hash mismatch: {name}")
    audit_path = resolved / "crossfit_empirical_null_pair_audit.tsv"
    audit = pd.read_csv(audit_path, sep="\t", keep_default_na=False)
    if len(audit) and not audit["block"].astype(str).eq(str(target_block)).all():
        raise ValueError("v4.47 pair audit contains a non-target block")
    return resolved, manifest, audit


def _model_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Apply the frozen four-feature zero-imputation rule."""

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


def _coefficient_sign_guard(coefficients: Sequence[float]) -> bool:
    if len(coefficients) != len(EXPECTED_COEFFICIENT_SIGNS):
        return False
    return all(
        float(coefficient) * expected_sign > 0.0
        for coefficient, expected_sign in zip(
            coefficients, EXPECTED_COEFFICIENT_SIGNS
        )
    )


def _truth_labels(
    frame: pd.DataFrame, truth_pairs_by_world: Mapping[str, set[frozenset[str]]]
) -> np.ndarray:
    labels: list[bool] = []
    for row in frame.itertuples(index=False):
        key = str(row.world_key)
        if key not in truth_pairs_by_world:
            raise ValueError(f"Truth was not loaded for training world {key}")
        labels.append(
            _pair_key(row.left_modality, row.right_modality)
            in truth_pairs_by_world[key]
        )
    return np.asarray(labels, dtype=bool)


def _fit_fold(
    training: pd.DataFrame,
    training_labels: np.ndarray,
    calibration: pd.DataFrame,
    calibration_labels: np.ndarray,
    target: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one deterministic nested fold and score calibration and target."""

    if len(training) != len(training_labels):
        raise ValueError("Training rows and labels differ in length")
    if len(calibration) != len(calibration_labels):
        raise ValueError("Calibration rows and labels differ in length")
    if len(np.unique(training_labels.astype(int))) != 2:
        raise ValueError("A nested training fold must contain both classes")
    if not (~calibration_labels).any():
        raise ValueError("A nested calibration fold lacks a null pair")

    scaler = StandardScaler()
    x_training = scaler.fit_transform(_model_matrix(training))
    model = LogisticRegression(
        C=MODEL_C,
        class_weight="balanced",
        fit_intercept=True,
        max_iter=MODEL_MAX_ITER,
        random_state=MODEL_RANDOM_STATE,
        solver="lbfgs",
        tol=MODEL_TOLERANCE,
    )
    model.fit(x_training, training_labels.astype(int))
    coefficients = model.coef_[0].astype(float)
    calibration_probability = model.predict_proba(
        scaler.transform(_model_matrix(calibration))
    )[:, 1]
    target_probability = model.predict_proba(
        scaler.transform(_model_matrix(target))
    )[:, 1]
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
        "standard_scaler_mean_json": _json_floats(scaler.mean_),
        "standard_scaler_scale_json": _json_floats(scaler.scale_),
        "standardized_logistic_coefficients_json": _json_floats(coefficients),
        "logistic_intercept": float(model.intercept_[0]),
        "expected_coefficient_signs_json": json.dumps(
            list(EXPECTED_COEFFICIENT_SIGNS), separators=(",", ":")
        ),
        "coefficient_sign_guard_passed": _coefficient_sign_guard(coefficients),
        "model_c": MODEL_C,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": MODEL_MAX_ITER,
        "tolerance": MODEL_TOLERANCE,
    }
    return calibration_probability, target_probability, audit


def calibrate_target_block(
    *,
    relaxed_roots: Sequence[Path],
    rescue_roots: Sequence[Path],
    v447_calibration_dir: Path,
    formal_worlds_root: Path,
    target_block: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Cross-fit a four-feature score and select one target block."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {out_dir}")
    relaxed_worlds = v447._discover_worlds(
        relaxed_roots,
        manifest_name="evidence_manifest.json",
        schema=relaxed.SCHEMA,
    )
    rescue_worlds = v447._discover_worlds(
        rescue_roots,
        manifest_name="rescue_manifest.json",
        schema=guarded_rescue.SCHEMA,
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

    v447_dir, v447_manifest, v447_audit = _verified_v447_calibration(
        v447_calibration_dir, target_block=target_block
    )
    target_safe_v447_pairs: dict[str, set[frozenset[str]]] = defaultdict(set)
    for row in v447_audit.itertuples(index=False):
        if _as_bool(row.retained_in_union):
            target_safe_v447_pairs[str(row.world_key)].add(
                _pair_key(row.left_modality, row.right_modality)
            )

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

        rescue_path = (
            rescue_dir / "heldout_synchronized_majority_candidates.tsv"
        )
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
        if target_block in truth_path.parts:
            raise RuntimeError("Attempted to open target-block truth")
        if truth_path.is_symlink() or not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        truth_pairs_by_world[key] = v447._truth_assay_pairs(truth_path)
        training_truth_hashes[key] = synchronization._sha256(truth_path)

    if not target_keys:
        raise ValueError("Target block has no cached worlds")
    if set(v447_manifest.get("target_world_keys", [])) != set(target_keys):
        raise ValueError("v4.47 target worlds differ from current evidence")

    # Bind the target-safe v4.47 output to the exact evidence and rescue inputs.
    v447_inputs = v447_manifest.get("input_sha256", {})
    for key in sorted(relaxed_worlds):
        if (
            v447_inputs.get("relaxed_evidence_worlds", {}).get(key)
            != evidence_input_hashes[key]
        ):
            raise ValueError(f"v4.47 evidence binding mismatch: {key}")
        if (
            v447_inputs.get("v446_guarded_rescue_worlds", {}).get(key)
            != rescue_input_hashes[key]
        ):
            raise ValueError(f"v4.47 rescue binding mismatch: {key}")

    target_model = pd.concat(
        [model_frames[key] for key in sorted(target_keys)], ignore_index=True
    )
    if target_model.empty:
        raise ValueError("Target block has no model-eligible assay pairs")

    target_fold_probabilities: list[np.ndarray] = []
    model_audit_rows: list[dict[str, Any]] = []
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
        calibration = pd.concat(
            [
                model_frames[key]
                for key in sorted(model_frames)
                if str(relaxed_worlds[key][1]["block"])
                == calibration_block
            ],
            ignore_index=True,
        )
        training_labels = _truth_labels(training, truth_pairs_by_world)
        calibration_labels = _truth_labels(calibration, truth_pairs_by_world)
        _, target_probability, fold_audit = _fit_fold(
            training,
            training_labels,
            calibration,
            calibration_labels,
            target_model,
        )
        target_fold_probabilities.append(target_probability)
        model_audit_rows.append(
            {
                "target_block": target_block,
                "calibration_block": calibration_block,
                "training_blocks": ";".join(training_blocks),
                **fold_audit,
            }
        )

    model_audit = pd.DataFrame(model_audit_rows, columns=MODEL_AUDIT_COLUMNS)
    all_sign_guards_pass = bool(
        model_audit["coefficient_sign_guard_passed"].astype(bool).all()
    )
    threshold = float(
        model_audit["maximum_calibration_null_probability"].max()
    )
    target_probabilities = np.vstack(target_fold_probabilities).T
    target_minimum_probability = target_probabilities.min(axis=1)
    model_selected_mask = (
        target_minimum_probability > threshold
        if all_sign_guards_pass
        else np.zeros(len(target_model), dtype=bool)
    )
    target_model = target_model.copy()
    target_model["nested_model_probabilities_json"] = [
        _json_floats(values) for values in target_probabilities
    ]
    target_model[SCORE_COLUMN] = target_minimum_probability
    target_model["selected_by_v448"] = model_selected_mask
    target_score_by_key_pair = {
        (
            str(row.world_key),
            _pair_key(row.left_modality, row.right_modality),
        ): (
            str(row.nested_model_probabilities_json),
            float(getattr(row, SCORE_COLUMN)),
            bool(row.selected_by_v448),
        )
        for row in target_model.itertuples(index=False)
    }

    candidate_rows: list[dict[str, Any]] = []
    pair_audit_rows: list[dict[str, Any]] = []
    total_new_pairs = 0
    total_model_selected_pairs = 0
    for key in sorted(target_keys):
        frame = evidence_frames[key]
        row_by_pair = {
            _pair_key(row.left_modality, row.right_modality): row
            for row in frame.itertuples(index=False)
        }
        model_pairs = {
            pair
            for (world_key, pair), (_, _, selected) in target_score_by_key_pair.items()
            if world_key == key and selected
        }
        baseline_pairs = target_safe_v447_pairs.get(key, set())
        union_pairs = model_pairs | baseline_pairs
        total_model_selected_pairs += len(model_pairs)
        total_new_pairs += len(model_pairs - baseline_pairs)
        component_id_by_pair: dict[frozenset[str], str] = {}
        for component_index, component in enumerate(
            v447._connected_pair_components(union_pairs), start=1
        ):
            hyperedge_id = f"crossfit_multivariate_shared_state_{component_index:03d}"
            modalities = sorted(set().union(*(set(pair) for pair in component)))
            pair_payload = [
                list(_pair_sort_key(pair))
                for pair in sorted(component, key=_pair_sort_key)
            ]
            feature_pairs: set[tuple[str, str]] = set()
            selection_audit: list[dict[str, Any]] = []
            sources: set[str] = set()
            for pair in sorted(component, key=_pair_sort_key):
                component_id_by_pair[pair] = hyperedge_id
                row = row_by_pair.get(pair)
                if row is None:
                    raise ValueError("Selected pair lacks relaxed evidence")
                parsed = json.loads(
                    str(row.candidate_cross_assay_feature_pairs_json)
                )
                feature_pairs.update(
                    tuple(sorted((str(left), str(right))))
                    for left, right in parsed
                )
                pair_sources: list[str] = []
                if pair in baseline_pairs:
                    pair_sources.append("target_safe_v447")
                    sources.add("target_safe_v447")
                if pair in model_pairs:
                    pair_sources.append("v448_nested_multivariate")
                    sources.add("v448_nested_multivariate")
                score_payload = target_score_by_key_pair.get((key, pair))
                selection_audit.append(
                    {
                        "assay_pair": list(_pair_sort_key(pair)),
                        "minimum_probability": (
                            score_payload[1] if score_payload else None
                        ),
                        "threshold": threshold,
                        "sources": pair_sources,
                    }
                )
            candidate_rows.append(
                {
                    "multivariate_hyperedge_id": hyperedge_id,
                    "world_key": key,
                    "block": str(frame["block"].iloc[0]),
                    "seed": int(frame["seed"].iloc[0]),
                    "modalities": ";".join(modalities),
                    "n_modalities": len(modalities),
                    "candidate_assay_pairs_json": json.dumps(
                        pair_payload, separators=(",", ":")
                    ),
                    "candidate_assay_pair_count": len(component),
                    "candidate_cross_assay_feature_pairs_json": json.dumps(
                        sorted(feature_pairs), separators=(",", ":")
                    ),
                    "candidate_cross_assay_feature_pair_count": len(feature_pairs),
                    "candidate_pair_selection_audit_json": json.dumps(
                        selection_audit,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "selection_sources": ";".join(sorted(sources)),
                    "calibration_score_name": SCORE_COLUMN,
                    THRESHOLD_COLUMN: threshold,
                    "all_coefficient_sign_guards_passed": all_sign_guards_pass,
                    "assay_participation_is_set_valued": len(component) != 1,
                    "exact_assay_participation_claimed": len(component) == 1,
                    "hypothesis_selected": True,
                    "selection_status": (
                        "selected_as_crossfit_multivariate_ambiguity"
                        if len(component) != 1
                        else "selected_as_crossfit_multivariate_point_hypothesis"
                    ),
                    "hypothesis_type": "cross_assay_shared_state_equivalence",
                    "within_assay_direct_structure_claimed": False,
                    "exact_cross_assay_pair_claimed": False,
                    "cross_assay_direction_claimed": False,
                    "candidate_pairs_may_be_expanded_as_edges": False,
                    "fixed_selection_rule": FIXED_SELECTION_RULE,
                }
            )

        for row in frame.itertuples(index=False):
            pair = _pair_key(row.left_modality, row.right_modality)
            v444 = _as_bool(row.v444_assay_pair_already_represented)
            v446 = pair in truth_blind_v446_pairs.get(key, set())
            score_payload = target_score_by_key_pair.get((key, pair))
            minimum_probability = score_payload[1] if score_payload else np.nan
            selected_by_model = bool(score_payload and score_payload[2])
            selected_by_v447 = pair in baseline_pairs
            values = {
                feature: float(
                    pd.to_numeric(
                        pd.Series([getattr(row, feature)]), errors="coerce"
                    ).fillna(0.0).iloc[0]
                )
                for feature in FEATURE_COLUMNS
            }
            pair_audit_rows.append(
                {
                    "world_key": key,
                    "block": str(row.block),
                    "seed": int(row.seed),
                    "left_modality": str(row.left_modality),
                    "right_modality": str(row.right_modality),
                    "v444_assay_pair_already_represented": v444,
                    "selected_by_v446_guarded_rescue": v446,
                    "eligible_for_nested_multivariate_model": not (v444 or v446),
                    **values,
                    "nested_model_probabilities_json": (
                        score_payload[0] if score_payload else "[]"
                    ),
                    SCORE_COLUMN: minimum_probability,
                    THRESHOLD_COLUMN: threshold,
                    "all_coefficient_sign_guards_passed": all_sign_guards_pass,
                    "strictly_exceeds_out_of_fold_null_envelope": selected_by_model,
                    "selected_by_target_safe_v447": selected_by_v447,
                    "newly_selected_by_v448_multivariate_rescue": (
                        selected_by_model and not selected_by_v447
                    ),
                    "retained_in_union": pair in union_pairs,
                    "merged_hyperedge_id": component_id_by_pair.get(pair, ""),
                }
            )

    candidates = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    pair_audit = pd.DataFrame(pair_audit_rows, columns=PAIR_AUDIT_COLUMNS)
    null_summary = model_audit.loc[
        :,
        [
            "calibration_block",
            "calibration_null_pair_count",
            "maximum_calibration_null_probability",
        ],
    ].copy()
    null_summary[THRESHOLD_COLUMN] = threshold

    out_dir.mkdir(parents=True)
    outputs = {
        "crossfit_multivariate_candidates.tsv": candidates,
        "crossfit_multivariate_pair_audit.tsv": pair_audit,
        "nested_model_audit.tsv": model_audit,
        "nested_calibration_null_summary.tsv": null_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(out_dir / name, sep="\t", index=False)

    v447_input_hashes = {
        "calibration_manifest.json": synchronization._sha256(
            v447_dir / "calibration_manifest.json"
        ),
        **{
            name: synchronization._sha256(v447_dir / name)
            for name in v447_manifest.get("output_sha256", {})
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
        "nested_model_count": len(model_audit),
        "model_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_imputation": "nonfinite_or_missing_to_zero",
        "model_c": MODEL_C,
        "model_class_weight": "balanced",
        "model_solver": "lbfgs",
        "model_random_state": MODEL_RANDOM_STATE,
        "model_max_iter": MODEL_MAX_ITER,
        "model_tolerance": MODEL_TOLERANCE,
        "expected_standardized_coefficient_signs": list(
            EXPECTED_COEFFICIENT_SIGNS
        ),
        "all_coefficient_sign_guards_passed": all_sign_guards_pass,
        "calibration_score_name": SCORE_COLUMN,
        THRESHOLD_COLUMN: threshold,
        "selected_union_hyperedge_count": len(candidates),
        "selected_union_assay_pair_count": int(
            candidates["candidate_assay_pair_count"].sum()
        ),
        "selected_by_multivariate_model_assay_pair_count": (
            total_model_selected_pairs
        ),
        "newly_selected_by_v448_assay_pair_count": total_new_pairs,
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
            "non_target_v447_selections_used_in_model_universe": False,
            "truth_blind_v444_and_v446_exclusions_only_in_model_universe": True,
            "target_safe_v447_used_only_as_target_baseline_union": True,
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
            Path(__file__).name: synchronization._sha256(Path(__file__).resolve())
        },
        "input_sha256": {
            "relaxed_evidence_worlds": evidence_input_hashes,
            "v446_guarded_rescue_worlds": rescue_input_hashes,
            "target_safe_v447_calibration": v447_input_hashes,
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
    parser.add_argument("--v447-calibration-dir", type=Path, required=True)
    parser.add_argument("--formal-worlds-root", type=Path, required=True)
    parser.add_argument("--target-block", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = calibrate_target_block(
        relaxed_roots=args.relaxed_root,
        rescue_roots=args.rescue_root,
        v447_calibration_dir=args.v447_calibration_dir,
        formal_worlds_root=args.formal_worlds_root,
        target_block=args.target_block,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
