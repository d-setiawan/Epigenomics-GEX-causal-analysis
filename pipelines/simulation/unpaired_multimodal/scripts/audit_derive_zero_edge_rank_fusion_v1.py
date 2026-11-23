#!/usr/bin/env python3
"""Reproduce the frozen zero-edge rank-fusion model from retired TRAIN only.

This is a development audit, not a runtime dependency.  It consumes the
already-frozen Phase-B candidate table for retired TRAIN330/360/370/380,
reconstructs the directed-edge design, reruns block-held-out model selection,
and asserts exact agreement with the constants shipped in
``crossfit_hierarchical_collider_v4``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "ca_boss_v4.zero_edge_rank_fusion_derivation.v1"
DERIVATION_CONTRACT_SCHEMA_VERSION = (
    "ca_boss_v4.zero_edge_rank_fusion_derivation_contract.v1"
)
ALLOWED_BLOCKS = ("TRAIN330", "TRAIN360", "TRAIN370", "TRAIN380")
MODEL_VERSION = "retired_train_330_380_sparse_l1_v1"
CONTEXTS = ("direct", "reverse", "outer")
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
EXACT_ASSAY_TOLERANCE = 1
RANDOM_STATE = 19
NONZERO_TOLERANCE = 1e-14
COEFFICIENT_DECIMAL_PLACES = 9
EXPECTED_CANDIDATE_SHA256 = (
    "2b4ff9c5f5b86d6a3ff65db2d539f6e1a3962e700115494bc8e31035f55cf256"
)
EXPECTED_PHASE_A_MANIFEST_SHA256 = (
    "4e13c3fd92086cd76cf0eadcf445b0e2e24bb716957857549409cdf7a5305b58"
)
EXPECTED_PHASE_B_MANIFEST_SHA256 = (
    "71ac33cce950aa26194e35a45ebd51fac383fb0fa7c3ba659129031835b2d7da"
)
EXPECTED_PAIR_MATRIX_SHA256 = (
    "f71af77c57b3a26161cdcc2d18c3f913cb50045d94af2b55812258f4c445be1f"
)
EXPECTED_CANDIDATE_ROWS = 34_560
EXPECTED_WORLDS = 96
EXPECTED_ASSAYS = 576
EXPECTED_DIRECTED_EDGE_ROWS = 17_280
EXPECTED_POSITIVE_DIRECTED_ARMS = 1_152
FORBIDDEN_INPUT_PATH_TOKENS = (
    "development340",
    "locked450",
    "train400",
    "train_400",
)

BASE_FEATURE_SCHEMA = (
    "n_loading_arm_views",
    "outer_positive_partial_state_count",
    "minimum_outer_partial_state_fit",
    "minimum_outer_conditional_wald_view_support",
    "minimum_outer_conditional_wald_above_null_95pct",
    "minimum_outer_reference_conditional_wald",
    "outer_feature_maxT_confirmation_count",
    "two_parent_positive_state_wald_consistency",
    "crossfit_joint_gain_over_baseline",
    "crossfit_drop_parent1_gain",
    "crossfit_drop_parent2_gain",
    "crossfit_min_drop_one_gain",
    "crossfit_joint_gain_fold_min",
    "laplace_mean_parent_coefficient_min",
    "laplace_fold_parent_coefficient_min",
    "laplace_min_parent_positive_posterior",
    "laplace_fold_min_parent_positive_posterior",
    "laplace_both_positive_fold_support",
    "laplace_both_practical_fold_support",
    "center_detection_rate",
    "minimum_parent_detection_rate",
    "maximum_parent_detection_rate",
    "minimum_arm_residual_correlation",
    "outer_absolute_residual_correlation",
    "residual_star_separation",
    "maximum_arm_empirical_positive_p",
    "outer_empirical_null_compatibility_p",
    "fold_minimum_arm_residual_correlation",
    "fold_median_arm_residual_correlation",
    "fold_maximum_outer_absolute_correlation",
    "crossfit_joint_gain_over_baseline_reverse_center_margin",
    "crossfit_min_drop_one_gain_reverse_center_margin",
    "laplace_min_parent_positive_posterior_reverse_center_margin",
    "residual_star_separation_reverse_center_margin",
)
PAIR_FEATURE_SCHEMA = tuple(
    f"{context}__{feature}"
    for context in CONTEXTS
    for feature in BASE_FEATURE_SCHEMA
)
IDENTITY_COLUMNS = (
    "block",
    "seed",
    "source_assay",
    "center",
    "outer1",
    "outer2",
)
EXACT_STAR_COLUMN = "truth_exact_star"

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    EVALUATION_ROOT
    / "outputs"
    / "ca_boss_v4"
    / "retired_crossfit_hierarchical_collider_audit_v1"
    / "phase_b_labeled_crossfit_star_candidates.tsv"
)
DEFAULT_OUTPUT = (
    EVALUATION_ROOT
    / "outputs"
    / "ca_boss_v4"
    / "development_only_zero_edge_rank_fusion_derivation_v1"
    / "manifest.json"
)


FloatMatrix = NDArray[np.float64]
BoolVector = NDArray[np.bool_]


@dataclass(frozen=True)
class PairMatrix:
    values: FloatMatrix
    labels: BoolVector
    metadata: pd.DataFrame
    feature_schema: tuple[str, ...]


@dataclass(frozen=True)
class FittedPipeline:
    imputer: SimpleImputer
    scaler: StandardScaler
    model: LogisticRegression


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(EVALUATION_ROOT.parent.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_input_binding(candidate_path: Path) -> dict[str, Any]:
    lowered = str(candidate_path.resolve()).lower()
    forbidden = [
        token for token in FORBIDDEN_INPUT_PATH_TOKENS if token in lowered
    ]
    if forbidden:
        raise ValueError(f"Forbidden input-path tokens: {forbidden}")
    if candidate_path.name != "phase_b_labeled_crossfit_star_candidates.tsv":
        raise ValueError("Expected the frozen retired Phase-B candidate table")
    root = candidate_path.parent
    manifest_path = root / "manifest.json"
    phase_a_path = root / "phase_a_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    if tuple(phase_a.get("blocks", ())) != ALLOWED_BLOCKS:
        raise ValueError("Retired input does not contain exactly the four allowed blocks")
    if phase_a.get("phase") != "observed_only_before_truth":
        raise ValueError("Retired Phase-A barrier is missing")
    if phase_a.get("truth_or_scorer_opened") is not False:
        raise ValueError("Retired Phase A reports scorer access")
    if phase_a.get("dev_or_locked_opened") is not False:
        raise ValueError("Retired Phase A reports DEV/LOCKED access")
    phase_a_sha = sha256_file(phase_a_path)
    if phase_a_sha != EXPECTED_PHASE_A_MANIFEST_SHA256:
        raise ValueError("Retired Phase-A manifest content hash drifted")
    if manifest.get("phase_a_manifest_sha256") != phase_a_sha:
        raise ValueError("Phase-A manifest hash does not match Phase B")
    candidate_sha = sha256_file(candidate_path)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("Retired candidate content hash drifted from the freeze")
    expected_candidate_sha = manifest.get("output_sha256", {}).get(
        candidate_path.name
    )
    if candidate_sha != expected_candidate_sha:
        raise ValueError("Retired labeled candidate content hash drifted")
    phase_b_sha = sha256_file(manifest_path)
    if phase_b_sha != EXPECTED_PHASE_B_MANIFEST_SHA256:
        raise ValueError("Retired Phase-B manifest content hash drifted")
    return {
        "candidate_table": {
            "path": _relative(candidate_path),
            "sha256": candidate_sha,
        },
        "phase_a_manifest": {
            "path": _relative(phase_a_path),
            "sha256": phase_a_sha,
        },
        "phase_b_manifest": {
            "path": _relative(manifest_path),
            "sha256": phase_b_sha,
        },
        "phase_a_source_sha256": phase_a.get("source_sha256"),
        "phase_a_verified_before_label_access": True,
    }


def _validate_candidates(frame: pd.DataFrame) -> None:
    required = set(IDENTITY_COLUMNS) | set(BASE_FEATURE_SCHEMA) | {
        EXACT_STAR_COLUMN
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if tuple(sorted(frame["block"].astype(str).unique())) != ALLOWED_BLOCKS:
        raise ValueError("Candidate rows do not contain exactly the allowed blocks")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Candidate star identities are not unique")
    grouped = frame.groupby(["block", "seed", "source_assay"], sort=True)
    if len(grouped) != 576:
        raise ValueError("Expected 576 retired block/world/assay groups")
    sizes = grouped.size()
    if not sizes.eq(60).all():
        raise ValueError("Every assay must contain all 60 directed stars")
    exact_counts = grouped[EXACT_STAR_COLUMN].sum()
    if not exact_counts.eq(1).all():
        raise ValueError("Every assay must contain exactly one labeled star")
    worlds = frame[["block", "seed"]].drop_duplicates()
    per_block = worlds.groupby("block").size()
    if not per_block.reindex(ALLOWED_BLOCKS).eq(24).all():
        raise ValueError("Expected 24 worlds in every retired block")


def _raw_score(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["crossfit_joint_gain_over_baseline"]
        + frame["crossfit_min_drop_one_gain"]
        + 0.02 * frame["laplace_min_parent_positive_posterior"]
        + 0.25 * frame["residual_star_separation"]
    )


def build_pair_matrix(candidates: pd.DataFrame) -> PairMatrix:
    """Map every assay's 60 stars to all 30 possible directed edges."""

    _validate_candidates(candidates)
    frame = candidates.copy()
    frame["_raw_score"] = _raw_score(frame)
    values: list[FloatMatrix] = []
    labels: list[bool] = []
    metadata: list[dict[str, Any]] = []
    group_keys = ["block", "seed", "source_assay"]
    for (block, seed, assay), group in frame.groupby(group_keys, sort=True):
        ordered = group.sort_values(
            ["_raw_score", "center", "outer1", "outer2"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        nodes = sorted(
            set(ordered["center"].astype(str))
            | set(ordered["outer1"].astype(str))
            | set(ordered["outer2"].astype(str))
        )
        if len(nodes) != 6:
            raise ValueError("Each retired assay must contain exactly six features")
        numeric = ordered.loc[:, BASE_FEATURE_SCHEMA].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        top: dict[str, dict[tuple[str, str], int]] = {
            context: {} for context in CONTEXTS
        }
        for row_index, row in enumerate(ordered.itertuples(index=False)):
            center = str(row.center)
            outer1 = str(row.outer1)
            outer2 = str(row.outer2)
            top["direct"].setdefault((outer1, center), row_index)
            top["direct"].setdefault((outer2, center), row_index)
            top["reverse"].setdefault((center, outer1), row_index)
            top["reverse"].setdefault((center, outer2), row_index)
            top["outer"].setdefault((outer1, outer2), row_index)
            top["outer"].setdefault((outer2, outer1), row_index)
        exact = ordered.loc[ordered[EXACT_STAR_COLUMN].astype(bool)].iloc[0]
        exact_center = str(exact.center)
        exact_outers = {str(exact.outer1), str(exact.outer2)}
        for parent in nodes:
            for child in nodes:
                if parent == child:
                    continue
                key = (parent, child)
                try:
                    context_rows = [top[context][key] for context in CONTEXTS]
                except KeyError as error:
                    raise ValueError(
                        f"Incomplete directional context for {block}/{seed}/{assay}"
                    ) from error
                values.append(np.concatenate([numeric[index] for index in context_rows]))
                labels.append(child == exact_center and parent in exact_outers)
                metadata.append(
                    {
                        "block": str(block),
                        "seed": int(seed),
                        "source_assay": str(assay),
                        "parent": parent,
                        "child": child,
                        "exact_center": exact_center,
                        "exact_outer1": min(exact_outers),
                        "exact_outer2": max(exact_outers),
                    }
                )
    matrix = np.vstack(values).astype(np.float64, copy=False)
    label_vector = np.asarray(labels, dtype=np.bool_)
    metadata_frame = pd.DataFrame(metadata)
    if matrix.shape != (EXPECTED_DIRECTED_EDGE_ROWS, 102):
        raise ValueError(f"Unexpected pair matrix shape: {matrix.shape}")
    if int(label_vector.sum()) != EXPECTED_POSITIVE_DIRECTED_ARMS:
        raise ValueError("Expected exactly two positive directed arms per assay")
    return PairMatrix(
        values=matrix,
        labels=label_vector,
        metadata=metadata_frame,
        feature_schema=PAIR_FEATURE_SCHEMA,
    )


def pair_matrix_sha256(matrix: PairMatrix) -> str:
    values = matrix.values.copy()
    values[np.isnan(values)] = np.nan
    digest = hashlib.sha256()
    digest.update(canonical_json(list(matrix.feature_schema)).encode("utf-8"))
    digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    digest.update(matrix.labels.astype(np.uint8).tobytes(order="C"))
    digest.update(
        matrix.metadata.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )
    return digest.hexdigest()


def fit_pipeline(
    matrix: PairMatrix,
    row_mask: NDArray[np.bool_],
    *,
    c_value: float,
) -> FittedPipeline:
    imputer = SimpleImputer(
        missing_values=np.nan,
        strategy="median",
        fill_value=None,
        copy=True,
        add_indicator=False,
        keep_empty_features=False,
    )
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    transformed = scaler.fit_transform(
        imputer.fit_transform(matrix.values[row_mask])
    )
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        dual=False,
        fit_intercept=True,
        intercept_scaling=1.0,
        l1_ratio=None,
        random_state=RANDOM_STATE,
        max_iter=5000,
        n_jobs=None,
        penalty="l1",
        solver="liblinear",
        tol=1e-4,
        verbose=0,
        warm_start=False,
    ).fit(transformed, matrix.labels[row_mask].astype(np.int8))
    return FittedPipeline(imputer=imputer, scaler=scaler, model=model)


def predict_probability(
    fitted: FittedPipeline,
    values: FloatMatrix,
) -> NDArray[np.float64]:
    transformed = fitted.scaler.transform(fitted.imputer.transform(values))
    return fitted.model.predict_proba(transformed)[:, 1].astype(np.float64)


def _score_assays(
    metadata: pd.DataFrame,
    probabilities: NDArray[np.float64],
) -> pd.DataFrame:
    scored = metadata.copy()
    scored["probability"] = probabilities
    rows: list[dict[str, Any]] = []
    keys = ["block", "seed", "source_assay"]
    for (block, seed, assay), group in scored.groupby(keys, sort=True):
        centers: list[tuple[float, str, str, str]] = []
        for child, incoming in group.groupby("child", sort=True):
            top_two = incoming.sort_values(
                ["probability", "parent"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[:2]
            outer1, outer2 = sorted(top_two["parent"].astype(str))
            centers.append(
                (
                    float(top_two["probability"].sum()),
                    str(child),
                    outer1,
                    outer2,
                )
            )
        centers.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))
        _, center, outer1, outer2 = centers[0]
        expected = group.iloc[0]
        rows.append(
            {
                "block": str(block),
                "seed": int(seed),
                "source_assay": str(assay),
                "exact": bool(
                    center == str(expected.exact_center)
                    and outer1 == str(expected.exact_outer1)
                    and outer2 == str(expected.exact_outer2)
                ),
            }
        )
    return pd.DataFrame(rows)


def _full_fit_nonzero_count(matrix: PairMatrix, c_value: float) -> int:
    fitted = fit_pipeline(
        matrix,
        np.ones(len(matrix.labels), dtype=np.bool_),
        c_value=c_value,
    )
    return int(np.count_nonzero(np.abs(fitted.model.coef_[0]) > NONZERO_TOLERANCE))


def evaluate_grid(matrix: PairMatrix) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    blocks = matrix.metadata["block"].astype(str).to_numpy()
    for c_value in C_GRID:
        held_scores: list[pd.DataFrame] = []
        block_metrics: list[dict[str, Any]] = []
        for held_block in ALLOWED_BLOCKS:
            training_mask = blocks != held_block
            held_mask = ~training_mask
            fitted = fit_pipeline(matrix, training_mask, c_value=c_value)
            scored = _score_assays(
                matrix.metadata.loc[held_mask].reset_index(drop=True),
                predict_probability(fitted, matrix.values[held_mask]),
            )
            held_scores.append(scored)
            perfect_worlds = int(
                scored.groupby(["block", "seed"], sort=True)["exact"].all().sum()
            )
            block_metrics.append(
                {
                    "held_block": held_block,
                    "exact_assays": int(scored["exact"].sum()),
                    "total_assays": int(len(scored)),
                    "all_six_exact_worlds": perfect_worlds,
                    "total_worlds": int(scored[["block", "seed"]].drop_duplicates().shape[0]),
                    "fit_nonzero_coefficients": int(
                        np.count_nonzero(
                            np.abs(fitted.model.coef_[0]) > NONZERO_TOLERANCE
                        )
                    ),
                }
            )
        aggregate = pd.concat(held_scores, ignore_index=True)
        results.append(
            {
                "C": c_value,
                "block_held_out": block_metrics,
                "exact_assays": int(aggregate["exact"].sum()),
                "total_assays": int(len(aggregate)),
                "all_six_exact_worlds": int(
                    aggregate.groupby(["block", "seed"], sort=True)["exact"]
                    .all()
                    .sum()
                ),
                "total_worlds": int(
                    aggregate[["block", "seed"]].drop_duplicates().shape[0]
                ),
                "full_retired_refit_nonzero_coefficients": (
                    _full_fit_nonzero_count(matrix, c_value)
                ),
            }
        )
    return results


def select_grid_candidate(grid: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best_exact = max(int(row["exact_assays"]) for row in grid)
    accuracy_eligible = [
        row
        for row in grid
        if int(row["exact_assays"]) >= best_exact - EXACT_ASSAY_TOLERANCE
    ]
    best_perfect_worlds = max(
        int(row["all_six_exact_worlds"]) for row in accuracy_eligible
    )
    world_eligible = [
        row
        for row in accuracy_eligible
        if int(row["all_six_exact_worlds"]) == best_perfect_worlds
    ]
    selected = min(
        world_eligible,
        key=lambda row: (
            int(row["full_retired_refit_nonzero_coefficients"]),
            float(row["C"]),
        ),
    )
    return {
        "maximum_block_oof_exact_assays": best_exact,
        "exact_assay_tolerance": EXACT_ASSAY_TOLERANCE,
        "minimum_eligible_exact_assays": best_exact - EXACT_ASSAY_TOLERANCE,
        "maximum_eligible_all_six_exact_worlds": best_perfect_worlds,
        "final_tie_break": (
            "fewest nonzero coefficients in the full retired refit, then lower C"
        ),
        "selected_C": float(selected["C"]),
    }


def _raw_scale_parameters(
    fitted: FittedPipeline,
) -> tuple[NDArray[np.int64], FloatMatrix, FloatMatrix, float]:
    raw_weights = fitted.model.coef_[0] / fitted.scaler.scale_
    raw_intercept = float(
        fitted.model.intercept_[0]
        - np.dot(raw_weights, fitted.scaler.mean_)
    )
    retained = np.flatnonzero(np.abs(raw_weights) > NONZERO_TOLERANCE)
    return (
        retained.astype(np.int64),
        fitted.imputer.statistics_[retained].astype(np.float64),
        np.round(
            raw_weights[retained].astype(np.float64),
            decimals=COEFFICIENT_DECIMAL_PLACES,
        ),
        round(raw_intercept, COEFFICIENT_DECIMAL_PLACES),
    )


def derivation_contract(binding: dict[str, Any]) -> dict[str, Any]:
    """Return the non-circular contract that identifies the frozen fit."""

    return {
        "schema_version": DERIVATION_CONTRACT_SCHEMA_VERSION,
        "input_binding": {
            "candidate_table_sha256": binding["candidate_table"]["sha256"],
            "phase_a_manifest_sha256": binding["phase_a_manifest"]["sha256"],
            "phase_b_manifest_sha256": binding["phase_b_manifest"]["sha256"],
        },
        "allowed_training_blocks": list(ALLOWED_BLOCKS),
        "input_cardinality": {
            "candidate_rows": EXPECTED_CANDIDATE_ROWS,
            "worlds": EXPECTED_WORLDS,
            "assays": EXPECTED_ASSAYS,
            "stars_per_assay": 60,
            "exact_star_labels_per_assay": 1,
            "directed_edge_rows": EXPECTED_DIRECTED_EDGE_ROWS,
            "positive_directed_arms": EXPECTED_POSITIVE_DIRECTED_ARMS,
        },
        "label_construction": (
            "for each exact directed two-parent star, label the two outer-to-center "
            "directed arms positive and all other ordered node pairs negative"
        ),
        "directed_edge_design": {
            "base_feature_schema": list(BASE_FEATURE_SCHEMA),
            "contexts": list(CONTEXTS),
            "full_feature_schema": list(PAIR_FEATURE_SCHEMA),
            "context_definitions": {
                "direct": "highest-score star whose center is the directed child and whose outer set contains the parent",
                "reverse": "highest-score star whose center is the directed parent and whose outer set contains the child",
                "outer": "highest-score star whose two outer nodes are the ordered edge endpoints",
            },
            "context_star_selection": (
                "descending raw generative score, then ascending center/outer1/outer2"
            ),
            "raw_generative_score": {
                "crossfit_joint_gain_over_baseline": 1.0,
                "crossfit_min_drop_one_gain": 1.0,
                "laplace_min_parent_positive_posterior": 0.02,
                "residual_star_separation": 0.25,
            },
            "pair_matrix_sha256": EXPECTED_PAIR_MATRIX_SHA256,
        },
        "block_held_out_selection": {
            "held_out_unit": "entire TRAIN block",
            "C_grid": list(C_GRID),
            "primary_metric": "exact assays",
            "exact_assay_tolerance_from_best": EXACT_ASSAY_TOLERANCE,
            "secondary_metric": "all-six-exact worlds",
            "final_tie_break": (
                "fewest nonzero coefficients in full retired refit, then lower C"
            ),
            "selected_C": 0.003,
        },
        "preprocessing": {
            "SimpleImputer": {
                "missing_values": "NaN",
                "strategy": "median",
                "fill_value": None,
                "copy": True,
                "add_indicator": False,
                "keep_empty_features": False,
            },
            "StandardScaler": {
                "copy": True,
                "with_mean": True,
                "with_std": True,
            },
        },
        "estimator": {
            "class": "sklearn.linear_model.LogisticRegression",
            "penalty": "l1",
            "C": 0.003,
            "class_weight": "balanced",
            "solver": "liblinear",
            "dual": False,
            "fit_intercept": True,
            "intercept_scaling": 1.0,
            "l1_ratio": None,
            "max_iter": 5000,
            "n_jobs": None,
            "random_state": RANDOM_STATE,
            "tol": 1e-4,
            "verbose": 0,
            "warm_start": False,
            "positive_label_dtype": "numpy.int8",
            "nonzero_tolerance": NONZERO_TOLERANCE,
        },
        "reference_fit_environment": {
            "numpy": "2.2.6",
            "pandas": "2.2.2",
            "scikit_learn": "1.6.1",
        },
        "runtime_canonicalization": {
            "imputation_medians": "unrounded IEEE-754 binary64",
            "raw_scale_weights": (
                f"numpy.round(binary64, decimals={COEFFICIENT_DECIMAL_PLACES})"
            ),
            "raw_scale_intercept": (
                f"Python round(binary64, ndigits={COEFFICIENT_DECIMAL_PLACES})"
            ),
            "reason": "remove sub-1e-9 liblinear/platform arithmetic drift",
        },
    }


def derivation_contract_sha256(contract: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(contract).encode("utf-8"))


def _runtime_payload(
    feature_schema: Sequence[str],
    medians: Sequence[float],
    weights: Sequence[float],
    intercept: float,
    contract_sha256: str,
) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "retired_training_blocks": list(ALLOWED_BLOCKS),
        "selection_rule": (
            "sparsest_l1_within_one_exact_assay_of_best_retired_block_oof"
        ),
        "fit": {
            "penalty": "l1",
            "C": 0.003,
            "class_weight": "balanced",
            "solver": "liblinear",
            "random_state": RANDOM_STATE,
        },
        "context_aggregation": "highest_raw_generative_score_star",
        "derivation_contract": {
            "schema_version": DERIVATION_CONTRACT_SCHEMA_VERSION,
            "sha256": contract_sha256,
            "retired_phase_b_manifest_sha256": EXPECTED_PHASE_B_MANIFEST_SHA256,
        },
        "feature_schema": list(feature_schema),
        "imputation_medians": [float(value) for value in medians],
        "raw_scale_weights": [float(value) for value in weights],
        "raw_scale_intercept": float(intercept),
        "coefficient_canonical_decimal_places": COEFFICIENT_DECIMAL_PLACES,
        "runtime_refit": False,
    }


def verify_runtime_freeze(
    feature_schema: Sequence[str],
    medians: NDArray[np.float64],
    weights: NDArray[np.float64],
    intercept: float,
    payload: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    import crossfit_hierarchical_collider_v4 as runtime

    schema_match = tuple(feature_schema) == tuple(
        f"{context}__{feature}"
        for context, feature in runtime.ZERO_EDGE_RANK_FUSION_SCHEMA
    )
    medians_match = np.array_equal(
        medians,
        np.asarray(runtime.ZERO_EDGE_RANK_FUSION_MEDIANS, dtype=np.float64),
    )
    weights_match = np.array_equal(
        weights,
        np.asarray(runtime.ZERO_EDGE_RANK_FUSION_WEIGHTS, dtype=np.float64),
    )
    intercept_match = float(intercept) == float(
        runtime.ZERO_EDGE_RANK_FUSION_INTERCEPT
    )
    payload_match = payload == runtime._zero_edge_rank_fusion_model_payload()
    payload_sha = sha256_bytes(canonical_json(payload).encode("utf-8"))
    payload_sha_match = payload_sha == runtime._zero_edge_rank_fusion_model_sha256()
    model_version_match = runtime.ZERO_EDGE_RANK_FUSION_MODEL_VERSION == (
        "retired_train_330_380_sparse_l1_v1"
    )
    contract_sha_match = contract_sha256 == (
        runtime.ZERO_EDGE_RANK_FUSION_DERIVATION_CONTRACT_SHA256
    )
    checks = {
        "runtime_model_version_exact": model_version_match,
        "derivation_contract_sha256_exact": contract_sha_match,
        "feature_schema_exact": schema_match,
        "imputation_medians_exact": medians_match,
        "raw_scale_weights_exact": weights_match,
        "raw_scale_intercept_exact": intercept_match,
        "payload_exact": payload_match,
        "payload_sha256_exact": payload_sha_match,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise AssertionError(f"Frozen runtime reproduction failed: {failed}")
    return {
        **checks,
        "all_exact": True,
        "runtime_payload_sha256": payload_sha,
        "runtime_source": _relative(Path(runtime.__file__)),
        "runtime_source_sha256": sha256_file(Path(runtime.__file__)),
    }


def derive_manifest(candidate_path: Path = DEFAULT_INPUT) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    binding = _validate_input_binding(candidate_path)
    candidates = pd.read_csv(candidate_path, sep="\t")
    pair_matrix = build_pair_matrix(candidates)
    matrix_sha = pair_matrix_sha256(pair_matrix)
    if matrix_sha != EXPECTED_PAIR_MATRIX_SHA256:
        raise AssertionError("Directed-edge matrix content hash drifted")
    grid = evaluate_grid(pair_matrix)
    selection = select_grid_candidate(grid)
    selected_c = float(selection["selected_C"])
    fitted = fit_pipeline(
        pair_matrix,
        np.ones(len(pair_matrix.labels), dtype=np.bool_),
        c_value=selected_c,
    )
    retained, medians, weights, intercept = _raw_scale_parameters(fitted)
    feature_schema = tuple(pair_matrix.feature_schema[index] for index in retained)
    contract = derivation_contract(binding)
    contract_sha = derivation_contract_sha256(contract)
    payload = _runtime_payload(
        feature_schema,
        medians,
        weights,
        intercept,
        contract_sha,
    )
    verification = verify_runtime_freeze(
        feature_schema,
        medians,
        weights,
        intercept,
        payload,
        contract_sha,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_self_verifying_retired_train_derivation",
        "runtime_source_closure_dependency": False,
        "allowed_training_blocks": list(ALLOWED_BLOCKS),
        "train400_inputs_used_for_fitting_or_selection": False,
        "dev_or_locked_inputs_opened": False,
        "derivation_contract": {
            "payload": contract,
            "sha256": contract_sha,
        },
        "input_binding": binding,
        "input_shape": {
            "candidate_rows": int(len(candidates)),
            "worlds": int(
                candidates[["block", "seed"]].drop_duplicates().shape[0]
            ),
            "assays": int(
                candidates[["block", "seed", "source_assay"]]
                .drop_duplicates()
                .shape[0]
            ),
            "exact_star_labels": int(candidates[EXACT_STAR_COLUMN].astype(bool).sum()),
            "directed_edge_rows": int(len(pair_matrix.labels)),
            "positive_directed_arms": int(pair_matrix.labels.sum()),
        },
        "directed_edge_design": {
            "base_feature_schema": list(BASE_FEATURE_SCHEMA),
            "contexts": list(CONTEXTS),
            "full_feature_schema": list(PAIR_FEATURE_SCHEMA),
            "context_star_selection": (
                "highest raw generative score, then center/outer1/outer2 lexical"
            ),
            "raw_generative_score_formula": (
                "joint_gain + min_drop_gain + 0.02*parent_posterior "
                "+ 0.25*residual_star_separation"
            ),
            "matrix_sha256": matrix_sha,
        },
        "preprocessing": contract["preprocessing"],
        "estimator": {
            **contract["estimator"],
            "C_grid": list(C_GRID),
        },
        "block_held_out_grid": grid,
        "selection": selection,
        "selected_model": {
            "feature_schema": list(feature_schema),
            "imputation_medians": medians.tolist(),
            "raw_scale_weights": weights.tolist(),
            "raw_scale_intercept": intercept,
            "coefficient_canonical_decimal_places": (
                COEFFICIENT_DECIMAL_PLACES
            ),
            "nonzero_coefficients": int(len(retained)),
            "runtime_payload": payload,
            "runtime_payload_sha256": sha256_bytes(
                canonical_json(payload).encode("utf-8")
            ),
        },
        "runtime_verification": verification,
        "software_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "derivation_source": {
            "path": _relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    manifest["manifest_payload_sha256"] = sha256_bytes(
        canonical_json(manifest).encode("utf-8")
    )
    return manifest


def verify_manifest_payload(manifest: dict[str, Any]) -> None:
    payload = dict(manifest)
    expected = str(payload.pop("manifest_payload_sha256"))
    observed = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if observed != expected:
        raise AssertionError("Derivation manifest payload hash mismatch")


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    if "development_only" not in str(output_path).lower():
        raise ValueError("Audit output must live in a development-only path")
    verify_manifest_payload(manifest)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"Refusing to overwrite a different audit manifest: {output_path}"
            )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = derive_manifest(args.input)
    write_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_C": manifest["selection"]["selected_C"],
                "nonzero_coefficients": manifest["selected_model"][
                    "nonzero_coefficients"
                ],
                "runtime_payload_sha256": manifest["selected_model"][
                    "runtime_payload_sha256"
                ],
                "derivation_contract_sha256": manifest["derivation_contract"][
                    "sha256"
                ],
                "exact_reproduction": manifest["runtime_verification"]["all_exact"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
