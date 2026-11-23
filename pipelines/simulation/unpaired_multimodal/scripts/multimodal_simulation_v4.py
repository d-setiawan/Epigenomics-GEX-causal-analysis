#!/usr/bin/env python3
"""Preregistered seven-assay simulation for unpaired causal discovery.

The simulator draws independent cells for RNA and six CUT&Tag assays.  It
separates its outputs into two hard information domains:

``discovery/``
    Observed background matrices and depths, observed focal measurements,
    background-only state estimates, and background-derived metacells.

``scorer/``
    Role assignments, the structural DAG, typed truth edges, coefficients, and
    per-cell latent state.  A discovery method must never read this directory.

The six focal slots in every mark are assigned the roles P, Q, C, S, N1, and
N2 only *after* slot-specific detection probabilities have been drawn.  The
structural motif is P -> C <- Q, while P and S share a latent-state parent.
One mark carries a slow, state-pairable P -> gene effect and a different mark
carries a fast, cell-specific residual challenge.  Because cells are unpaired,
the latter is intentionally not identifiable from observational cross-assay
covariance.

DOE factor D is a raw-data background-observability stress: its high arm adds
moderate assay-private noise to a background-only proxy of biological state
before background counts are sampled.  Focal molecular activities continue to
use the true state.  An optional supplied-coordinate derangement remains only
as a separate invariance diagnostic and is never selected by D.

No feature identity, cell pairing, focal value, simulator label, or privileged
embedding is passed to the alignment adapter.  The adapter contract is the
public API in :mod:`unpaired_state_alignment`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.special import expit

try:
    import unpaired_state_alignment as state_alignment
except ImportError as exc:  # pragma: no cover - exercised only in partial checkouts
    raise ImportError(
        "multimodal_simulation_v4 requires unpaired_state_alignment.py with "
        "AlignmentConfig and align_unpaired_background_states(background_matrices, "
        "background_depths, *, reference_modality, config)."
    ) from exc


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MARKS = (
    "H3K27ac",
    "H3K27me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9me3",
)
MODALITIES = ("rna", *(f"chrom_{mark}" for mark in MARKS))
ROLES = ("P", "Q", "C", "S", "N1", "N2")
GENE = "expr__SIMG"
AGGREGATION_CELL_TARGETS = (24, 14, 8)
AGGREGATION_SEEDS = (0, 1)
TRUTH_TYPES = {
    "within_assay_direct",
    "within_assay_shared_state",
    "cross_assay_shared_state",
    "slow_peak_gene",
    "fast_peak_gene_challenge",
    "sentinel_null",
}
# TRAIN-only paired sensitivity runs showed that 0.35 and above collapses the
# observable cross-assay synchronization graph, while 0.20 remains a material
# stress (reduced cycle support in three of four paired worlds) without
# annihilating every chromatin assay.  Keep this fixed before any DEV/LOCKED
# evaluation; it is an observation-layer challenge, not a tuning parameter for
# discovery on an individual world.
BACKGROUND_PROXY_NOISE_SD = 0.20

# A standard 2^(4-1) half fraction with D = A*B*C.  Preregistered world seeds
# are one-based within each seed block: design index = (seed - 1) mod 8.
_HALF_FRACTION = (
    (-1, -1, -1),
    (-1, -1, +1),
    (-1, +1, -1),
    (-1, +1, +1),
    (+1, -1, -1),
    (+1, -1, +1),
    (+1, +1, -1),
    (+1, +1, +1),
)
_CELL_MULTIPLIERS = (0.65, 0.80, 1.00, 1.00, 1.20, 1.35, 1.50)
_MIXTURE_CENTERS = np.asarray(
    [
        [-1.45, -0.55, +0.15],
        [-0.25, +1.35, -0.35],
        [+1.25, +0.20, +0.75],
        [-0.95, +0.75, +1.15],
        [+0.45, -1.25, -0.85],
        [+1.35, +0.55, -0.65],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SimulationV4Config:
    """Configuration for one independently drawn biological world."""

    seed: int = 0
    base_cells_override: int | None = None
    n_background_features: int = 64
    latent_dim: int = 3
    n_microstates: int = 24
    min_cells_per_microstate: int = 20
    aggregation_cell_targets: tuple[int, ...] = AGGREGATION_CELL_TARGETS
    aggregation_seeds: tuple[int, ...] = AGGREGATION_SEEDS
    # This override exists only for paired metamorphic tests of the observation
    # model.  Preregistered benchmark worlds leave it at ``None`` so D is still
    # fixed by the half-fraction design.
    background_observability_stress_override: bool | None = None
    # Output derangement is an optional invariance diagnostic.  It is not a DOE
    # treatment and is disabled for all ordinary simulation worlds.
    alignment_derangement_diagnostic: bool = False

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.base_cells_override is not None and self.base_cells_override < 16:
            raise ValueError("base_cells_override must be at least 16")
        if self.n_background_features < 4:
            raise ValueError("n_background_features must be at least four")
        if self.latent_dim != 3:
            raise ValueError("v4 is preregistered with exactly three latent dimensions")
        if self.n_microstates <= self.latent_dim:
            raise ValueError("n_microstates must exceed latent_dim")
        if self.min_cells_per_microstate < 2:
            raise ValueError("min_cells_per_microstate must be at least two")
        if not self.aggregation_cell_targets or min(self.aggregation_cell_targets) < 2:
            raise ValueError("aggregation cell targets must all be at least two")
        if not self.aggregation_seeds:
            raise ValueError("at least one aggregation seed is required")


@dataclass
class _World:
    config: SimulationV4Config
    design: dict[str, Any]
    parameters: dict[str, Any]
    cell_ids: dict[str, NDArray[np.str_]]
    latent_states: dict[str, FloatArray]
    latent_modes: dict[str, IntArray]
    background_state_proxy: dict[str, FloatArray]
    background_proxy_noise: dict[str, FloatArray]
    background_technical_scale: dict[str, FloatArray]
    background: dict[str, FloatArray]
    background_depth: dict[str, FloatArray]
    focal: dict[str, pd.DataFrame]
    roles: pd.DataFrame
    truth_edges: pd.DataFrame
    dag_nodes: pd.DataFrame
    dag_edges: pd.DataFrame


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discovery_safe_alignment_provenance(value: Any) -> Any:
    """Redact simulator stress assignments while retaining fit diagnostics."""

    forbidden_keys = {
        "config",
        "deranged",
        "derangement_property_summary",
        "inferred_assignment",
        "applied_assignment",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _discovery_safe_alignment_provenance(item)
            for key, item in value.items()
            if str(key) not in forbidden_keys
        }
    if isinstance(value, list):
        return [_discovery_safe_alignment_provenance(item) for item in value]
    return value


def design_for_seed(
    seed: int,
    *,
    base_cells_override: int | None = None,
    background_observability_stress_override: bool | None = None,
    alignment_derangement_diagnostic: bool = False,
) -> dict[str, Any]:
    """Return the preregistered DOE cell for ``seed`` without drawing data."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    design_index = (seed - 1) % len(_HALF_FRACTION)
    a, b, c = _HALF_FRACTION[design_index]
    d = a * b * c
    stress = bool(d > 0)
    if background_observability_stress_override is not None:
        stress = bool(background_observability_stress_override)
    base_cells = int(base_cells_override or (1200 if a < 0 else 2400))
    return {
        "half_fraction_cell": int(design_index + 1),
        "half_fraction_index_zero_based": int(design_index),
        "A_cells": int(a),
        "B_detection": int(b),
        "C_population_divergence": int(c),
        "D_background_observability": int(d),
        "base_cells": base_cells,
        "detection_stress": bool(b > 0),
        "population_divergence": bool(c > 0),
        "background_integration_stress": stress,
        "background_observability_stress_override": (
            None
            if background_observability_stress_override is None
            else bool(background_observability_stress_override)
        ),
        "alignment_derangement_diagnostic": bool(alignment_derangement_diagnostic),
        "generator": "2^(4-1), D_background_observability=A*B*C",
    }


def _seeded_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _zscore(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    scale = float(values.std())
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / scale


def _draw_parameter_table(config: SimulationV4Config) -> tuple[dict[str, Any], pd.DataFrame]:
    """Draw detection first, then randomize semantic roles over the six slots."""

    seed = config.seed
    design = design_for_seed(
        seed,
        base_cells_override=config.base_cells_override,
        background_observability_stress_override=(
            config.background_observability_stress_override
        ),
        alignment_derangement_diagnostic=config.alignment_derangement_diagnostic,
    )
    rng = _seeded_rng(seed, 101)

    modality_order = np.asarray(MODALITIES, dtype=object)
    multipliers = np.asarray(_CELL_MULTIPLIERS, dtype=np.float64)
    rng.shuffle(multipliers)
    cell_multipliers = {
        str(modality): float(multiplier)
        for modality, multiplier in zip(modality_order, multipliers, strict=True)
    }
    cell_counts = {
        modality: int(max(16, round(design["base_cells"] * cell_multipliers[modality])))
        for modality in MODALITIES
    }

    if design["population_divergence"]:
        raw = rng.normal(size=len(MODALITIES))
        raw -= raw.mean()
        raw /= max(float(np.max(np.abs(raw))), 1e-12)
        shifts = 0.15 * raw
    else:
        shifts = np.zeros(len(MODALITIES), dtype=np.float64)
    population_fractions = {
        modality: float(np.clip(0.60 + shift, 0.25, 0.90))
        for modality, shift in zip(MODALITIES, shifts, strict=True)
    }

    stress_marks: set[str] = set()
    if design["detection_stress"]:
        stress_marks = set(rng.choice(MARKS, size=3, replace=False).tolist())

    mark_rotation = rng.permutation(np.asarray(MARKS, dtype=object)).tolist()
    slow_mark, fast_mark = str(mark_rotation[0]), str(mark_rotation[1])
    mark_pairs = [
        (str(mark_rotation[index]), str(mark_rotation[index + 1]))
        for index in range(0, len(mark_rotation), 2)
    ]

    state_dimension: dict[str, int] = {}
    state_sign: dict[str, float] = {}
    for pair_index, pair in enumerate(mark_pairs):
        dimension = pair_index % config.latent_dim
        sign = float(rng.choice([-1.0, 1.0]))
        for mark in pair:
            state_dimension[mark] = dimension
            state_sign[mark] = sign

    beta_p = {mark: float(rng.uniform(0.65, 1.00)) for mark in MARKS}
    beta_q = {mark: float(rng.uniform(0.55, 0.90)) for mark in MARKS}
    gamma_p_detection = {mark: float(rng.uniform(1.40, 1.80)) for mark in MARKS}
    gamma_q_detection = {mark: float(rng.uniform(1.40, 1.80)) for mark in MARKS}
    q_dimension = {
        mark: int((state_dimension[mark] + 1 + rng.integers(0, 2)) % config.latent_dim)
        for mark in MARKS
    }
    q_sign = {mark: float(rng.choice([-1.0, 1.0])) for mark in MARKS}
    p_state_loading = {
        mark: 1.30 if mark == slow_mark else 0.14 if mark == fast_mark else 0.90
        for mark in MARKS
    }
    p_residual_scale = {
        mark: 0.25 if mark == slow_mark else 1.15 if mark == fast_mark else 0.52
        for mark in MARKS
    }

    role_rows: list[dict[str, Any]] = []
    feature_nodes: dict[str, dict[str, str]] = {}
    for mark_index, mark in enumerate(MARKS):
        base_detection = (
            float(rng.uniform(0.035, 0.055))
            if mark in stress_marks
            else float(rng.uniform(0.070, 0.100))
        )
        # These six draws deliberately happen before the role permutation.
        slot_detection = np.clip(
            base_detection * np.exp(rng.normal(scale=0.12, size=len(ROLES))),
            0.025,
            0.13,
        )
        starts = rng.choice(
            np.arange(100_000 + mark_index * 1_000_000, 900_000 + mark_index * 1_000_000, 250),
            size=len(ROLES),
            replace=False,
        )
        slot_nodes = [
            f"chr{mark_index + 1}:{int(start)}-{int(start) + 199}__{mark}"
            for start in starts
        ]
        role_permutation = rng.permutation(np.asarray(ROLES, dtype=object)).tolist()
        feature_nodes[mark] = {}
        for slot, role_object in enumerate(role_permutation):
            role = str(role_object)
            feature_nodes[mark][role] = slot_nodes[slot]
            role_rows.append(
                {
                    "node": slot_nodes[slot],
                    "mark": mark,
                    "role": role,
                    "slot": slot,
                    "detection_draw_index": slot,
                    "detection_probability": float(slot_detection[slot]),
                    "detection_draw_stage": "before_role_assignment",
                    "state_dimension": (
                        state_dimension[mark]
                        if role in {"P", "S"}
                        else q_dimension[mark]
                        if role == "Q"
                        else ""
                    ),
                }
            )

    roles = pd.DataFrame(role_rows)
    parameters: dict[str, Any] = {
        "schema_version": "multimodal_simulation_v4.parameters.v2",
        "seed": seed,
        "marks": list(MARKS),
        "modalities": list(MODALITIES),
        "roles": list(ROLES),
        "gene": GENE,
        "design": design,
        "cell_multipliers": cell_multipliers,
        "cell_counts": cell_counts,
        "population_fractions": population_fractions,
        "population_shift_mean": float(np.mean(shifts)),
        "stress_marks": sorted(stress_marks),
        "mark_rotation": mark_rotation,
        "slow_mark": slow_mark,
        "fast_mark": fast_mark,
        "mark_pairs": mark_pairs,
        "state_dimension": state_dimension,
        "state_sign": state_sign,
        "q_dimension": q_dimension,
        "q_sign": q_sign,
        "p_state_loading": p_state_loading,
        "p_residual_scale": p_residual_scale,
        "beta_p_to_c": beta_p,
        "beta_q_to_c": beta_q,
        "gamma_p_to_c_detection_log_odds": gamma_p_detection,
        "gamma_q_to_c_detection_log_odds": gamma_q_detection,
        "slow_peak_gene_coefficient": float(rng.uniform(0.65, 0.90)),
        "fast_peak_gene_coefficient": float(rng.uniform(0.90, 1.20)),
        "background_observation_model": {
            "schema_version": "multimodal_simulation_v4.background_observation.v1",
            "background_latent_input": "assay_private_background_state_proxy",
            "focal_activity_latent_input": "true_latent_state",
            "background_integration_stress": bool(
                design["background_integration_stress"]
            ),
            "background_state_proxy_noise_sd": (
                BACKGROUND_PROXY_NOISE_SD
                if design["background_integration_stress"]
                else 0.0
            ),
            "background_state_proxy_noise_family": "independent_gaussian",
            "background_state_proxy_noise_is_assay_private": True,
            "background_state_proxy_rng_stream_by_modality": {
                modality: 300 + index
                for index, modality in enumerate(MODALITIES)
            },
            "background_signal_scale": {
                modality: 0.42 if modality == "rna" else 0.55
                for modality in MODALITIES
            },
            "technical_scale_log_sd": 0.34,
            "background_log_rate_noise_sd": 0.10,
        },
        "conditional_dag": {
            "schema_version": "multimodal_simulation_v4.conditional_dag.v1",
            "world_condition": "background_integration_stress",
            "doe_source": "D_background_observability",
            "background_path": [
                "latent_state",
                "assay_private_background_state_proxy",
                "observed_background_count",
            ],
            "conditional_mechanism": "assay_private_background_proxy_noise",
            "conditional_mechanism_active_when": (
                "background_integration_stress == true"
            ),
            "truth_topology_invariant_across_condition": True,
            "focal_activities_use_true_latent_state": True,
            "alignment_derangement_is_doe_factor": False,
        },
        "detection_draws_precede_role_assignment": True,
        "all_cells_retained": True,
    }
    parameters["feature_nodes"] = feature_nodes
    return parameters, roles


def _draw_latent_state(
    n_cells: int,
    target_fraction: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    within = np.asarray([0.46, 0.34, 0.20], dtype=np.float64)
    probabilities = np.concatenate(
        [target_fraction * within, (1.0 - target_fraction) * within]
    )
    probabilities /= probabilities.sum()
    modes = np.asarray(rng.choice(6, size=n_cells, p=probabilities), dtype=np.int64)
    z = _MIXTURE_CENTERS[modes] + rng.normal(scale=0.48, size=(n_cells, 3))
    z[:, 1] += 0.22 * z[:, 0]
    return np.asarray(z, dtype=np.float64), modes


def _draw_background(
    background_state_proxy: FloatArray,
    *,
    modality: str,
    n_features: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Generate an assay-private observed matrix from a background-only proxy."""

    loadings = rng.normal(size=(n_features, background_state_proxy.shape[1]))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    if modality == "rna":
        baseline = rng.uniform(3.0, 8.0, size=n_features)
        signal_scale = 0.42
    else:
        baseline = rng.uniform(0.45, 2.2, size=n_features)
        signal_scale = 0.55
    technical_scale = np.exp(
        rng.normal(scale=0.34, size=background_state_proxy.shape[0])
    )
    log_rate = np.log(baseline)[None, :] + signal_scale * (
        background_state_proxy @ loadings.T
    )
    log_rate += rng.normal(scale=0.10, size=log_rate.shape)
    rate = np.exp(np.clip(log_rate, -6.0, 5.0)) * technical_scale[:, None]
    counts = rng.poisson(rate).astype(np.float64)
    depth = counts.sum(axis=1)
    # Zero-depth rows are retained and receive a single background event.  This
    # is an observation-model safeguard, never a label- or state-based filter.
    zero = np.flatnonzero(depth <= 0)
    if zero.size:
        chosen = rng.integers(0, n_features, size=zero.size)
        counts[zero, chosen] = 1.0
        depth = counts.sum(axis=1)
    return (
        counts,
        np.asarray(depth, dtype=np.float64),
        np.asarray(technical_scale, dtype=np.float64),
    )


def _activities_for_mark(
    z: FloatArray,
    mark: str,
    parameters: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, FloatArray]:
    dimension = int(parameters["state_dimension"][mark])
    sign = float(parameters["state_sign"][mark])
    q_dimension = int(parameters["q_dimension"][mark])
    q_sign = float(parameters["q_sign"][mark])
    p_loading = float(parameters["p_state_loading"][mark])
    p_noise = float(parameters["p_residual_scale"][mark])
    p = sign * p_loading * z[:, dimension] + rng.normal(scale=p_noise, size=len(z))
    q = q_sign * 0.85 * z[:, q_dimension] + rng.normal(scale=0.55, size=len(z))
    s = sign * 1.05 * z[:, dimension] + rng.normal(scale=0.38, size=len(z))
    c = (
        float(parameters["beta_p_to_c"][mark]) * p
        + float(parameters["beta_q_to_c"][mark]) * q
        + rng.normal(scale=0.48, size=len(z))
    )
    return {
        "P": np.asarray(p, dtype=np.float64),
        "Q": np.asarray(q, dtype=np.float64),
        "C": np.asarray(c, dtype=np.float64),
        "S": np.asarray(s, dtype=np.float64),
        "N1": np.asarray(rng.normal(size=len(z)), dtype=np.float64),
        "N2": np.asarray(rng.normal(size=len(z)), dtype=np.float64),
    }


def _calibrated_binary_observation(
    activity: FloatArray,
    background_depth: FloatArray,
    target_detection: float,
    rng: np.random.Generator,
    logit_offset: FloatArray | None = None,
) -> IntArray:
    depth_score = _zscore(np.log1p(background_depth))
    score = 0.78 * _zscore(activity) + 0.16 * depth_score
    if logit_offset is not None:
        offset = np.asarray(logit_offset, dtype=np.float64)
        if offset.shape != score.shape or not np.isfinite(offset).all():
            raise ValueError("logit_offset must be finite and aligned to activity")
        score = score + offset
    low, high = -14.0, 3.0
    for _ in range(50):
        midpoint = (low + high) / 2.0
        if float(expit(midpoint + score).mean()) < target_detection:
            low = midpoint
        else:
            high = midpoint
    probabilities = expit((low + high) / 2.0 + score)
    return np.asarray(rng.binomial(1, probabilities), dtype=np.int64)


def _draw_focal_tables(
    parameters: Mapping[str, Any],
    roles: pd.DataFrame,
    cell_ids: Mapping[str, NDArray[np.str_]],
    latent_states: Mapping[str, FloatArray],
    background_depth: Mapping[str, FloatArray],
    seed: int,
) -> dict[str, pd.DataFrame]:
    focal: dict[str, pd.DataFrame] = {}
    role_lookup = roles.set_index(["mark", "role"])

    rna_rng = _seeded_rng(seed, 401)
    rna_z = latent_states["rna"]
    slow = _activities_for_mark(rna_z, str(parameters["slow_mark"]), parameters, rna_rng)
    fast = _activities_for_mark(rna_z, str(parameters["fast_mark"]), parameters, rna_rng)
    gene_activity = (
        float(parameters["slow_peak_gene_coefficient"]) * slow["P"]
        + float(parameters["fast_peak_gene_coefficient"]) * fast["P"]
        + 0.28 * rna_z[:, 2]
        + rna_rng.normal(scale=0.65, size=len(rna_z))
    )
    gene_mean = np.exp(
        np.clip(
            1.45
            + 0.48 * _zscore(gene_activity)
            + 0.10 * _zscore(np.log1p(background_depth["rna"])),
            -5.0,
            6.0,
        )
    )
    dispersion = 10.0
    probability = dispersion / (dispersion + gene_mean)
    gene_counts = rna_rng.negative_binomial(dispersion, probability)
    focal["rna"] = pd.DataFrame(
        {
            "cell": cell_ids["rna"],
            "modality": "rna",
            "observed_background_library_size": background_depth["rna"],
            "observed_library_size": background_depth["rna"] + gene_counts,
            GENE: np.log1p(gene_counts.astype(np.float64)),
        }
    )

    for mark_index, mark in enumerate(MARKS):
        modality = f"chrom_{mark}"
        rng = _seeded_rng(seed, 500 + mark_index)
        activities = _activities_for_mark(
            latent_states[modality], mark, parameters, rng
        )
        columns: dict[str, Any] = {
            "cell": cell_ids[modality],
            "modality": modality,
            "observed_background_library_size": background_depth[modality],
        }
        total_focal = np.zeros(len(cell_ids[modality]), dtype=np.int64)
        observed_by_role: dict[str, IntArray] = {}
        observation_order = ("P", "Q", "S", "N1", "N2", "C")
        for role in observation_order:
            row = role_lookup.loc[(mark, role)]
            node = str(row["node"])
            logit_offset = None
            if role == "C":
                logit_offset = (
                    float(parameters["gamma_p_to_c_detection_log_odds"][mark])
                    * observed_by_role["P"]
                    + float(parameters["gamma_q_to_c_detection_log_odds"][mark])
                    * observed_by_role["Q"]
                )
            observed = _calibrated_binary_observation(
                activities[role],
                background_depth[modality],
                float(row["detection_probability"]),
                rng,
                logit_offset=logit_offset,
            )
            observed_by_role[role] = observed
            columns[node] = observed
            total_focal += observed
        columns["observed_library_size"] = background_depth[modality] + total_focal
        ordered = [
            "cell",
            "modality",
            "observed_background_library_size",
            "observed_library_size",
            *roles.loc[roles["mark"] == mark].sort_values("slot")["node"].tolist(),
        ]
        focal[modality] = pd.DataFrame(columns)[ordered]
    return focal


def _truth_and_dag(
    parameters: Mapping[str, Any], roles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = roles.set_index(["mark", "role"])
    truth_rows: list[dict[str, Any]] = []

    def pure_state_nodes(mark: str, dimension: int) -> tuple[str, ...]:
        """Return observed features with a single loading on ``dimension``.

        P and S share the assay's primary state axis.  Q is an independent
        parent with its own (also pure) state axis.  C is deliberately omitted:
        as a collider of P and Q it generally mixes two axes, so treating it as
        an interchangeable state reporter would erase the direct-structure
        estimand.  Signs are not used because the discovery layer predeclares
        sign-invariant loading equivalence.
        """

        nodes: list[str] = []
        if int(parameters["state_dimension"][mark]) == int(dimension):
            nodes.extend(
                [
                    str(lookup.loc[(mark, "P"), "node"]),
                    str(lookup.loc[(mark, "S"), "node"]),
                ]
            )
        if int(parameters["q_dimension"][mark]) == int(dimension):
            nodes.append(str(lookup.loc[(mark, "Q"), "node"]))
        return tuple(dict.fromkeys(nodes))

    def truth(
        node1: str,
        node2: str,
        truth_type: str,
        label: str,
        mark: str,
        *,
        parent: str = "",
        child: str = "",
        primary: bool = False,
        orientation_credit: bool = False,
        node_set1: tuple[str, ...] | None = None,
        node_set2: tuple[str, ...] | None = None,
        estimand: str = "exact_pair",
    ) -> None:
        truth_rows.append(
            {
                "node1": node1,
                "node2": node2,
                "truth_type": truth_type,
                "label": label,
                "mark": mark,
                "parent": parent,
                "child": child,
                "primary": bool(primary),
                "orientation_credit": bool(orientation_credit),
                "node_set1": ";".join(node_set1 or (node1,)),
                "node_set2": ";".join(node_set2 or (node2,)),
                "estimand": estimand,
            }
        )

    for mark in MARKS:
        nodes = {role: str(lookup.loc[(mark, role), "node"]) for role in ROLES}
        for parent_role in ("P", "Q"):
            truth(
                nodes[parent_role],
                nodes["C"],
                "within_assay_direct",
                f"{mark}:{parent_role}->C",
                mark,
                parent=nodes[parent_role],
                child=nodes["C"],
                primary=True,
                orientation_credit=True,
            )
        truth(
            nodes["P"],
            nodes["S"],
            "within_assay_shared_state",
            f"{mark}:P-S_shared_state",
            mark,
            primary=True,
        )

        # Detection-matched nulls are fixed from design probabilities, never
        # chosen using observed outcomes.  Each null is matched to its nearest
        # non-null slot within the same assay.
        for null_role in ("N1", "N2"):
            null_rate = float(lookup.loc[(mark, null_role), "detection_probability"])
            comparison_role = min(
                ("P", "Q", "C", "S"),
                key=lambda role: abs(
                    float(lookup.loc[(mark, role), "detection_probability"]) - null_rate
                ),
            )
            truth(
                nodes[null_role],
                nodes[comparison_role],
                "sentinel_null",
                f"{mark}:{null_role}-{comparison_role}_detection_matched_null",
                mark,
            )

    for left_mark, right_mark in parameters["mark_pairs"]:
        left_s = str(lookup.loc[(left_mark, "S"), "node"])
        right_s = str(lookup.loc[(right_mark, "S"), "node"])
        shared_dimension = int(parameters["state_dimension"][left_mark])
        if int(parameters["state_dimension"][right_mark]) != shared_dimension:
            raise ValueError("Paired marks must share their preregistered state dimension")
        truth(
            left_s,
            right_s,
            "cross_assay_shared_state",
            f"{left_mark}|{right_mark}:S-S_shared_state",
            f"{left_mark}|{right_mark}",
            primary=True,
            node_set1=pure_state_nodes(left_mark, shared_dimension),
            node_set2=pure_state_nodes(right_mark, shared_dimension),
            estimand="latent_loading_equivalence_class",
        )
        combinations = [
            (left_role, right_role)
            for left_role in ("N1", "N2")
            for right_role in ("N1", "N2")
        ]
        left_role, right_role = min(
            combinations,
            key=lambda pair: abs(
                float(lookup.loc[(left_mark, pair[0]), "detection_probability"])
                - float(lookup.loc[(right_mark, pair[1]), "detection_probability"])
            ),
        )
        truth(
            str(lookup.loc[(left_mark, left_role), "node"]),
            str(lookup.loc[(right_mark, right_role), "node"]),
            "sentinel_null",
            f"{left_mark}|{right_mark}:{left_role}-{right_role}_detection_matched_null",
            f"{left_mark}|{right_mark}",
        )

    slow_mark = str(parameters["slow_mark"])
    fast_mark = str(parameters["fast_mark"])
    slow_node = str(lookup.loc[(slow_mark, "P"), "node"])
    fast_node = str(lookup.loc[(fast_mark, "P"), "node"])
    slow_dimension = int(parameters["state_dimension"][slow_mark])
    slow_loading_group = tuple(
        node
        for mark in MARKS
        for node in pure_state_nodes(mark, slow_dimension)
    )
    truth(
        slow_node,
        GENE,
        "slow_peak_gene",
        f"{slow_mark}:P->gene_slow_state_pairable",
        slow_mark,
        parent=slow_node,
        child=GENE,
        primary=True,
        orientation_credit=False,
        node_set1=slow_loading_group,
        node_set2=(GENE,),
        estimand="latent_loading_equivalence_class_support_hypothesis",
    )
    truth(
        fast_node,
        GENE,
        "fast_peak_gene_challenge",
        f"{fast_mark}:P->gene_fast_unpaired_challenge",
        fast_mark,
        parent=fast_node,
        child=GENE,
        primary=False,
        orientation_credit=False,
    )
    truth_edges = pd.DataFrame(truth_rows)

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    def dag_node(
        node: str,
        node_type: str,
        assay: str = "",
        observed: bool = False,
        mark: str = "",
        role: str = "",
    ) -> None:
        node_rows.append(
            {
                "node": node,
                "node_type": node_type,
                "assay": assay,
                "observed": observed,
                "mark": mark,
                "role": role,
            }
        )

    def dag_edge(
        source: str,
        target: str,
        mechanism: str,
        coefficient: float | str,
        *,
        condition: str = "always",
        active_in_world: bool = True,
    ) -> None:
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "mechanism": mechanism,
                "coefficient": coefficient,
                "condition": condition,
                "active_in_world": bool(active_in_world),
            }
        )

    # Plate-expanded state/background mechanisms.  Every assay samples a
    # distinct cell population, so its latent-state variables are separate
    # instances even though all assays share the same structural template.
    for modality in MODALITIES:
        population_node = f"population_fraction__{modality}"
        mode_node = f"mixture_mode__{modality}"
        technical_node = f"technical_scale__{modality}"
        depth_node = f"depth__{modality}"
        total_library_node = f"observed_library_size__{modality}"
        dag_node(population_node, "population_parameter", modality)
        dag_node(mode_node, "latent_mixture_mode", modality)
        dag_node(technical_node, "latent_technical_scale", modality)
        dag_node(depth_node, "observed_background_depth", modality, True)
        dag_node(total_library_node, "observed_total_library_size", modality, True)
        dag_edge(population_node, mode_node, "mixture_probability", "assay_fraction")
        for dimension in range(3):
            state_node = f"state__{modality}__dim{dimension}"
            proxy_node = f"background_state_proxy__{modality}__dim{dimension}"
            proxy_noise_node = f"background_proxy_noise__{modality}__dim{dimension}"
            dag_node(state_node, "latent_state", modality)
            dag_node(proxy_node, "latent_background_state_proxy", modality)
            dag_node(proxy_noise_node, "latent_background_proxy_noise", modality)
            dag_edge(mode_node, state_node, "mixture_center", "mode_specific")
            dag_edge(
                state_node,
                proxy_node,
                "background_proxy_identity",
                1.0,
            )
            dag_edge(
                proxy_noise_node,
                proxy_node,
                "assay_private_background_proxy_noise",
                float(
                    parameters["background_observation_model"][
                        "background_state_proxy_noise_sd"
                    ]
                ),
                condition="background_integration_stress",
                active_in_world=bool(
                    parameters["design"]["background_integration_stress"]
                ),
            )
        dag_edge(
            f"state__{modality}__dim0",
            f"state__{modality}__dim1",
            "within_state_dependence",
            0.22,
        )
        n_background = int(parameters["n_background_features_by_modality"][modality])
        for feature_index in range(n_background):
            background_node = f"background_count__{modality}__{feature_index:03d}"
            dag_node(background_node, "observed_background_count", modality, True)
            for dimension in range(3):
                dag_edge(
                    f"background_state_proxy__{modality}__dim{dimension}",
                    background_node,
                    "assay_private_background_loading",
                    "drawn_private_loading",
                )
            dag_edge(technical_node, background_node, "technical_scale", 1.0)
            dag_edge(background_node, depth_node, "deterministic_sum", 1.0)
        dag_edge(depth_node, total_library_node, "background_component", 1.0)

    dag_node("activity__gene", "molecular_activity", "rna", False, "RNA", "gene")
    dag_node("count__gene", "observed_gene_count", "rna", True, "RNA", "gene")
    dag_node(GENE, "observed_focal", "rna", True, "RNA", "gene")
    dag_edge("activity__gene", "count__gene", "negative_binomial_mean", 0.48)
    dag_edge("depth__rna", "count__gene", "technical_observation", 0.10)
    dag_edge("count__gene", GENE, "log1p_transform", 1.0)
    dag_edge("count__gene", "observed_library_size__rna", "focal_component", 1.0)
    dag_edge("state__rna__dim2", "activity__gene", "baseline_state", 0.28)

    for mark in MARKS:
        modality = f"chrom_{mark}"
        observed_nodes = {role: str(lookup.loc[(mark, role), "node"]) for role in ROLES}
        for role in ROLES:
            activity_node = f"activity__{modality}__{role}"
            dag_node(activity_node, "molecular_activity", modality, False, mark, role)
            dag_node(observed_nodes[role], "observed_focal", modality, True, mark, role)
            dag_edge(activity_node, observed_nodes[role], "sparse_observation", 0.78)
            dag_edge(f"depth__{modality}", observed_nodes[role], "technical_observation", 0.16)
            dag_edge(
                observed_nodes[role],
                f"observed_library_size__{modality}",
                "focal_component",
                1.0,
            )
        dimension = int(parameters["state_dimension"][mark])
        q_dimension = int(parameters["q_dimension"][mark])
        dag_edge(
            f"state__{modality}__dim{dimension}",
            f"activity__{modality}__P",
            "shared_state",
            float(parameters["state_sign"][mark])
            * float(parameters["p_state_loading"][mark]),
        )
        dag_edge(
            f"state__{modality}__dim{dimension}",
            f"activity__{modality}__S",
            "shared_state",
            float(parameters["state_sign"][mark]) * 1.05,
        )
        dag_edge(
            f"state__{modality}__dim{q_dimension}",
            f"activity__{modality}__Q",
            "state_loading",
            float(parameters["q_sign"][mark]) * 0.85,
        )
        dag_edge(
            f"activity__{modality}__P",
            f"activity__{modality}__C",
            "direct",
            float(parameters["beta_p_to_c"][mark]),
        )
        dag_edge(
            f"activity__{modality}__Q",
            f"activity__{modality}__C",
            "direct",
            float(parameters["beta_q_to_c"][mark]),
        )
        dag_edge(
            observed_nodes["P"],
            observed_nodes["C"],
            "observed_coaccessibility_log_odds",
            float(parameters["gamma_p_to_c_detection_log_odds"][mark]),
        )
        dag_edge(
            observed_nodes["Q"],
            observed_nodes["C"],
            "observed_coaccessibility_log_odds",
            float(parameters["gamma_q_to_c_detection_log_odds"][mark]),
        )

    for mark in (slow_mark, fast_mark):
        counterfactual = f"activity__rna_counterfactual__{mark}__P"
        dag_node(counterfactual, "counterfactual_molecular_activity", "rna", False, mark, "P")
        dag_edge(
            f"state__rna__dim{int(parameters['state_dimension'][mark])}",
            counterfactual,
            "shared_state",
            float(parameters["state_sign"][mark])
            * float(parameters["p_state_loading"][mark]),
        )
    dag_edge(
        f"activity__rna_counterfactual__{slow_mark}__P",
        "activity__gene",
        "slow_state_pairable_direct",
        float(parameters["slow_peak_gene_coefficient"]),
    )
    dag_edge(
        f"activity__rna_counterfactual__{fast_mark}__P",
        "activity__gene",
        "fast_residual_unpaired_direct",
        float(parameters["fast_peak_gene_coefficient"]),
    )
    return truth_edges, pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def generate_world(config: SimulationV4Config) -> _World:
    """Generate one world in memory without writing artifacts."""

    parameters, roles = _draw_parameter_table(config)
    design = dict(parameters["design"])
    cell_ids: dict[str, NDArray[np.str_]] = {}
    latent_states: dict[str, FloatArray] = {}
    latent_modes: dict[str, IntArray] = {}
    background_state_proxy: dict[str, FloatArray] = {}
    background_proxy_noise: dict[str, FloatArray] = {}
    background_technical_scale: dict[str, FloatArray] = {}
    background: dict[str, FloatArray] = {}
    background_depth: dict[str, FloatArray] = {}
    parameters["n_background_features_by_modality"] = {}

    for modality_index, modality in enumerate(MODALITIES):
        rng = _seeded_rng(config.seed, 200 + modality_index)
        n_cells = int(parameters["cell_counts"][modality])
        prefix = "rna" if modality == "rna" else modality.removeprefix("chrom_").lower()
        cell_ids[modality] = np.asarray(
            [f"{prefix}_w{config.seed:04d}_{index:06d}" for index in range(n_cells)]
        )
        z, modes = _draw_latent_state(
            n_cells,
            float(parameters["population_fractions"][modality]),
            rng,
        )
        latent_states[modality] = z
        latent_modes[modality] = modes
        proxy_rng = _seeded_rng(config.seed, 300 + modality_index)
        proxy_noise_sd = float(
            parameters["background_observation_model"][
                "background_state_proxy_noise_sd"
            ]
        )
        proxy_noise = np.asarray(
            proxy_rng.normal(scale=proxy_noise_sd, size=z.shape),
            dtype=np.float64,
        )
        proxy = np.asarray(z + proxy_noise, dtype=np.float64)
        background_proxy_noise[modality] = proxy_noise
        background_state_proxy[modality] = proxy
        # Unequal assay-private feature counts make the no-feature-link boundary
        # explicit while retaining a common latent geometry.
        n_features = config.n_background_features + (modality_index % 3)
        parameters["n_background_features_by_modality"][modality] = int(n_features)
        matrix, depth, technical_scale = _draw_background(
            proxy,
            modality=modality,
            n_features=n_features,
            rng=rng,
        )
        background[modality] = matrix
        background_depth[modality] = depth
        background_technical_scale[modality] = technical_scale

    focal = _draw_focal_tables(
        parameters,
        roles,
        cell_ids,
        latent_states,
        background_depth,
        config.seed,
    )
    truth_edges, dag_nodes, dag_edges = _truth_and_dag(parameters, roles)
    return _World(
        config=config,
        design=design,
        parameters=parameters,
        cell_ids=cell_ids,
        latent_states=latent_states,
        latent_modes=latent_modes,
        background_state_proxy=background_state_proxy,
        background_proxy_noise=background_proxy_noise,
        background_technical_scale=background_technical_scale,
        background=background,
        background_depth=background_depth,
        focal=focal,
        roles=roles,
        truth_edges=truth_edges,
        dag_nodes=dag_nodes,
        dag_edges=dag_edges,
    )


def _alignment(world: _World) -> Any:
    config = state_alignment.AlignmentConfig(
        n_components=world.config.latent_dim,
        n_microstates=world.config.n_microstates,
        min_cells_per_microstate=world.config.min_cells_per_microstate,
        derange=bool(world.config.alignment_derangement_diagnostic),
        random_seed=world.config.seed,
    )
    # This is the complete information boundary.  Do not add cells, labels,
    # focal tables, feature names, or any scorer-only object to this call.
    return state_alignment.align_cross_fitted_background_states(
        world.background,
        world.background_depth,
        reference_modality="rna",
        n_feature_folds=2,
        config=config,
    )


def _base_microstate_alignment(alignment: Any) -> Any:
    """Return the preregistered fold used only for metacell stratification.

    Cross-fitted consensus coordinates have no single cluster labeling.  Fold
    zero is fixed in advance as the base result; selecting it does not inspect
    focal observations, scorer truth, or fold performance.
    """

    fold_results = getattr(alignment, "fold_results", ())
    if not fold_results:
        raise TypeError("Cross-fitted alignment result must expose fold_results")
    return fold_results[0]


def _aligned_labels(alignment: Any, modality: str) -> IntArray:
    base = _base_microstate_alignment(alignment)
    local = np.asarray(base.microstate_labels[modality], dtype=np.int64)
    assignment = np.asarray(base.microstate_assignments[modality], dtype=np.int64)
    return np.asarray(assignment[local], dtype=np.int64)


def _build_nested_run(
    world: _World,
    alignment: Any,
    *,
    target_cells: int,
    aggregation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create complete inferred-state metacells while consuming every cell."""

    labels = {modality: _aligned_labels(alignment, modality) for modality in MODALITIES}
    base_alignment = _base_microstate_alignment(alignment)
    n_states = int(base_alignment.provenance["effective_n_microstates"])
    rng = _seeded_rng(world.config.seed, 10_000 + 101 * target_cells + aggregation_seed)
    membership_rows: list[dict[str, str]] = []
    observed_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    used = {
        modality: np.zeros(len(world.cell_ids[modality]), dtype=bool)
        for modality in MODALITIES
    }

    for state_index in range(n_states):
        state_indices: dict[str, IntArray] = {}
        for modality in MODALITIES:
            indices = np.flatnonzero(labels[modality] == state_index)
            rng.shuffle(indices)
            state_indices[modality] = np.asarray(indices, dtype=np.int64)
        if any(len(indices) == 0 for indices in state_indices.values()):
            raise RuntimeError("Alignment returned an empty matched microstate")
        n_groups = min(
            max(1, int(round(len(indices) / target_cells)))
            for indices in state_indices.values()
        )
        partitions = {
            modality: [
                np.asarray(chunk, dtype=np.int64)
                for chunk in np.array_split(indices, n_groups)
            ]
            for modality, indices in state_indices.items()
        }
        for group_index in range(n_groups):
            sample_id = (
                f"cells{target_cells}_seed{aggregation_seed}_"
                f"state{state_index:02d}_group{group_index:03d}"
            )
            observed_row: dict[str, Any] = {"sample_id": sample_id}
            state_row: dict[str, Any] = {"sample_id": sample_id}
            assay_centroids: list[FloatArray] = []
            assay_crossfit_uncertainty: list[float] = []
            for modality in MODALITIES:
                indices = partitions[modality][group_index]
                used[modality][indices] = True
                membership_rows.extend(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "cell": str(world.cell_ids[modality][index]),
                    }
                    for index in indices
                )
                table = world.focal[modality]
                biological = [
                    column
                    for column in table.columns
                    if column.startswith("chr") or column.startswith("expr__")
                ]
                for column in biological:
                    observed_row[column] = float(table.iloc[indices][column].mean())
                channel = "rna" if modality == "rna" else modality.removeprefix("chrom_")
                observed_row[f"libsize_background__{channel}"] = float(
                    world.background_depth[modality][indices].mean()
                )
                centroid = np.asarray(alignment.embeddings[modality][indices].mean(axis=0))
                assay_centroids.append(centroid)
                for dimension, value in enumerate(centroid):
                    state_row[f"state__{modality}__dim{dimension}"] = float(value)
                uncertainty = float(
                    np.sqrt(
                        np.mean(
                            np.asarray(
                                alignment.embedding_uncertainty[modality][indices],
                                dtype=np.float64,
                            )
                            ** 2
                        )
                    )
                )
                state_row[f"state__{modality}__crossfit_uncertainty"] = uncertainty
                assay_crossfit_uncertainty.append(uncertainty)
            stacked = np.vstack(assay_centroids)
            consensus = stacked.mean(axis=0)
            for dimension, value in enumerate(consensus):
                state_row[f"state__consensus__dim{dimension}"] = float(value)
            state_row["state__assay_uncertainty"] = float(
                np.sqrt(np.mean((stacked - consensus[None, :]) ** 2))
            )
            state_row["state__crossfit_uncertainty"] = float(
                np.sqrt(np.mean(np.square(assay_crossfit_uncertainty)))
            )
            observed_rows.append(observed_row)
            state_rows.append(state_row)

    cells_used = {modality: int(flags.sum()) for modality, flags in used.items()}
    cells_total = {modality: int(len(flags)) for modality, flags in used.items()}
    if cells_used != cells_total:
        raise RuntimeError("Nested aggregation failed to retain every cell exactly once")
    provenance = {
        "schema_version": "multimodal_simulation_v4.aggregation.v1",
        "target_cells_per_assay": target_cells,
        "aggregation_seed": aggregation_seed,
        "assignment_source": "background_only_cross_fitted_unpaired_state_alignment_fold0",
        "coordinate_source": "two_feature_fold_consensus",
        "membership_is_analysis_derived": True,
        "paired_cell_identity_used": False,
        "feature_identity_links_used": False,
        "focal_values_used_for_assignment": False,
        "simulator_labels_used": False,
        "all_cells_used_exactly_once": True,
        "cells_used": cells_used,
        "cells_total": cells_total,
        "n_complete_metacells": len(observed_rows),
    }
    return (
        pd.DataFrame(observed_rows),
        pd.DataFrame(state_rows),
        pd.DataFrame(membership_rows),
        provenance,
    )


def _write_discovery(world: _World, alignment: Any, directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    embedding_rows: list[pd.DataFrame] = []
    for modality in MODALITIES:
        background_path = directory / f"background_matrix__{modality}.npz"
        np.savez_compressed(
            background_path,
            matrix=world.background[modality],
            cell_ids=world.cell_ids[modality],
            background_depth=world.background_depth[modality],
        )
        written.append(str(background_path.relative_to(directory)))
        focal_path = directory / f"observed_focal__{modality}.tsv"
        world.focal[modality].to_csv(focal_path, sep="\t", index=False)
        written.append(str(focal_path.relative_to(directory)))

        embedding = pd.DataFrame(
            {
                "modality": modality,
                "cell": world.cell_ids[modality],
                **{
                    f"integration__dim{dimension}": alignment.embeddings[modality][:, dimension]
                    for dimension in range(alignment.embeddings[modality].shape[1])
                },
                "integration_uncertainty": alignment.embedding_uncertainty[modality],
            }
        )
        embedding_rows.append(embedding)
    embeddings_path = directory / "analysis_embeddings.tsv"
    pd.concat(embedding_rows, ignore_index=True).to_csv(
        embeddings_path, sep="\t", index=False
    )
    written.append(str(embeddings_path.relative_to(directory)))
    provenance_path = directory / "alignment_provenance.json"
    _write_json(
        provenance_path,
        _discovery_safe_alignment_provenance(alignment.provenance),
    )
    written.append(str(provenance_path.relative_to(directory)))

    for target_cells in world.config.aggregation_cell_targets:
        for aggregation_seed in world.config.aggregation_seeds:
            run_directory = directory / "nested" / f"cells{target_cells}_seed{aggregation_seed}"
            run_directory.mkdir(parents=True, exist_ok=True)
            observed, state, membership, provenance = _build_nested_run(
                world,
                alignment,
                target_cells=int(target_cells),
                aggregation_seed=int(aggregation_seed),
            )
            for filename, table in (
                ("observed_metacell_matrix.tsv", observed),
                ("background_state_centroids.tsv", state),
                ("analysis_membership.tsv", membership),
            ):
                path = run_directory / filename
                table.to_csv(path, sep="\t", index=False)
                written.append(str(path.relative_to(directory)))
            aggregation_path = run_directory / "aggregation_provenance.json"
            _write_json(aggregation_path, provenance)
            written.append(str(aggregation_path.relative_to(directory)))

    manifest = {
        "schema_version": "multimodal_simulation_v4.discovery.v2",
        "modalities": list(MODALITIES),
        "assay_cells_are_independently_sampled": True,
        "information_boundary": {
            "available": [
                "observed_unpaired_assay_variables",
                "assay_labels",
                "observed_background_depth",
                "background_only_state_estimates",
                "background_only_per_cell_integration_uncertainty",
                "background_derived_metacell_membership",
            ],
            "unavailable": [
                "simulator_state",
                "simulator_population_labels",
                "focal_role_assignments",
                "generative_graph",
                "paired_cell_identity",
                "cross_assay_feature_links",
                "external_information",
            ],
        },
        "nested_design": {
            "cell_targets": list(world.config.aggregation_cell_targets),
            "seeds": list(world.config.aggregation_seeds),
            "all_cells_retained_in_each_run": True,
        },
        "files": sorted(written),
        "file_sha256": {
            relative: _file_sha256(directory / relative)
            for relative in sorted(written)
        },
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    written.append(str(manifest_path.relative_to(directory)))
    return written


def _write_scorer(world: _World, directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    tables = (
        ("truth_typed_edges.tsv", world.truth_edges),
        ("role_assignments.tsv", world.roles),
        ("generative_dag_nodes.tsv", world.dag_nodes),
        ("generative_dag_edges.tsv", world.dag_edges),
    )
    for filename, table in tables:
        path = directory / filename
        table.to_csv(path, sep="\t", index=False)
        written.append(filename)
    parameters_path = directory / "world_parameters.json"
    _write_json(parameters_path, world.parameters)
    written.append(parameters_path.name)
    for modality in MODALITIES:
        path = directory / f"cell_ground_truth__{modality}.npz"
        np.savez_compressed(
            path,
            cell_ids=world.cell_ids[modality],
            latent_state=world.latent_states[modality],
            population_mode=world.latent_modes[modality],
            background_state_proxy=world.background_state_proxy[modality],
            background_proxy_noise=world.background_proxy_noise[modality],
            background_technical_scale=world.background_technical_scale[modality],
        )
        written.append(path.name)
    manifest_path = directory / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "multimodal_simulation_v4.scorer.v2",
            "scorer_only": True,
            "must_not_be_read_by_discovery": True,
            "truth_types": sorted(TRUTH_TYPES),
            "files": sorted(written),
            "file_sha256": {
                relative: _file_sha256(directory / relative)
                for relative in sorted(written)
            },
        },
    )
    written.append(manifest_path.name)
    return written


def simulate_world(config: SimulationV4Config, out_dir: Path | str) -> dict[str, Any]:
    """Generate, align, aggregate, and write one v4 simulation world."""

    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    world = generate_world(config)
    alignment = _alignment(world)
    discovery_files = _write_discovery(world, alignment, output / "discovery")
    scorer_files = _write_scorer(world, output / "scorer")
    summary = {
        "schema_version": "multimodal_simulation_v4.run.v2",
        "seed": config.seed,
        "discovery_directory": "discovery",
        "scorer_directory": "scorer",
        "n_modalities": len(MODALITIES),
        "cell_counts": world.parameters["cell_counts"],
        "nested_runs": len(config.aggregation_cell_targets) * len(config.aggregation_seeds),
        "discovery_file_count": len(discovery_files),
        "scorer_file_count": len(scorer_files),
        "information_domains_separated_by_directory": True,
        "raw_only_preparer_required_for_discovery": True,
        "discovery_manifest_sha256": _file_sha256(output / "discovery" / "manifest.json"),
        "scorer_manifest_sha256": _file_sha256(output / "scorer" / "manifest.json"),
        "domain_payloads_content_addressed": True,
    }
    _write_json(output / "run_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-cells", type=int, default=None)
    parser.add_argument("--n-background-features", type=int, default=64)
    parser.add_argument("--n-microstates", type=int, default=24)
    parser.add_argument("--min-cells-per-microstate", type=int, default=20)
    parser.add_argument(
        "--alignment-derangement-diagnostic",
        action="store_true",
        help=(
            "Apply the supplied-coordinate derangement invariance diagnostic; "
            "this is deliberately separate from the DOE factors."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SimulationV4Config(
        seed=args.seed,
        base_cells_override=args.base_cells,
        n_background_features=args.n_background_features,
        n_microstates=args.n_microstates,
        min_cells_per_microstate=args.min_cells_per_microstate,
        alignment_derangement_diagnostic=args.alignment_derangement_diagnostic,
    )
    summary = simulate_world(config, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AGGREGATION_CELL_TARGETS",
    "AGGREGATION_SEEDS",
    "BACKGROUND_PROXY_NOISE_SD",
    "GENE",
    "MARKS",
    "MODALITIES",
    "ROLES",
    "SimulationV4Config",
    "TRUTH_TYPES",
    "design_for_seed",
    "generate_world",
    "simulate_world",
]
