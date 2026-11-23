#!/usr/bin/env python3
"""Rebuild leakage-safe CA-BOSS v4 inputs from observed discovery artifacts.

This command accepts one simulator ``discovery/`` directory and reads exactly
two whitelisted artifact families:

* ``background_matrix__<modality>.npz``
* ``observed_focal__<modality>.tsv``

It deliberately does not read a supplied embedding, alignment provenance,
manifest, metacell membership, scorer directory, or truth artifact.  Instead it
recomputes a two-feature-fold unpaired background alignment with derangement
disabled, then creates complete 24/14/8-cell metacell analyses for seeds 0 and
1.  Every observed cell is assigned exactly once in every nested analysis.

The narrow input contract is what makes a simulator arm with deliberately
shuffled *supplied* alignment recoverable: CA-BOSS reconstructs state from the
raw observed background rather than inheriting the shuffled artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import pandas as pd

try:
    import soft_distributional_alignment as soft_alignment
    import unpaired_state_alignment as state_alignment
except ImportError as exc:  # pragma: no cover - partial checkout only
    raise ImportError(
        "prepare_ca_boss_v4_inputs requires unpaired_state_alignment.py and "
        "soft_distributional_alignment.py."
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
CELL_TARGETS = (24, 14, 8)
AGGREGATION_SEEDS = (0, 1)
_BACKGROUND_PREFIX = "background_matrix__"
_FOCAL_PREFIX = "observed_focal__"
_FORBIDDEN_INPUT_COMPONENTS = {
    "scorer",
    "truth",
    "ground_truth",
    "oracle",
}
_FORBIDDEN_FOCAL_TOKENS = (
    "truth",
    "true_",
    "role",
    "parent",
    "child",
    "latent",
    "oracle",
    "scorer",
    "correspondence",
    "pairing",
    "population_mode",
)


@dataclass(frozen=True)
class PrepareConfig:
    """Frozen, observable-only preprocessing settings."""

    n_components: int = 3
    # TRAIN seed 30001 selected the coarser observable alignment because it
    # reduced cross-fold/assay Procrustes residual without consulting truth.
    # The downstream 24/14/8 values remain aggregation resolutions, not state
    # cluster counts.
    n_microstates: int = 6
    min_cells_per_microstate: int = 20
    alignment_seed: int = 0
    n_feature_folds: int = 2
    alignment_mode: str = "hard"
    soft_n_microstates: int = 24
    soft_max_rotation_dispersion: float = 0.35
    soft_low_confidence_action: str = "partial_soft"
    cell_targets: tuple[int, ...] = CELL_TARGETS
    aggregation_seeds: tuple[int, ...] = AGGREGATION_SEEDS

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least two")
        if self.n_microstates <= self.n_components:
            raise ValueError("n_microstates must exceed n_components")
        if self.min_cells_per_microstate < 2:
            raise ValueError("min_cells_per_microstate must be at least two")
        if self.alignment_seed < 0:
            raise ValueError("alignment_seed must be non-negative")
        if self.n_feature_folds != 2:
            raise ValueError("CA-BOSS v4 input preparation is fixed to two feature folds")
        if self.alignment_mode not in {"hard", "soft"}:
            raise ValueError("alignment_mode must be 'hard' or 'soft'")
        if self.soft_n_microstates <= self.n_components:
            raise ValueError("soft_n_microstates must exceed n_components")
        if (
            not np.isfinite(self.soft_max_rotation_dispersion)
            or self.soft_max_rotation_dispersion < 0.0
        ):
            raise ValueError("soft_max_rotation_dispersion must be finite and non-negative")
        if self.soft_low_confidence_action not in {
            "partial_soft",
            "fallback_hard",
            "abstain",
        }:
            raise ValueError(
                "soft_low_confidence_action must be 'partial_soft', "
                "'fallback_hard', or 'abstain'"
            )
        if tuple(self.cell_targets) != CELL_TARGETS:
            raise ValueError("CA-BOSS v4 requires cell targets (24, 14, 8)")
        if tuple(self.aggregation_seeds) != AGGREGATION_SEEDS:
            raise ValueError("CA-BOSS v4 requires aggregation seeds (0, 1)")


@dataclass
class DiscoveryInputs:
    """Validated observed inputs, with focal rows ordered like background rows."""

    root: Path
    matrices: dict[str, FloatArray]
    depths: dict[str, FloatArray]
    cell_ids: dict[str, NDArray[np.str_]]
    focal: dict[str, pd.DataFrame]
    biological_columns: dict[str, tuple[str, ...]]
    input_hashes: dict[str, str]


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_discovery_root(discovery_dir: Path | str) -> Path:
    lexical = Path(discovery_dir).absolute()
    if lexical.name.casefold() != "discovery":
        raise ValueError("Input must be the simulator/world discovery/ directory")
    if not lexical.exists() or not lexical.is_dir():
        raise ValueError("Input discovery/ directory does not exist")
    resolved = lexical.resolve(strict=True)
    if resolved.name.casefold() != "discovery":
        raise ValueError("Input discovery/ may not be a symlink to another data domain")
    forbidden = {
        component.casefold()
        for component in resolved.parts
        if component.casefold() in _FORBIDDEN_INPUT_COMPONENTS
    }
    if forbidden:
        raise ValueError(
            f"Refusing scorer/truth input path; forbidden components={sorted(forbidden)}"
        )
    return resolved


def _whitelisted_files(root: Path, prefix: str, suffix: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.glob(f"{prefix}*{suffix}")):
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError(f"Input artifact must be a direct regular file: {path.name}")
        modality = path.name.removeprefix(prefix).removesuffix(suffix)
        if not modality or modality in files:
            raise ValueError(f"Invalid or duplicate modality artifact: {path.name}")
        files[modality] = path
    if set(files) != set(MODALITIES):
        missing = sorted(set(MODALITIES) - set(files))
        extra = sorted(set(files) - set(MODALITIES))
        raise ValueError(
            f"Expected exactly RNA plus six CUT&Tag modalities; missing={missing}, extra={extra}"
        )
    return files


def _load_background(path: Path) -> tuple[FloatArray, NDArray[np.str_], FloatArray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"matrix", "cell_ids", "background_depth"}:
                raise ValueError(
                    f"{path.name} must contain only matrix, cell_ids, and background_depth"
                )
            matrix = np.asarray(archive["matrix"], dtype=np.float64)
            raw_ids = np.asarray(archive["cell_ids"])
            depth = np.asarray(archive["background_depth"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load safe numeric background archive {path.name}") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 4:
        raise ValueError(f"{path.name} needs at least four cells and four features")
    if raw_ids.ndim != 1 or depth.ndim != 1:
        raise ValueError(f"{path.name} cell_ids and background_depth must be vectors")
    if len(raw_ids) != matrix.shape[0] or len(depth) != matrix.shape[0]:
        raise ValueError(f"{path.name} row, cell ID, and depth counts disagree")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError(f"{path.name} matrix must be finite and non-negative")
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError(f"{path.name} background depth must be finite and positive")
    if not np.allclose(matrix.sum(axis=1), depth, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{path.name} depth is not the observed background row sum")
    cell_ids = np.asarray(raw_ids, dtype=str)
    if np.any(np.char.str_len(cell_ids) == 0) or len(set(cell_ids.tolist())) != len(cell_ids):
        raise ValueError(f"{path.name} cell IDs must be non-empty and unique")
    return matrix, cell_ids, depth


def _biological_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        str(column)
        for column in frame.columns
        if str(column).startswith("chr") or str(column).startswith("expr__")
    )


def _load_focal(
    path: Path,
    *,
    modality: str,
    expected_ids: NDArray[np.str_],
    expected_depth: FloatArray,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    try:
        frame = pd.read_csv(path, sep="\t")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read observed focal table {path.name}") from exc
    required = {
        "cell",
        "modality",
        "observed_background_library_size",
        "observed_library_size",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns {missing}")
    lowered = {str(column).casefold(): str(column) for column in frame.columns}
    forbidden = sorted(
        original
        for lowered_name, original in lowered.items()
        if any(token in lowered_name for token in _FORBIDDEN_FOCAL_TOKENS)
    )
    if forbidden:
        raise ValueError(f"{path.name} contains forbidden discovery columns {forbidden}")
    biological = _biological_columns(frame)
    allowed = required | set(biological)
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise ValueError(f"{path.name} contains non-whitelisted columns {unexpected}")
    if not biological:
        raise ValueError(f"{path.name} contains no observed biological variables")
    if frame["cell"].isna().any():
        raise ValueError(f"{path.name} has missing cell IDs")
    frame = frame.copy()
    frame["cell"] = frame["cell"].astype(str)
    if frame["cell"].duplicated().any():
        raise ValueError(f"{path.name} cell IDs must be unique")
    if set(frame["cell"]) != set(expected_ids.tolist()):
        raise ValueError(f"{path.name} cell IDs disagree with its background archive")
    if set(frame["modality"].astype(str)) != {modality}:
        raise ValueError(f"{path.name} modality column does not equal {modality!r}")
    frame = frame.set_index("cell", drop=False).loc[expected_ids.tolist()].reset_index(drop=True)
    numeric = [
        "observed_background_library_size",
        "observed_library_size",
        *biological,
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path.name} numeric observations must be finite")
    focal_depth = frame["observed_background_library_size"].to_numpy(dtype=np.float64)
    if not np.allclose(focal_depth, expected_depth, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{path.name} background depth disagrees with its archive")
    if np.any(
        frame["observed_library_size"].to_numpy(dtype=np.float64) < focal_depth
    ):
        raise ValueError(f"{path.name} total library size is below background depth")
    if modality == "rna":
        if biological != ("expr__SIMG",):
            raise ValueError("RNA focal table must contain exactly expr__SIMG")
    else:
        mark = modality.removeprefix("chrom_")
        if len(biological) != 6 or not all(
            column.startswith("chr") and column.endswith(f"__{mark}")
            for column in biological
        ):
            raise ValueError(f"{modality} focal variables must carry the assay mark suffix")
    return frame, biological


def load_discovery_inputs(discovery_dir: Path | str) -> DiscoveryInputs:
    """Load the two whitelisted observed artifact families and nothing else."""

    root = _validated_discovery_root(discovery_dir)
    background_paths = _whitelisted_files(root, _BACKGROUND_PREFIX, ".npz")
    focal_paths = _whitelisted_files(root, _FOCAL_PREFIX, ".tsv")
    matrices: dict[str, FloatArray] = {}
    depths: dict[str, FloatArray] = {}
    cell_ids: dict[str, NDArray[np.str_]] = {}
    focal: dict[str, pd.DataFrame] = {}
    biological_columns: dict[str, tuple[str, ...]] = {}
    input_hashes: dict[str, str] = {}
    seen_cells: set[str] = set()
    seen_features: set[str] = set()

    for modality in MODALITIES:
        matrix, ids, depth = _load_background(background_paths[modality])
        if not seen_cells.isdisjoint(ids.tolist()):
            raise ValueError("Cell IDs must be disjoint across independently sampled assays")
        seen_cells.update(ids.tolist())
        table, biological = _load_focal(
            focal_paths[modality],
            modality=modality,
            expected_ids=ids,
            expected_depth=depth,
        )
        overlap = seen_features.intersection(biological)
        if overlap:
            raise ValueError(f"Biological variable names repeat across assays: {sorted(overlap)}")
        seen_features.update(biological)
        matrices[modality] = matrix
        depths[modality] = depth
        cell_ids[modality] = ids
        focal[modality] = table
        biological_columns[modality] = biological
        input_hashes[background_paths[modality].name] = _sha256(background_paths[modality])
        input_hashes[focal_paths[modality].name] = _sha256(focal_paths[modality])
    return DiscoveryInputs(
        root=root,
        matrices=matrices,
        depths=depths,
        cell_ids=cell_ids,
        focal=focal,
        biological_columns=biological_columns,
        input_hashes=input_hashes,
    )


def _hard_alignment(inputs: DiscoveryInputs, config: PrepareConfig) -> Any:
    """Compute the fixed hard cross-fit baseline from observed backgrounds."""

    alignment_config = state_alignment.AlignmentConfig(
        n_components=config.n_components,
        n_microstates=config.n_microstates,
        min_cells_per_microstate=config.min_cells_per_microstate,
        derange=False,
        random_seed=config.alignment_seed,
    )
    return state_alignment.align_cross_fitted_background_states(
        inputs.matrices,
        inputs.depths,
        reference_modality="rna",
        n_feature_folds=2,
        config=alignment_config,
    )


def _soft_confidence_gate(
    provenance: Mapping[str, Any],
    config: PrepareConfig,
) -> dict[str, Any]:
    """Decide whether distributional rotations are sufficiently identified."""

    synchronization = provenance.get("synchronization", {})
    if isinstance(synchronization, Mapping):
        synchronized = synchronization.get("assay_synchronized")
        confidence = synchronization.get("assay_alignment_confidence")
        if isinstance(synchronized, Mapping) and isinstance(confidence, Mapping):
            failed: dict[str, str] = {}
            observed: dict[str, float] = {}
            for modality in MODALITIES:
                if modality == "rna":
                    continue
                if synchronized.get(modality) is not True:
                    failed[modality] = "rotation_synchronization_abstained"
                    continue
                try:
                    value = float(confidence[modality])
                except (KeyError, TypeError, ValueError):
                    failed[modality] = "missing_or_invalid_synchronization_confidence"
                    continue
                observed[modality] = value
                if not np.isfinite(value) or value <= 0.0:
                    failed[modality] = "nonpositive_synchronization_confidence"
            return {
                "metric": "confidence_gated_multi_assay_rotation_synchronization",
                "maximum_allowed": None,
                "observed_by_modality": observed,
                "failed_modalities": failed,
                "passed": not failed,
            }

    raw_diagnostics = provenance.get("distribution_diagnostics", {})
    if not isinstance(raw_diagnostics, Mapping):
        raw_diagnostics = {}
    failed: dict[str, str] = {}
    observed: dict[str, float] = {}
    for modality in MODALITIES:
        if modality == "rna":
            continue
        diagnostics = raw_diagnostics.get(modality)
        if not isinstance(diagnostics, Mapping):
            failed[modality] = "missing_distribution_diagnostics"
            continue
        try:
            dispersion = float(diagnostics["near_optimal_rotation_dispersion"])
        except (KeyError, TypeError, ValueError):
            failed[modality] = "missing_or_invalid_rotation_dispersion"
            continue
        observed[modality] = dispersion
        if not np.isfinite(dispersion):
            failed[modality] = "non_finite_rotation_dispersion"
        elif dispersion > config.soft_max_rotation_dispersion:
            failed[modality] = "rotation_dispersion_above_threshold"
    return {
        "metric": "near_optimal_rotation_dispersion",
        "maximum_allowed": config.soft_max_rotation_dispersion,
        "observed_by_modality": observed,
        "failed_modalities": failed,
        "passed": not failed,
    }


def _selected_alignment(
    hard: Any,
    *,
    embeddings: Mapping[str, FloatArray],
    uncertainty: Mapping[str, FloatArray],
    selection: Mapping[str, Any],
) -> Any:
    """Combine selected coordinates with hard baseline membership estimates."""

    if set(embeddings) != set(MODALITIES) or set(uncertainty) != set(MODALITIES):
        raise ValueError("Selected alignment must contain exactly the seven assay modalities")
    selected_embeddings: dict[str, FloatArray] = {}
    selected_uncertainty: dict[str, FloatArray] = {}
    for modality in MODALITIES:
        coordinates = np.asarray(embeddings[modality], dtype=np.float64)
        errors = np.asarray(uncertainty[modality], dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape != (
            len(hard.embeddings[modality]),
            hard.embeddings[modality].shape[1],
        ):
            raise ValueError(f"Selected {modality} embedding has an invalid shape")
        if errors.shape != (len(coordinates),):
            raise ValueError(f"Selected {modality} uncertainty has an invalid shape")
        if not np.isfinite(coordinates).all() or not np.isfinite(errors).all():
            raise ValueError(f"Selected {modality} coordinates must be finite")
        if np.any(errors < 0.0):
            raise ValueError(f"Selected {modality} uncertainty must be non-negative")
        selected_embeddings[modality] = coordinates
        selected_uncertainty[modality] = errors
    provenance = dict(hard.provenance)
    provenance["preparation_alignment_selection"] = dict(selection)
    return state_alignment.CrossFittedAlignmentResult(
        embeddings=selected_embeddings,
        embedding_uncertainty=selected_uncertainty,
        fold_results=hard.fold_results,
        provenance=provenance,
    )


def recompute_alignment(inputs: DiscoveryInputs, config: PrepareConfig) -> Any:
    """Cross-fit observed backgrounds and optionally select soft coordinates.

    Hard alignment is always the membership baseline.  In ``soft`` mode the
    distributional coordinates are accepted when every non-reference assay
    passes the observable synchronization gate.  With ``partial_soft``, good
    synchronized assays remain eligible for cross-assay axes; failed assays
    retain useful within-assay soft coordinates but are explicitly excluded
    from cross-assay clustering.  Whole-world hard fallback and explicit
    abstention remain available as diagnostic policies.
    """

    hard = _hard_alignment(inputs, config)
    if config.alignment_mode == "hard":
        return _selected_alignment(
            hard,
            embeddings=hard.embeddings,
            uncertainty=hard.embedding_uncertainty,
            selection={
                "requested_mode": "hard",
                "selected_mode": "hard",
                "soft_alignment_attempted": False,
                "soft_alignment_abstained": False,
                "low_confidence_action": config.soft_low_confidence_action,
                "confidence_gate": None,
            },
        )

    soft_config = soft_alignment.SoftAlignmentConfig(
        n_components=config.n_components,
        n_microstates=config.soft_n_microstates,
        min_cells_per_microstate=config.min_cells_per_microstate,
        n_feature_folds=config.n_feature_folds,
        random_seed=config.alignment_seed,
    )
    soft = soft_alignment.align_soft_distributional_background_states(
        inputs.matrices,
        inputs.depths,
        reference_modality="rna",
        config=soft_config,
    )
    confidence = _soft_confidence_gate(soft.provenance, config)
    failed_modalities = tuple(sorted(confidence["failed_modalities"]))
    synchronized_modalities = tuple(
        modality for modality in MODALITIES if modality not in failed_modalities
    )
    partial = bool(
        not confidence["passed"]
        and config.soft_low_confidence_action == "partial_soft"
    )
    selection = {
        "requested_mode": "soft",
        "selected_mode": (
            "soft" if confidence["passed"] else "soft_partial" if partial else "hard"
        ),
        "soft_alignment_attempted": True,
        "soft_alignment_abstained": not confidence["passed"],
        "cross_assay_axis_eligible_modalities": list(synchronized_modalities),
        "cross_assay_axis_abstained_modalities": list(failed_modalities),
        "low_confidence_action": config.soft_low_confidence_action,
        "confidence_gate": confidence,
        "soft_method": soft.provenance.get("method", ""),
        "soft_schema_version": soft.provenance.get("schema_version", ""),
        "soft_config": asdict(soft_config),
    }
    if confidence["passed"] or partial:
        return _selected_alignment(
            hard,
            embeddings=soft.embeddings,
            uncertainty=soft.embedding_uncertainty,
            selection=selection,
        )
    if config.soft_low_confidence_action == "abstain":
        failed = sorted(confidence["failed_modalities"])
        raise RuntimeError(
            "Soft distributional alignment confidence gate abstained; "
            f"failed modalities={failed}"
        )
    return _selected_alignment(
        hard,
        embeddings=hard.embeddings,
        uncertainty=hard.embedding_uncertainty,
        selection=selection,
    )


def _base_alignment(alignment: Any) -> Any:
    folds = getattr(alignment, "fold_results", ())
    if len(folds) != 2:
        raise TypeError("Expected exactly two cross-fitted base alignment results")
    return folds[0]


def _aligned_labels(alignment: Any, modality: str) -> IntArray:
    base = _base_alignment(alignment)
    local = np.asarray(base.microstate_labels[modality], dtype=np.int64)
    assignment = np.asarray(base.microstate_assignments[modality], dtype=np.int64)
    return np.asarray(assignment[local], dtype=np.int64)


def _alignment_selection(alignment: Any) -> Mapping[str, Any]:
    selection = alignment.provenance.get("preparation_alignment_selection", {})
    if not isinstance(selection, Mapping):
        raise TypeError("Alignment selection provenance is malformed")
    return selection


def _aggregation_rng(alignment_seed: int, target_cells: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([alignment_seed, 91_001, target_cells, seed])
    )


def build_complete_metacells(
    inputs: DiscoveryInputs,
    alignment: Any,
    *,
    target_cells: int,
    aggregation_seed: int,
    alignment_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate complete inferred-state groups, consuming every assay cell once."""

    base = _base_alignment(alignment)
    selection = _alignment_selection(alignment)
    selected_mode = str(selection.get("selected_mode", "hard"))
    n_states = int(base.provenance["effective_n_microstates"])
    labels = {modality: _aligned_labels(alignment, modality) for modality in MODALITIES}
    rng = _aggregation_rng(alignment_seed, target_cells, aggregation_seed)
    membership_rows: list[dict[str, str]] = []
    observed_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    used = {
        modality: np.zeros(len(inputs.cell_ids[modality]), dtype=bool)
        for modality in MODALITIES
    }

    for state_index in range(n_states):
        state_indices: dict[str, IntArray] = {}
        for modality in MODALITIES:
            indices = np.flatnonzero(labels[modality] == state_index)
            rng.shuffle(indices)
            state_indices[modality] = np.asarray(indices, dtype=np.int64)
        if any(len(indices) == 0 for indices in state_indices.values()):
            raise RuntimeError("Cross-fitted alignment produced an empty matched microstate")
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
            assay_fold_uncertainty: list[float] = []
            for modality in MODALITIES:
                indices = partitions[modality][group_index]
                used[modality][indices] = True
                membership_rows.extend(
                    {
                        "sample_id": sample_id,
                        "modality": modality,
                        "cell": str(inputs.cell_ids[modality][index]),
                    }
                    for index in indices
                )
                table = inputs.focal[modality]
                for column in inputs.biological_columns[modality]:
                    observed_row[column] = float(table.iloc[indices][column].mean())
                channel = "rna" if modality == "rna" else modality.removeprefix("chrom_")
                observed_row[f"libsize_background__{channel}"] = float(
                    inputs.depths[modality][indices].mean()
                )
                centroid = np.asarray(
                    alignment.embeddings[modality][indices].mean(axis=0),
                    dtype=np.float64,
                )
                assay_centroids.append(centroid)
                for dimension, value in enumerate(centroid):
                    state_row[f"state__{modality}__dim{dimension}"] = float(value)
                fold_uncertainty = float(
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
                state_row[f"state__{modality}__crossfit_uncertainty"] = fold_uncertainty
                assay_fold_uncertainty.append(fold_uncertainty)
            stacked = np.vstack(assay_centroids)
            consensus = stacked.mean(axis=0)
            for dimension, value in enumerate(consensus):
                state_row[f"state__consensus__dim{dimension}"] = float(value)
            state_row["state__assay_uncertainty"] = float(
                np.sqrt(np.mean((stacked - consensus[None, :]) ** 2))
            )
            state_row["state__crossfit_uncertainty"] = float(
                np.sqrt(np.mean(np.square(assay_fold_uncertainty)))
            )
            observed_rows.append(observed_row)
            state_rows.append(state_row)

    cells_used = {modality: int(flags.sum()) for modality, flags in used.items()}
    cells_total = {modality: int(len(flags)) for modality, flags in used.items()}
    if cells_used != cells_total:
        raise RuntimeError("Complete aggregation did not consume every observed cell")
    membership = pd.DataFrame(membership_rows)
    if membership.duplicated(["modality", "cell"]).any():
        raise RuntimeError("A cell was assigned more than once")
    provenance = {
        "schema_version": "prepare_ca_boss_v4_inputs.aggregation.v1",
        "target_cells_per_assay": target_cells,
        "aggregation_seed": aggregation_seed,
        "assignment_source": "recomputed_crossfit_fold0_background_microstates",
        "coordinate_source": (
            "recomputed_soft_distributional_background_coordinates"
            if selected_mode.startswith("soft")
            else "recomputed_two_feature_fold_background_consensus"
        ),
        "alignment_mode_requested": selection.get("requested_mode", "hard"),
        "alignment_mode_selected": selected_mode,
        "soft_alignment_abstained": bool(
            selection.get("soft_alignment_abstained", False)
        ),
        "cross_assay_axis_eligible_modalities": list(
            selection.get(
                "cross_assay_axis_eligible_modalities",
                MODALITIES,
            )
        ),
        "cross_assay_axis_abstained_modalities": list(
            selection.get("cross_assay_axis_abstained_modalities", ())
        ),
        "paired_cell_identity_used": False,
        "feature_identity_links_used": False,
        "focal_values_used_for_assignment": False,
        "supplied_alignment_used": False,
        "scorer_or_truth_used": False,
        "all_cells_used_exactly_once": True,
        "cells_used": cells_used,
        "cells_total": cells_total,
        "n_complete_metacells": len(observed_rows),
    }
    return (
        pd.DataFrame(observed_rows),
        pd.DataFrame(state_rows),
        membership,
        provenance,
    )


def _safe_alignment_provenance(
    inputs: DiscoveryInputs,
    alignment: Any,
    config: PrepareConfig,
) -> dict[str, Any]:
    base = _base_alignment(alignment)
    selection = _alignment_selection(alignment)
    selected_mode = str(selection.get("selected_mode", "hard"))
    return {
        "schema_version": "prepare_ca_boss_v4_inputs.alignment.v1",
        "method": (
            "recomputed_soft_distributional_background_coordinates"
            if selected_mode.startswith("soft")
            else "recomputed_independent_two_feature_fold_consensus"
        ),
        "alignment_mode_requested": selection.get("requested_mode", "hard"),
        "alignment_mode_selected": selected_mode,
        "alignment_selection": selection,
        "reference_modality": alignment.provenance["reference_modality"],
        "n_feature_folds": 2,
        "effective_n_components": base.provenance["effective_n_components"],
        "effective_n_microstates": base.provenance["effective_n_microstates"],
        "alignment_config": {
            "n_components": config.n_components,
            "n_microstates": config.n_microstates,
            "min_cells_per_microstate": config.min_cells_per_microstate,
            "random_seed": config.alignment_seed,
            "derange": False,
            "alignment_mode": config.alignment_mode,
            "soft_n_microstates": config.soft_n_microstates,
            "soft_max_rotation_dispersion": config.soft_max_rotation_dispersion,
            "soft_low_confidence_action": config.soft_low_confidence_action,
        },
        "fold_feature_counts": alignment.provenance["fold_feature_counts"],
        "fold_alignment": alignment.provenance["fold_alignment"],
        "input_file_sha256": inputs.input_hashes,
        "input_policy": {
            "files_read": sorted(inputs.input_hashes),
            "allowed_patterns": [
                "background_matrix__*.npz",
                "observed_focal__*.tsv",
            ],
            "supplied_analysis_embeddings_read": False,
            "supplied_alignment_provenance_read": False,
            "supplied_metacells_or_membership_read": False,
            "manifest_read": False,
            "scorer_or_truth_read": False,
            "external_data_read": False,
        },
        "information_boundary": {
            "feature_folds_are_within_assay": True,
            "feature_columns_compared_across_assays": False,
            "cell_rows_compared_across_assays": False,
            "focal_values_used_for_alignment": False,
            "assay_labels_used": True,
            "background_depth_used": True,
        },
    }


def _write_embeddings(inputs: DiscoveryInputs, alignment: Any, path: Path) -> None:
    rows: list[pd.DataFrame] = []
    for modality in MODALITIES:
        rows.append(
            pd.DataFrame(
                {
                    "modality": modality,
                    "cell": inputs.cell_ids[modality],
                    **{
                        f"integration__dim{dimension}": alignment.embeddings[modality][
                            :, dimension
                        ]
                        for dimension in range(alignment.embeddings[modality].shape[1])
                    },
                    "integration_uncertainty": alignment.embedding_uncertainty[modality],
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(path, sep="\t", index=False)


def _validate_output_path(input_root: Path, out_dir: Path | str) -> Path:
    output = Path(out_dir).absolute()
    resolved_output = output.resolve(strict=False)
    if resolved_output == input_root or input_root in resolved_output.parents:
        raise ValueError("Output directory must be separate from the input discovery/ tree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve(strict=True)


def prepare_inputs(
    discovery_dir: Path | str,
    out_dir: Path | str,
    config: PrepareConfig | None = None,
) -> dict[str, Any]:
    """Prepare all CA-BOSS v4 discovery inputs from one raw discovery directory."""

    cfg = config or PrepareConfig()
    inputs = load_discovery_inputs(discovery_dir)
    output = _validate_output_path(inputs.root, out_dir)
    alignment = recompute_alignment(inputs, cfg)
    selection = _alignment_selection(alignment)

    embeddings_path = output / "analysis_embeddings.tsv"
    _write_embeddings(inputs, alignment, embeddings_path)
    provenance_path = output / "alignment_provenance.json"
    _write_json(provenance_path, _safe_alignment_provenance(inputs, alignment, cfg))

    nested_files: list[str] = []
    nested_counts: dict[str, int] = {}
    for target_cells in cfg.cell_targets:
        for aggregation_seed in cfg.aggregation_seeds:
            run_name = f"cells{target_cells}_seed{aggregation_seed}"
            run_dir = output / "nested" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            observed, state, membership, provenance = build_complete_metacells(
                inputs,
                alignment,
                target_cells=target_cells,
                aggregation_seed=aggregation_seed,
                alignment_seed=cfg.alignment_seed,
            )
            tables = {
                "observed_metacell_matrix.tsv": observed,
                "background_state_centroids.tsv": state,
                "analysis_membership.tsv": membership,
            }
            for filename, table in tables.items():
                path = run_dir / filename
                table.to_csv(path, sep="\t", index=False)
                nested_files.append(str(path.relative_to(output)))
            run_provenance = run_dir / "aggregation_provenance.json"
            _write_json(run_provenance, provenance)
            nested_files.append(str(run_provenance.relative_to(output)))
            nested_counts[run_name] = int(len(observed))

    summary = {
        "schema_version": "prepare_ca_boss_v4_inputs.run.v1",
        "n_modalities": len(MODALITIES),
        "cell_counts": {
            modality: int(len(inputs.cell_ids[modality])) for modality in MODALITIES
        },
        "alignment_method": (
            "recomputed_soft_distributional_background_coordinates"
            if str(selection.get("selected_mode", "hard")).startswith("soft")
            else "recomputed_two_feature_fold_background_consensus"
        ),
        "alignment_mode_requested": selection.get("requested_mode", "hard"),
        "alignment_mode_selected": selection.get("selected_mode", "hard"),
        "soft_alignment_abstained": bool(
            selection.get("soft_alignment_abstained", False)
        ),
        "cross_assay_axis_eligible_modalities": list(
            selection.get(
                "cross_assay_axis_eligible_modalities",
                MODALITIES,
            )
        ),
        "cross_assay_axis_abstained_modalities": list(
            selection.get("cross_assay_axis_abstained_modalities", ())
        ),
        "derange": False,
        "nested_runs": len(cfg.cell_targets) * len(cfg.aggregation_seeds),
        "nested_metacell_counts": nested_counts,
        "all_cells_retained_in_every_run": True,
        "files_read": sorted(inputs.input_hashes),
        "files_written": sorted(
            ["analysis_embeddings.tsv", "alignment_provenance.json", *nested_files]
        ),
    }
    _write_json(output / "preparation_manifest.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--n-microstates", type=int, default=6)
    parser.add_argument("--min-cells-per-microstate", type=int, default=20)
    parser.add_argument("--alignment-seed", type=int, default=0)
    parser.add_argument("--alignment-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--soft-n-microstates", type=int, default=24)
    parser.add_argument("--soft-max-rotation-dispersion", type=float, default=0.35)
    parser.add_argument(
        "--soft-low-confidence-action",
        choices=("partial_soft", "fallback_hard", "abstain"),
        default="partial_soft",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PrepareConfig(
        n_components=args.n_components,
        n_microstates=args.n_microstates,
        min_cells_per_microstate=args.min_cells_per_microstate,
        alignment_seed=args.alignment_seed,
        alignment_mode=args.alignment_mode,
        soft_n_microstates=args.soft_n_microstates,
        soft_max_rotation_dispersion=args.soft_max_rotation_dispersion,
        soft_low_confidence_action=args.soft_low_confidence_action,
    )
    summary = prepare_inputs(args.discovery_dir, args.out_dir, config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "AGGREGATION_SEEDS",
    "CELL_TARGETS",
    "DiscoveryInputs",
    "MARKS",
    "MODALITIES",
    "PrepareConfig",
    "build_complete_metacells",
    "load_discovery_inputs",
    "prepare_inputs",
    "recompute_alignment",
]
