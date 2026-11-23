#!/usr/bin/env python3
"""Conservative observed-only gate for a direct-reference alignment fallback.

The ordinary soft synchronizer can lose an otherwise coherent reference star
when cycle filtering removes intermediate assay--assay edges and the remaining
graph fails its degree-core requirement.  This module evaluates a narrower
fallback: use each assay's already fitted direct rotation into the RNA frame,
but only when those rotations are also supported by several independent
assay triangles and by a strong global consistency audit.

Strong pairwise transport alone is intentionally insufficient.  In a nearly
symmetric background distribution, finite-sample fluctuations can make every
direct assay--RNA fit appear unique while the fitted rotations disagree
globally.  Such a world must remain abstained rather than acquiring arbitrary
cross-assay axes.

This file is standalone during TRAIN development.  It does not alter the
production soft aligner or the benchmark.  If integrated later, the accepted
assays would use the direct assay-to-reference rotations already computed by
``soft_distributional_alignment``; no cell matching is introduced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import soft_distributional_alignment as soft_alignment


@dataclass(frozen=True)
class ReferenceStarFallbackConfig:
    """Frozen observable thresholds for the direct-reference rescue gate."""

    min_direct_transport_separation: float = 0.85
    max_direct_rotation_dispersion: float = 1e-8
    min_direct_cycle_support: int = 2
    min_direct_consistent_cycles: int = 2
    min_direct_consistent_cycle_fraction: float = 2.0 / 3.0
    max_direct_median_cycle_inconsistency: float = 0.35
    max_direct_minimum_cycle_inconsistency: float = 0.35
    min_strong_direct_assays: int = 3
    min_strong_direct_assay_fraction: float = 1.0
    min_global_consistent_cycle_fraction: float = 0.75
    max_global_median_cycle_inconsistency: float = 0.35

    def __post_init__(self) -> None:
        probabilities = (
            self.min_direct_transport_separation,
            self.min_direct_consistent_cycle_fraction,
            self.min_strong_direct_assay_fraction,
            self.min_global_consistent_cycle_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Separation and fraction thresholds must lie in [0, 1]")
        if self.max_direct_rotation_dispersion < 0.0:
            raise ValueError("Rotation-dispersion threshold may not be negative")
        if min(
            self.max_direct_median_cycle_inconsistency,
            self.max_direct_minimum_cycle_inconsistency,
            self.max_global_median_cycle_inconsistency,
        ) <= 0.0:
            raise ValueError("Cycle-inconsistency thresholds must be positive")
        if min(
            self.min_direct_cycle_support,
            self.min_direct_consistent_cycles,
            self.min_strong_direct_assays,
        ) < 1:
            raise ValueError("Support and assay-count thresholds must be positive")
        if self.min_direct_consistent_cycles > self.min_direct_cycle_support:
            raise ValueError("Required consistent cycles cannot exceed cycle support")


@dataclass(frozen=True)
class ReferenceStarFallbackDecision:
    """Decision plus complete observable diagnostics for audit and freezing."""

    accepted: bool
    reference_modality: str
    eligible_modalities: tuple[str, ...]
    abstained_modalities: tuple[str, ...]
    reason: str
    assay_evidence: tuple[dict[str, Any], ...]
    global_evidence: dict[str, Any]
    provenance: dict[str, Any]


def _pair_key(left: str, right: str) -> str:
    return "||".join(sorted((str(left), str(right))))


def _finite_float(value: Any, *, default: float = float("inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def evaluate_reference_star_fallback(
    alignment_provenance: Mapping[str, Any],
    *,
    config: ReferenceStarFallbackConfig | None = None,
) -> ReferenceStarFallbackDecision:
    """Gate direct RNA rotations using pair strength and global cycle support.

    Only diagnostics already produced from held-out observed background counts
    are consumed.  Focal values, feature identities, truth, scorer artifacts,
    and cell correspondences are neither accepted nor needed by this API.
    """

    cfg = config or ReferenceStarFallbackConfig()
    reference = str(alignment_provenance.get("reference_modality", ""))
    if not reference:
        raise ValueError("Alignment provenance is missing its reference modality")
    synchronization = alignment_provenance.get("synchronization")
    pairwise = alignment_provenance.get("pairwise_distribution_diagnostics")
    if not isinstance(synchronization, Mapping) or not isinstance(pairwise, Mapping):
        raise ValueError("Alignment provenance lacks synchronization diagnostics")
    assay_status = synchronization.get("assay_synchronized")
    if not isinstance(assay_status, Mapping) or reference not in assay_status:
        raise ValueError("Alignment provenance lacks per-assay synchronization status")

    modalities = sorted(str(modality) for modality in assay_status)
    non_reference = [modality for modality in modalities if modality != reference]
    if len(non_reference) < cfg.min_strong_direct_assays:
        raise ValueError("Too few non-reference assays for a global star audit")

    rows: list[dict[str, Any]] = []
    for modality in non_reference:
        diagnostic = pairwise.get(_pair_key(modality, reference))
        if not isinstance(diagnostic, Mapping):
            diagnostic = {}
        separation = _finite_float(diagnostic.get("transport_separation"), default=-1.0)
        dispersion = _finite_float(
            diagnostic.get("near_optimal_rotation_dispersion")
        )
        cycle_support = int(diagnostic.get("cycle_support", 0) or 0)
        consistent_cycles = int(diagnostic.get("consistent_cycle_support", 0) or 0)
        consistent_fraction = (
            float(consistent_cycles / cycle_support) if cycle_support > 0 else 0.0
        )
        median_cycle = _finite_float(
            diagnostic.get("median_cycle_inconsistency")
        )
        minimum_cycle = _finite_float(
            diagnostic.get("minimum_cycle_inconsistency")
        )
        strong_checks = {
            "preliminary_pair_gate": bool(
                diagnostic.get("preliminary_gate_accepted", False)
            ),
            "transport_separation": (
                separation >= cfg.min_direct_transport_separation
            ),
            "rotation_dispersion": (
                dispersion <= cfg.max_direct_rotation_dispersion
            ),
        }
        strong_direct_anchor = bool(all(strong_checks.values()))
        consistency_checks = {
            "cycle_support": cycle_support >= cfg.min_direct_cycle_support,
            "consistent_cycles": (
                consistent_cycles >= cfg.min_direct_consistent_cycles
            ),
            "consistent_cycle_fraction": (
                consistent_fraction >= cfg.min_direct_consistent_cycle_fraction
            ),
            "median_cycle_inconsistency": (
                median_cycle <= cfg.max_direct_median_cycle_inconsistency
            ),
            "minimum_cycle_inconsistency": (
                minimum_cycle <= cfg.max_direct_minimum_cycle_inconsistency
            ),
        }
        locally_eligible = bool(
            strong_direct_anchor and all(consistency_checks.values())
        )
        failures = [
            f"strong:{name}"
            for name, passed in strong_checks.items()
            if not passed
        ] + [
            f"consistency:{name}"
            for name, passed in consistency_checks.items()
            if not passed
        ]
        rows.append(
            {
                "modality": modality,
                "direct_pair_key": _pair_key(modality, reference),
                "already_synchronized": bool(assay_status.get(modality, False)),
                "transport_separation": separation,
                "near_optimal_rotation_dispersion": dispersion,
                "cycle_support": cycle_support,
                "consistent_cycle_support": consistent_cycles,
                "consistent_cycle_fraction": consistent_fraction,
                "median_cycle_inconsistency": median_cycle,
                "minimum_cycle_inconsistency": minimum_cycle,
                "strong_direct_anchor": strong_direct_anchor,
                "locally_eligible": locally_eligible,
                "failed_checks": failures,
            }
        )

    strong_rows = [row for row in rows if row["strong_direct_anchor"]]
    locally_eligible_rows = [row for row in rows if row["locally_eligible"]]
    strong_fraction = float(len(strong_rows) / len(non_reference))
    total_cycles = int(sum(row["cycle_support"] for row in strong_rows))
    total_consistent = int(
        sum(row["consistent_cycle_support"] for row in strong_rows)
    )
    global_cycle_fraction = (
        float(total_consistent / total_cycles) if total_cycles > 0 else 0.0
    )
    strong_cycle_medians = [
        float(row["median_cycle_inconsistency"])
        for row in strong_rows
        if np.isfinite(row["median_cycle_inconsistency"])
    ]
    global_median_cycle = (
        float(np.median(strong_cycle_medians))
        if strong_cycle_medians
        else float("inf")
    )
    global_checks = {
        "enough_strong_direct_assays": (
            len(strong_rows) >= cfg.min_strong_direct_assays
        ),
        "strong_direct_assay_fraction": (
            strong_fraction >= cfg.min_strong_direct_assay_fraction
        ),
        "all_strong_anchors_locally_consistent": (
            len(locally_eligible_rows) == len(strong_rows) and bool(strong_rows)
        ),
        "global_consistent_cycle_fraction": (
            global_cycle_fraction >= cfg.min_global_consistent_cycle_fraction
        ),
        "global_median_cycle_inconsistency": (
            global_median_cycle <= cfg.max_global_median_cycle_inconsistency
        ),
    }
    accepted = bool(all(global_checks.values()))
    if accepted:
        eligible = tuple(
            sorted({reference, *(row["modality"] for row in locally_eligible_rows)})
        )
        reason = "accepted:strong_direct_reference_star_with_global_cycle_support"
    else:
        # This standalone gate is atomic: a failed global audit may not expose a
        # cherry-picked partial star.  The original synchronizer's safe assays
        # remain usable independently of this fallback.
        eligible = tuple(
            sorted(
                modality
                for modality, synchronized in assay_status.items()
                if bool(synchronized)
            )
        )
        failed = [name for name, passed in global_checks.items() if not passed]
        reason = "abstained:" + ",".join(failed)
    abstained = tuple(sorted(set(modalities) - set(eligible)))

    global_evidence = {
        "n_non_reference_assays": len(non_reference),
        "n_strong_direct_assays": len(strong_rows),
        "strong_direct_assay_fraction": strong_fraction,
        "n_locally_consistent_direct_assays": len(locally_eligible_rows),
        "total_direct_cycle_support": total_cycles,
        "total_consistent_direct_cycles": total_consistent,
        "global_consistent_cycle_fraction": global_cycle_fraction,
        "global_median_cycle_inconsistency": global_median_cycle,
        "checks": global_checks,
    }
    provenance = {
        "schema_version": "reference_star_fallback_v4.decision.v1",
        "method": "direct_reference_star_with_multi_neighbor_global_cycle_gate",
        "config": asdict(cfg),
        "information_boundary": {
            "consumed": [
                "observed_background_pair_transport_diagnostics",
                "observed_background_rotation_ambiguity",
                "observed_background_multi_assay_cycle_diagnostics",
                "assay_key",
            ],
            "not_consumed": [
                "focal_features",
                "feature_identity_correspondence",
                "true_latent_state",
                "truth_graph",
                "scorer_metrics",
                "paired_cells",
                "cell_identity_correspondence",
                "external_data",
            ],
            "direct_pair_strength_alone_is_sufficient": False,
            "fallback_is_atomic_under_global_gate_failure": True,
            "cross_assay_direction_identified": False,
        },
        "integration_status": "standalone_train_diagnostic_not_production_wired",
    }
    return ReferenceStarFallbackDecision(
        accepted=accepted,
        reference_modality=reference,
        eligible_modalities=eligible,
        abstained_modalities=abstained,
        reason=reason,
        assay_evidence=tuple(rows),
        global_evidence=global_evidence,
        provenance=provenance,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_reference_star_diagnostic(
    discovery_dir: Path | str,
    out_dir: Path | str,
    *,
    soft_config: soft_alignment.SoftAlignmentConfig | None = None,
    fallback_config: ReferenceStarFallbackConfig | None = None,
) -> dict[str, Any]:
    """Recompute background-only evidence and freeze the fallback decision."""

    soft_cfg = soft_config or soft_alignment.SoftAlignmentConfig()
    fallback_cfg = fallback_config or ReferenceStarFallbackConfig()
    _, matrices, depths, _, input_hashes = soft_alignment._load_background_archives(
        discovery_dir
    )
    result = soft_alignment.align_soft_distributional_background_states(
        matrices,
        depths,
        reference_modality="rna",
        config=soft_cfg,
    )
    decision = evaluate_reference_star_fallback(
        result.provenance, config=fallback_cfg
    )
    output = Path(out_dir).absolute()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    decision_payload = {
        "accepted": decision.accepted,
        "reference_modality": decision.reference_modality,
        "eligible_modalities": decision.eligible_modalities,
        "abstained_modalities": decision.abstained_modalities,
        "reason": decision.reason,
        "assay_evidence": decision.assay_evidence,
        "global_evidence": decision.global_evidence,
        "provenance": decision.provenance,
    }
    decision_path = output / "reference_star_fallback_decision.json"
    decision_path.write_text(
        json.dumps(_json_ready(decision_payload), indent=2, sort_keys=True) + "\n"
    )
    source_path = Path(__file__).resolve(strict=True)
    soft_source = Path(soft_alignment.__file__).resolve(strict=True)
    manifest = {
        "schema_version": "reference_star_fallback_v4.freeze.v1",
        "development_status": "TRAIN_posthoc_diagnostic_not_blind_validation",
        "candidate_decision_frozen_before_new_candidate_scorer_comparison": True,
        "algorithm_source_sha256": {
            source_path.name: _file_sha256(source_path),
            soft_source.name: _file_sha256(soft_source),
        },
        "soft_config": asdict(soft_cfg),
        "fallback_config": asdict(fallback_cfg),
        "input_file_sha256": input_hashes,
        "output_file_sha256": {
            decision_path.name: _file_sha256(decision_path),
        },
        "information_boundary": {
            "allowed_input_pattern": "background_matrix__*.npz",
            "files_read": sorted(input_hashes),
            "focal_files_read": False,
            "supplied_embeddings_read": False,
            "manifest_read": False,
            "scorer_or_truth_read_by_runner": False,
            "external_data_read": False,
        },
    }
    manifest_path = output / "candidate_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the observed-only direct-reference-star fallback audit."
    )
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    freeze_reference_star_diagnostic(
        args.discovery,
        args.out,
        soft_config=soft_alignment.SoftAlignmentConfig(random_seed=args.random_seed),
    )


__all__ = [
    "ReferenceStarFallbackConfig",
    "ReferenceStarFallbackDecision",
    "evaluate_reference_star_fallback",
    "freeze_reference_star_diagnostic",
]


if __name__ == "__main__":
    main()
