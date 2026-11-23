from __future__ import annotations

import inspect
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import reference_star_fallback_v4 as star  # noqa: E402


def _direct(
    *,
    separation: float = 0.92,
    dispersion: float = 0.0,
    cycle_support: int = 3,
    consistent_cycles: int = 3,
    median_cycle: float = 0.10,
    minimum_cycle: float = 0.06,
) -> dict[str, object]:
    return {
        "transport_separation": separation,
        "near_optimal_rotation_dispersion": dispersion,
        "preliminary_gate_accepted": True,
        "cycle_support": cycle_support,
        "consistent_cycle_support": consistent_cycles,
        "median_cycle_inconsistency": median_cycle,
        "minimum_cycle_inconsistency": minimum_cycle,
    }


def _provenance(
    direct: dict[str, dict[str, object]],
) -> dict[str, object]:
    modalities = ["rna", *sorted(direct)]
    return {
        "reference_modality": "rna",
        "synchronization": {
            "assay_synchronized": {
                modality: modality == "rna" for modality in modalities
            }
        },
        "pairwise_distribution_diagnostics": {
            "||".join(sorted((modality, "rna"))): diagnostic
            for modality, diagnostic in direct.items()
        },
    }


def test_consistent_direct_reference_star_is_accepted() -> None:
    provenance = _provenance(
        {modality: _direct() for modality in ("chrom_A", "chrom_B", "chrom_C")}
    )
    decision = star.evaluate_reference_star_fallback(provenance)
    assert decision.accepted is True
    assert decision.eligible_modalities == (
        "chrom_A",
        "chrom_B",
        "chrom_C",
        "rna",
    )
    assert decision.abstained_modalities == ()
    assert decision.global_evidence["global_consistent_cycle_fraction"] == 1.0


def test_strong_pairwise_stars_abstain_when_globally_inconsistent() -> None:
    # This is the important D-high failure mode: each direct pair has a unique,
    # high-separation optimum, but none of the independent assay paths agrees.
    provenance = _provenance(
        {
            modality: _direct(
                cycle_support=3,
                consistent_cycles=0,
                median_cycle=0.95,
                minimum_cycle=0.82,
            )
            for modality in ("chrom_A", "chrom_B", "chrom_C")
        }
    )
    decision = star.evaluate_reference_star_fallback(provenance)
    assert decision.accepted is False
    assert decision.eligible_modalities == ("rna",)
    assert set(decision.abstained_modalities) == {"chrom_A", "chrom_B", "chrom_C"}
    assert decision.global_evidence["n_strong_direct_assays"] == 3
    assert decision.global_evidence["total_consistent_direct_cycles"] == 0
    assert "global_consistent_cycle_fraction" in decision.reason


def test_one_ambiguous_anchor_prevents_atomic_global_fallback() -> None:
    direct = {
        modality: _direct()
        for modality in ("chrom_A", "chrom_B", "chrom_C", "chrom_D")
    }
    direct["chrom_D"] = _direct(dispersion=0.31)
    decision = star.evaluate_reference_star_fallback(_provenance(direct))
    assert decision.accepted is False
    assert decision.global_evidence["strong_direct_assay_fraction"] == 0.75
    assert decision.global_evidence["checks"][
        "all_strong_anchors_locally_consistent"
    ] is True
    # The atomic decision does not quietly expose a partial star when one assay
    # lacks a reproducible direct reference rotation.
    assert decision.eligible_modalities == ("rna",)


def test_direct_cycle_support_threshold_is_not_replaced_by_one_lucky_path() -> None:
    provenance = _provenance(
        {
            modality: _direct(cycle_support=1, consistent_cycles=1)
            for modality in ("chrom_A", "chrom_B", "chrom_C")
        }
    )
    decision = star.evaluate_reference_star_fallback(provenance)
    assert decision.accepted is False
    assert all(not row["locally_eligible"] for row in decision.assay_evidence)


def test_api_and_provenance_have_no_truth_or_correspondence_inputs() -> None:
    parameters = set(
        inspect.signature(star.evaluate_reference_star_fallback).parameters
    )
    assert parameters.isdisjoint(
        {
            "truth",
            "true_z",
            "labels",
            "focal_features",
            "pairs",
            "cell_correspondence",
            "scorer_metrics",
        }
    )
    decision = star.evaluate_reference_star_fallback(
        _provenance(
            {modality: _direct() for modality in ("chrom_A", "chrom_B", "chrom_C")}
        )
    )
    boundary = decision.provenance["information_boundary"]
    assert boundary["direct_pair_strength_alone_is_sufficient"] is False
    assert boundary["fallback_is_atomic_under_global_gate_failure"] is True
    assert "true_latent_state" in boundary["not_consumed"]
    assert "paired_cells" in boundary["not_consumed"]


def test_invalid_fallback_configuration_is_rejected() -> None:
    for kwargs in (
        {"min_direct_transport_separation": 1.1},
        {"max_direct_rotation_dispersion": -0.1},
        {"min_direct_cycle_support": 0},
        {"min_direct_cycle_support": 1, "min_direct_consistent_cycles": 2},
        {"max_global_median_cycle_inconsistency": 0.0},
    ):
        try:
            star.ReferenceStarFallbackConfig(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive assertion style
            raise AssertionError(f"Expected invalid configuration to fail: {kwargs}")
