"""Focused tests for the formal-TRAIN loading recall diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from score_formal_loading_recall_diagnostic_train_v4 import (  # noqa: E402
    RescueConfig,
    _baseline_mask,
    _select,
)


def _candidate(
    pair_key: str,
    modality: str,
    *,
    lower_cosine: float,
    margin: float,
    null_support: float,
    parent_win: float = 1.0,
    state: bool = True,
    coherence: float = 0.5,
) -> dict[str, object]:
    return {
        "seed": 36_001,
        "modality": modality,
        "pair_key": pair_key,
        "passes_state_membership_gate": state,
        "lower_absolute_loading_cosine": lower_cosine,
        "relative_median_parent_margin": margin,
        "parent_win_view_support": parent_win,
        "empirical_null_support_p_le_0_125": null_support,
        "empirical_null_support_p_le_0_25": null_support,
        "empirical_null_support_p_le_0_50": null_support,
        "median_minimum_loading_stability": 0.8,
        "view_q90__candidate_pair_max_relative_uncertainty": 0.5,
        "median_coherence_group_score": coherence,
        "median_absolute_loading_cosine": lower_cosine + 0.01,
    }


def test_empirical_null_rescue_never_replaces_a_baseline_selection() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "baseline",
                "chrom_A",
                lower_cosine=0.90,
                margin=0.20,
                null_support=0.0,
                coherence=0.20,
            ),
            _candidate(
                "strong_rescue_but_same_assay",
                "chrom_A",
                lower_cosine=0.80,
                margin=0.50,
                null_support=1.0,
                coherence=0.99,
            ),
        ]
    )
    assert _baseline_mask(candidates).tolist() == [True, False]
    selected, _ = _select(candidates, RescueConfig("test"))
    assert selected["pair_key"].tolist() == ["baseline"]
    assert selected["selection_stage"].tolist() == ["formal_baseline"]


def test_empirical_null_rescue_requires_a_unique_unanimous_candidate() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "unique",
                "chrom_A",
                lower_cosine=0.60,
                margin=0.30,
                null_support=1.0,
            ),
            _candidate(
                "ambiguous_1",
                "chrom_B",
                lower_cosine=0.70,
                margin=0.30,
                null_support=1.0,
            ),
            _candidate(
                "ambiguous_2",
                "chrom_B",
                lower_cosine=0.75,
                margin=0.40,
                null_support=1.0,
            ),
        ]
    )
    selected, decisions = _select(candidates, RescueConfig("test"))
    assert selected["pair_key"].tolist() == ["unique"]
    assert selected["selection_stage"].tolist() == [
        "empirical_null_unanimity_rescue"
    ]
    ambiguous = decisions.loc[
        (decisions["seed"].eq(36_001))
        & decisions["modality"].eq("chrom_B")
    ].iloc[0]
    assert ambiguous["selection_status"] == "unresolved"
    assert ambiguous["abstention_reason"] == "ambiguous_rescue_candidates"


def test_rescue_rejects_nonunanimous_null_parent_or_state_evidence() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                "null_not_unanimous",
                "chrom_A",
                lower_cosine=0.80,
                margin=0.30,
                null_support=5.0 / 6.0,
            ),
            _candidate(
                "parent_not_unanimous",
                "chrom_B",
                lower_cosine=0.80,
                margin=0.30,
                null_support=1.0,
                parent_win=5.0 / 6.0,
            ),
            _candidate(
                "state_not_selected",
                "chrom_C",
                lower_cosine=0.80,
                margin=0.30,
                null_support=1.0,
                state=False,
            ),
        ]
    )
    selected, _ = _select(candidates, RescueConfig("test"))
    assert selected.empty
