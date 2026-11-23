from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_cached_layered_contract_v4 as audit  # noqa: E402


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["core1", "core2", "core3", "core4", "tail", "gene"],
            "modality": ["chrom_A"] * 5 + ["rna"],
            "selection_status": ["selected"] * 6,
            "abstention_reason": [""] * 6,
            "feature_global_maxT_p": [0.001, 0.002, 0.003, 0.01, 0.02, 0.001],
        }
    )


def test_strong_core_replay_is_subtractive_and_idempotent() -> None:
    first = audit.replay_strong_core_guard(_membership())

    assert set(first.loc[first["audit_selected"], "feature"]) == {
        "core1",
        "core2",
        "core3",
        "core4",
        "gene",
    }
    assert first.loc[first["feature"].eq("tail"), "audit_within_assay_weak_maxT_tail_demoted"].item()
    assert set(first.loc[first["audit_selected"], "modality"]) == {"chrom_A", "rna"}

    already_guarded = _membership()
    tail = already_guarded["feature"].eq("tail")
    already_guarded.loc[tail, "selection_status"] = "unresolved"
    already_guarded.loc[tail, "abstention_reason"] = (
        "within_assay_strong_maxT_core_weak_tail"
    )
    already_guarded["within_assay_weak_maxT_tail_demoted"] = tail
    second = audit.replay_strong_core_guard(already_guarded)

    pd.testing.assert_series_equal(
        first["audit_selected"], second["audit_selected"], check_names=False
    )
    pd.testing.assert_series_equal(
        first["audit_within_assay_weak_maxT_tail_demoted"],
        second["audit_within_assay_weak_maxT_tail_demoted"],
        check_names=False,
    )


def test_selection_payload_requires_cached_only_truth_closed_manifest(
    tmp_path: Path,
) -> None:
    payload = {
        "new_worlds_or_datasets_created": False,
        "scorer_or_truth_opened": False,
        "selected_worlds": [],
    }
    payload["selection_payload_sha256"] = audit._canonical_sha256(payload)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert audit._selection_payload(path)["selected_worlds"] == []

    payload["scorer_or_truth_opened"] = True
    unsigned = dict(payload)
    unsigned.pop("selection_payload_sha256")
    payload["selection_payload_sha256"] = audit._canonical_sha256(unsigned)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="opened scorer/truth"):
        audit._selection_payload(path)


def test_phase_a_observable_guard_rejects_scorer_path(tmp_path: Path) -> None:
    scorer = tmp_path / "world" / "scorer"
    scorer.mkdir(parents=True)
    truth = scorer / "role_assignments.tsv"
    truth.write_text("node\trole\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refuses scorer/truth"):
        audit._assert_observable(truth)
