from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import bootstrap_dual_null_disagreement_rescue_v4 as stage
import score_bootstrap_dual_null_disagreement_rescue_v4 as scorer


def baseline(pair=("a", "b")):
    return pd.DataFrame([{
        "world_key": "train__world_seed1", "block": "train", "seed": 1,
        "candidate_assay_pairs_json": json.dumps([pair]),
        "candidate_cross_assay_feature_pairs_json": json.dumps([[x + "1" for x in pair]]),
        "candidate_pair_selection_audit_json": "[]",
    }])


def selected(pair=("b", "c")):
    return pd.DataFrame([{
        "world_key": "train__world_seed1", "block": "train", "seed": 1,
        "left_modality": pair[0], "right_modality": pair[1],
        "candidate_cross_assay_feature_pairs_json": json.dumps([[x + "1" for x in pair]]),
        "bootstrap_selection_count": 100,
        stage.parent.diagnostic.LOCAL_CONTEXT_COLUMN: 2,
        "two_context_stratum_null_maximum_selection_count": 88,
        "unstratified_global_null_maximum_selection_count": 96,
        "posterior_probability_exceeds_two_context_stratum_null": 0.999,
    }])


def test_connected_proposals_merge_without_transitive_pair_expansion():
    result = stage.union_candidates(baseline(), selected())
    assert len(result) == 1
    row = result.iloc[0]
    assert row.candidate_assay_pair_count == 2
    assert set(map(tuple, json.loads(row.candidate_assay_pairs_json))) == {("a", "b"), ("b", "c")}
    assert row.candidate_cross_assay_feature_pair_count == 2
    assert row.assay_participation_is_set_valued
    assert not row.exact_assay_participation_claimed
    assert not row.candidate_pairs_may_be_expanded_as_edges


def test_disjoint_addition_preserves_baseline_candidate_content():
    result = stage.union_candidates(baseline(), selected(("c", "d")))
    assert len(result) == 2
    assert result.candidate_assay_pair_count.tolist() == [1, 1]
    assert all(~result.cross_assay_direction_claimed)
    assert "target_safe_v458" in set(result.selection_sources)


def test_union_normalizes_with_common_scorer_and_rejects_core_duplicates():
    result = stage.union_candidates(baseline(), pd.DataFrame())
    core = pd.DataFrame(columns=["hypothesis_selected", "candidate_assay_pairs_json", "selection_route"])
    normalized = scorer.normalize_predictions(core, result)
    assert len(normalized) == 1
    assert normalized.prediction_source.iloc[0] == "v459_dual_null_disagreement_rescue"
    core = pd.DataFrame([{"hypothesis_selected": True, "candidate_assay_pairs_json": '[["a","b"]]'}])
    with pytest.raises(ValueError, match="duplicate"):
        scorer.normalize_predictions(core, result)


@pytest.mark.parametrize("context,count,stratum,global_null,expected", [
    (2, 100, 88, 96, True), (2, 96, 0, 96, False),
    (2, 100, 100, 96, False), (3, 100, 0, 0, False),
    (2, 50, 0, 0, False), (2, 100, 99, 0, False),
])
def test_dual_null_selection_ceilings_and_beta_guard(context, count, stratum, global_null, expected):
    decision, _ = stage.diagnostic._selection_decision(
        nominal_new_proposal=True, local_context_count=context,
        candidate_count=count, stratum_null_count=stratum, global_null_count=global_null)
    assert decision is expected


def test_bound_file_mutation_is_rejected(tmp_path):
    path = tmp_path / "evidence.tsv"
    path.write_text("original")
    source = Path(stage.__file__)
    manifest = {"source_sha256": {source.name: stage.geometry._sha256(source)},
                "output_sha256": {path.name: stage.geometry._sha256(path)}}
    stage.verify_files(tmp_path, manifest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Changed bound artifact"):
        stage.verify_files(tmp_path, manifest)
