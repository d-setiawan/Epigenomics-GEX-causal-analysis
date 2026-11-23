from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import layered_state_evidence_v4 as layered  # noqa: E402
import replay_cached_layered_state_v4 as replay  # noqa: E402
from replay_existing_world_panel_v4 import ExistingWorld  # noqa: E402
from test_layered_state_evidence_v4 import (  # noqa: E402
    _empirical_null_calibrator,
    _groups,
    _membership,
    _multievidence_calibrator,
    _structure_evidence,
    _subspace,
    _topology_consensus_calibrator,
)


def _world() -> ExistingWorld:
    return ExistingWorld(
        block="train_cached",
        seed=17,
        run_root=Path("/tmp/run"),
        world_root=Path("/tmp/worlds"),
        artifact_root=Path("/tmp/artifacts"),
    )


def test_cached_direct_rows_become_oriented_consensus_candidates() -> None:
    cache = pd.DataFrame(
        [
            {
                "block": "train_cached",
                "seed": 17,
                "source_assay": "chrom_A",
                "parent": "peak_A",
                "child": "peak_C",
            }
        ]
    )
    direct = replay._cached_direct_consensus(cache, _world())

    assert replay._direct_tuples(direct) == {
        ("peak_A", "TAIL", "peak_C", "ARROW")
    }
    assert direct.loc[0, "edge_type"] == "within_assay_direct_candidate"
    assert direct.loc[0, "source_assay"] == "chrom_A"


def test_cached_membership_upgrade_is_canonical_and_idempotent() -> None:
    membership = _membership()
    old = membership.drop(
        columns=[
            "within_assay_strong_core_member",
            "within_assay_strong_core_count",
            "within_assay_weak_maxT_tail_demoted",
        ]
    )
    upgraded = replay._upgrade_cached_membership(old)
    repeated = replay._upgrade_cached_membership(upgraded)

    assert tuple(upgraded.columns) == tuple(
        replay.layered_scorer.subspace_scorer.MEMBERSHIP_COLUMNS
    )
    pd.testing.assert_frame_equal(upgraded, repeated)


def test_layer_replay_never_expands_equivalence_sets_or_changes_direct_edges() -> None:
    membership = replay._upgrade_cached_membership(_membership())
    groups, group_membership = _groups()
    direct = pd.DataFrame(
        [
            {
                "node1": "peak_A",
                "endpoint1": "TAIL",
                "node2": "peak_C",
                "endpoint2": "ARROW",
                "edge_type": "within_assay_direct_candidate",
                "source_assay": "chrom_A",
            }
        ]
    )
    result = layered.build_layered_state_evidence(
        membership,
        _subspace(),
        direct,
        groups,
        group_membership,
        _structure_evidence(),
        empirical_null_calibrator=_empirical_null_calibrator(),
        multievidence_calibrator=_multievidence_calibrator(),
        topology_consensus_calibrator=_topology_consensus_calibrator(),
    )
    diagnostics = replay._world_diagnostics(_world(), result, direct)

    assert diagnostics["direct_candidates_unchanged_by_layering"]
    assert diagnostics["strict_membership_unchanged_by_equivalence"]
    assert diagnostics["n_state_equivalence_pairwise_relations"] == 0
    assert diagnostics["unsupported_equivalence_claims"] == 0
    assert diagnostics["empirical_null_pairwise_relations"] == 0
    assert diagnostics["empirical_null_direct_relations_created"] == 0
    assert diagnostics["multievidence_pairwise_relations"] == 0
    assert diagnostics["multievidence_direct_relations_created"] == 0
    assert diagnostics["topology_consensus_pairwise_relations"] == 0
    assert diagnostics["topology_consensus_direct_relations_created"] == 0
    assert diagnostics[
        "topology_consensus_direct_structure_alone_is_insufficient"
    ]
