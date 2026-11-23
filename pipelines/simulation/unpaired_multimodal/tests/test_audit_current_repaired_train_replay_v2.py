from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_current_repaired_train_replay_v2 as audit  # noqa: E402


def test_replay_tsv_reader_preserves_threshold_boundary_floats(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observable.tsv"
    path.write_text("value\n0.10000000000000002\n", encoding="utf-8")
    hashes: dict[str, str] = {}

    observed = audit._read_tsv(path, hashes)

    assert observed.loc[0, "value"] == 0.10000000000000002
    assert str(path.resolve()) in hashes


def test_replay_tsv_reader_accepts_empty_only_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.tsv"
    path.write_bytes(b"\n")

    with pytest.raises(ValueError, match="Observed TSV is empty"):
        audit._read_tsv(path, {})

    hashes: dict[str, str] = {}
    observed = audit._read_tsv(path, hashes, allow_empty=True)
    assert observed.empty
    assert len(observed.columns) == 0
    assert str(path.resolve()) in hashes


def test_replay_gate_cannot_invent_an_upstream_state_hyperedge() -> None:
    participating = {*(f"chrom_M{index}" for index in range(6)), "rna"}
    payload = audit._gate_payload(
        state_score={
            "n_selected": 7,
            "n_true_selected": 7,
            "n_sentinel_selected": 0,
            "n_unknown_selected": 0,
            "precision": 1.0,
            "recall": 0.28,
        },
        participating=participating,
        direct={"precision": 1.0, "recall": 1.0, "reversals": 0},
        loading={
            "precision": 1.0,
            "recall": 0.8,
            "sentinel_members": 0,
        },
        density=17,
        state_hyperedge_selected=False,
    )

    validation = audit.evaluate_layered_world_reliability(payload)

    assert payload["cross_assay_state_hyperedge"]["n_selected"] == 0
    assert not validation["gate_checks"][
        "exactly_one_selected_cross_assay_state_hyperedge"
    ]
    assert not validation["world_reliability_success"]


def test_phase_a_verifier_rejects_a_stale_schema_before_outputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "phase_a_manifest.json").write_text(
        '{"schema_version":"stale"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="schema version"):
        audit._verify_phase_a(tmp_path)


def test_registered_worlds_are_exactly_the_five_opened_train_blocks() -> None:
    assert audit.SCHEMA == "ca_boss_v4.current_repaired_train_replay.v4"
    assert [block.name for block in audit.BLOCKS] == [
        "TRAIN330",
        "TRAIN360",
        "TRAIN370",
        "TRAIN380",
        "TRAIN400",
    ]
    assert len(audit.ALLOWED_SEEDS) == 120
    assert min(audit.ALLOWED_SEEDS) == 33_001
    assert max(audit.ALLOWED_SEEDS) == 40_024
    assert 34_001 not in audit.ALLOWED_SEEDS
    assert 41_001 not in audit.ALLOWED_SEEDS
    assert 45_001 not in audit.ALLOWED_SEEDS


@pytest.mark.parametrize(
    "path",
    [
        "pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/dev_34001_34016/file.tsv",
        "pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_41001_41024/file.tsv",
        "pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/locked_45001_45040/file.tsv",
    ],
)
def test_protected_split_paths_are_rejected_before_file_access(path: str) -> None:
    with pytest.raises(ValueError, match="protected path"):
        audit._assert_path(Path(path))


def test_phase_a_rejects_truth_paths_before_file_access() -> None:
    with pytest.raises(ValueError, match="scorer/truth path"):
        audit._assert_path(
            Path("pipelines/simulation/unpaired_multimodal/worlds/formal_worlds/train_40001_40024/world_seed40001/scorer/role_assignments.tsv")
        )


def test_union_schema_edge_parser_uses_populated_representation() -> None:
    frame = pd.DataFrame(
        {
            "source_assay": ["chrom_A", "chrom_A"],
            "parent": [float("nan"), float("nan")],
            "child": [float("nan"), float("nan")],
            "node1": ["P", "Q"],
            "endpoint1": ["TAIL", "TAIL"],
            "node2": ["C", "C"],
            "endpoint2": ["ARROW", "ARROW"],
        }
    )
    assert audit._edge_set(frame) == {
        ("chrom_A", "P", "C"),
        ("chrom_A", "Q", "C"),
    }


def test_replay_recomputes_complete_null_omnibus_annotations() -> None:
    rows: list[dict[str, object]] = []
    for index in range(6):
        modality = f"chrom_M{index}"
        rows.extend(
            [
                {
                    "feature": f"peak_M{index}_signal",
                    "modality": modality,
                    "formal_test_eligible": True,
                    "empirical_null_p": 0.001,
                    "reference_conditional_wald": 0.40,
                    "conditional_wald_view_support": 1.0,
                    "pre_redundancy_selection_status": "selected",
                    "selection_status": "selected",
                },
                {
                    "feature": f"peak_M{index}_null",
                    "modality": modality,
                    "formal_test_eligible": True,
                    "empirical_null_p": 0.8,
                    "reference_conditional_wald": 0.01,
                    "conditional_wald_view_support": 0.0,
                    "pre_redundancy_selection_status": "unresolved",
                    "selection_status": "unresolved",
                },
            ]
        )
    rows.append(
        {
            "feature": "expr__gene",
            "modality": "rna",
            "formal_test_eligible": True,
            "empirical_null_p": 0.001,
            "reference_conditional_wald": 0.45,
            "conditional_wald_view_support": 1.0,
            "pre_redundancy_selection_status": "selected",
            "selection_status": "selected",
        }
    )

    state = audit._current_state_membership(pd.DataFrame(rows))
    participation = audit._participation_table(state).set_index("modality")

    chrom = participation.loc[participation.index.str.startswith("chrom_")]
    assert chrom["omnibus_route_count"].eq(2).all()
    assert chrom["single_feature_p"].eq(0.002).all()
    assert chrom["omnibus_p"].eq(0.004).all()
    assert chrom["winning_route"].eq("single_feature").all()
    assert chrom["selection_route"].eq("single_feature").all()
    assert chrom["selected"].all()
    assert participation.loc["rna", "omnibus_route_count"] == 1
    assert participation.loc["rna", "winning_route"] == "single_feature"


def test_replay_does_not_reconstruct_demoted_motif_as_direct_arrows() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": "chrom_A",
                "action": (
                    "demote_retained_orientation_ambiguous_production_motif"
                ),
                "selected_replacement": False,
                "demote_production_motif": True,
                "selected_unoriented_adjacency": False,
                "production_exact_two_arm_star": True,
                "production_center": "C",
                "production_outer1": "P",
                "production_outer2": "Q",
                "selected_center": "",
                "selected_outer1": "",
                "selected_outer2": "",
            }
        ]
    )
    resolved = audit._resolved_direct_frame(
        decisions,
        Path("unused_for_a_demoted_motif"),
        {},
    )
    assert resolved.empty


def test_replay_preserves_selected_unoriented_adjacency_without_arrows() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_assay": "chrom_A",
                "action": audit.UNORIENTED_RESCUE_OVERLAP_ACTION,
                "selected_replacement": False,
                "demote_production_motif": False,
                "selected_unoriented_adjacency": True,
                "unoriented_adjacency_node1": "C",
                "unoriented_adjacency_node2": "Q",
                "production_exact_two_arm_star": False,
                "production_center": "",
                "production_outer1": "",
                "production_outer2": "",
                "selected_center": "",
                "selected_outer1": "",
                "selected_outer2": "",
            }
        ]
    )
    resolved = audit._resolved_direct_frame(decisions, Path("unused"), {})
    assert len(resolved) == 1
    assert resolved.loc[0, "endpoint1"] == "CIRCLE"
    assert resolved.loc[0, "endpoint2"] == "CIRCLE"
    assert resolved.loc[0, "collider_center"] == ""
    assert not bool(resolved.loc[0, "direction_claimed"])
    assert audit._edge_set(resolved) == set()
