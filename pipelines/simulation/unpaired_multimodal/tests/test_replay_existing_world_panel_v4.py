from __future__ import annotations

import json
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import replay_existing_world_panel_v4 as replay  # noqa: E402
import replay_cached_layered_state_v4 as layered_replay  # noqa: E402


def test_exclusion_ledger_verifies_hash_and_collects_worlds(
    tmp_path: Path,
) -> None:
    selection = {
        "schema_version": "test.selection.v1",
        "selected_worlds": [
            {"block": "train_1_2", "seed": 1},
            {"block": "development_3_4", "seed": 4},
        ],
        "scorer_or_truth_opened": False,
    }
    selection["selection_payload_sha256"] = replay._canonical_sha256(
        selection
    )
    path = tmp_path / "phase_a_selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")

    keys, hashes = replay.load_excluded_world_keys([path])

    assert keys == {("train_1_2", 1), ("development_3_4", 4)}
    assert hashes == {str(path.resolve()): replay._sha256(path)}


def test_exclusion_ledger_rejects_changed_selection(tmp_path: Path) -> None:
    path = tmp_path / "phase_a_selection.json"
    path.write_text(
        json.dumps(
            {
                "selected_worlds": [{"block": "train_1_2", "seed": 1}],
                "selection_payload_sha256": "not-the-payload-hash",
            }
        ),
        encoding="utf-8",
    )

    try:
        replay.load_excluded_world_keys([path])
    except ValueError as error:
        assert "payload hash mismatch" in str(error)
    else:
        raise AssertionError("Changed selection should not be accepted")


def test_layered_replay_resolves_exact_frozen_pretruth_selection(
    tmp_path: Path,
) -> None:
    eligible = (
        replay.ExistingWorld(
            block="train_1_2",
            seed=1,
            run_root=tmp_path / "runs" / "train_1_2",
            world_root=tmp_path / "worlds" / "train_1_2",
            artifact_root=tmp_path / "artifacts" / "world_seed1",
        ),
        replay.ExistingWorld(
            block="train_1_2",
            seed=2,
            run_root=tmp_path / "runs" / "train_1_2",
            world_root=tmp_path / "worlds" / "train_1_2",
            artifact_root=tmp_path / "artifacts" / "world_seed2",
        ),
    )
    selection = {
        "schema_version": "test.selection.v1",
        "selection_seed": 17,
        "selected_worlds": [{"block": "train_1_2", "seed": 2}],
        "scorer_or_truth_opened": False,
        "new_worlds_or_datasets_created": False,
    }
    selection["selection_payload_sha256"] = replay._canonical_sha256(
        selection
    )
    path = tmp_path / "phase_a_selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")

    selected, binding = layered_replay._selected_from_frozen_pretruth_file(
        path, eligible
    )

    assert [(world.block, world.seed) for world in selected] == [
        ("train_1_2", 2)
    ]
    assert binding["selection_seed"] == 17
    assert binding["selection_payload_sha256"] == selection[
        "selection_payload_sha256"
    ]
