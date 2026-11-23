from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import multimodal_simulation_v4 as simulation  # noqa: E402
import prepare_ca_boss_v4_inputs as preparer  # noqa: E402
from channel_aware_boss_v4 import load_state_centroids  # noqa: E402


@pytest.fixture(scope="module")
def source_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("ca-boss-v4-source") / "world"
    config = simulation.SimulationV4Config(
        seed=0,
        base_cells_override=96,
        n_background_features=10,
        n_microstates=5,
        min_cells_per_microstate=4,
        alignment_derangement_diagnostic=True,
    )
    simulation.simulate_world(config, output)
    return output


@pytest.fixture(scope="module")
def prepared_inputs(
    source_world: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    output = tmp_path_factory.mktemp("ca-boss-v4-prepared") / "prepared"
    preparer.prepare_inputs(
        source_world / "discovery",
        output,
        preparer.PrepareConfig(
            n_microstates=5,
            min_cells_per_microstate=4,
        ),
    )
    return output


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _copy_discovery(source_world: Path, destination: Path) -> Path:
    copied = destination / "world" / "discovery"
    copied.parent.mkdir(parents=True)
    shutil.copytree(source_world / "discovery", copied)
    return copied


def _fake_soft_result(
    matrices: dict[str, np.ndarray],
    *,
    dispersion: float,
) -> SimpleNamespace:
    embeddings: dict[str, np.ndarray] = {}
    uncertainty: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for modality, matrix in matrices.items():
        values = np.log1p(np.asarray(matrix[:, :3], dtype=np.float64))
        values -= values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, ddof=0, keepdims=True)
        scale[scale <= 1e-12] = 1.0
        embeddings[modality] = values / scale
        uncertainty[modality] = np.full(len(values), 0.05, dtype=np.float64)
        diagnostics[modality] = {
            "near_optimal_rotation_dispersion": 0.0 if modality == "rna" else dispersion,
            "reference_coordinate_system": modality == "rna",
        }
    return SimpleNamespace(
        embeddings=embeddings,
        embedding_uncertainty=uncertainty,
        provenance={
            "schema_version": "soft_distributional_alignment.v1",
            "method": "test_observed_only_soft_distributional_alignment",
            "reference_modality": "rna",
            "distribution_diagnostics": diagnostics,
        },
    )


def test_recomputation_overrides_deranged_supplied_alignment(
    source_world: Path,
    prepared_inputs: Path,
) -> None:
    supplied = _read_json(source_world / "discovery" / "alignment_provenance.json")
    scorer_parameters = _read_json(source_world / "scorer" / "world_parameters.json")
    recomputed = _read_json(prepared_inputs / "alignment_provenance.json")

    # Scorer-side validation establishes that this fixture explicitly enabled
    # the supplied-coordinate invariance diagnostic.  It is not DOE factor D;
    # the discovery-side supplied provenance redacts the assignment either way.
    assert (
        scorer_parameters["design"]["alignment_derangement_diagnostic"] is True
    )
    assert supplied["n_feature_folds"] == 2
    assert all("config" not in fold for fold in supplied["fold_provenance"])
    assert recomputed["n_feature_folds"] == 2
    assert recomputed["alignment_config"]["derange"] is False
    policy = recomputed["input_policy"]
    assert policy["supplied_analysis_embeddings_read"] is False
    assert policy["supplied_alignment_provenance_read"] is False
    assert policy["supplied_metacells_or_membership_read"] is False
    assert policy["scorer_or_truth_read"] is False
    assert len(policy["files_read"]) == 14
    assert all(
        filename.startswith("background_matrix__")
        or filename.startswith("observed_focal__")
        for filename in policy["files_read"]
    )

    supplied_embedding = pd.read_csv(
        source_world / "discovery" / "analysis_embeddings.tsv", sep="\t"
    )
    safe_embedding = pd.read_csv(prepared_inputs / "analysis_embeddings.tsv", sep="\t")
    key = ["modality", "cell"]
    dimensions = [column for column in safe_embedding if column.startswith("integration__")]
    merged = safe_embedding.merge(
        supplied_embedding,
        on=key,
        suffixes=("_safe", "_supplied"),
        validate="one_to_one",
    )
    non_rna = merged["modality"] != "rna"
    differences = np.column_stack(
        [
            merged.loc[non_rna, f"{column}_safe"]
            - merged.loc[non_rna, f"{column}_supplied"]
            for column in dimensions
        ]
    )
    assert np.mean(np.abs(differences)) > 0.05


def test_nested_outputs_are_complete_and_use_every_observed_cell_once(
    source_world: Path,
    prepared_inputs: Path,
) -> None:
    manifest = _read_json(prepared_inputs / "preparation_manifest.json")
    assert manifest["derange"] is False
    assert manifest["nested_runs"] == 6
    assert manifest["all_cells_retained_in_every_run"] is True

    source_cells: dict[str, set[str]] = {}
    for modality in preparer.MODALITIES:
        with np.load(
            source_world / "discovery" / f"background_matrix__{modality}.npz",
            allow_pickle=False,
        ) as archive:
            source_cells[modality] = {str(cell) for cell in archive["cell_ids"]}

    expected_runs = {
        f"cells{cells}_seed{seed}"
        for cells in preparer.CELL_TARGETS
        for seed in preparer.AGGREGATION_SEEDS
    }
    run_dirs = {path.name: path for path in (prepared_inputs / "nested").iterdir()}
    assert set(run_dirs) == expected_runs
    for run_name, run_dir in run_dirs.items():
        membership = pd.read_csv(run_dir / "analysis_membership.tsv", sep="\t")
        observed = pd.read_csv(run_dir / "observed_metacell_matrix.tsv", sep="\t")
        states = pd.read_csv(run_dir / "background_state_centroids.tsv", sep="\t")
        provenance = _read_json(run_dir / "aggregation_provenance.json")

        assert provenance["supplied_alignment_used"] is False
        assert provenance["scorer_or_truth_used"] is False
        assert provenance["all_cells_used_exactly_once"] is True
        assert not membership.duplicated(["modality", "cell"]).any()
        for modality in preparer.MODALITIES:
            cells = set(membership.loc[membership["modality"] == modality, "cell"])
            assert cells == source_cells[modality]
        assert set(membership.groupby("sample_id")["modality"].nunique()) == {7}
        assert observed["sample_id"].is_unique
        assert states["sample_id"].is_unique
        assert set(observed["sample_id"]) == set(states["sample_id"])
        assert len(observed) == provenance["n_complete_metacells"] > 0
        assert sum(column.startswith("chr") for column in observed) == 36
        assert sum(column.startswith("expr__") for column in observed) == 1
        assert sum(column.startswith("libsize_background__") for column in observed) == 7
        assert "state__crossfit_uncertainty" in states
        assert np.isfinite(states["state__crossfit_uncertainty"]).all()

        loaded_states, loading_provenance = load_state_centroids(
            prepared_inputs / "analysis_embeddings.tsv",
            run_dir / "analysis_membership.tsv",
            observed["sample_id"].astype(str).tolist(),
        )
        shared_coordinates = sorted(
            column
            for column in loaded_states
            if column.startswith("state__") and "uncertainty" not in column
        )
        expected = states[["sample_id", *shared_coordinates]].sort_values("sample_id")
        actual = loaded_states[["sample_id", *shared_coordinates]].sort_values("sample_id")
        assert expected["sample_id"].tolist() == actual["sample_id"].tolist()
        np.testing.assert_allclose(
            expected[shared_coordinates],
            actual[shared_coordinates],
            rtol=1e-10,
            atol=1e-10,
        )
        assert loading_provenance["n_modalities"] == 7
        assert loading_provenance["n_state_coordinates"] == 3


def test_only_whitelisted_raw_observations_are_read(
    source_world: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = _copy_discovery(source_world, tmp_path)
    # These supplied analysis artifacts are intentionally unparsable.  A safe
    # preparer succeeds because neither is part of its read whitelist.
    (discovery / "analysis_embeddings.tsv").write_text("not\ta\tvalid\tembedding\n")
    (discovery / "alignment_provenance.json").write_text("{not-json\n")

    original_read_csv = preparer.pd.read_csv
    original_np_load = preparer.np.load
    paths_read: list[str] = []

    def guarded_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        name = Path(path).name
        assert name.startswith("observed_focal__") and name.endswith(".tsv")
        paths_read.append(name)
        return original_read_csv(path, *args, **kwargs)

    def guarded_np_load(path: object, *args: object, **kwargs: object) -> object:
        name = Path(path).name
        assert name.startswith("background_matrix__") and name.endswith(".npz")
        paths_read.append(name)
        return original_np_load(path, *args, **kwargs)

    monkeypatch.setattr(preparer.pd, "read_csv", guarded_read_csv)
    monkeypatch.setattr(preparer.np, "load", guarded_np_load)
    captured: dict[str, object] = {}
    original_alignment = preparer.state_alignment.align_cross_fitted_background_states

    def guarded_alignment(
        matrices: dict[str, np.ndarray],
        depths: dict[str, np.ndarray],
        *,
        reference_modality: str | None = None,
        n_feature_folds: int = 2,
        config: object | None = None,
    ) -> object:
        captured["modalities"] = set(matrices)
        captured["depth_modalities"] = set(depths)
        captured["reference"] = reference_modality
        captured["folds"] = n_feature_folds
        captured["derange"] = getattr(config, "derange")
        return original_alignment(
            matrices,
            depths,
            reference_modality=reference_modality,
            n_feature_folds=n_feature_folds,
            config=config,
        )

    monkeypatch.setattr(
        preparer.state_alignment,
        "align_cross_fitted_background_states",
        guarded_alignment,
    )
    output = tmp_path / "prepared"
    summary = preparer.prepare_inputs(
        discovery,
        output,
        preparer.PrepareConfig(n_microstates=5, min_cells_per_microstate=4),
    )

    assert len(paths_read) == 14
    assert set(paths_read) == set(summary["files_read"])
    assert captured == {
        "modalities": set(preparer.MODALITIES),
        "depth_modalities": set(preparer.MODALITIES),
        "reference": "rna",
        "folds": 2,
        "derange": False,
    }
    assert (output / "analysis_embeddings.tsv").is_file()


def test_scorer_truth_and_symlink_paths_are_strictly_rejected(
    source_world: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="discovery"):
        preparer.load_discovery_inputs(source_world / "scorer")

    truth_root = tmp_path / "truth"
    truth_discovery = truth_root / "discovery"
    truth_root.mkdir()
    shutil.copytree(source_world / "discovery", truth_discovery)
    with pytest.raises(ValueError, match="scorer/truth"):
        preparer.load_discovery_inputs(truth_discovery)

    linked = tmp_path / "linked" / "discovery"
    linked.parent.mkdir()
    linked.symlink_to(source_world / "scorer", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        preparer.load_discovery_inputs(linked)


def test_hidden_truth_column_and_output_overlap_are_rejected(
    source_world: Path,
    tmp_path: Path,
) -> None:
    discovery = _copy_discovery(source_world, tmp_path)
    focal_path = discovery / "observed_focal__rna.tsv"
    focal = pd.read_csv(focal_path, sep="\t")
    focal["true_z0"] = 0.0
    focal.to_csv(focal_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="forbidden discovery columns"):
        preparer.load_discovery_inputs(discovery)

    with pytest.raises(ValueError, match="separate"):
        preparer.prepare_inputs(
            source_world / "discovery",
            source_world / "discovery" / "prepared",
            preparer.PrepareConfig(n_microstates=5, min_cells_per_microstate=4),
        )


def test_soft_mode_is_explicit_and_preserves_the_prepared_output_contract(
    source_world: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def accepted_soft(
        matrices: dict[str, np.ndarray],
        depths: dict[str, np.ndarray],
        *,
        reference_modality: str,
        config: object,
    ) -> SimpleNamespace:
        calls.append(
            {
                "matrix_modalities": set(matrices),
                "depth_modalities": set(depths),
                "reference": reference_modality,
                "n_feature_folds": getattr(config, "n_feature_folds"),
                "n_microstates": getattr(config, "n_microstates"),
            }
        )
        return _fake_soft_result(matrices, dispersion=0.08)

    monkeypatch.setattr(
        preparer.soft_alignment,
        "align_soft_distributional_background_states",
        accepted_soft,
    )
    output = tmp_path / "soft-prepared"
    summary = preparer.prepare_inputs(
        source_world / "discovery",
        output,
        preparer.PrepareConfig(
            n_microstates=5,
            min_cells_per_microstate=4,
            alignment_mode="soft",
            soft_max_rotation_dispersion=0.10,
        ),
    )

    assert calls == [
        {
            "matrix_modalities": set(preparer.MODALITIES),
            "depth_modalities": set(preparer.MODALITIES),
            "reference": "rna",
            "n_feature_folds": 2,
            "n_microstates": 24,
        }
    ]
    assert summary["alignment_mode_requested"] == "soft"
    assert summary["alignment_mode_selected"] == "soft"
    assert summary["soft_alignment_abstained"] is False
    provenance = _read_json(output / "alignment_provenance.json")
    assert provenance["alignment_mode_requested"] == "soft"
    assert provenance["alignment_mode_selected"] == "soft"
    assert provenance["alignment_selection"]["confidence_gate"]["passed"] is True
    assert provenance["alignment_config"]["soft_max_rotation_dispersion"] == 0.10
    assert provenance["input_policy"]["scorer_or_truth_read"] is False
    assert len(provenance["input_policy"]["files_read"]) == 14

    embedding = pd.read_csv(output / "analysis_embeddings.tsv", sep="\t")
    assert set(embedding["modality"]) == set(preparer.MODALITIES)
    assert sum(column.startswith("integration__dim") for column in embedding) == 3
    assert np.isfinite(embedding.select_dtypes(include=[np.number])).all().all()
    expected_runs = {
        f"cells{cells}_seed{seed}"
        for cells in preparer.CELL_TARGETS
        for seed in preparer.AGGREGATION_SEEDS
    }
    assert {path.name for path in (output / "nested").iterdir()} == expected_runs
    for run_name in expected_runs:
        run_dir = output / "nested" / run_name
        assert {path.name for path in run_dir.iterdir()} == {
            "aggregation_provenance.json",
            "analysis_membership.tsv",
            "background_state_centroids.tsv",
            "observed_metacell_matrix.tsv",
        }
        aggregation = _read_json(run_dir / "aggregation_provenance.json")
        assert aggregation["alignment_mode_selected"] == "soft"
        assert aggregation["coordinate_source"] == (
            "recomputed_soft_distributional_background_coordinates"
        )
        assert aggregation["all_cells_used_exactly_once"] is True


def test_soft_confidence_gate_falls_back_or_explicitly_abstains(
    source_world: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = preparer.load_discovery_inputs(source_world / "discovery")
    hard = preparer.recompute_alignment(
        inputs,
        preparer.PrepareConfig(
            n_microstates=5,
            min_cells_per_microstate=4,
            alignment_mode="hard",
        ),
    )
    monkeypatch.setattr(preparer, "_hard_alignment", lambda *_args, **_kwargs: hard)
    monkeypatch.setattr(
        preparer.soft_alignment,
        "align_soft_distributional_background_states",
        lambda matrices, *_args, **_kwargs: _fake_soft_result(
            matrices, dispersion=0.40
        ),
    )
    fallback = preparer.recompute_alignment(
        inputs,
        preparer.PrepareConfig(
            n_microstates=5,
            min_cells_per_microstate=4,
            alignment_mode="soft",
            soft_max_rotation_dispersion=0.10,
            soft_low_confidence_action="fallback_hard",
        ),
    )
    selection = fallback.provenance["preparation_alignment_selection"]
    assert selection["requested_mode"] == "soft"
    assert selection["selected_mode"] == "hard"
    assert selection["soft_alignment_abstained"] is True
    assert selection["confidence_gate"]["passed"] is False
    assert set(selection["confidence_gate"]["failed_modalities"]) == {
        modality for modality in preparer.MODALITIES if modality != "rna"
    }
    for modality in preparer.MODALITIES:
        np.testing.assert_allclose(fallback.embeddings[modality], hard.embeddings[modality])

    partial = preparer.recompute_alignment(
        inputs,
        preparer.PrepareConfig(
            n_microstates=5,
            min_cells_per_microstate=4,
            alignment_mode="soft",
            soft_max_rotation_dispersion=0.10,
            soft_low_confidence_action="partial_soft",
        ),
    )
    partial_selection = partial.provenance["preparation_alignment_selection"]
    assert partial_selection["selected_mode"] == "soft_partial"
    assert partial_selection["soft_alignment_abstained"] is True
    assert partial_selection["cross_assay_axis_eligible_modalities"] == ["rna"]
    assert set(partial_selection["cross_assay_axis_abstained_modalities"]) == {
        modality for modality in preparer.MODALITIES if modality != "rna"
    }
    assert any(
        not np.allclose(partial.embeddings[modality], hard.embeddings[modality])
        for modality in preparer.MODALITIES
    )

    with pytest.raises(RuntimeError, match="confidence gate abstained"):
        preparer.recompute_alignment(
            inputs,
            preparer.PrepareConfig(
                n_microstates=5,
                min_cells_per_microstate=4,
                alignment_mode="soft",
                soft_max_rotation_dispersion=0.10,
                soft_low_confidence_action="abstain",
            ),
        )


def test_synchronized_soft_gate_can_resolve_a_weak_reference_anchor() -> None:
    non_reference = [
        modality for modality in preparer.MODALITIES if modality != "rna"
    ]
    provenance = {
        "distribution_diagnostics": {
            modality: {"near_optimal_rotation_dispersion": 0.90}
            for modality in non_reference
        },
        "synchronization": {
            "assay_synchronized": {
                modality: True for modality in preparer.MODALITIES
            },
            "assay_alignment_confidence": {
                modality: 0.75 for modality in preparer.MODALITIES
            },
        },
    }

    accepted = preparer._soft_confidence_gate(
        provenance,
        preparer.PrepareConfig(soft_max_rotation_dispersion=0.10),
    )
    assert accepted["passed"] is True
    assert accepted["metric"] == (
        "confidence_gated_multi_assay_rotation_synchronization"
    )

    provenance["synchronization"]["assay_synchronized"][non_reference[0]] = False
    rejected = preparer._soft_confidence_gate(
        provenance,
        preparer.PrepareConfig(soft_max_rotation_dispersion=0.10),
    )
    assert rejected["passed"] is False
    assert rejected["failed_modalities"][non_reference[0]] == (
        "rotation_synchronization_abstained"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alignment_mode": "automatic"}, "alignment_mode"),
        ({"soft_n_microstates": 3}, "soft_n_microstates"),
        ({"soft_max_rotation_dispersion": -0.1}, "rotation_dispersion"),
        ({"soft_low_confidence_action": "ignore"}, "low_confidence_action"),
    ],
)
def test_soft_alignment_configuration_is_frozen_and_validated(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        preparer.PrepareConfig(**kwargs)
