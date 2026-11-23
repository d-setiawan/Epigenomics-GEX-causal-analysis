from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import multimodal_simulation_v4 as simulation  # noqa: E402


@pytest.fixture(scope="module")
def written_world(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("multimodal-v4") / "world"
    config = simulation.SimulationV4Config(
        seed=0,
        base_cells_override=96,
        n_background_features=10,
        n_microstates=5,
        min_cells_per_microstate=4,
    )
    simulation.simulate_world(config, output)
    return output


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _is_acyclic(nodes: set[str], edges: pd.DataFrame) -> bool:
    children = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in edges.itertuples(index=False):
        children[str(edge.source)].append(str(edge.target))
        indegree[str(edge.target)] += 1
    queue = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == len(nodes)


def test_preregistered_doe_and_role_randomization_are_seeded() -> None:
    for seed in range(1, 9):
        design = simulation.design_for_seed(seed)
        assert design["D_background_observability"] == (
            design["A_cells"]
            * design["B_detection"]
            * design["C_population_divergence"]
        )
        assert design["base_cells"] in {1200, 2400}
        assert design["background_integration_stress"] == (
            design["D_background_observability"] > 0
        )
        assert design["background_observability_stress_override"] is None
        assert design["alignment_derangement_diagnostic"] is False
        assert "D_alignment" not in design
        assert "deranged_alignment" not in design
        assert design["half_fraction_cell"] == seed
        assert design["half_fraction_index_zero_based"] == seed - 1

    for block_seed in (30_001, 31_001, 42_001):
        design = simulation.design_for_seed(block_seed)
        assert design["half_fraction_cell"] == 1
        assert design["half_fraction_index_zero_based"] == 0
        assert (
            design["A_cells"],
            design["B_detection"],
            design["C_population_divergence"],
            design["D_background_observability"],
        ) == (-1, -1, -1, -1)

    diagnostic = simulation.design_for_seed(
        1, alignment_derangement_diagnostic=True
    )
    assert diagnostic["D_background_observability"] == -1
    assert diagnostic["background_integration_stress"] is False
    assert diagnostic["alignment_derangement_diagnostic"] is True

    config0 = simulation.SimulationV4Config(
        seed=0,
        base_cells_override=16,
        n_background_features=4,
        n_microstates=4,
        min_cells_per_microstate=2,
    )
    config1 = simulation.SimulationV4Config(
        seed=1,
        base_cells_override=16,
        n_background_features=4,
        n_microstates=4,
        min_cells_per_microstate=2,
    )
    parameters0, roles0 = simulation._draw_parameter_table(config0)
    parameters1, roles1 = simulation._draw_parameter_table(config1)
    for roles in (roles0, roles1):
        assert set(roles["detection_draw_stage"]) == {"before_role_assignment"}
        for mark in simulation.MARKS:
            mark_roles = roles.loc[roles["mark"] == mark]
            assert set(mark_roles["role"]) == set(simulation.ROLES)
            assert set(mark_roles["slot"]) == set(range(6))
            assert set(mark_roles["detection_draw_index"]) == set(range(6))
    slot_roles0 = tuple(roles0.sort_values(["mark", "slot"])["role"])
    slot_roles1 = tuple(roles1.sort_values(["mark", "slot"])["role"])
    assert slot_roles0 != slot_roles1
    assert parameters0["slow_mark"] != parameters0["fast_mark"]
    assert parameters1["slow_mark"] != parameters1["fast_mark"]
    assert parameters0["slow_mark"] != parameters1["slow_mark"]
    assert (
        parameters0["p_state_loading"][parameters0["slow_mark"]]
        > parameters0["p_state_loading"][parameters0["fast_mark"]]
    )
    assert (
        parameters0["p_residual_scale"][parameters0["slow_mark"]]
        < parameters0["p_residual_scale"][parameters0["fast_mark"]]
    )
    for parameter in (parameters0, parameters1):
        for field in (
            "gamma_p_to_c_detection_log_odds",
            "gamma_q_to_c_detection_log_odds",
        ):
            assert all(1.4 <= float(value) <= 1.8 for value in parameter[field].values())


def test_background_observability_treatment_changes_only_the_observation_path() -> None:
    common = dict(
        seed=17,
        base_cells_override=24,
        n_background_features=5,
        n_microstates=4,
        min_cells_per_microstate=2,
    )
    low = simulation.generate_world(
        simulation.SimulationV4Config(
            **common,
            background_observability_stress_override=False,
        )
    )
    high = simulation.generate_world(
        simulation.SimulationV4Config(
            **common,
            background_observability_stress_override=True,
        )
    )

    assert low.design["D_background_observability"] == high.design[
        "D_background_observability"
    ]
    assert low.design["background_integration_stress"] is False
    assert high.design["background_integration_stress"] is True
    pd.testing.assert_frame_equal(low.roles, high.roles)
    pd.testing.assert_frame_equal(low.truth_edges, high.truth_edges)
    pd.testing.assert_frame_equal(low.dag_nodes, high.dag_nodes)
    pd.testing.assert_frame_equal(
        low.dag_edges[["source", "target", "mechanism", "condition"]],
        high.dag_edges[["source", "target", "mechanism", "condition"]],
    )
    low_noise_edges = low.dag_edges.loc[
        low.dag_edges["mechanism"] == "assay_private_background_proxy_noise"
    ]
    high_noise_edges = high.dag_edges.loc[
        high.dag_edges["mechanism"] == "assay_private_background_proxy_noise"
    ]
    assert len(low_noise_edges) == len(high_noise_edges) == 21
    assert not low_noise_edges["active_in_world"].any()
    assert high_noise_edges["active_in_world"].all()
    assert set(low_noise_edges["coefficient"].astype(float)) == {0.0}
    assert set(high_noise_edges["coefficient"].astype(float)) == {
        simulation.BACKGROUND_PROXY_NOISE_SD
    }

    for modality in simulation.MODALITIES:
        np.testing.assert_array_equal(
            low.latent_states[modality], high.latent_states[modality]
        )
        np.testing.assert_array_equal(
            low.latent_modes[modality], high.latent_modes[modality]
        )
        np.testing.assert_array_equal(
            low.background_technical_scale[modality],
            high.background_technical_scale[modality],
        )
        np.testing.assert_array_equal(
            low.background_state_proxy[modality], low.latent_states[modality]
        )
        assert np.count_nonzero(high.background_proxy_noise[modality]) > 0
        np.testing.assert_allclose(
            high.background_state_proxy[modality],
            high.latent_states[modality] + high.background_proxy_noise[modality],
        )
        # These are exactly the raw, whitelisted matrices written to discovery/.
        assert not np.array_equal(low.background[modality], high.background[modality])

    low_model = low.parameters["background_observation_model"]
    high_model = high.parameters["background_observation_model"]
    assert low_model["background_state_proxy_noise_sd"] == 0.0
    assert high_model["background_state_proxy_noise_sd"] == (
        simulation.BACKGROUND_PROXY_NOISE_SD
    )
    assert low.parameters["conditional_dag"][
        "truth_topology_invariant_across_condition"
    ] is True
    assert high.parameters["conditional_dag"][
        "focal_activities_use_true_latent_state"
    ] is True


def test_alignment_adapter_receives_background_and_depth_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = simulation.SimulationV4Config(
        seed=0,
        base_cells_override=32,
        n_background_features=6,
        n_microstates=4,
        min_cells_per_microstate=2,
    )
    world = simulation.generate_world(config)
    original = simulation.state_alignment.align_cross_fitted_background_states
    captured: dict[str, object] = {}

    def spy(
        background_matrices: dict[str, np.ndarray],
        background_depths: dict[str, np.ndarray],
        *,
        reference_modality: str | None = None,
        n_feature_folds: int = 2,
        config: object | None = None,
    ) -> object:
        captured["matrices"] = background_matrices
        captured["depths"] = background_depths
        captured["reference"] = reference_modality
        captured["n_feature_folds"] = n_feature_folds
        captured["config"] = config
        return original(
            background_matrices,
            background_depths,
            reference_modality=reference_modality,
            n_feature_folds=n_feature_folds,
            config=config,
        )

    monkeypatch.setattr(
        simulation.state_alignment, "align_cross_fitted_background_states", spy
    )
    result = simulation._alignment(world)

    assert captured["matrices"] is world.background
    assert captured["depths"] is world.background_depth
    assert captured["reference"] == "rna"
    assert captured["n_feature_folds"] == 2
    assert captured["config"].derange is False
    assert world.design["background_integration_stress"] is True
    assert set(result.embeddings) == set(simulation.MODALITIES)
    assert len(result.fold_results) == 2
    assert result.provenance["n_feature_folds"] == 2
    for modality in simulation.MODALITIES:
        assert result.embedding_uncertainty[modality].shape == (
            len(world.cell_ids[modality]),
        )
        assert np.isfinite(result.embedding_uncertainty[modality]).all()
        assert np.all(result.embedding_uncertainty[modality] >= 0)
    boundary = result.provenance["information_boundary"]
    assert boundary["feature_columns_compared_across_assays"] is False
    assert result.provenance["information_boundary"]["cell_rows_compared_across_assays"] is False


def test_discovery_artifacts_contain_observations_not_simulator_truth(
    written_world: Path,
) -> None:
    discovery = written_world / "discovery"
    scorer = written_world / "scorer"
    manifest = _read_json(discovery / "manifest.json")
    summary = _read_json(written_world / "run_summary.json")
    embeddings = pd.read_csv(discovery / "analysis_embeddings.tsv", sep="\t")
    alignment_text = (discovery / "alignment_provenance.json").read_text().lower()

    assert scorer.is_dir()
    assert not any(path.name.startswith("truth") for path in discovery.rglob("*"))
    assert all(not str(name).startswith("../scorer") for name in manifest["files"])
    assert manifest["assay_cells_are_independently_sampled"] is True
    assert "integration_uncertainty" in embeddings
    assert len(embeddings) == sum(summary["cell_counts"].values())
    assert np.isfinite(embeddings["integration_uncertainty"]).all()
    assert (embeddings["integration_uncertainty"] >= 0).all()
    assert "derang" not in alignment_text
    assert "applied_assignment" not in alignment_text
    assert "inferred_assignment" not in alignment_text
    assert manifest["nested_design"] == {
        "all_cells_retained_in_each_run": True,
        "cell_targets": [24, 14, 8],
        "seeds": [0, 1],
    }

    forbidden_columns = ("true", "truth", "role", "parent", "child", "population_mode")
    for path in discovery.rglob("*.tsv"):
        columns = [str(column).lower() for column in pd.read_csv(path, sep="\t", nrows=0)]
        assert not any(
            token in column for token in forbidden_columns for column in columns
        ), path

    all_ids: set[str] = set()
    background_widths: set[int] = set()
    for modality in simulation.MODALITIES:
        archive_path = discovery / f"background_matrix__{modality}.npz"
        with np.load(archive_path) as archive:
            assert set(archive.files) == {"matrix", "cell_ids", "background_depth"}
            assert archive["matrix"].shape[0] == summary["cell_counts"][modality]
            assert archive["background_depth"].shape == (summary["cell_counts"][modality],)
            assert np.all(archive["background_depth"] > 0)
            background_widths.add(int(archive["matrix"].shape[1]))
            ids = {str(cell) for cell in archive["cell_ids"]}
        assert all_ids.isdisjoint(ids)
        all_ids.update(ids)

        focal = pd.read_csv(discovery / f"observed_focal__{modality}.tsv", sep="\t")
        assert len(focal) == summary["cell_counts"][modality]
        assert focal["cell"].is_unique
        assert set(focal["modality"]) == {modality}
        biological = [
            column
            for column in focal
            if column.startswith("chr") or column.startswith("expr__")
        ]
        assert len(biological) == (1 if modality == "rna" else 6)
    # Unequal widths prevent accidental positional background-feature matching.
    assert len(background_widths) > 1

    # Observation proxies and their technical realizations remain scorer-only.
    assert not any("proxy" in path.name for path in discovery.rglob("*"))
    for modality in simulation.MODALITIES:
        with np.load(scorer / f"cell_ground_truth__{modality}.npz") as archive:
            assert set(archive.files) == {
                "cell_ids",
                "latent_state",
                "population_mode",
                "background_state_proxy",
                "background_proxy_noise",
                "background_technical_scale",
            }
            np.testing.assert_allclose(
                archive["background_state_proxy"],
                archive["latent_state"] + archive["background_proxy_noise"],
            )
            assert np.all(archive["background_technical_scale"] > 0)


def test_typed_truth_and_explicit_dag_match_the_planted_motifs(written_world: Path) -> None:
    scorer = written_world / "scorer"
    truth = pd.read_csv(scorer / "truth_typed_edges.tsv", sep="\t")
    roles = pd.read_csv(scorer / "role_assignments.tsv", sep="\t")
    parameters = _read_json(scorer / "world_parameters.json")

    required = {
        "node1",
        "node2",
        "truth_type",
        "label",
        "mark",
        "parent",
        "child",
        "primary",
        "orientation_credit",
        "node_set1",
        "node_set2",
        "estimand",
    }
    assert required <= set(truth.columns)
    assert set(truth["truth_type"]) == simulation.TRUTH_TYPES
    counts = truth.groupby("truth_type").size().to_dict()
    assert counts == {
        "cross_assay_shared_state": 3,
        "fast_peak_gene_challenge": 1,
        "sentinel_null": 15,
        "slow_peak_gene": 1,
        "within_assay_direct": 12,
        "within_assay_shared_state": 6,
    }

    node_to_role = roles.set_index("node")["role"].to_dict()
    direct = truth.loc[truth["truth_type"] == "within_assay_direct"]
    assert direct["orientation_credit"].all()
    assert direct["primary"].all()
    assert all(node_to_role[parent] in {"P", "Q"} for parent in direct["parent"])
    assert all(node_to_role[child] == "C" for child in direct["child"])
    assert (direct["node1"] == direct["parent"]).all()
    assert (direct["node2"] == direct["child"]).all()

    slow = truth.loc[truth["truth_type"] == "slow_peak_gene"].iloc[0]
    fast = truth.loc[truth["truth_type"] == "fast_peak_gene_challenge"].iloc[0]
    assert slow["mark"] == parameters["slow_mark"]
    assert fast["mark"] == parameters["fast_mark"]
    assert slow["mark"] != fast["mark"]
    assert slow["child"] == fast["child"] == simulation.GENE
    assert not bool(slow["orientation_credit"])
    assert not bool(fast["orientation_credit"])
    assert slow["estimand"] == "latent_loading_equivalence_class_support_hypothesis"

    role_lookup = roles.set_index(["mark", "role"])
    slow_dimension = int(parameters["state_dimension"][parameters["slow_mark"]])

    def expected_pure_nodes(mark: str, dimension: int) -> set[str]:
        expected: set[str] = set()
        if int(parameters["state_dimension"][mark]) == dimension:
            expected |= {
                str(role_lookup.loc[(mark, "P"), "node"]),
                str(role_lookup.loc[(mark, "S"), "node"]),
            }
        if int(parameters["q_dimension"][mark]) == dimension:
            expected.add(str(role_lookup.loc[(mark, "Q"), "node"]))
        return expected

    expected_slow_group = set().union(
        *(expected_pure_nodes(mark, slow_dimension) for mark in simulation.MARKS)
    )
    assert set(str(slow["node_set1"]).split(";")) == expected_slow_group
    assert slow["parent"] in expected_slow_group

    cross = truth.loc[truth["truth_type"] == "cross_assay_shared_state"]
    assert set(cross["estimand"]) == {"latent_loading_equivalence_class"}
    for row in cross.itertuples(index=False):
        left_mark, right_mark = str(row.mark).split("|")
        dimension = int(parameters["state_dimension"][left_mark])
        assert int(parameters["state_dimension"][right_mark]) == dimension
        assert set(str(row.node_set1).split(";")) == expected_pure_nodes(
            left_mark, dimension
        )
        assert set(str(row.node_set2).split(";")) == expected_pure_nodes(
            right_mark, dimension
        )

    sentinel = truth.loc[truth["truth_type"] == "sentinel_null"]
    assert sentinel["label"].str.contains("detection_matched_null").all()
    assert not sentinel["primary"].any()
    assert not sentinel["orientation_credit"].any()

    dag_nodes = pd.read_csv(scorer / "generative_dag_nodes.tsv", sep="\t")
    dag_edges = pd.read_csv(scorer / "generative_dag_edges.tsv", sep="\t")
    nodes = set(dag_nodes["node"].astype(str))
    assert set(dag_edges["source"].astype(str)) <= nodes
    assert set(dag_edges["target"].astype(str)) <= nodes
    assert _is_acyclic(nodes, dag_edges)
    assert (dag_edges["mechanism"] == "slow_state_pairable_direct").sum() == 1
    assert (dag_edges["mechanism"] == "fast_residual_unpaired_direct").sum() == 1
    assert (dag_edges["mechanism"] == "direct").sum() == 12
    assert (dag_edges["mechanism"] == "observed_coaccessibility_log_odds").sum() == 12
    assert (dag_edges["mechanism"] == "within_state_dependence").sum() == 7
    assert (dag_nodes["node_type"] == "latent_mixture_mode").sum() == 7
    assert (dag_nodes["node_type"] == "latent_technical_scale").sum() == 7
    assert (dag_nodes["node_type"] == "observed_background_count").sum() > 0
    assert (dag_nodes["node_type"] == "latent_background_state_proxy").sum() == 21
    assert (dag_nodes["node_type"] == "latent_background_proxy_noise").sum() == 21
    assert dag_nodes["node"].is_unique
    assert set(dag_edges["condition"]) == {
        "always",
        "background_integration_stress",
    }
    assert dag_edges["active_in_world"].all()

    # Incidence audit: raw background variables receive state information only
    # through assay-private proxies, plus the separate technical scale.
    node_types = dag_nodes.set_index("node")["node_type"].to_dict()
    incoming = {
        node: dag_edges.loc[dag_edges["target"] == node]
        for node in nodes
    }
    outgoing = {
        node: dag_edges.loc[dag_edges["source"] == node]
        for node in nodes
    }
    for modality in simulation.MODALITIES:
        n_background = int(parameters["n_background_features_by_modality"][modality])
        background_nodes = {
            node
            for node, node_type in node_types.items()
            if node_type == "observed_background_count"
            and node.startswith(f"background_count__{modality}__")
        }
        proxy_nodes = {
            f"background_state_proxy__{modality}__dim{dimension}"
            for dimension in range(3)
        }
        noise_nodes = {
            f"background_proxy_noise__{modality}__dim{dimension}"
            for dimension in range(3)
        }
        assert len(background_nodes) == n_background
        for background_node in background_nodes:
            parents = incoming[background_node]
            assert set(parents["source"]) == proxy_nodes | {f"technical_scale__{modality}"}
            assert set(parents["mechanism"]) == {
                "assay_private_background_loading",
                "technical_scale",
            }
            assert not any(
                str(source).startswith(f"state__{modality}__")
                for source in parents["source"]
            )
            children = outgoing[background_node]
            assert set(children["target"]) == {f"depth__{modality}"}
            assert set(children["mechanism"]) == {"deterministic_sum"}
        for dimension, proxy_node in enumerate(sorted(proxy_nodes)):
            proxy_parents = incoming[proxy_node]
            assert set(proxy_parents["source"]) == {
                f"state__{modality}__dim{dimension}",
                f"background_proxy_noise__{modality}__dim{dimension}",
            }
            assert set(outgoing[proxy_node]["target"]) == background_nodes
        for noise_node in noise_nodes:
            assert incoming[noise_node].empty
            noise_children = outgoing[noise_node]
            assert len(noise_children) == 1
            assert set(noise_children["mechanism"]) == {
                "assay_private_background_proxy_noise"
            }
            assert set(noise_children["condition"]) == {
                "background_integration_stress"
            }

    model = parameters["background_observation_model"]
    conditional = parameters["conditional_dag"]
    assert model["background_latent_input"] == "assay_private_background_state_proxy"
    assert model["focal_activity_latent_input"] == "true_latent_state"
    assert conditional["doe_source"] == "D_background_observability"
    assert conditional["background_path"] == [
        "latent_state",
        "assay_private_background_state_proxy",
        "observed_background_count",
    ]
    assert conditional["alignment_derangement_is_doe_factor"] is False


def test_all_six_nested_runs_retain_every_cell_once(written_world: Path) -> None:
    discovery = written_world / "discovery"
    summary = _read_json(written_world / "run_summary.json")
    expected = {
        f"cells{cells}_seed{seed}"
        for cells in simulation.AGGREGATION_CELL_TARGETS
        for seed in simulation.AGGREGATION_SEEDS
    }
    run_directories = {path.name: path for path in (discovery / "nested").iterdir()}
    assert set(run_directories) == expected

    for name, directory in run_directories.items():
        provenance = _read_json(directory / "aggregation_provenance.json")
        membership = pd.read_csv(directory / "analysis_membership.tsv", sep="\t")
        observed = pd.read_csv(directory / "observed_metacell_matrix.tsv", sep="\t")
        state = pd.read_csv(directory / "background_state_centroids.tsv", sep="\t")

        assert provenance["all_cells_used_exactly_once"] is True
        assert provenance["simulator_labels_used"] is False
        assert provenance["focal_values_used_for_assignment"] is False
        assert provenance["paired_cell_identity_used"] is False
        assert provenance["cells_used"] == provenance["cells_total"] == summary["cell_counts"]
        assert not membership.duplicated(["modality", "cell"]).any()
        assert membership.groupby("modality").size().to_dict() == summary["cell_counts"]
        assert set(membership.groupby("sample_id")["modality"].nunique()) == {7}
        assert observed["sample_id"].is_unique
        assert state["sample_id"].is_unique
        assert set(observed["sample_id"]) == set(state["sample_id"])
        assert len(observed) == provenance["n_complete_metacells"] > 0
        assert sum(column.startswith("chr") for column in observed) == 36
        assert simulation.GENE in observed
        assert sum(column.startswith("state__consensus__") for column in state) == 3
        assert "state__assay_uncertainty" in state
        assert "state__crossfit_uncertainty" in state
        assert (state["state__crossfit_uncertainty"] >= 0).all()


def test_simulation_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("user data\n")
    config = simulation.SimulationV4Config(
        seed=0,
        base_cells_override=16,
        n_background_features=4,
        n_microstates=4,
        min_cells_per_microstate=2,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        simulation.simulate_world(config, output)
    assert (output / "keep.txt").read_text() == "user data\n"
