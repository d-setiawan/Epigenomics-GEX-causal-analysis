"""Run with unittest; no pytest dependency or benchmark generation required."""

from __future__ import annotations

import importlib.util
import itertools
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("recoverability_evaluate", MODULE)
ev = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ev
SPEC.loader.exec_module(ev)
HAVE_CAUSALLEARN = importlib.util.find_spec("causallearn") is not None
HAVE_ANNDATA = importlib.util.find_spec("anndata") is not None


def fixture_truth():
    assays = np.repeat(ev.ASSAYS, 3)
    names = [f"{assay}_{role}" for assay in ev.ASSAYS for role in ("core", "sentinel")]
    counts = np.arange(9 * 6).reshape(9, 6)
    return dict(
        cell_ids=np.asarray([f"opaque_{index}" for index in range(9)]),
        assigned_assay=assays, state=np.arange(9)[:, None] / 9,
        activities=counts.astype(float) - 10,
        exposures=np.ones((9, 3)), candidate_names=np.asarray(names),
        candidate_assays=np.repeat(ev.ASSAYS, 2),
        candidate_roles=np.tile(("core", "sentinel"), 3),
        candidate_feature_ids=np.asarray([f"f{index}" for index in range(6)]),
        counts_realistic=counts,
    )


def candidates_from_truth(truth):
    return [dict(name=str(name), assay=str(assay), role=str(role))
            for name, assay, role in zip(truth["candidate_names"], truth["candidate_assays"], truth["candidate_roles"])]


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.truth = ev.validate_truth(fixture_truth(), "realistic")
        self.observed = {assay: self.truth["cell_ids"][self.truth["assigned_assay"] == assay] for assay in ev.ASSAYS}
        self.run = {assay: {"cell_ids": ids.copy(), "shared": np.arange(len(ids))[:, None].astype(float)}
                    for assay, ids in self.observed.items()}

    def test_common_pool_intersects_candidate_ids_not_only_anchor(self):
        second = {assay: dict(value) for assay, value in self.run.items()}
        second["H3K27ac"] = {"cell_ids": self.run["H3K27ac"]["cell_ids"][1:],
                             "shared": self.run["H3K27ac"]["shared"][1:]}
        pools, audit = ev.select_common_pools({"first": self.run, "second": second}, self.observed)
        self.assertEqual({len(pool) for pool in pools.values()}, {2})
        self.assertNotIn(self.observed["H3K27ac"][0], pools["H3K27ac"])
        repeat, _ = ev.select_common_pools({"second": second, "first": self.run}, self.observed)
        for assay in pools:
            np.testing.assert_array_equal(pools[assay], repeat[assay])
        self.assertEqual(audit["by_assay"]["RNA"]["common_before_equalization"], 3)

    def test_reject_nonfinite_embeddings_not_silent_filter(self):
        self.run["RNA"]["shared"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            ev.select_common_pools({"bad": self.run}, self.observed)

    def test_reject_unknown_ids(self):
        self.run["RNA"]["cell_ids"][0] = "stranger"
        with self.assertRaisesRegex(ValueError, "non-observed"):
            ev.select_common_pools({"bad": self.run}, self.observed)

    def test_euclidean_global_assignment_and_no_reuse(self):
        assignment = ev.match_one_to_one(self.observed, self.run)
        for assay in ev.ASSAYS:
            self.assertEqual(len(set(assignment["cell_ids"][assay])), 3)
        self.assertEqual(assignment["costs"]["H3K27ac"]["sum_euclidean"], 0)
        readout = ev.assemble_readout(assignment, self.truth)
        np.testing.assert_array_equal(readout["counts"][:, 2:4], self.truth["counts"][3:6, 2:4])
        # These values are explicitly NOT the RNA anchor's hidden mark counts.
        self.assertFalse(np.array_equal(readout["counts"][:, 2:4], self.truth["counts"][:3, 2:4]))

    def test_random_assignment_reproducible_without_truth(self):
        first = ev.match_one_to_one(self.observed, random_seed=17)
        second = ev.match_one_to_one(self.observed, random_seed=17)
        for assay in ev.ASSAYS:
            np.testing.assert_array_equal(first["cell_ids"][assay], second["cell_ids"][assay])
            self.assertEqual(set(first["cell_ids"][assay]), set(self.observed[assay]))

    def test_readout_rejects_hidden_partner_substitution(self):
        assignment = ev.match_one_to_one(self.observed, self.run)
        assignment["cell_ids"]["H3K27ac"] = assignment["cell_ids"]["RNA"]
        with self.assertRaisesRegex(ValueError, "not observed"):
            ev.assemble_readout(assignment, self.truth)

    def test_readout_rejects_cached_cell_reuse(self):
        assignment = ev.match_one_to_one(self.observed, self.run)
        assignment["cell_ids"]["H3K27ac"][1] = assignment["cell_ids"]["H3K27ac"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            ev.assemble_readout(assignment, self.truth)

    def test_unequal_pools_not_implicitly_resampled(self):
        self.observed["RNA"] = self.observed["RNA"][:-1]
        with self.assertRaisesRegex(ValueError, "equal-size"):
            ev.match_one_to_one(self.observed, random_seed=0)

    def test_dependence_constant_is_undefined_not_zero(self):
        values = np.column_stack((np.zeros(30), np.arange(30)))
        candidates = [dict(name="x", assay="RNA", role="core"), dict(name="y", assay="H3K27ac", role="core")]
        result = ev.dependence_metrics(values, values, candidates, scale="activities")[0]
        self.assertIsNone(result["pearson"])
        self.assertEqual(result["status"], "undefined_constant")


class GraphTruthTests(unittest.TestCase):
    def test_dsep_chain_and_collider(self):
        chain = [("a", "b"), ("b", "r")]
        collider = [("a", "r"), ("b", "r")]
        self.assertFalse(ev.d_separated(chain, "a", "r"))
        self.assertTrue(ev.d_separated(chain, "a", "r", ["b"]))
        self.assertTrue(ev.d_separated(collider, "a", "b"))
        self.assertFalse(ev.d_separated(collider, "a", "b", ["r"]))

    def test_dsep_collider_descendant(self):
        dag = [("a", "r"), ("b", "r"), ("r", "d")]
        self.assertFalse(ev.d_separated(dag, "a", "b", ["d"]))

    def test_reject_cyclic_truth(self):
        with self.assertRaisesRegex(ValueError, "not a DAG"):
            ev.d_separated([("a", "b"), ("b", "a")], "a", "b")

    def test_measurement_graph_does_not_reuse_activity_ci(self):
        candidates = [dict(name="a", assay="H3K27ac"), dict(name="b", assay="H3K4me3"),
                      dict(name="r", assay="RNA"), dict(name="n", assay="RNA")]
        edges, observed = ev.measurement_dag([("a", "b"), ("b", "r")], candidates)
        self.assertFalse(ev.d_separated(edges, "count::a", "count::r", ["count::b"]))
        self.assertFalse(ev.d_separated(edges, "count::r", "count::n"))
        self.assertTrue(ev.d_separated(edges, "count::a", "count::n"))
        self.assertEqual(len(observed), 4)

    def test_three_node_family_has_25_not_27_dags(self):
        self.assertEqual(len(ev.enumerate_three_node_dags(["a", "b", "r"])), 25)
        chain = ev.markov_equivalence_class_three_nodes(["a", "b", "r"], [("a", "b"), ("b", "r")])
        collider = ev.markov_equivalence_class_three_nodes(["a", "b", "r"], [("a", "r"), ("b", "r")])
        self.assertEqual(chain["oracle_markov_equivalence_size"], 3)
        self.assertEqual(collider["oracle_markov_equivalence_size"], 1)

    @unittest.skipUnless(HAVE_CAUSALLEARN, "causal-learn lives in a separate environment")
    def test_exact_oracle_graphs(self):
        edges = [("a", "b"), ("b", "r")]
        oracle = ev.oracle_graph(edges, ["a", "b", "r", "n"], method="pc")
        self.assertEqual(oracle["status"], "ok", oracle)
        self.assertEqual(set(ev._edge_map(oracle["edges"])), {("a", "b"), ("b", "r")})
        count_edges, count_names = ev.measurement_dag(edges, [dict(name=n, assay=m) for n, m in zip(("a", "b", "r"), ev.ASSAYS)])
        count_oracle = ev.oracle_graph(count_edges, count_names, method="fci")
        self.assertEqual(count_oracle["status"], "ok", count_oracle)
        self.assertEqual(len(count_oracle["edges"]), 3)


class DiscoveryTests(unittest.TestCase):
    def test_constants_fail_not_empty_graph(self):
        result = ev.discover(np.zeros((30, 3)), ["a", "b", "c"], method="fci", scale="counts")
        self.assertEqual(result["status"], "failed")
        self.assertIn("Constant columns", result["error"])
        self.assertIsNone(result["edges"])

    def test_logged_counts_cannot_be_logged_twice(self):
        values = np.log1p(np.arange(90).reshape(30, 3))
        result = ev.discover(values, ["a", "b", "c"], method="fci", scale="counts")
        self.assertEqual(result["status"], "failed")
        self.assertIn("raw nonnegative integers", result["error"])

    def test_no_lpd4_approximation_fallback(self):
        values = np.arange(90).reshape(30, 3)
        result = ev.discover(values, ["a", "b", "c"], method="pc", scale="activities",
                             config=ev.DiscoveryConfig(approximation="lpd4"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("fixed RCIT approximation", result["error"])

    @unittest.skipUnless(HAVE_CAUSALLEARN, "causal-learn lives in a separate environment")
    def test_rcit_reproducible_and_rng_restored(self):
        values = np.random.default_rng(9).normal(size=(100, 2))
        np.random.seed(124)
        before = np.random.get_state()
        first = ev.discover(values, ["a", "b"], method="pc", scale="activities")
        after = np.random.get_state()
        second = ev.discover(values, ["a", "b"], method="pc", scale="activities")
        self.assertEqual(first["status"], "ok", first)
        self.assertEqual(first["backend"]["variant"], "RCIT")
        self.assertEqual(first["tests"], second["tests"])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    @unittest.skipUnless(HAVE_CAUSALLEARN, "causal-learn lives in a separate environment")
    def test_nonfinite_pvalue_explicit_failure(self):
        from causallearn.utils.RCIT.RCIT import RIT
        values = np.random.default_rng(9).normal(size=(40, 2))
        with patch.object(RIT, "compute_pvalue", return_value=(float("nan"), 0)):
            result = ev.discover(values, ["a", "b"], method="pc", scale="activities")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["edges"])
        self.assertEqual(result["tests"][-1]["status"], "failed")
        self.assertIn("invalid p-value", result["error"])


class MetricsTests(unittest.TestCase):
    def test_unsupported_outputs_are_na(self):
        metrics = ev.optional_structure_metrics(candidates_from_truth(fixture_truth()))
        self.assertEqual(metrics["hyperedges"]["status"], "not_applicable")
        self.assertEqual(metrics["equivalence_class"]["status"], "not_applicable")

    def test_graph_failure_does_not_score_as_zero_false_positives(self):
        candidates = candidates_from_truth(fixture_truth())
        result = ev.score_graph({"status": "failed", "edges": None}, {"status": "ok", "edges": []}, candidates, target="activity")
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["metrics"])

    def test_endpoints_and_directions_separate(self):
        candidates = [dict(name="a", assay="H3K27ac", role="core"), dict(name="r", assay="RNA", role="core")]
        predicted = dict(status="ok", edges=[dict(node1="a", node2="r", endpoint1="CIRCLE", endpoint2="ARROW")])
        oracle = dict(status="ok", edges=[dict(node1="a", node2="r", endpoint1="CIRCLE", endpoint2="ARROW")])
        score = ev.score_graph(predicted, oracle, candidates, target="count_measurement_pag", biological_edges=[("a", "r")])
        self.assertEqual(score["adjacency"]["recall"], 1)
        self.assertEqual(score["endpoints"]["recall"], 1)
        self.assertEqual(score["exact_cross_assay_stress_test"]["direction"]["recall"], 0)

    def test_actual_hyperedges_and_bounded_ambiguity(self):
        candidates = candidates_from_truth(fixture_truth())
        names = [row["name"] for row in candidates if row["role"] == "core"]
        result = ev.optional_structure_metrics(candidates, predicted_hyperedges=[names], true_hyperedges=[names],
                                               compatible_models=["rho0", "rho1"], candidate_family=["rho0", "rho1"], true_model="rho1")
        self.assertEqual(result["hyperedges"]["exact_membership"]["precision"], 1)
        self.assertEqual(result["hyperedges"]["sentinel_participation_rate"], 0)
        self.assertTrue(result["equivalence_class"]["coverage"])
        self.assertEqual(result["equivalence_class"]["ambiguity_set_size"], 2)


class ProbeTests(unittest.TestCase):
    def test_preprocessing_dimensions_can_differ_only_for_probes(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_probe_contract_") as directory:
            root = Path(directory)
            (root / "probe").mkdir()
            for dimension, assay in enumerate(ev.ASSAYS, start=1):
                np.savez_compressed(root / "probe" / f"{assay}.npz",
                                    cell_ids=np.asarray(["probe0", "probe1"]),
                                    shared=np.ones((2, dimension)))
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                ev._load_embedding_run(root, "probe")
            _, representations, _ = ev._load_embedding_run(root, "probe", require_common_dimension=False)
            self.assertEqual([representations[assay]["shared"].shape[1] for assay in ev.ASSAYS], [1, 2, 3])

    def test_cli_keeps_preprocessing_out_of_matching_inputs(self):
        with patch.object(ev, "evaluate_world", return_value={"status": "completed", "failed_integrations": []}) as evaluate:
            with redirect_stdout(io.StringIO()):
                ev.main(["evaluate", "--world", "fixture", "--regime", "realistic",
                         "--embeddings", "integrated", "--preprocessing-embeddings", "native_representations",
                         "--output", "fixture_results"])
        self.assertEqual(evaluate.call_args.args[1], [Path("integrated")])
        self.assertEqual(evaluate.call_args.kwargs["preprocessing_directories"], [Path("native_representations")])

    def make_inputs(self):
        rng = np.random.default_rng(50)
        def cohort(n, prefix):
            state = rng.normal(size=(n, 1))
            activity = state + rng.normal(size=(n, 3))
            truth = dict(cell_ids=np.asarray([f"{prefix}_{index}" for index in range(n)]), state=state,
                         activities=activity, candidate_names=np.asarray([f"v_{assay}" for assay in ev.ASSAYS]),
                         candidate_assays=np.asarray(ev.ASSAYS), conditional_mean_activities=np.repeat(state, 3, axis=1))
            embeddings = {assay: dict(cell_ids=truth["cell_ids"], shared=state.copy()) for assay in ev.ASSAYS}
            embeddings["RNA"]["private"] = activity[:, :1] - state
            return truth, embeddings
        cal, cal_embed = cohort(60, "cal")
        assessment, assessment_embed = cohort(40, "test")
        return cal, cal_embed, assessment, assessment_embed

    def test_overlapping_probe_cohorts_rejected(self):
        truth, embedding, _, _ = self.make_inputs()
        with self.assertRaisesRegex(ValueError, "overlap"):
            ev.evaluate_probes(embedding, embedding, truth, truth)

    def test_private_residual_information_and_train_only_scaling(self):
        cal, cal_embed, assessment, assessment_embed = self.make_inputs()
        result = ev.evaluate_probes(cal_embed, assessment_embed, cal, assessment)
        self.assertEqual(result["status"], "ok", result)
        rows = [row for row in result["rows"] if row["source_assay"] == "RNA" and row["probe"] == "ridge"
                and row["representation"] == "shared_plus_private" and row["target"] == "residual::v_RNA"]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["embedding_only"]["r2"], 0.99)
        self.assertGreater(rows[0]["mse_gain_over_true_state"], 0.5)


@unittest.skipUnless(HAVE_CAUSALLEARN, "causal-learn lives in a separate environment")
class EndToEndTests(unittest.TestCase):
    def make_fixture(self, root):
        """In-memory-sized unit fixture, not a production world or model fit."""
        world, integration = root / "unit_world", root / "unit_model"
        truth = fixture_truth()
        (world / "truth").mkdir(parents=True)
        np.savez_compressed(world / "truth/train.npz", **truth)
        manifest = {"candidates": candidates_from_truth(truth), "biological_dag_edges": [
            ["S0", "H3K27ac_core"], ["H3K27ac_core", "H3K4me3_core"], ["H3K4me3_core", "RNA_core"]]}
        ev._write_json_new(world / "truth/manifest.json", manifest)
        ev._write_json_new(world / "manifest.json", {
            "status": "complete", "count_calibration_status": "unit_fixture_not_calibrated",
            "focal_calibration_passed": False, "panel_calibration_passed": False,
            "panel_calibration_audits": {"unit_fixture": {"passed": False}},
        })
        # This unit fixture supplies a validation record for identical compact
        # source arrays. Production validation must use validate-observed/H5AD.
        sources = {}
        for assay in ev.ASSAYS:
            ids = truth["cell_ids"][truth["assigned_assay"] == assay]
            columns = np.flatnonzero(truth["candidate_assays"] == assay)
            source = world / f"unit_observed_{assay}.npz"
            np.savez_compressed(source, cell_ids=ids, values=truth["counts_realistic"][truth["assigned_assay"] == assay][:, columns])
            sources[assay] = {"path": str(source), "sha256": ev._sha256(source)}
            (integration / "train").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(integration / "train" / f"{assay}.npz", cell_ids=ids, shared=np.arange(len(ids))[:, None])
        ev._write_json_new(integration / "embedding_manifest.json", {"label": "unit_model", "method": "unit_fixture"})
        ev._write_json_new(world / "truth/observation_validation_realistic.json", {
            "status": "ok", "regime": "realistic", "truth_manifest_sha256": ev._sha256(world / "truth/manifest.json"),
            "cohorts": {"train": {"truth_sha256": ev._sha256(world / "truth/train.npz"), "assays": sources}},
        })
        return world, integration

    def test_failed_discovery_report_and_immutable_cache(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_evaluator_test_") as directory:
            root = Path(directory)
            world, integration = self.make_fixture(root)
            output = root / "evaluation"
            with redirect_stdout(io.StringIO()):
                report = ev.evaluate_world(world, [integration], output, regime="realistic")
                cached = ev.evaluate_world(world, [integration], output, regime="realistic")
            self.assertEqual(report["status"], "completed_with_failures")
            self.assertFalse(report["count_calibration"]["focal_calibration_passed"])
            self.assertFalse(report["inputs"]["count_calibration"]["panel_calibration_passed"])
            self.assertEqual(report["inputs"]["world_manifest_sha256"], ev._sha256(world / "manifest.json"))
            self.assertEqual(len(report["failed_diagnostics"]), 8)
            self.assertEqual(report["input_fingerprint"], cached["input_fingerprint"])
            self.assertTrue((output / "readouts/paired_anchor_reference.npz").exists())
            self.assertTrue(list((output / "stages").glob("*.json")))
            for result in report["results"].values():
                self.assertIsNone(result["activities"]["edges"])
            with self.assertRaisesRegex(FileExistsError, "different inputs"):
                ev.evaluate_world(world, [integration], output, regime="realistic", config=ev.DiscoveryConfig(seed=4))

    def test_root_calibration_change_invalidates_evaluation_cache(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_evaluator_test_") as directory:
            root = Path(directory)
            world, integration = self.make_fixture(root)
            output = root / "evaluation"
            with redirect_stdout(io.StringIO()):
                ev.evaluate_world(world, [integration], output, regime="realistic")
            # Unit-fixture mutation is deliberate: scientific outputs must not
            # silently reuse a result labeled under different calibration.
            updated = json.loads((world / "manifest.json").read_text())
            updated["count_calibration_status"] = "changed_unit_fixture_label"
            (world / "manifest.json").write_text(json.dumps(updated))
            with self.assertRaisesRegex(FileExistsError, "different inputs"):
                ev.evaluate_world(world, [integration], output, regime="realistic")

    def test_no_successful_integration_records_failure(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_evaluator_test_") as directory:
            root = Path(directory)
            world, _ = self.make_fixture(root)
            report = ev.evaluate_world(world, [root / "missing"], root / "evaluation", regime="realistic")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["failed_integrations"]), 1)
            self.assertEqual(report["results"], {})

    def test_observed_source_change_invalidates_validation(self):
        with tempfile.TemporaryDirectory(prefix="recoverability_evaluator_test_") as directory:
            root = Path(directory)
            world, integration = self.make_fixture(root)
            np.savez_compressed(world / "unit_observed_RNA.npz", modified=np.array([1]))
            with self.assertRaisesRegex(ValueError, "Observed counts changed"):
                ev.evaluate_world(world, [integration], root / "evaluation", regime="realistic")


@unittest.skipUnless(HAVE_ANNDATA, "AnnData validation lives in a separate environment")
class AnnDataValidationTests(unittest.TestCase):
    def test_exact_original_count_validation_with_reordered_columns(self):
        import anndata as ad
        import pandas as pd
        from scipy import sparse
        truth = fixture_truth()
        with tempfile.TemporaryDirectory(prefix="recoverability_observed_validation_test_") as directory:
            world = Path(directory)
            (world / "truth").mkdir()
            np.savez_compressed(world / "truth/train.npz", **truth)
            ev._write_json_new(world / "truth/manifest.json", {"candidates": candidates_from_truth(truth)})
            for assay in ev.ASSAYS:
                rows = np.flatnonzero(truth["assigned_assay"] == assay)
                columns = np.flatnonzero(truth["candidate_assays"] == assay)[::-1]
                data = ad.AnnData(sparse.csr_matrix(truth["counts_realistic"][np.ix_(rows, columns)]),
                                 obs=pd.DataFrame(index=truth["cell_ids"][rows]),
                                 var=pd.DataFrame(index=truth["candidate_feature_ids"][columns]))
                destination = world / "observed/realistic/train" / f"{assay}.h5ad"
                destination.parent.mkdir(parents=True, exist_ok=True)
                data.write_h5ad(destination)
            result = ev.validate_observed_files(world, "realistic")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["cohorts"]["train"]["assays"]["RNA"]["n_scored_features"], 2)


if __name__ == "__main__":
    unittest.main()
