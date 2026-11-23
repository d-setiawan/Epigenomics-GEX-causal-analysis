"""Direct annotation-prior contracts; no model fitting or causal orientation."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import networkx as nx

MODULE = Path(__file__).resolve().parents[1] / "guidance_baseline.py"
SPEC = importlib.util.spec_from_file_location("recoverability_guidance_baseline", MODULE)
prior = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prior
SPEC.loader.exec_module(prior)

CANDIDATES = [dict(feature_id="f_a", name="A", assay="RNA", role="core"),
              dict(feature_id="f_b", name="B", assay="H3K27ac", role="core"),
              dict(feature_id="f_c", name="C", assay="H3K4me3", role="sentinel")]
FEATURE_MAP = [{key: row[key] for key in ("feature_id", "name", "assay")} for row in CANDIDATES]


def graph_fixture():
    graph = nx.DiGraph()
    graph.add_edges_from([(x, x, {"weight": 1.0, "sign": 1}) for x in ("f_a", "f_b", "f_c", "background")])
    graph.add_edge("f_a", "f_b", weight=1.0, sign=-1)
    graph.add_edge("f_b", "f_a", weight=1.0, sign=-1)
    graph.add_edge("f_b", "background", weight=1.0, sign=1)
    graph.add_edge("background", "f_c", weight=1.0, sign=1)
    return graph


def world_fixture(root, tag):
    """File identity fixtures; baseline never parses observed biological data."""
    world = root / tag
    truth_dir = world / "truth"
    truth_dir.mkdir(parents=True)
    truth = truth_dir / "manifest.json"
    truth.write_text(json.dumps({"candidates": CANDIDATES, "biological_dag_edges": [["A", "B"]]}))
    files = {"truth/manifest.json": prior.file_hash(truth)}
    for assay in ("RNA", "H3K27ac", "H3K4me3"):
        path = world / "observed" / "realistic" / "train" / f"{assay}.h5ad"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{tag}:{assay}:observed file identity".encode())
        files[str(path.relative_to(world))] = prior.file_hash(path)
    (world / "manifest.json").write_text(json.dumps({"status": "complete", "regimes": ["realistic"], "files": files}))
    return world, truth


def preprocessing_fixture(root, world):
    prep = root / "preprocessing"
    prep.mkdir()
    graph_path = prep / "guidance.graphml"
    nx.write_graphml(graph_fixture(), graph_path)
    manifest = {
        "status": "complete", "guidance": {"path": str(graph_path), "sha256": prior.file_hash(graph_path)},
        "assays": {"RNA": {"feature_ids": ["f_a", "background"]},
                   "H3K27ac": {"feature_ids": ["f_b"]}, "H3K4me3": {"feature_ids": ["f_c"]}},
        "inputs": {assay: {"sha256": prior.file_hash(world / "observed" / "realistic" / "train" / f"{assay}.h5ad")}
                   for assay in ("RNA", "H3K27ac", "H3K4me3")},
    }
    (prep / "manifest.json").write_text(json.dumps(manifest))
    return prep


class GuidanceTests(unittest.TestCase):
    def test_direct_links_only_and_no_orientation(self):
        result = prior.extract_prior(graph_fixture(), FEATURE_MAP)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertEqual({edge["node1"], edge["node2"]}, {"A", "B"})
        self.assertNotIn("endpoint1", edge)
        self.assertNotIn("source", edge)
        self.assertEqual(edge["edge_type"], "undirected_annotation_prior")
        self.assertEqual(result["direction"]["status"], "not_applicable")

    def test_truth_roles_cannot_enter_extraction(self):
        with self.assertRaisesRegex(ValueError, "never truth roles"):
            prior.extract_prior(graph_fixture(), CANDIDATES)

    def test_no_represented_features_is_not_empty_graph_success(self):
        result = prior.extract_prior(nx.Graph([("unscored1", "unscored2")]), FEATURE_MAP)
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["edges"])
        scored = prior.score_prior(result, CANDIDATES, [("A", "B")])
        self.assertIsNone(scored["adjacency"])
        self.assertEqual(scored["coverage"]["feature_coverage"], 0)

    def test_only_one_represented_feature_has_no_pair_universe(self):
        graph = nx.Graph()
        graph.add_node("f_a")
        result = prior.extract_prior(graph, FEATURE_MAP)
        self.assertEqual(result["status"], "not_applicable")
        self.assertIsNone(result["edges"])

    def test_truth_only_affects_scoring_not_prior_edges(self):
        result = prior.extract_prior(graph_fixture(), FEATURE_MAP)
        good = prior.score_prior(result, CANDIDATES, [("A", "B")])
        other = prior.score_prior(result, CANDIDATES, [("B", "C")])
        self.assertEqual(good["adjacency"]["precision"], 1)
        self.assertEqual(other["adjacency"]["precision"], 0)
        self.assertEqual(result["edges"][0]["node1"], "A")

    def test_omitted_vertices_are_reported_as_abstention(self):
        graph = graph_fixture()
        graph.remove_node("f_c")
        result = prior.extract_prior(graph, FEATURE_MAP)
        scored = prior.score_prior(result, CANDIDATES, [("A", "B"), ("B", "C")])
        self.assertEqual(scored["coverage"]["n_true_biological_pairs_unrepresented"], 1)
        self.assertEqual(scored["adjacency"]["recall"], 1)
        self.assertEqual(scored["whole_panel_comparison"]["status"], "not_applicable")

    def test_all_masked_candidates_are_not_null_recovery(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(row["feature_id"], row["feature_id"], {"weight": 1.0, "sign": 1})
                              for row in FEATURE_MAP])
        result = prior.extract_prior(graph, FEATURE_MAP, excluded_feature_ids=[row["feature_id"] for row in FEATURE_MAP])
        scored = prior.score_prior(result, CANDIDATES, [("A", "B")])
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["baseline_kind"], "masked_scored_feature_negative_control")
        self.assertEqual(result["n_direct_scored_links_in_exposed_graph"], 0)
        self.assertIsNone(result["edges"])
        self.assertIsNone(scored["adjacency"])
        self.assertEqual(scored["coverage"]["feature_coverage"], 1)
        self.assertEqual(scored["coverage"]["n_link_eligible_features"], 0)
        self.assertEqual(scored["coverage"]["n_true_biological_pairs_masked"], 1)

    def test_partially_masked_pairs_are_excluded_from_denominator(self):
        graph = graph_fixture()
        graph.remove_edge("background", "f_c")
        result = prior.extract_prior(graph, FEATURE_MAP, excluded_feature_ids=["f_c"])
        scored = prior.score_prior(result, CANDIDATES, [("A", "B"), ("B", "C")])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(scored["adjacency"]["n_possible_pairs"], 1)
        self.assertEqual(scored["adjacency"]["recall"], 1)
        self.assertEqual(scored["coverage"]["n_represented_pairs"], 3)
        self.assertEqual(scored["coverage"]["n_link_eligible_pairs"], 1)
        self.assertEqual(scored["coverage"]["n_true_biological_pairs_masked"], 1)
        self.assertEqual(scored["sentinel_incident_pairs"]["status"], "not_applicable")
        self.assertEqual(scored["whole_panel_comparison"]["status"], "not_applicable")

    def test_declared_mask_must_match_exposed_graph(self):
        with self.assertRaisesRegex(ValueError, "contradicts.*mask"):
            prior.extract_prior(graph_fixture(), FEATURE_MAP, excluded_feature_ids=["f_c"])

    def test_preprocessing_must_belong_to_scored_world(self):
        with tempfile.TemporaryDirectory(prefix="guidance_world_binding_") as directory:
            root = Path(directory)
            first, first_truth = world_fixture(root, "first")
            _, other_truth = world_fixture(root, "other")
            prep = preprocessing_fixture(root, first)
            result = prior.run_baseline(prep, first_truth, root / "first_result.json")
            self.assertEqual(result["inputs"]["matched_count_regimes"], ["realistic"])
            self.assertEqual(result["status"], "ok")
            with self.assertRaisesRegex(ValueError, "do not belong to the scored world"):
                prior.run_baseline(prep, other_truth, root / "other_result.json")
            self.assertFalse((root / "other_result.json").exists())

    def test_modified_truth_receipt_rejected(self):
        with tempfile.TemporaryDirectory(prefix="guidance_truth_receipt_") as directory:
            root = Path(directory)
            world, truth = world_fixture(root, "first")
            prep = preprocessing_fixture(root, world)
            truth.write_text(json.dumps({"candidates": CANDIDATES, "biological_dag_edges": [["B", "C"]]}))
            with self.assertRaisesRegex(ValueError, "differs from the world's frozen truth"):
                prior.run_baseline(prep, truth, root / "result.json")

    def test_modified_world_counts_rejected(self):
        with tempfile.TemporaryDirectory(prefix="guidance_count_receipt_") as directory:
            root = Path(directory)
            world, truth = world_fixture(root, "first")
            prep = preprocessing_fixture(root, world)
            (world / "observed" / "realistic" / "train" / "RNA.h5ad").write_bytes(b"changed after freeze")
            with self.assertRaisesRegex(ValueError, "observed training counts changed"):
                prior.run_baseline(prep, truth, root / "result.json")

    def test_changed_graph_receipt_rejected(self):
        with tempfile.TemporaryDirectory(prefix="guidance_baseline_unit_") as directory:
            root = Path(directory)
            graph_path = root / "guidance.graphml"
            nx.write_graphml(graph_fixture(), graph_path)
            manifest = {"status": "complete", "guidance": {"path": str(graph_path), "sha256": "incorrect"},
                        "assays": {"unit": {"feature_ids": list(graph_fixture().nodes)}}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            truth = root / "unit_truth.json"
            truth.write_text(json.dumps({"candidates": CANDIDATES, "biological_dag_edges": [["A", "B"]]}))
            with self.assertRaisesRegex(ValueError, "changed after preprocessing"):
                prior.run_baseline(root, truth, root / "result.json")


if __name__ == "__main__":
    unittest.main()
