"""CPU-only adapter contract tests; these are not model-recovery experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

MODULE = Path(__file__).resolve().parents[1] / "scglue_adapter.py"
SPEC = importlib.util.spec_from_file_location("recoverability_scglue_adapter", MODULE)
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def inputs(n=12):
    rng = np.random.default_rng(100)
    prepared = {}
    graph = nx.DiGraph()
    for assay in adapter.ASSAYS:
        ids = [f"unit_{assay}_{index}" for index in range(n)]
        features = [f"unit_{assay}_f{index}" for index in range(4)]
        values = sparse.csr_matrix(rng.poisson(2, size=(n, 4)))
        data = ad.AnnData(values, obs=pd.DataFrame({"assay": assay}, index=ids),
                         var=pd.DataFrame({"highly_variable": True}, index=features))
        data.layers["counts"] = values.copy()
        data.obsm[adapter.REPRESENTATIONS[assay]] = rng.normal(size=(n, 5)).astype(np.float32)
        prepared[assay] = data
        graph.add_nodes_from(features)
        graph.add_edges_from((feature, feature, {"weight": 1.0, "sign": 1}) for feature in features)
        if assay != "RNA":
            for gene, region in zip(prepared["RNA"].var_names, features):
                graph.add_edge(gene, region, weight=1.0, sign=1)
                graph.add_edge(region, gene, weight=1.0, sign=1)
    return prepared, graph


def untrained_native_model(prepared, graph, **kwargs):
    """Test double: construct a real native network, deliberately do not fit."""
    import scglue
    for value in prepared.values():
        value.obs["balancing_weight"] = 1.0
    return scglue.models.SCGLUEModel(prepared, sorted(graph.nodes), **kwargs["init_kws"])


class InputContractTests(unittest.TestCase):
    def test_defaults_match_local_training_protocol(self):
        config = adapter.ScGLUEConfig()
        kwargs = adapter.native_configuration(config, Path("unit_checkpoints"))
        self.assertEqual(kwargs["init_kws"]["latent_dim"], 30)
        self.assertEqual(kwargs["fit_kws"]["max_epochs"], 120)
        self.assertEqual(kwargs["fit_kws"]["graph_batch_size"], 8192)
        self.assertEqual(kwargs["fit_kws"]["align_burnin"], 20)
        self.assertNotIn("skip_balance", kwargs)
        self.assertNotIn("safe_burnin", kwargs["fit_kws"])

    def test_feature_order_cannot_silently_change(self):
        prepared, _ = inputs()
        reference = adapter.validate_prepared_adata(prepared["RNA"], "RNA")
        with self.assertRaisesRegex(ValueError, "identity/order"):
            adapter.validate_prepared_adata(prepared["RNA"][:, ::-1], "RNA", expected=reference)

    def test_forbidden_truth_metadata_rejected(self):
        prepared, _ = inputs()
        prepared["RNA"].obs["true_state"] = 0
        with self.assertRaisesRegex(ValueError, "Forbidden truth"):
            adapter.validate_prepared_adata(prepared["RNA"], "RNA")

    def test_nonfinite_representation_not_silently_filtered(self):
        prepared, _ = inputs()
        prepared["RNA"].obsm["X_pca"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            adapter.validate_prepared_adata(prepared["RNA"], "RNA")

    def test_noninteger_count_layer_rejected(self):
        prepared, _ = inputs()
        prepared["RNA"].layers["counts"] = prepared["RNA"].layers["counts"].astype(float)
        prepared["RNA"].layers["counts"].data[0] = 0.5
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            adapter.validate_prepared_adata(prepared["RNA"], "RNA")

    def test_zero_reconstruction_row_kept_when_encoder_is_valid(self):
        prepared, _ = inputs()
        matrix = prepared["RNA"].layers["counts"].toarray()
        matrix[0] = 0
        prepared["RNA"].layers["counts"] = sparse.csr_matrix(matrix)
        self.assertEqual(adapter.validate_prepared_adata(prepared["RNA"], "RNA")["zero_reconstruction_count_rows"], 1)

    def test_guidance_selfloops_and_reciprocity(self):
        prepared, graph = inputs()
        self.assertEqual(adapter.validate_guidance(graph, prepared)["n_self_loops"], 12)
        feature = str(prepared["RNA"].var_names[0])
        graph.remove_edge(feature, feature)
        with self.assertRaisesRegex(ValueError, "self-loop"):
            adapter.validate_guidance(graph, prepared)

    def test_nonbinary_guidance_sign_is_not_truncated(self):
        prepared, graph = inputs()
        first, second = next(iter(graph.edges))
        graph.edges[first, second]["sign"] = 1.5
        with self.assertRaisesRegex(ValueError, "weight/sign"):
            adapter.validate_guidance(graph, prepared)


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.prepared, self.graph = inputs()
        self.config = adapter.ScGLUEConfig(device="cpu", latent_dim=3, h_dim=8, h_depth=1)

    def test_scheduler_shim_is_temporary_and_objective_neutral(self):
        import torch
        import scglue
        import scglue.models.plugins as plugins
        before = plugins.ReduceLROnPlateau
        original_cpu = scglue.config.CPU_ONLY
        old_rng = torch.random.get_rng_state().clone()
        with adapter.native_runtime(self.config) as audit:
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
            scheduler = plugins.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            self.assertEqual(audit["objective_changes"], [])
            scheduler.step(1.0)
            scheduler.step(1.0)
            scheduler.step(1.0)
            self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)
            torch.rand(3)
        self.assertIs(plugins.ReduceLROnPlateau, before)
        self.assertEqual(scglue.config.CPU_ONLY, original_cpu)
        self.assertTrue(torch.equal(old_rng, torch.random.get_rng_state()))

    def test_native_fit_invocation_does_not_skip_balancing(self):
        import scglue
        with tempfile.TemporaryDirectory(prefix="scglue_adapter_unit_") as directory:
            with patch.object(scglue.models, "fit_SCGLUE", side_effect=untrained_native_model) as fit:
                model, audit = adapter.fit_native(self.prepared, self.graph, self.config, Path(directory))
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.kwargs["fit_kws"]["max_epochs"], 120)
        self.assertEqual(audit["training_sequence"], ["native_pretraining", "native_balancing_weights", "native_fine_tuning"])
        self.assertIsInstance(model, scglue.models.SCGLUEModel)

    def test_disjoint_training_ids_enforced(self):
        self.prepared["H3K27ac"].obs_names = self.prepared["RNA"].obs_names.copy()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            adapter.fit_native(self.prepared, self.graph, self.config, Path("unused_unit_path"))

    def test_networkx_graph_input_native_conversion(self):
        from scglue.models.data import GraphDataset
        dataset = GraphDataset(self.graph, pd.Index(sorted(self.graph.nodes)), "weight", "sign")
        self.assertEqual(dataset.eidx.shape[1], self.graph.number_of_edges())
        np.testing.assert_array_equal(dataset.esgn, np.ones(self.graph.number_of_edges()))

    def test_real_untrained_network_checkpoint_and_encoder_roundtrip(self):
        import scglue
        with tempfile.TemporaryDirectory(prefix="scglue_adapter_unit_") as directory:
            with patch.object(scglue.models, "fit_SCGLUE", side_effect=untrained_native_model):
                model, audit = adapter.fit_native(self.prepared, self.graph, self.config, Path(directory))
            checkpoint = Path(directory) / "untrained_unit_model.dill"
            with adapter.native_runtime(self.config):
                before = adapter.model_state_hash(model)
                model.save(checkpoint)
                loaded = scglue.models.load_model(checkpoint)
                self.assertEqual(before, adapter.model_state_hash(loaded))
                for assay in adapter.ASSAYS:
                    encoded = adapter.encode_frozen(loaded, assay, self.prepared[assay], audit["input_contracts"][assay], batch_size=5)
                    self.assertEqual(encoded["shared"].shape, (12, 3))
                    self.assertTrue(np.isfinite(encoded["shared"]).all())
                self.assertEqual(before, adapter.model_state_hash(loaded))


class TrainingHistoryDependencyTests(unittest.TestCase):
    """Exercise actual writer/reader dependencies, not mocked fit histories."""

    def test_tensorboard_reader_parses_native_writer_fixture(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboardX import SummaryWriter
        with tempfile.TemporaryDirectory(prefix="scglue_history_unit_") as directory:
            root = Path(directory)
            for phase in ("pretrain", "fine-tune"):
                writer = SummaryWriter(log_dir=str(root / phase / "tensorboard"))
                try:
                    writer.add_scalar("train/vae_loss", 2.5, 1)
                    writer.add_scalar("train/vae_loss", 1.25, 2)
                    writer.add_scalar("val/vae_loss", 1.5, 2)
                finally:
                    writer.close()
            native_reader = EventAccumulator(str(root / "pretrain/tensorboard")).Reload()
            self.assertIn("train/vae_loss", native_reader.Tags()["scalars"])
            result = adapter.training_event_summary(root)
            self.assertEqual(set(result), {"pretrain", "fine-tune"})
            for phase in result.values():
                self.assertEqual(phase["last_recorded_epoch"], 2)
                self.assertEqual(phase["final_recorded_scalars"]["train/vae_loss"], 1.25)
                self.assertEqual(phase["final_recorded_scalars"]["val/vae_loss"], 1.5)

    def test_nonfinite_recorded_loss_is_rejected(self):
        from tensorboardX import SummaryWriter
        with tempfile.TemporaryDirectory(prefix="scglue_history_unit_") as directory:
            root = Path(directory)
            writer = SummaryWriter(log_dir=str(root / "pretrain/tensorboard"))
            try:
                writer.add_scalar("train/vae_loss", float("nan"), 1)
            finally:
                writer.close()
            with self.assertRaisesRegex(FloatingPointError, "nonfinite losses"):
                adapter.training_event_summary(root)


class LifecycleTests(unittest.TestCase):
    def test_full_lifecycle_contract_without_optimizer_steps(self):
        import scglue
        prepared, graph = inputs()
        config = adapter.ScGLUEConfig(device="cpu", latent_dim=3, h_dim=8, h_depth=1)
        original_provenance = adapter.package_provenance()
        with tempfile.TemporaryDirectory(prefix="scglue_adapter_lifecycle_unit_") as directory:
            root = Path(directory)
            preproc, output = root / "prepared", root / "model"
            (preproc / "train").mkdir(parents=True)
            (preproc / "transforms").mkdir()
            adapter.write_json_new(preproc / "transforms/unit_test_only.json", {"unit_fixture": True})
            for assay, data in prepared.items():
                data.write_h5ad(preproc / "train" / f"{assay}.h5ad")
            nx.write_graphml(graph, preproc / "guidance.graphml.gz")
            with (patch.object(scglue.models, "fit_SCGLUE", side_effect=untrained_native_model) as fit,
                  patch.object(adapter, "package_provenance", return_value=original_provenance),
                  patch.object(adapter, "training_event_summary", return_value={"unit_test_only_no_training": True})):
                result = adapter.fit_prepared(preproc, preproc / "guidance.graphml.gz", output, config=config, label="unit_fixture_only")
                cached = adapter.fit_prepared(preproc, preproc / "guidance.graphml.gz", output, config=config, label="unit_fixture_only")
            self.assertEqual(fit.call_count, 1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["input_fingerprint"], cached["input_fingerprint"])
            manifest = json.loads((output / "model_manifest.json").read_text())
            self.assertIn("latent_dim", manifest["protocol_overrides"])
            self.assertEqual(len(manifest["checkpoint_roundtrip_max_abs_error"]), 3)
            for assay in adapter.ASSAYS:
                with np.load(output / "train" / f"{assay}.npz") as arrays:
                    self.assertEqual(set(arrays.files), {"cell_ids", "shared"})
            adapter.write_json_new(preproc / "transforms/changed_unit_transform.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "different fitted preprocessing"):
                adapter.encode_prepared(output, preproc, root / "new_embeddings")


if __name__ == "__main__":
    unittest.main()
