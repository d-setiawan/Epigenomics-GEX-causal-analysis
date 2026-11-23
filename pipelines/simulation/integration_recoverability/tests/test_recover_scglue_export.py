"""Recovery contracts on tiny CPU native networks; no scientific model fitting."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import torch

# This suite is also collected as standalone files by repository-wide pytest;
# the benchmark directories intentionally do not require Python packages.
TEST_DIRECTORY = Path(__file__).resolve().parent
for import_directory in (TEST_DIRECTORY.parent, TEST_DIRECTORY):
    if str(import_directory) not in sys.path:
        sys.path.insert(0, str(import_directory))

import recover_scglue_export as recovery
from test_scglue_adapter import inputs

adapter = recovery.adapter


def native_fixture(root):
    """Real serialization/history fixtures, explicitly untrained unit networks."""
    import scglue
    from tensorboardX import SummaryWriter
    prepared, graph = inputs()
    config = adapter.ScGLUEConfig(device="cpu", latent_dim=3, h_dim=8, h_depth=1)
    prep, run = root / "prepared", root / "run"
    (prep / "train").mkdir(parents=True)
    (prep / "transforms").mkdir()
    (prep / "transforms" / "unit.json").write_text('{"unit_fixture": true}\n')
    run.mkdir()
    graph_path = prep / "guidance.graphml"
    nx.write_graphml(graph, graph_path)
    training = {}
    for assay, data in prepared.items():
        path = prep / "train" / f"{assay}.h5ad"
        data.write_h5ad(path)
        training[assay] = {"path": str(path.resolve()), "sha256": adapter.file_hash(path)}
    fit_inputs = {
        "training_files": training, "guidance": {"path": str(graph_path.resolve()), "sha256": adapter.file_hash(graph_path)},
        "transformations": adapter.transformation_fingerprint(prep), "configuration": asdict(config),
        "label": "untrained_cpu_serialization_fixture", "package_provenance": adapter.package_provenance(),
        "adapter_sha256": adapter.file_hash(Path(adapter.__file__)),
    }
    fingerprint = adapter.content_hash(fit_inputs)
    adapter.write_json_new(run / "fit_inputs.json", {"input_fingerprint": fingerprint, "inputs": fit_inputs})
    adapter.write_json_new(run / "failure.json", {
        "status": "failed", "stage": "native_training_history_validation", "error_type": "ModuleNotFoundError",
        "error": "No module named 'tensorboard'", "input_fingerprint": fingerprint, "runtime_seconds": 1.5,
    })
    with adapter.native_runtime(config):
        for phase in ("pretrain", "fine-tune"):
            folder = run / "checkpoints" / phase
            folder.mkdir(parents=True)
            for assay, data in prepared.items():
                data.obs["balancing_weight"] = 1.0
                scglue.models.configure_dataset(data, "NB", use_highly_variable=True, use_layer="counts",
                                                use_rep=adapter.REPRESENTATIONS[assay],
                                                use_dsc_weight="balancing_weight" if phase == "fine-tune" else None)
            model = scglue.models.SCGLUEModel(prepared, sorted(graph.nodes), **adapter.native_configuration(config, folder)["init_kws"])
            model.save(folder / f"{phase}.dill")
            checkpoint = {"net": model.net.state_dict(), "trainer": {"vae_optim": {}, "dsc_optim": {}}}
            torch.save(checkpoint, folder / "checkpoint_3.pt")
            # A retained earlier checkpoint is intentionally different. Recovery
            # must prove equality to native final selection, not load it silently.
            first = {key: value.clone() for key, value in checkpoint["net"].items()}
            first[next(key for key, value in first.items() if value.is_floating_point())].add_(1)
            torch.save({"net": first, "trainer": checkpoint["trainer"]}, folder / "checkpoint_1.pt")
            writer = SummaryWriter(log_dir=str(folder / "tensorboard"))
            try:
                for epoch in range(1, 5):
                    writer.add_scalar("train/vae_loss", 1.0 + epoch / 10, epoch)
                    writer.add_scalar("val/vae_loss", 1.0 + epoch / 10, epoch)
            finally:
                writer.close()
    return run, prep, config


class RecoveryTests(unittest.TestCase):
    def test_exact_snapshot_export_roundtrip_and_no_retraining(self):
        import scglue
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, prep, config = native_fixture(Path(directory))
            original_failure = (run / "failure.json").read_bytes()
            original_snapshot = (run / "checkpoints/fine-tune/fine-tune.dill").read_bytes()
            with patch.object(scglue.models.SCGLUEModel, "fit", side_effect=AssertionError("No retraining permitted")), \
                 patch.object(scglue.models, "estimate_balancing_weight", side_effect=AssertionError("No rebalancing permitted")):
                result = recovery.recover_export(run, prep, cohorts=["train"])
                cached = recovery.recover_export(run, prep, cohorts=["train"])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result, cached)
            self.assertEqual((run / "failure.json").read_bytes(), original_failure)
            self.assertEqual((run / "scglue_model.dill").read_bytes(), original_snapshot)
            receipt = json.loads((run / "recovery_provenance.json").read_text())
            manifest = json.loads((run / "model_manifest.json").read_text())
            self.assertEqual(receipt["optimization_steps"], 0)
            self.assertEqual(receipt["balancing_reestimations"], 0)
            self.assertEqual(receipt["native_phases"]["fine-tune"]["epoch"], 3)
            self.assertEqual(receipt["roundtrip_max_abs_error"], {assay: 0.0 for assay in adapter.ASSAYS})
            self.assertEqual(manifest["configuration"], asdict(config))
            self.assertEqual(set(result["files"]), {f"train/{assay}.npz" for assay in adapter.ASSAYS})

    def test_other_training_failure_cannot_be_called_completed(self):
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, prep, _ = native_fixture(Path(directory))
            path = run / "failure.json"
            failure = json.loads(path.read_text())
            failure["stage"] = "native_fit_SCGLUE"
            path.write_text(json.dumps(failure))
            with self.assertRaisesRegex(ValueError, "only supports the post-fit"):
                recovery.recover_export(run, prep)
            self.assertFalse((run / "model_manifest.json").exists())

    def test_missing_final_snapshot_is_not_rebuilt_from_optimizer_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, prep, _ = native_fixture(Path(directory))
            snapshot = run / "checkpoints/fine-tune/fine-tune.dill"
            snapshot.rename(snapshot.with_suffix(".unavailable"))
            with self.assertRaisesRegex(FileNotFoundError, "final snapshots are required"):
                recovery.recover_export(run, prep)
            self.assertFalse((run / "scglue_model.dill").exists())

    def test_mismatched_restored_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, prep, _ = native_fixture(Path(directory))
            latest = run / "checkpoints/fine-tune/checkpoint_3.pt"
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            key = next(key for key, value in checkpoint["net"].items() if value.is_floating_point())
            checkpoint["net"][key].add_(2)
            torch.save(checkpoint, latest)
            with self.assertRaisesRegex(ValueError, "differs from restored checkpoint"):
                recovery.recover_export(run, prep)
            self.assertFalse((run / "scglue_model.dill").exists())

    def test_checkpoint_after_history_rejected(self):
        import scglue
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, _, config = native_fixture(Path(directory))
            with adapter.native_runtime(config):
                model = scglue.models.load_model(run / "checkpoints/fine-tune/fine-tune.dill")
                with self.assertRaisesRegex(ValueError, "after the recorded"):
                    recovery.validate_native_checkpoint(model, run / "checkpoints/fine-tune", {"last_recorded_epoch": 2})

    def test_pretraining_snapshot_cannot_be_presented_as_final_finetune(self):
        import scglue
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, prep, config = native_fixture(Path(directory))
            prepared, graph = inputs()
            contracts = {assay: adapter.validate_prepared_adata(data, assay) for assay, data in prepared.items()}
            with adapter.native_runtime(config):
                model = scglue.models.load_model(run / "checkpoints/pretrain/pretrain.dill")
                with self.assertRaisesRegex(ValueError, "domain mismatch: use_dsc_weight"):
                    recovery.validate_snapshot_configuration(model, contracts, graph, config, phase="fine-tune")

    def test_checkpoint_agreement_does_not_bypass_feature_identity(self):
        import scglue
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, _, config = native_fixture(Path(directory))
            prepared, graph = inputs()
            contracts = {assay: adapter.validate_prepared_adata(data, assay) for assay, data in prepared.items()}
            contracts["RNA"]["feature_ids"] = contracts["RNA"]["feature_ids"][::-1]
            with adapter.native_runtime(config):
                model = scglue.models.load_model(run / "checkpoints/fine-tune/fine-tune.dill")
                with self.assertRaisesRegex(ValueError, "domain mismatch: features"):
                    recovery.validate_snapshot_configuration(model, contracts, graph, config, phase="fine-tune")

    def test_changed_training_input_rejected_before_export(self):
        with tempfile.TemporaryDirectory(prefix="native_recovery_unit_") as directory:
            run, prep, _ = native_fixture(Path(directory))
            with (prep / "train/RNA.h5ad").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "prepared training data changed"):
                recovery.recover_export(run, prep)
            self.assertFalse((run / "scglue_model.dill").exists())

    def test_changed_native_source_not_silently_accepted(self):
        with patch.dict(recovery.AUDITED_NATIVE_SOURCES, {"scglue.models.plugins": "wrong"}):
            with self.assertRaisesRegex(ValueError, "Unaudited native"):
                recovery.native_source_audit()

    def test_existing_different_export_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="native_recovery_copy_") as directory:
            source, destination = Path(directory) / "native.dill", Path(directory) / "export.dill"
            source.write_bytes(b"native")
            destination.write_bytes(b"different")
            with self.assertRaisesRegex(FileExistsError, "not the native final snapshot"):
                recovery.copy_native_snapshot(source, destination)
            self.assertEqual(destination.read_bytes(), b"different")


if __name__ == "__main__":
    unittest.main()
