"""Audit and export a completed native scGLUE fit after history-reader failure.

This deliberately narrow recovery path requires the native final fine-tune.dill
snapshot, not just an arbitrary optimizer checkpoint. No fitting, balancing,
initialization, checkpoint selection by a new criterion, or objective change is
performed. The original failure and every native artifact remain untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

try:
    from . import scglue_adapter as adapter
except ImportError:
    import scglue_adapter as adapter


# These are the installed 0.2.3 sources audited before enabling this recovery.
# In this version COMPLETED/TERMINATE restores the largest retained epoch,
# except a final nonfinite checkpoint. It does NOT reselect the global loss min.
AUDITED_NATIVE_SOURCES = {
    "scglue.models.plugins": "340851db358fb4db642f0667a7c7057ecc6f4180097f1d21993fb58ae5f0a2f0",
    "scglue.models.base": "786b098224ecac49ea4a8ff1dbfab54fc9929fca61053908d1ebf981788e879a",
}


def validate_receipts(run: Path, preprocessing: Path) -> tuple[dict, dict]:
    """Bind the recovery to the exact frozen adapter and observed fit inputs."""
    fit = json.loads((run / "fit_inputs.json").read_text())
    failure = json.loads((run / "failure.json").read_text())
    if adapter.content_hash(fit["inputs"]) != fit["input_fingerprint"]:
        raise ValueError("Original fit-input fingerprint is invalid")
    if (failure.get("status") != "failed"
            or failure.get("stage") != "native_training_history_validation"
            or failure.get("error_type") != "ModuleNotFoundError"
            or failure.get("error") != "No module named 'tensorboard'"
            or failure.get("input_fingerprint") != fit["input_fingerprint"]):
        raise ValueError("This recovery only supports the post-fit TensorBoard-reader failure")
    inputs = fit["inputs"]
    if inputs["adapter_sha256"] != adapter.file_hash(Path(adapter.__file__)):
        raise ValueError("Frozen adapter source changed after the failed export")
    if set(inputs["training_files"]) != set(adapter.ASSAYS):
        raise ValueError("Recovery requires the original three observed training assays")
    for assay in adapter.ASSAYS:
        record = inputs["training_files"][assay]
        expected_path = (preprocessing / "train" / f"{assay}.h5ad").resolve()
        if Path(record["path"]).resolve() != expected_path or adapter.file_hash(expected_path) != record["sha256"]:
            raise ValueError(f"Original prepared training data changed: {assay}")
    graph = inputs["guidance"]
    if adapter.file_hash(Path(graph["path"])) != graph["sha256"]:
        raise ValueError("Original guidance graph changed")
    if adapter.transformation_fingerprint(preprocessing) != inputs["transformations"]:
        raise ValueError("Frozen preprocessing transformations changed")
    current = adapter.package_provenance()
    if current != inputs["package_provenance"]:
        raise ValueError("Model packages or native fit sources changed; exact recovery is unverified")
    return fit, failure


def native_source_audit() -> dict[str, Any]:
    import importlib
    if importlib.metadata.version("scglue") != "0.2.3":
        raise ValueError("Native early-stopping recovery was only audited for scGLUE 0.2.3")
    audit = {}
    for name, expected in AUDITED_NATIVE_SOURCES.items():
        module = importlib.import_module(name)
        path = Path(inspect.getsourcefile(module))
        actual = adapter.file_hash(path)
        if actual != expected:
            raise ValueError(f"Unaudited native serialization/early-stopping source: {name}")
        audit[name] = {"path": str(path), "sha256": actual}
    return audit


def validate_snapshot_configuration(model, contracts: Mapping, graph, config, *, phase: str) -> dict:
    """Validate the serialized architecture/domain configuration without rebuild."""
    import scglue
    if type(model) is not scglue.models.SCGLUEModel:
        raise TypeError("Expected the exact native SCGLUEModel snapshot class")
    if model.random_seed != config.seed or set(model.domains) != set(adapter.ASSAYS):
        raise ValueError("Native snapshot seed/assays differ from fit inputs")
    if list(map(str, model.vertices)) != sorted(map(str, graph.nodes)):
        raise ValueError("Native snapshot guidance-vertex order differs")
    balancing = {}
    for assay in adapter.ASSAYS:
        contract = contracts[assay]
        domain = model.domains[assay]
        expected = {
            "features": contract["feature_ids"], "prob_model": "NB", "use_highly_variable": True,
            "use_layer": "counts", "use_rep": adapter.REPRESENTATIONS[assay],
            "rep_dim": contract["representation_dimension"], "use_batch": None,
            "use_cell_type": None, "use_uid": None,
            "use_dsc_weight": "balancing_weight" if phase == "fine-tune" else None,
        }
        for key, value in expected.items():
            if domain.get(key) != value:
                raise ValueError(f"Native {phase}/{assay} domain mismatch: {key}")
        if len(domain["batches"]) or len(domain["cell_types"]):
            raise ValueError("Unexpected native batch/cell-type supervision")
        encoder = model.net.x2u[assay]
        linear = [name for name, _ in encoder.named_children() if re.fullmatch(r"linear_\d+", name)]
        if len(linear) != config.h_depth:
            raise ValueError("Native encoder hidden depth differs from the frozen configuration")
        for index in range(config.h_depth):
            layer = getattr(encoder, f"linear_{index}")
            expected_input = contract["representation_dimension"] if index == 0 else config.h_dim
            if (layer.in_features, layer.out_features) != (expected_input, config.h_dim):
                raise ValueError("Native encoder hidden width differs from the frozen configuration")
            if getattr(encoder, f"dropout_{index}").p != config.dropout:
                raise ValueError("Native encoder dropout differs from the frozen configuration")
        if encoder.loc.out_features != config.latent_dim or encoder.std_lin.out_features != config.latent_dim:
            raise ValueError("Native encoder latent dimension differs from the frozen configuration")
        expected_index = model.vertices.get_indexer(contract["feature_ids"])
        actual_index = getattr(model.net, f"{assay}_idx").detach().cpu().numpy()
        if not np.array_equal(actual_index, expected_index):
            raise ValueError("Native feature-to-graph index differs from frozen feature identities")
        balancing[assay] = domain["use_dsc_weight"]
    if getattr(model, "_trainer", None) is not None:
        raise ValueError("Native exported snapshot unexpectedly contains a trainer")
    return {"domain_balancing_weight_keys": balancing,
            "architecture_and_feature_order": "verified against original fit inputs"}


def validate_native_checkpoint(model, directory: Path, history: Mapping) -> dict[str, Any]:
    """Check the final saved object against the audited native restoration rule.

The history has already passed a finite-loss check. The current adapter also
returned from fit_native before its recorded failure, which means finite model
state and balancing weights were checked there. We reject missing/mismatched
snapshots; a newest file alone is never accepted as the final fitted model.
"""
    import torch
    checkpoints = []
    for path in directory.glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match is None:
            raise ValueError("Unexpected native checkpoint filename")
        checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise ValueError("No retained native optimizer checkpoints")
    epoch, selected = max(checkpoints)
    if epoch > history["last_recorded_epoch"]:
        raise ValueError("Restored checkpoint lies after the recorded native history")
    # All paths are local artifacts of this explicitly requested run. weights_only
    # is kept enabled: this check only needs the native tensor state dictionaries.
    checkpoint = torch.load(selected, map_location="cpu", weights_only=True)
    if set(checkpoint) != {"net", "trainer"} or set(checkpoint["trainer"]) != {"vae_optim", "dsc_optim"}:
        raise ValueError("Unsupported native checkpoint structure")
    state = model.net.state_dict()
    stored = checkpoint["net"]
    if set(state) != set(stored):
        raise ValueError("Final native snapshot and restored checkpoint have different state keys")
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all() or not torch.isfinite(stored[name]).all():
            raise FloatingPointError("Nonfinite native model/checkpoint state")
        if tensor.dtype != stored[name].dtype or not torch.equal(tensor.detach().cpu(), stored[name]):
            raise ValueError(f"Final native snapshot differs from restored checkpoint: {name}")
    return {
        "epoch": epoch, "path": str(selected.resolve()), "sha256": adapter.file_hash(selected),
        "retained_checkpoints": {path.name: adapter.file_hash(path) for _, path in sorted(checkpoints)},
        "snapshot_state_sha256": adapter.model_state_hash(model),
        "selection": "Exactly matches native final .dill and largest retained epoch under audited finite-history restoration",
        "not_claimed": "This is not a new selection of the globally minimum validation-loss epoch",
        "optimizer_state": "Native VAE/discriminator optimizer states preserved in original .pt; not resumed",
    }


def copy_native_snapshot(source: Path, destination: Path) -> None:
    """Byte-preserving, no-overwrite copy, including safe restart after a copy."""
    expected = adapter.file_hash(source)
    if destination.exists():
        if adapter.file_hash(destination) != expected:
            raise FileExistsError("Existing exported model is not the native final snapshot")
        return
    with source.open("rb") as original, destination.open("xb") as exported:
        shutil.copyfileobj(original, exported, length=1024 * 1024)
    if adapter.file_hash(destination) != expected:
        raise IOError("Native snapshot byte-copy failed verification")


def recover_export(run: Path, preprocessing: Path, *, device: str = "cpu",
                   cohorts: Sequence[str] | None = None) -> dict[str, Any]:
    """Recover the exact already-fitted native snapshot, then frozen-encode."""
    import anndata as ad
    import scglue
    # Fail before making any output if the reader dependency is still missing.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
    run, preprocessing = Path(run).resolve(), Path(preprocessing).resolve()
    fit, failure = validate_receipts(run, preprocessing)
    native_sources = native_source_audit()
    inputs = fit["inputs"]
    original_config = adapter.ScGLUEConfig(**inputs["configuration"])
    runtime_config = replace(original_config, device=device)
    runtime_config.validate()
    snapshots = {phase: run / "checkpoints" / phase / f"{phase}.dill" for phase in ("pretrain", "fine-tune")}
    if not all(path.is_file() for path in snapshots.values()):
        raise FileNotFoundError("Native final snapshots are required; reconstruction from an arbitrary .pt is not supported")
    preserved = {str(path.relative_to(run)): adapter.file_hash(path)
                 for path in [run / "fit_inputs.json", run / "failure.json", *snapshots.values()]
                 + sorted(path for path in (run / "checkpoints").rglob("*") if path.is_file())}
    recovery_inputs = {"preserved_original_files": preserved, "recovery_source_sha256": adapter.file_hash(Path(__file__)),
                       "runtime_device": device, "native_source_audit": native_sources}
    recovery_fingerprint = adapter.content_hash(recovery_inputs)
    model_manifest_path = run / "model_manifest.json"
    if model_manifest_path.exists():
        existing = json.loads(model_manifest_path.read_text())
        if (existing.get("input_fingerprint") != fit["input_fingerprint"]
                or existing.get("recovery", {}).get("input_fingerprint") != recovery_fingerprint):
            raise FileExistsError("Existing model completion is not this exact audited recovery")
        recovery_path = run / "recovery_provenance.json"
        if adapter.file_hash(recovery_path) != existing["recovery"]["provenance_sha256"]:
            raise ValueError("Recovery provenance changed")
        return adapter.encode_prepared(run, preprocessing, run, cohorts=cohorts, device=device)
    started = time.monotonic()
    history = adapter.training_event_summary(run / "checkpoints")
    prepared = {assay: ad.read_h5ad(Path(inputs["training_files"][assay]["path"])) for assay in adapter.ASSAYS}
    contracts = {assay: adapter.validate_prepared_adata(data, assay) for assay, data in prepared.items()}
    all_ids = [str(cell) for data in prepared.values() for cell in data.obs_names]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Frozen training assays unexpectedly contain paired/repeated cell IDs")
    graph = nx.read_graphml(inputs["guidance"]["path"])
    graph_audit = adapter.validate_guidance(graph, prepared)
    phase_audits = {}
    with adapter.native_runtime(runtime_config) as runtime:
        for phase in ("pretrain", "fine-tune"):
            model = scglue.models.load_model(snapshots[phase])
            domain_audit = validate_snapshot_configuration(model, contracts, graph, original_config, phase=phase)
            phase_audits[phase] = {**validate_native_checkpoint(model, run / "checkpoints" / phase, history[phase]),
                                   **domain_audit, "native_snapshot_sha256": adapter.file_hash(snapshots[phase])}
            if phase == "pretrain":
                del model
        state_hash = adapter.model_state_hash(model)
        samples = {assay: data[:min(32, data.n_obs)].copy() for assay, data in prepared.items()}
        before = {assay: adapter.encode_frozen(model, assay, data, contracts[assay],
                                               batch_size=runtime_config.encode_batch_size)["shared"]
                  for assay, data in samples.items()}
        if adapter.model_state_hash(model) != state_hash:
            raise RuntimeError("Native sample encoding modified the restored state")
        checkpoint = run / "scglue_model.dill"
        copy_native_snapshot(snapshots["fine-tune"], checkpoint)
        reloaded = scglue.models.load_model(checkpoint)
        if adapter.model_state_hash(reloaded) != state_hash:
            raise RuntimeError("Exported checkpoint state is not the exact native final state")
        errors = {}
        for assay, data in samples.items():
            after = adapter.encode_frozen(reloaded, assay, data, contracts[assay],
                                           batch_size=runtime_config.encode_batch_size)["shared"]
            errors[assay] = float(np.max(np.abs(after - before[assay])))
            if not np.array_equal(before[assay], after):
                raise RuntimeError("Byte-identical native snapshots produce different sample embeddings")
        if adapter.model_state_hash(reloaded) != state_hash:
            raise RuntimeError("Round-trip sample encoding modified model parameters/buffers")
    for relative, digest in preserved.items():
        if adapter.file_hash(run / relative) != digest:
            raise RuntimeError("Recovery modified an original native artifact")
    balancing = {
        "configuration": "Native fine-tune snapshot specifies balancing_weight for all assays",
        "numeric_values": "Not persisted by native model/optimizer serialization; not regenerated",
        "prior_validation": "Frozen adapter returned from fit_native, which checks finite/nonnegative positive-sum weights, before this recorded history-reader failure",
        "inference_requirement": "Balancing weights affect training only and are not required by native eval-mode encoding",
    }
    recovery = {
        "status": "complete", "operation": "post_fit_export_recovery_without_retraining",
        "input_fingerprint": recovery_fingerprint, "inputs": recovery_inputs,
        "original_failure": failure, "original_failure_preserved": True,
        "native_phases": phase_audits, "balancing_weights": balancing,
        "serialization": "scglue_model.dill is a byte-identical copy of native fine-tune/fine-tune.dill",
        "optimization_steps": 0, "balancing_reestimations": 0,
        "roundtrip_max_abs_error": errors,
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("tensorboard", "tensorboardX")},
        "integrity_scope": "Native artifact hashes were first recorded during recovery because export failed before a completion receipt; cross-checks establish internal consistency, not an independent pre-failure anti-tampering attestation",
        "scope": "Restores the final fitted network/domain configuration, not a resumable trainer or unpersisted balancing-weight vectors",
    }
    recovery_path = run / "recovery_provenance.json"
    adapter.write_json_new(recovery_path, recovery)
    default = asdict(adapter.ScGLUEConfig(seed=original_config.seed, device=original_config.device))
    audit = {
        "input_contracts": contracts, "guidance": graph_audit,
        "runtime": {"original_training_device": original_config.device, "recovery_inference": runtime,
                    "original_runtime_caveat": "Runtime audit returned in memory was not persisted before failure; no invented original runtime values"},
        "native_configuration": adapter.native_configuration(original_config, run / "checkpoints"),
        "training_sequence": ["native_pretraining", "native_balancing_weights", "native_fine_tuning"],
        "epoch_budget_interpretation": "max_epochs applies separately to pretraining and fine-tuning",
        "recovery": "Verified native completed-phase snapshots; no training repeated",
    }
    manifest = {
        "status": "complete", "label": inputs["label"], "method": "scglue", "seed": original_config.seed,
        "configuration": asdict(original_config),
        "protocol_overrides": {key: value for key, value in asdict(original_config).items() if value != default[key]},
        "input_fingerprint": fit["input_fingerprint"], "inputs": inputs, "training_audit": audit,
        "training_history": history, "model_state_sha256": state_hash,
        "checkpoint_sha256": adapter.file_hash(checkpoint), "checkpoint_roundtrip_max_abs_error": errors,
        "checkpoint_scope": "Native network/domain snapshot, not trainer; original optimizer checkpoints preserved under checkpoints/",
        "runtime_seconds": failure["runtime_seconds"], "runtime_scope": "Original failed export attempt, including native training",
        "recovery_runtime_seconds": time.monotonic() - started,
        "recovery": {"input_fingerprint": recovery_fingerprint, "provenance_path": str(recovery_path),
                     "provenance_sha256": adapter.file_hash(recovery_path)},
    }
    adapter.write_json_new(model_manifest_path, manifest)
    del prepared, model, reloaded
    return adapter.encode_prepared(run, preprocessing, run, cohorts=cohorts, device=device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preprocessing-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True,
                        help="Explicit inference device; match the original fit device for ordinary embedding-cache reuse")
    parser.add_argument("--cohorts", nargs="+")
    args = parser.parse_args(argv)
    result = recover_export(args.run_dir, args.preprocessing_dir, device=args.device, cohorts=args.cohorts)
    print(json.dumps({"status": result["status"], "label": result["label"], "run": str(args.run_dir),
                      "operation": "recovered_native_snapshot_export_no_retraining"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
