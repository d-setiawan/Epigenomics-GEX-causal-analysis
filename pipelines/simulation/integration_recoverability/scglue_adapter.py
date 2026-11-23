"""Native scGLUE adapter for already-frozen recoverability preprocessing.

This module never loads simulator truth.  Only prepared training AnnData and an
annotation-derived guidance graph enter fit_SCGLUE.  Private evaluation views
may be encoded *after* the fitted native checkpoint has been saved, without
optimization, to support evaluator-only information-retention diagnostics.

The installed upstream fit_SCGLUE procedure is used unchanged: pretraining,
balancing-weight estimation, and fine-tuning.  Scheduler API compatibility is
scoped to a context manager and does not alter losses, optimization, or epochs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence
import time

import networkx as nx
import numpy as np
from scipy import sparse


ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
REPRESENTATIONS = {"RNA": "X_pca", "H3K27ac": "X_lsi", "H3K4me3": "X_lsi"}
FORBIDDEN_KEYS = {
    "state", "s0", "s1", "true_state", "true_z", "true_depth", "true_exposure",
    "exposure", "exposures", "activities", "activity", "candidate_role", "role",
    "partner_id", "paired_id", "biological_cell_id", "epsilon", "innovation",
    "sentinel", "truth", "true_cell_type",
}


@dataclass(frozen=True)
class ScGLUEConfig:
    seed: int = 0
    latent_dim: int = 30
    h_depth: int = 2
    h_dim: int = 256
    dropout: float = 0.2
    lam_data: float = 1.0
    lam_kl: float = 1.0
    lam_graph: float = 0.02
    lam_align: float = 0.05
    lam_sup: float = 0.02
    learning_rate: float = 0.002
    neg_samples: int = 10
    val_split: float = 0.1
    data_batch_size: int = 128
    graph_batch_size: int = 8192
    align_burnin: int = 20
    max_epochs: int = 120
    patience: int = 8
    reduce_lr_patience: int = 4
    wait_n_lrs: int = 1
    encode_batch_size: int = 256
    device: str = "cuda"

    def validate(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be explicitly cpu or cuda")
        if not 0 < self.val_split < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid validation fraction or dropout")
        for key in ("latent_dim", "h_depth", "h_dim", "neg_samples", "data_batch_size",
                    "graph_batch_size", "max_epochs", "patience", "reduce_lr_patience", "encode_batch_size"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.align_burnin < 0 or self.wait_n_lrs < 0:
            raise ValueError("Burn-in and scheduler waits must be nonnegative")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Refusing to overwrite changed artifact: {path}")
        return
    with path.open("x") as handle:
        handle.write(text)


def native_configuration(config: ScGLUEConfig, directory: Path) -> dict[str, dict[str, Any]]:
    config.validate()
    return {
        "init_kws": {
            "latent_dim": config.latent_dim, "h_depth": config.h_depth, "h_dim": config.h_dim,
            "dropout": config.dropout, "shared_batches": False, "random_seed": config.seed,
        },
        "compile_kws": {
            "lam_data": config.lam_data, "lam_kl": config.lam_kl, "lam_graph": config.lam_graph,
            "lam_align": config.lam_align, "lam_sup": config.lam_sup, "lr": config.learning_rate,
        },
        "fit_kws": {
            "directory": str(directory), "neg_samples": config.neg_samples, "val_split": config.val_split,
            "data_batch_size": config.data_batch_size, "graph_batch_size": config.graph_batch_size,
            "align_burnin": config.align_burnin, "max_epochs": config.max_epochs, "patience": config.patience,
            "reduce_lr_patience": config.reduce_lr_patience, "wait_n_lrs": config.wait_n_lrs,
        },
    }


def reject_truth_metadata(adata, *, context: str) -> None:
    for namespace, keys in (("obs", adata.obs.columns), ("var", adata.var.columns),
                            ("obsm", adata.obsm.keys()), ("layers", adata.layers.keys()), ("uns", adata.uns.keys())):
        blocked = [str(key) for key in keys if str(key).lower() in FORBIDDEN_KEYS or str(key).lower().startswith("true_")]
        if blocked:
            raise ValueError(f"Forbidden truth metadata in {context}/{namespace}: {blocked}")


def validate_prepared_adata(adata, assay: str, *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if assay not in ASSAYS:
        raise ValueError(f"Unknown assay {assay}")
    reject_truth_metadata(adata, context=assay)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"Empty prepared data for {assay}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"Repeated cell or feature IDs in {assay}")
    if "assay" in adata.obs and not np.all(np.asarray(adata.obs["assay"], dtype=str) == assay):
        raise ValueError(f"Prepared assay labels disagree with input key {assay}")
    if "counts" not in adata.layers:
        raise ValueError(f"Prepared {assay} lacks the raw counts layer")
    matrix = adata.layers["counts"]
    nonzero = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(nonzero).all() or np.any(nonzero < 0) or np.any(nonzero != np.floor(nonzero)):
        raise ValueError(f"Prepared {assay} counts must be finite nonnegative integers")
    representation = REPRESENTATIONS[assay]
    if representation not in adata.obsm:
        raise ValueError(f"Prepared {assay} lacks {representation}")
    values = np.asarray(adata.obsm[representation])
    if values.ndim != 2 or values.shape[0] != adata.n_obs or not values.shape[1]:
        raise ValueError(f"Invalid {representation} dimensions")
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite prepared representation in {assay}; preprocessing must record exclusions")
    if "highly_variable" not in adata.var or not np.asarray(adata.var["highly_variable"], dtype=bool).all():
        raise ValueError("Prepared matrices must contain only training-selected reconstruction features")
    features = list(map(str, adata.var_names))
    result = {
        "feature_ids": features, "feature_order_sha256": content_hash(features),
        "representation": representation, "representation_dimension": values.shape[1],
        "n_cells": adata.n_obs, "n_reconstruction_features": adata.n_vars,
        "zero_reconstruction_count_rows": int(np.count_nonzero(np.asarray(matrix.sum(axis=1)).reshape(-1) == 0)),
        "cell_ids_sha256": content_hash(list(map(str, adata.obs_names))),
    }
    if expected is not None:
        if result["feature_ids"] != list(expected["feature_ids"]):
            raise ValueError(f"Frozen feature identity/order mismatch for {assay}; no implicit reorder")
        if result["representation_dimension"] != expected["representation_dimension"]:
            raise ValueError(f"Frozen representation dimension mismatch for {assay}")
    return result


def validate_guidance(graph, prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Validate supplied guidance without opening a planted-edge/truth file."""
    features = [str(feature) for adata in prepared.values() for feature in adata.var_names]
    if len(features) != len(set(features)):
        raise ValueError("Global guidance requires distinct feature IDs across assays")
    if set(graph.nodes) != set(features):
        raise ValueError("Guidance vertices must equal the frozen reconstruction feature universe")
    if set(nx.nodes_with_selfloops(graph)) != set(features):
        raise ValueError("Every reconstruction feature requires an explicit guidance self-loop")
    for first, second, attrs in graph.edges(data=True):
        if "weight" not in attrs or "sign" not in attrs:
            raise ValueError("Guidance edge lacks weight/sign")
        weight, sign = float(attrs["weight"]), float(attrs["sign"])
        if not np.isfinite(weight) or not 0 < weight <= 1 or sign not in {-1, 1}:
            raise ValueError("Invalid guidance weight/sign")
        if first == second and (sign != 1 or weight != 1):
            raise ValueError("Guidance self-loops must have positive unit weight")
        if graph.is_directed() and not graph.has_edge(second, first):
            raise ValueError("Synthetic guidance must contain reciprocal links; directions are not causal priors")
    return {"n_vertices": graph.number_of_nodes(), "n_edges": graph.number_of_edges(),
            "n_self_loops": nx.number_of_selfloops(graph), "directed": graph.is_directed(),
            "truth_access": "none; graph supplied by annotation-only preprocessing"}


@contextmanager
def native_runtime(config: ScGLUEConfig):
    """Temporary runtime/API settings, restored even if native fitting fails."""
    import scglue
    import scglue.models.plugins as plugins
    import torch
    config.validate()
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
    original_cpu = scglue.config.CPU_ONLY
    original_scheduler = plugins.ReduceLROnPlateau
    original_numpy = np.random.get_state()
    original_python = random.getstate()
    original_torch = torch.random.get_rng_state()
    original_cuda = torch.cuda.get_rng_state_all() if config.device == "cuda" else None
    scheduler_patch = "verbose" not in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters
    audit = {"device_requested": config.device, "scheduler_verbose_api_shim": scheduler_patch,
             "networkx_api_shim": False, "objective_changes": [],
             "scglue_runtime": {key: getattr(scglue.config, key) for key in (
                 "DATALOADER_NUM_WORKERS", "ARRAY_SHUFFLE_NUM_WORKERS", "GRAPH_SHUFFLE_NUM_WORKERS", "CUDNN_MODE")}}
    try:
        scglue.config.CPU_ONLY = config.device == "cpu"
        np.random.seed(config.seed)
        random.seed(config.seed)
        if scheduler_patch:
            # Return the native torch scheduler instance; only the removed
            # presentation-only keyword is discarded. No global torch patch.
            def scheduler_compat(*args, verbose=None, **kwargs):
                return original_scheduler(*args, **kwargs)
            plugins.ReduceLROnPlateau = scheduler_compat
        yield audit
    finally:
        plugins.ReduceLROnPlateau = original_scheduler
        scglue.config.CPU_ONLY = original_cpu
        np.random.set_state(original_numpy)
        random.setstate(original_python)
        torch.random.set_rng_state(original_torch)
        if original_cuda is not None:
            torch.cuda.set_rng_state_all(original_cuda)


def package_provenance() -> dict[str, Any]:
    import scglue
    sources = {}
    for label, obj in (("fit_SCGLUE", scglue.models.fit_SCGLUE), ("SCGLUEModel", scglue.models.SCGLUEModel)):
        path = Path(inspect.getsourcefile(obj))
        sources[label] = {"path": str(path), "sha256": file_hash(path)}
    packages = ("scglue", "torch", "anndata", "scanpy", "networkx", "numpy", "scipy", "pytorch-ignite")
    return {"packages": {name: importlib.metadata.version(name) for name in packages}, "sources": sources,
            "documentation": "https://scglue.readthedocs.io/en/latest/api/scglue.models.fit_SCGLUE.html",
            "version_policy": "Installed source/version is authoritative; current online docs may describe a newer version"}


def model_state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.net.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_native(prepared: Mapping[str, Any], graph, config: ScGLUEConfig, checkpoint_directory: Path):
    """Call the installed upstream two-phase training routine, not a rewrite."""
    import scglue
    if set(prepared) != set(ASSAYS):
        raise ValueError("The primary adapter requires RNA, H3K27ac, and H3K4me3")
    contracts = {assay: validate_prepared_adata(prepared[assay], assay) for assay in ASSAYS}
    cell_ids = [str(cell) for adata in prepared.values() for cell in adata.obs_names]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("Training views must be disjoint; paired IDs are prohibited")
    for assay in ASSAYS:
        if prepared[assay].n_obs < max(3, int(np.ceil(1 / config.val_split))):
            raise ValueError(f"Too few {assay} training cells for the fixed validation fraction")
    graph_audit = validate_guidance(graph, prepared)
    with native_runtime(config) as runtime:
        for assay in ASSAYS:
            scglue.models.configure_dataset(
                prepared[assay], "NB", use_highly_variable=True, use_layer="counts",
                use_rep=REPRESENTATIONS[assay], use_batch=None, use_cell_type=None,
                use_dsc_weight=None, use_uid=None,
            )
        kwargs = native_configuration(config, checkpoint_directory)
        model = scglue.models.fit_SCGLUE(prepared, graph, **kwargs)
        if str(model.net.device).split(":")[0] != config.device:
            raise RuntimeError(f"Requested {config.device}, but native scGLUE used {model.net.device}")
        for assay, adata in prepared.items():
            if "balancing_weight" not in adata.obs:
                raise RuntimeError(f"Native balancing weights were not produced for {assay}")
            weights = np.asarray(adata.obs["balancing_weight"], dtype=float)
            if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                raise FloatingPointError(f"Invalid native balancing weights for {assay}")
        for name, value in model.net.state_dict().items():
            if not bool(value.isfinite().all()):
                raise FloatingPointError(f"Nonfinite fitted model state: {name}")
    return model, {"input_contracts": contracts, "guidance": graph_audit,
                   "runtime": runtime, "native_configuration": kwargs,
                   "training_sequence": ["native_pretraining", "native_balancing_weights", "native_fine_tuning"],
                   "epoch_budget_interpretation": "max_epochs applies separately to pretraining and fine-tuning"}


def encode_frozen(model, assay: str, adata, contract: Mapping[str, Any], *, batch_size: int) -> dict[str, np.ndarray]:
    validate_prepared_adata(adata, assay, expected=contract)
    shared = np.asarray(model.encode_data(assay, adata, batch_size=batch_size))
    if shared.shape != (adata.n_obs, model.net.x2u[assay].loc.out_features):
        raise ValueError("Native embedding shape differs from encoder latent dimension")
    if not np.isfinite(shared).all():
        raise FloatingPointError(f"Native {assay} embedding contains nonfinite values")
    return {"cell_ids": np.asarray(adata.obs_names, dtype=str), "shared": shared}


def transformation_fingerprint(preprocessing: Path) -> dict[str, Any]:
    paths = sorted(path for path in (preprocessing / "transforms").rglob("*") if path.is_file())
    if not paths:
        raise ValueError("Frozen preprocessing transformation artifacts are missing")
    files = {str(path.relative_to(preprocessing)): file_hash(path) for path in paths}
    return {"files": files, "fingerprint": content_hash(files)}


def training_event_summary(directory: Path) -> dict[str, Any]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    summary = {}
    for phase in ("pretrain", "fine-tune"):
        path = directory / phase / "tensorboard"
        if not path.exists():
            raise RuntimeError(f"Native {phase} TensorBoard history is missing")
        events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        tags = events.Tags().get("scalars", [])
        histories = {tag: events.Scalars(tag) for tag in tags}
        if not histories:
            raise RuntimeError(f"Native {phase} has no recorded loss history")
        nonfinite = [tag for tag, history in histories.items() if not all(np.isfinite(event.value) for event in history)]
        if nonfinite:
            raise FloatingPointError(f"Native {phase} recorded nonfinite losses: {nonfinite}")
        summary[phase] = {
            "last_recorded_epoch": max(event.step for history in histories.values() for event in history),
            "final_recorded_scalars": {tag: float(history[-1].value) for tag, history in histories.items() if history},
            "checkpoints": sorted(path.name for path in (directory / phase).glob("checkpoint_*.pt")),
        }
    return summary


def _cohort_path(preprocessing: Path, cohort: str, assay: str) -> Path:
    return preprocessing / "train" / f"{assay}.h5ad" if cohort == "train" else preprocessing / "cohorts" / cohort / f"{assay}.h5ad"


def _available_cohorts(preprocessing: Path) -> list[str]:
    values = ["train"]
    root = preprocessing / "cohorts"
    if root.exists():
        values.extend(path.name for path in sorted(root.iterdir()) if path.is_dir() and path.name != "train")
    return values


def _save_embedding(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], arrays[key]) for key in arrays):
                raise FileExistsError(f"Refusing to overwrite changed embedding {path}")
        return
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def encode_prepared(
    model_directory: Path, preprocessing: Path, output: Path, *,
    cohorts: Sequence[str] | None = None, device: str | None = None,
) -> dict[str, Any]:
    import anndata as ad
    import scglue
    model_manifest_path = model_directory / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    if model_manifest["status"] != "complete":
        raise ValueError("A complete native model checkpoint is required for encoding")
    checkpoint = model_directory / "scglue_model.dill"
    if file_hash(checkpoint) != model_manifest["checkpoint_sha256"]:
        raise ValueError("Native model checkpoint changed")
    transformations = transformation_fingerprint(preprocessing)
    if transformations["fingerprint"] != model_manifest["inputs"]["transformations"]["fingerprint"]:
        raise ValueError("Encoding uses different fitted preprocessing transformations")
    config_fields = dict(model_manifest["configuration"])
    if device is not None:
        config_fields["device"] = device
    config = ScGLUEConfig(**config_fields)
    selected_cohorts = list(cohorts or _available_cohorts(preprocessing))
    if len(set(selected_cohorts)) != len(selected_cohorts):
        raise ValueError("Repeated encode cohort")
    source_files = {f"{cohort}/{assay}": {"path": str(_cohort_path(preprocessing, cohort, assay).resolve()),
                                            "sha256": file_hash(_cohort_path(preprocessing, cohort, assay))}
                    for cohort in selected_cohorts for assay in ASSAYS}
    provenance = package_provenance()
    inputs = {"model_manifest_sha256": file_hash(model_manifest_path), "source_files": source_files,
              "configuration": asdict(config), "package_provenance": provenance,
              "adapter_sha256": file_hash(Path(__file__))}
    fingerprint = content_hash(inputs)
    manifest_path = output / "embedding_manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached["input_fingerprint"] != fingerprint:
            raise FileExistsError("Existing embeddings used different inputs; choose another output directory")
        for path, digest in cached["files"].items():
            if file_hash(output / path) != digest:
                raise ValueError("Cached embedding artifact changed")
        return cached
    started = time.monotonic()
    files, retention = {}, {}
    with native_runtime(config) as runtime:
        model = scglue.models.load_model(checkpoint)
        before = model_state_hash(model)
        if before != model_manifest["model_state_sha256"]:
            raise ValueError("Reloaded model weights differ from the fitted checkpoint")
        for cohort in selected_cohorts:
            for assay in ASSAYS:
                data = ad.read_h5ad(_cohort_path(preprocessing, cohort, assay))
                contract = model_manifest["training_audit"]["input_contracts"][assay]
                arrays = encode_frozen(model, assay, data, contract, batch_size=config.encode_batch_size)
                path = output / cohort / f"{assay}.npz"
                _save_embedding(path, arrays)
                files[str(path.relative_to(output))] = file_hash(path)
                retention[f"{cohort}/{assay}"] = {"n_cells": data.n_obs, "n_features": data.n_vars,
                                                  "latent_dimension": arrays["shared"].shape[1]}
                del data
        if model_state_hash(model) != before:
            raise RuntimeError("Frozen encoding modified model parameters or buffers")
    result = {
        "status": "complete", "label": model_manifest["label"], "method": "scglue", "seed": config.seed,
        "input_fingerprint": fingerprint, "inputs": inputs, "files": files, "retention": retention,
        "model_directory": str(model_directory.resolve()), "checkpoint_sha256": file_hash(checkpoint),
        "runtime": runtime, "runtime_seconds": time.monotonic() - started,
        "fit_access": "Prepared observed training views only",
        "encode_access": "Frozen inference; probe/assessment views are evaluator-only and never used in optimization",
    }
    write_json_new(manifest_path, result)
    return result


def fit_prepared(
    preprocessing: Path, guidance: Path, output: Path, *,
    config: ScGLUEConfig | None = None, label: str | None = None,
    cohorts: Sequence[str] | None = None,
) -> dict[str, Any]:
    import anndata as ad
    import scglue
    config = config or ScGLUEConfig()
    config.validate()
    preprocessing, guidance, output = preprocessing.resolve(), guidance.resolve(), output.resolve()
    label = label or f"scglue_seed{config.seed}"
    transformations = transformation_fingerprint(preprocessing)
    training_paths = {assay: preprocessing / "train" / f"{assay}.h5ad" for assay in ASSAYS}
    inputs = {
        "training_files": {assay: {"path": str(path), "sha256": file_hash(path)} for assay, path in training_paths.items()},
        "guidance": {"path": str(guidance), "sha256": file_hash(guidance)},
        "transformations": transformations, "configuration": asdict(config), "label": label,
        "package_provenance": package_provenance(), "adapter_sha256": file_hash(Path(__file__)),
    }
    fingerprint = content_hash(inputs)
    model_manifest_path = output / "model_manifest.json"
    if model_manifest_path.exists():
        existing = json.loads(model_manifest_path.read_text())
        if existing["input_fingerprint"] != fingerprint:
            raise FileExistsError("Cached scGLUE model has different training inputs/configuration")
        return encode_prepared(output, preprocessing, output, cohorts=cohorts)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to restart training in a nonempty partial run directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json_new(output / "fit_inputs.json", {"input_fingerprint": fingerprint, "inputs": inputs})
    started = time.monotonic()
    stage = "load_training_inputs"
    try:
        prepared = {assay: ad.read_h5ad(path) for assay, path in training_paths.items()}
        graph = nx.read_graphml(guidance)
        stage = "native_fit_SCGLUE"
        model, audit = fit_native(prepared, graph, config, output / "checkpoints")
        stage = "native_training_history_validation"
        history = training_event_summary(output / "checkpoints")
        stage = "save_and_validate_checkpoint"
        checkpoint = output / "scglue_model.dill"
        state_hash = model_state_hash(model)
        with native_runtime(config):
            sample_before = {
                assay: encode_frozen(model, assay, prepared[assay][:min(32, prepared[assay].n_obs)],
                                     audit["input_contracts"][assay], batch_size=config.encode_batch_size)["shared"]
                for assay in ASSAYS
            }
            model.save(checkpoint)
            reloaded = scglue.models.load_model(checkpoint)
            if model_state_hash(reloaded) != state_hash:
                raise RuntimeError("Native checkpoint failed the exact state round-trip")
            errors = {}
            for assay in ASSAYS:
                after = encode_frozen(reloaded, assay, prepared[assay][:len(sample_before[assay])],
                                      audit["input_contracts"][assay], batch_size=config.encode_batch_size)["shared"]
                errors[assay] = float(np.max(np.abs(after - sample_before[assay])))
                if not np.allclose(after, sample_before[assay], rtol=1e-5, atol=1e-6):
                    raise RuntimeError(f"Native checkpoint changes {assay} embeddings")
        default = asdict(ScGLUEConfig(seed=config.seed, device=config.device))
        overrides = {key: value for key, value in asdict(config).items() if value != default[key]}
        model_manifest = {
            "status": "complete", "label": label, "method": "scglue", "seed": config.seed,
            "configuration": asdict(config), "protocol_overrides": overrides,
            "input_fingerprint": fingerprint, "inputs": inputs, "training_audit": audit,
            "training_history": history, "model_state_sha256": state_hash,
            "checkpoint_sha256": file_hash(checkpoint), "checkpoint_roundtrip_max_abs_error": errors,
            "checkpoint_scope": "Native .dill stores network/configuration, not trainer; native optimizer checkpoints are under checkpoints/",
            "runtime_seconds": time.monotonic() - started,
        }
        write_json_new(model_manifest_path, model_manifest)
        del prepared, model, reloaded
        stage = "encode_frozen_cohorts"
        return encode_prepared(output, preprocessing, output, cohorts=cohorts)
    except Exception as error:
        failure = {"status": "failed", "stage": stage, "error_type": type(error).__name__, "error": str(error),
                   "input_fingerprint": fingerprint, "runtime_seconds": time.monotonic() - started}
        write_json_new(output / "failure.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Run native pretraining, balancing, fine-tuning, then frozen encoding")
    fit.add_argument("--preprocessing-dir", type=Path, required=True)
    fit.add_argument("--guidance", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    fit.add_argument("--max-epochs", type=int, default=120,
                     help="Per native phase; nondefault values are explicitly recorded as protocol overrides")
    fit.add_argument("--label")
    fit.add_argument("--cohorts", nargs="+")
    encode = commands.add_parser("encode", help="Encode additional views with a frozen saved model; no optimization")
    encode.add_argument("--model-dir", type=Path, required=True)
    encode.add_argument("--preprocessing-dir", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--device", choices=("cuda", "cpu"))
    encode.add_argument("--cohorts", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "fit":
        result = fit_prepared(args.preprocessing_dir, args.guidance, args.output,
                              config=ScGLUEConfig(seed=args.seed, device=args.device, max_epochs=args.max_epochs),
                              label=args.label, cohorts=args.cohorts)
    else:
        result = encode_prepared(args.model_dir, args.preprocessing_dir, args.output,
                                 cohorts=args.cohorts, device=args.device)
    print(json.dumps({"status": result["status"], "label": result["label"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
