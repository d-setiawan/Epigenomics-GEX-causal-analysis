"""Pinned scConfluence count-input pilot with train-only preprocessing and exports."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

METHOD_HOME = Path(__file__).resolve().parent
REPO = METHOD_HOME.parents[4]
SIMULATION = REPO / "pipelines/simulation/integration_recoverability"
sys.path.insert(0, str(SIMULATION))
from gene_features import annotation_matrix

ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
UPSTREAM_SHA = "7f93822b48f33585a3406ca05fc325136ec3725d73aa555c9194f21bce9535db"


@dataclass(frozen=True)
class Config:
    seed: int = 0
    device: str = "cuda"
    n_latent: int = 30
    n_hidden: int = 128
    batch_size: int = 128
    max_epochs: int = 120
    patience: int = 8
    ratio_val: float = 0.2
    learning_rate: float = 0.003
    mass: float = 0.5
    reach: float = 0.3
    blur: float = 0.01
    iot_loss_weight: float = 0.01
    sinkhorn_loss_weight: float = 0.1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def native():
    upstream = METHOD_HOME / "upstream"
    if sha256(upstream / "UPSTREAM.json") != UPSTREAM_SHA:
        raise ValueError("Pinned upstream manifest changed")
    record = json.loads((upstream / "UPSTREAM.json").read_text())
    for name, expected in record["files"].items():
        if sha256(upstream / name) != expected:
            raise ValueError(f"Pinned upstream source changed: {name}")
    sys.path.insert(0, str(upstream))
    import scconfluence
    from scconfluence.model import ScConfluence
    from scconfluence.unimodal import AutoEncoder
    from loss_patch import corrected_loss, nonempty_latent_logs
    if Path(scconfluence.__file__).resolve().parent != upstream / "scconfluence":
        raise ValueError("A different scConfluence package was already imported")
    class CorrectedScConfluence(ScConfluence):
        loss = corrected_loss
        log_latent_norms = nonempty_latent_logs
    return CorrectedScConfluence, AutoEncoder


def input_records(directory):
    return {a: {"path": str((Path(directory) / f"{a}.h5ad").resolve()),
                "sha256": sha256(Path(directory) / f"{a}.h5ad")} for a in ASSAYS}


def counts(data):
    x = sparse.csr_matrix(data.X)
    x.sum_duplicates()
    x.eliminate_zeros()
    if (not data.obs_names.is_unique or not data.var_names.is_unique
            or not np.isfinite(x.data).all() or np.any(x.data < 0)
            or np.any(x.data != np.floor(x.data))):
        raise ValueError("Require unique IDs and finite nonnegative integer counts in X")
    return x.astype(np.float32)


def normalize(x):
    library = np.asarray(x.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("Cannot normalize zero count libraries")
    result = sparse.csr_matrix(sparse.diags(1e4 / library) @ x)
    result.data = np.log1p(result.data)
    return result.astype(np.float32)


def prepare(directory, contract=None, selection_paths=None):
    """Only training defines the feature mask; observed covariates are not copied."""
    import anndata as ad
    import pandas as pd
    fitting = contract is None
    if selection_paths is not None and (not fitting or set(selection_paths) != set(ASSAYS)):
        raise ValueError("Selection references apply only to all three training assays")
    contract = {"assays": {}, "selection_references": {},
                "normalization": "log1p(10000 * counts / observed library)"} if fitting else contract
    prepared, dropped = {}, {}
    seen_ids = set()
    for assay in ASSAYS:
        data = ad.read_h5ad(Path(directory) / f"{assay}.h5ad")
        selected_features = None
        if selection_paths is not None:
            import h5py
            reference_path = Path(selection_paths[assay]).resolve()
            with h5py.File(reference_path, "r") as reference:
                selected_ids = reference["obs"][reference["obs"].attrs["_index"]].asstr()[:].tolist()
                selected_features = reference["var"][reference["var"].attrs["_index"]].asstr()[:].tolist()
            if (len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(data.obs_names)
                    or len(set(selected_features)) != len(selected_features)
                    or not set(selected_features) <= set(data.var_names)):
                raise ValueError("Selection reference contains duplicate or unavailable IDs")
            data = data[selected_ids].copy()
            contract["selection_references"][assay] = {
                "path": str(reference_path), "sha256": sha256(reference_path),
                "cell_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
                "feature_ids_sha256": hashlib.sha256("\n".join(selected_features).encode()).hexdigest()}
        x = counts(data)
        names = data.var_names.astype(str).tolist()
        if "gene_id" not in data.var:
            raise ValueError("Require permitted gene_id annotations")
        annotations = data.var["gene_id"].astype(object).fillna("").astype(str).tolist()
        if fitting:
            used = (data.var_names.get_indexer(selected_features).tolist() if selected_features is not None
                    else np.flatnonzero(np.asarray(x.sum(axis=0)).ravel() > 0).tolist())
            contract["assays"][assay] = {"input_features": names, "input_gene_annotations": annotations,
                                         "used_feature_indices": used}
        info = contract["assays"][assay]
        if names != info["input_features"] or annotations != info["input_gene_annotations"]:
            raise ValueError(f"Held-out feature order or annotations changed for {assay}")
        used = info["used_feature_indices"]
        raw = x[:, used]
        keep = np.asarray(raw.sum(axis=1)).ravel() > 0
        ids = data.obs_names[keep].astype(str).tolist()
        if not ids or seen_ids.intersection(ids):
            raise ValueError("Require nonempty assay views with disjoint cell IDs")
        if fitting:
            info["training_cell_ids"] = ids
        elif set(ids).intersection(info["training_cell_ids"]):
            raise ValueError("Held-out cohort overlaps training cell IDs")
        seen_ids.update(ids)
        dropped[assay] = data.obs_names[~keep].astype(str).tolist()
        raw = raw[keep].copy()
        observed = ad.AnnData(normalize(raw), obs=pd.DataFrame(index=ids),
                              var=pd.DataFrame({"gene_id": [annotations[i] for i in used]},
                                               index=[names[i] for i in used]))
        observed.layers["counts"] = raw
        prepared[assay] = observed
    return prepared, contract, dropped


def cross_costs(prepared):
    """Tutorial-style correlation costs from all common variable training genes."""
    sets = [{str(g) for g in data.var["gene_id"] if str(g) not in {"", "nan", "None"}}
            for data in prepared.values()]
    genes = sorted(set.intersection(*sets))
    if len(genes) < 2:
        raise ValueError("Insufficient common annotated genes for alignment")
    values = {}
    for assay, data in prepared.items():
        mapping = annotation_matrix(data.var_names.tolist(), data.var["gene_id"].tolist(), genes)
        values[assay] = normalize(data.layers["counts"] @ mapping).toarray().astype(np.float64)
    varying = np.logical_and.reduce([x.std(axis=0, ddof=1) > 0 for x in values.values()])
    if varying.sum() < 2:
        raise ValueError("Insufficient variable common training genes")
    scaling = {}
    for assay, x in values.items():
        x = x[:, varying]
        mean, std = x.mean(axis=0), x.std(axis=0, ddof=1)
        values[assay] = (x - mean) / std
        scaling[assay] = {"mean": mean.tolist(), "std": std.tolist()}
    costs = {}
    for first, second in itertools.combinations(ASSAYS, 2):
        cost = cdist(values[first], values[second], metric="correlation")
        if not np.isfinite(cost).all():
            raise ValueError("Degenerate guide row; cannot invent an alignment cost")
        costs[f"{first}+{second}"] = cost
    return costs, {"genes_before_constant_filter": genes,
                   "genes_used": np.asarray(genes)[varying].tolist(), "scaling": scaling,
                   "cost": "Pearson correlation distance after per-assay log-library normalization and training feature scaling"}


def make_encoders(prepared, config):
    _, AutoEncoder = native()
    return {a: AutoEncoder(prepared[a], modality=a, rep_in=None, rep_out="counts",
                           type_loss="nb", n_hidden=config.n_hidden, n_latent=config.n_latent)
            for a in ASSAYS}


def embeddings(encoders, prepared, device, batch_size):
    import torch
    result = {}
    for assay in ASSAYS:
        encoder = encoders[assay].to(device).eval()
        values = []
        with torch.inference_mode():
            for start in range(0, prepared[assay].n_obs, batch_size):
                x = torch.as_tensor(prepared[assay].X[start:start + batch_size].toarray(), device=device)
                batch = torch.zeros((len(x), 1), dtype=torch.long, device=device)
                mean, _, _ = encoder.z_encoder(x, batch)
                values.append(mean.cpu().numpy())
        z = np.concatenate(values)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite frozen embeddings")
        result[assay] = z
    return result


def export(directory, cohort, prepared, values, dropped, inputs, checkpoint_sha, preprocessing_sha):
    path = Path(directory) / cohort
    path.mkdir(exist_ok=False)
    records = {}
    for assay in ASSAYS:
        ids = prepared[assay].obs_names.to_numpy(dtype=str)
        target = path / f"{assay}.npz"
        np.savez_compressed(target, cell_ids=ids, shared=values[assay])
        records[assay] = {"path": str(target.resolve()), "sha256": sha256(target),
                          "n_cells": len(ids), "latent_dim": values[assay].shape[1],
                          "cell_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                          "dropped_cell_ids": dropped[assay]}
    record = {"status": "complete", "cohort": cohort, "inputs": inputs,
              "checkpoint_sha256": checkpoint_sha, "preprocessing_sha256": preprocessing_sha,
              "exports": records, "encoder": "frozen posterior mean; no held-out fitting or cross-cost computation"}
    write_new(path / "encoding_manifest.json", record)
    return record


def reload_encoders(directory, prepared, config, device):
    import torch
    encoders = make_encoders(prepared, config)
    checkpoint = torch.load(Path(directory) / "checkpoint.pt", map_location="cpu", weights_only=True)
    if checkpoint["configuration"] != asdict(config):
        raise ValueError("Checkpoint configuration mismatch")
    for assay, encoder in encoders.items():
        prefix = f"aes.{assay}."
        encoder.load_state_dict({k[len(prefix):]: v for k, v in checkpoint["state_dict"].items()
                                 if k.startswith(prefix)}, strict=True)
        encoder.to(device).eval()
    return encoders


def fit_run(train_dir, output, config=Config(), selection_paths=None):
    import mudata as md
    import pytorch_lightning as pl
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        if config.device not in {"cuda", "cpu"} or not 0 < config.ratio_val < 1:
            raise ValueError("Unsupported device or validation fraction")
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Declared CUDA fit requires an available GPU")
        pl.seed_everything(config.seed, workers=True)
        inputs = input_records(train_dir)
        prepared, contract, dropped = prepare(train_dir, selection_paths=selection_paths)
        costs, guide = cross_costs(prepared)
        contract["guidance"] = guide
        write_new(output / "preprocessing.json", contract)
        preprocessing_sha = sha256(output / "preprocessing.json")
        data = md.MuData(prepared)
        data.uns["cross_keys"] = [f"cross_{pair}" for pair in costs]
        for pair, value in costs.items():
            data.uns[f"cross_{pair}"] = value
        Model, _ = native()
        encoders = make_encoders(prepared, config)
        model = Model(data, encoders, mass=config.mass, reach=config.reach, blur=config.blur,
                      iot_loss_weight=config.iot_loss_weight, sinkhorn_loss_weight=config.sinkhorn_loss_weight)
        model.fit(output / "native_fit", use_cuda=config.device == "cuda", ratio_val=config.ratio_val,
                  batch_size=config.batch_size, max_epochs=config.max_epochs, lr=config.learning_rate,
                  patience=config.patience, save_models=True, num_workers=0,
                  enable_progress_bar=False, enable_model_summary=False)
        torch.save({"configuration": asdict(config), "preprocessing_sha256": preprocessing_sha,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, output / "checkpoint.pt")
        before = embeddings(model.aes, prepared, config.device, config.batch_size)
        restored = reload_encoders(output, prepared, config, config.device)
        after = embeddings(restored, prepared, config.device, config.batch_size)
        errors = {a: float(np.max(np.abs(before[a] - after[a]))) for a in ASSAYS}
        if any(error > 1e-6 for error in errors.values()):
            raise ValueError("Checkpoint roundtrip altered embeddings")
        checkpoint_sha = sha256(output / "checkpoint.pt")
        export(output, "train", prepared, after, dropped, inputs, checkpoint_sha, preprocessing_sha)
        sources = [Path(__file__), METHOD_HOME / "loss_patch.py", METHOD_HOME / "upstream/UPSTREAM.json", SIMULATION / "gene_features.py"]
        record = {"status": "complete", "method": "scconfluence", "configuration": asdict(config),
                  "inputs": inputs, "checkpoint_sha256": checkpoint_sha, "preprocessing_sha256": preprocessing_sha,
                  "selection_references": contract["selection_references"],
                  "source_files": {str(p.resolve()): sha256(p) for p in sources},
                  "checkpoint_roundtrip_max_abs_error": errors,
                  "training_cells": {a: prepared[a].n_obs for a in ASSAYS},
                  "native_encoder": {a: {"rep_in": None, "rep_out": "counts", "type_loss": "nb",
                                          "input_width": prepared[a].n_vars} for a in ASSAYS},
                  "cross_cost_shapes": {p: list(v.shape) for p, v in costs.items()},
                  "versions": {p: importlib.metadata.version(p) for p in
                               ("numpy", "scipy", "torch", "pytorch-lightning", "anndata", "mudata", "geomloss", "POT")},
                  "python": sys.version, "upstream_commit": "14d81e2ca679e161ec5c6806103a66c14fcdcd9a",
                  "scope": "Train-only counts and permitted gene annotations; no states, paired observations, labels or time covariates"}
        write_new(output / "model_manifest.json", record)
        return record
    except Exception as error:
        write_new(output / "failure.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})
        raise


def encode_run(model_dir, input_dir, cohort, device="cpu"):
    if cohort not in {"val", "discovery", "test"}:
        raise ValueError("Use a declared public held-out cohort")
    model_dir = Path(model_dir)
    record = json.loads((model_dir / "model_manifest.json").read_text())
    if (record["status"] != "complete" or record["checkpoint_sha256"] != sha256(model_dir / "checkpoint.pt")
            or record["preprocessing_sha256"] != sha256(model_dir / "preprocessing.json")
            or any(sha256(Path(p)) != expected for p, expected in record["source_files"].items())):
        raise ValueError("Model, preprocessing or implementation changed after fit")
    contract = json.loads((model_dir / "preprocessing.json").read_text())
    prepared, _, dropped = prepare(input_dir, contract)
    config = Config(**record["configuration"])
    model = reload_encoders(model_dir, prepared, config, device)
    values = embeddings(model, prepared, device, config.batch_size)
    return export(model_dir, cohort, prepared, values, dropped, input_records(input_dir),
                  record["checkpoint_sha256"], record["preprocessing_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "encode"))
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cohort", choices=("val", "discovery", "test"))
    parser.add_argument("--selection-dir")
    args = parser.parse_args()
    selection = ({a: Path(args.selection_dir) / f"{a}.h5ad" for a in ASSAYS} if args.selection_dir else None)
    result = (fit_run(args.input_dir, args.output, Config(seed=args.seed, device=args.device), selection)
              if args.action == "fit" else encode_run(args.output, args.input_dir, args.cohort, args.device))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
