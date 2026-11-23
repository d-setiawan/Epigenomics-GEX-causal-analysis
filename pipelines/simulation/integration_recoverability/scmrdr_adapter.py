"""Benchmark I/O around paper-faithful scMRDR; no truth-dependent fitting.

Fit reads only learner-visible train/validation count files. Encode is a
separate operation with a frozen checkpoint; private cohort inputs are allowed
only in this post-fit diagnostic operation. No held-out feature selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from gene_features import GeneConfig, fit_gene_features, transform_gene_features

ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
METHOD_HOME = Path(__file__).resolve().parents[2] / "empirical/integration/methods/scmrdr"
sys.path.insert(0, str(METHOD_HOME))
from paper_faithful import SCmRDRConfig, ScMRDRAdapter, fit


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


def _paths(directory):
    return {assay: Path(directory) / f"{assay}.h5ad" for assay in ASSAYS}


def _reuse_gene_manifest(directory, train, config, exclude, train_limit):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["config"] != asdict(config) or manifest["train_limit"] != train_limit
            or manifest["excluded_features"] != sorted(exclude)):
        raise ValueError("Cached gene transform has different fitting configuration")
    for assay, path in train.items():
        if manifest["assays"][assay]["source_sha256"] != sha256(path):
            raise ValueError("Cached gene transform has different training counts")
    if manifest["source_sha256"] != sha256(Path(__file__).with_name("gene_features.py")):
        raise ValueError("Gene preprocessing implementation changed; use a new transform directory")
    return manifest


def fit_run(train_dir, out_dir, *, validation_dir=None, gene_transform_dir=None,
            seed=0, device="cuda", max_epochs=200, train_limit=2000,
            excluded_features=(), n_top_genes=2000, label=None):
    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(f"Run already exists: {root}")
    train = _paths(train_dir)
    gene_config = GeneConfig(n_top_genes=n_top_genes)
    transform_root = Path(gene_transform_dir) if gene_transform_dir else root / "gene_transform"
    started = time.monotonic()
    # Do not create the run before checking reusable preprocessing.
    if transform_root.exists():
        gene = _reuse_gene_manifest(transform_root, train, gene_config, excluded_features, train_limit)
    else:
        gene = fit_gene_features(train, transform_root, gene_config,
                                 tuple(excluded_features), train_limit=train_limit)
    root.mkdir(parents=True, exist_ok=True)
    config = SCmRDRConfig(seed=seed, device=device, max_epochs=max_epochs)
    features = gene["feature_names"]
    inputs = {a: transform_gene_features(p, a, gene, train_limit) for a, p in train.items()}
    validation = ({a: transform_gene_features(p, a, gene) for a, p in _paths(validation_dir).items()}
                   if validation_dir is not None else None)
    record = {"schema_version": 1, "method": "scmrdr_paper_faithful", "seed": seed,
              "label": label or f"scmrdr_paper_seed{seed}", "status": "running",
              "config": asdict(config), "gene_transform_dir": str(transform_root.resolve()),
              "gene_manifest_sha256": sha256(transform_root / "manifest.json"),
              "feature_count": len(features), "train_limit": train_limit,
              "train_inputs": {a: {"path": str(p.resolve()), "sha256": sha256(p),
                                    "retained_cells": len(inputs[a]["cell_ids"]),
                                    "dropped_cell_ids": inputs[a]["dropped_cell_ids"].tolist()}
                               for a, p in train.items()},
              "validation_inputs": ({a: {"path": str(p.resolve()), "sha256": sha256(p)}
                                     for a, p in _paths(validation_dir).items()} if validation else None),
              "versions": {package: importlib.metadata.version(package) for package in
                           ("numpy", "scipy", "torch", "scanpy", "anndata", "scikit-misc")},
              "adapter_source_sha256": sha256(Path(__file__)),
              "protocol_note": "2000 assay-wise gene HVGs is a benchmark choice, not a paper-stated number"}
    write_new(root / "input_manifest.json", record)
    try:
        model = fit({a: x["counts"] for a, x in inputs.items()},
                    {a: x["availability"] for a, x in inputs.items()}, features,
                    root / "model", config=config,
                    validation_counts_by_modality=({a: x["counts"] for a, x in validation.items()} if validation else None))
        for assay, x in inputs.items():
            values = model.encode(x["counts"], assay, features)
            path = root / "train" / f"{assay}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                np.savez_compressed(handle, cell_ids=x["cell_ids"], shared=values["shared"],
                                    private=values["specific"])
        record.update(status="complete", runtime_seconds=time.monotonic() - started,
                      checkpoint_sha256=sha256(root / "model/checkpoint.pt"))
        write_new(root / "manifest.json", record)
        return record
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error),
                      runtime_seconds=time.monotonic() - started)
        write_new(root / "failure.json", record)
        raise


def encode_run(run_dir, input_dir, cohort, *, limit=None, device="cuda"):
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    transform = Path(manifest["gene_transform_dir"])
    if sha256(transform / "manifest.json") != manifest["gene_manifest_sha256"]:
        raise ValueError("Frozen gene manifest changed")
    if sha256(root / "model/checkpoint.pt") != manifest["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if Path(cohort).name != cohort or cohort in {"", ".", "..", "train"}:
        raise ValueError("Use a distinct simple cohort name; train is exported during fit")
    destination = root / cohort
    if destination.exists():
        raise FileExistsError(f"Encoded cohort already exists: {destination}")
    gene = json.loads((transform / "manifest.json").read_text())
    model = ScMRDRAdapter.load(root / "model/checkpoint.pt", device=device)
    destination.mkdir()
    audit = {"cohort": cohort, "limit": limit, "checkpoint_sha256": manifest["checkpoint_sha256"],
             "post_fit_only": True, "inputs": {}}
    for assay, path in _paths(input_dir).items():
        data = transform_gene_features(path, assay, gene, limit=limit)
        encoded = model.encode(data["counts"], assay, gene["feature_names"])
        with (destination / f"{assay}.npz").open("xb") as handle:
            np.savez_compressed(handle, cell_ids=data["cell_ids"], shared=encoded["shared"],
                                private=encoded["specific"])
        audit["inputs"][assay] = {"path": str(path.resolve()), "sha256": sha256(path),
                                  "retained_cells": len(data["cell_ids"]),
                                  "dropped_cell_ids": data["dropped_cell_ids"].tolist()}
    write_new(destination / "encoding_manifest.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("fit")
    training.add_argument("--train-dir", type=Path, required=True)
    training.add_argument("--validation-dir", type=Path)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--gene-transform-dir", type=Path)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    training.add_argument("--epochs", type=int, default=200)
    training.add_argument("--train-limit", type=int, default=2000)
    training.add_argument("--top-genes", type=int, default=2000)
    training.add_argument("--label")
    training.add_argument("--excluded-features", type=Path, help="JSON list; explicit background-only ablation")
    encoding = sub.add_parser("encode")
    encoding.add_argument("--run", type=Path, required=True)
    encoding.add_argument("--input-dir", type=Path, required=True)
    encoding.add_argument("--cohort", required=True)
    encoding.add_argument("--limit", type=int)
    encoding.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.command == "fit":
        result = fit_run(args.train_dir, args.out, validation_dir=args.validation_dir,
                         gene_transform_dir=args.gene_transform_dir, seed=args.seed,
                         device=args.device, max_epochs=args.epochs, train_limit=args.train_limit,
                         n_top_genes=args.top_genes, label=args.label,
                         excluded_features=json.loads(args.excluded_features.read_text()) if args.excluded_features else ())
    else:
        result = encode_run(args.run, args.input_dir, args.cohort, limit=args.limit, device=args.device)
    print(json.dumps({"status": result.get("status", "encoded"), "output": str(args.out if args.command == "fit" else args.run)}))


if __name__ == "__main__":
    main()
