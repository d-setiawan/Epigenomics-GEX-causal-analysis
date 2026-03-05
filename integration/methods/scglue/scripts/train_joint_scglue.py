#!/usr/bin/env python3
"""Train one joint scGLUE model across RNA + all histone marks."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def load_graph(nodes_tsv: Path, edges_tsv: Path) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    if nodes_tsv.exists():
        nodes = pd.read_csv(nodes_tsv, sep="\t")
        for row in nodes.itertuples(index=False):
            g.add_node(
                str(getattr(row, "node")),
                node_type=str(getattr(row, "node_type", "")),
                modality=str(getattr(row, "modality", "")),
                mark=str(getattr(row, "mark", "")),
            )
    edges = pd.read_csv(edges_tsv, sep="\t")
    for row in edges.itertuples(index=False):
        w = float(getattr(row, "weight"))
        if not np.isfinite(w) or w <= 0:
            continue
        s = int(getattr(row, "sign"))
        s = 1 if s >= 0 else -1
        g.add_edge(
            str(getattr(row, "source")),
            str(getattr(row, "target")),
            weight=min(w, 1.0),
            sign=s,
            type=str(getattr(row, "type", "")),
            mark=str(getattr(row, "mark", "")),
            distance_bp=int(getattr(row, "distance_bp", 0)),
        )
    if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
        raise RuntimeError(f"Invalid graph loaded from {nodes_tsv} / {edges_tsv}")
    return g


def subset_adata_to_graph(adata: ad.AnnData, graph_nodes: set[str]) -> Tuple[ad.AnnData, int]:
    mask = adata.var_names.isin(graph_nodes)
    dropped = int((~mask).sum())
    return adata[:, mask].copy(), dropped


def drop_nonfinite_obs_by_rep(adata: ad.AnnData, rep_key: str) -> Tuple[ad.AnnData, int]:
    rep = adata.obsm.get(rep_key)
    if rep is None:
        return adata, 0
    keep = np.isfinite(rep).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        adata = adata[keep].copy()
    return adata, dropped


def build_embedding_df(mat: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i+1}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=index, columns=cols)


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def patch_scglue_scheduler_verbose_compat() -> bool:
    import torch
    import scglue.models.plugins as plugins

    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau)
    if "verbose" in sig.parameters:
        return False

    base_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

    class _CompatReduceLROnPlateau(base_cls):
        def __init__(self, *args, verbose=None, **kwargs):
            super().__init__(*args, **kwargs)

    plugins.ReduceLROnPlateau = _CompatReduceLROnPlateau
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Train joint scGLUE model")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--preprocess-dir", required=True, help="Joint preprocess directory")
    p.add_argument("--graph-dir", required=True, help="Joint guidance graph directory")
    p.add_argument("--out-dir", default=None, help="Default: <preprocess-parent>/train")

    p.add_argument("--rna-key", default="rna")
    p.add_argument("--rna-layer", default="counts")
    p.add_argument("--chrom-layer", default="counts")
    p.add_argument("--rna-rep", default="X_pca")
    p.add_argument("--chrom-rep", default="X_lsi")
    p.add_argument("--no-use-highly-variable", action="store_true")
    p.add_argument("--no-subset-to-graph", action="store_true")

    p.add_argument("--latent-dim", type=int, default=30)
    p.add_argument("--h-depth", type=int, default=2)
    p.add_argument("--h-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--lam-data", type=float, default=1.0)
    p.add_argument("--lam-kl", type=float, default=1.0)
    p.add_argument("--lam-graph", type=float, default=0.02)
    p.add_argument("--lam-align", type=float, default=0.05)
    p.add_argument("--lam-sup", type=float, default=0.02)
    p.add_argument("--learning-rate", type=float, default=2e-3)

    p.add_argument("--neg-samples", type=int, default=10)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--data-batch-size", type=int, default=128)
    p.add_argument("--graph-batch-size", type=int, default=8192)
    p.add_argument("--align-burnin", type=int, default=20)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--reduce-lr-patience", type=int, default=4)
    p.add_argument("--wait-n-lrs", type=int, default=1)

    p.add_argument("--cpu-only", action="store_true", default=False)
    p.add_argument("--no-cpu-only", action="store_false", dest="cpu_only")
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--compute-consistency", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    preprocess_dir = resolve_path(repo_root, args.preprocess_dir)
    graph_dir = resolve_path(repo_root, args.graph_dir)
    out_dir = resolve_path(repo_root, args.out_dir) if args.out_dir else preprocess_dir.parent / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    modalities_dir = out_dir / "modalities"
    modalities_dir.mkdir(parents=True, exist_ok=True)

    rna_path = preprocess_dir / "rna_preprocessed.h5ad"
    chrom_manifest_path = preprocess_dir / "chrom_preprocessed_manifest.tsv"
    edges_path = graph_dir / "guidance_edges.tsv"
    nodes_path = graph_dir / "guidance_nodes.tsv"
    if not all(p.exists() for p in [rna_path, chrom_manifest_path, edges_path, nodes_path]):
        raise FileNotFoundError("Missing required preprocess/graph inputs for joint training")

    rna = ad.read_h5ad(rna_path)
    chrom_manifest = pd.read_csv(chrom_manifest_path, sep="\t")
    adatas: Dict[str, ad.AnnData] = {args.rna_key: rna}
    key_to_mark: Dict[str, str] = {args.rna_key: "RNA"}

    for row in chrom_manifest.to_dict(orient="records"):
        mark = str(row["mark"])
        key = f"chrom_{mark}"
        cpath = Path(str(row["chrom_h5ad"]))
        if not cpath.is_absolute():
            cpath = (repo_root / cpath).resolve()
        adatas[key] = ad.read_h5ad(cpath)
        key_to_mark[key] = mark

    graph = load_graph(nodes_path, edges_path)
    dropped: Dict[str, int] = {}
    if not args.no_subset_to_graph:
        gnodes = set(map(str, graph.nodes))
        for k in list(adatas.keys()):
            adatas[k], dropped[k] = subset_adata_to_graph(adatas[k], gnodes)
    else:
        dropped = {k: 0 for k in adatas}

    dropped_nonfinite_rep_rows: Dict[str, int] = {}
    for k in list(adatas.keys()):
        rep_key = args.rna_rep if k == args.rna_key else args.chrom_rep
        adatas[k], dropped_nonfinite_rep_rows[k] = drop_nonfinite_obs_by_rep(adatas[k], rep_key)
        if adatas[k].n_obs < 2:
            raise RuntimeError(
                f"Too few cells in modality '{k}' after dropping non-finite rows from {rep_key}. "
                "Re-run preprocessing to remove zero-count / non-finite rows."
            )
        if dropped_nonfinite_rep_rows[k] > 0:
            print(
                f"[WARN] Dropped {dropped_nonfinite_rep_rows[k]} cells in {k} "
                f"due to non-finite values in {rep_key}."
            )

    import scglue

    patched_scheduler = patch_scglue_scheduler_verbose_compat()
    if patched_scheduler:
        print("Applied compatibility patch: scGLUE LR scheduler verbose argument")

    if args.cpu_only:
        scglue.config.CPU_ONLY = True
        print("Runtime: CPU-only mode enabled")
    else:
        scglue.config.CPU_ONLY = False
        try:
            import torch

            print(f"Runtime: GPU-preferred mode (torch.cuda.is_available={torch.cuda.is_available()})")
        except Exception:
            print("Runtime: GPU-preferred mode")

    use_hv_global = not args.no_use_highly_variable
    for k, adata in adatas.items():
        use_hv = (
            use_hv_global
            and ("highly_variable" in adata.var)
            and (int(adata.var["highly_variable"].sum()) > 0)
        )
        if k == args.rna_key:
            scglue.models.configure_dataset(
                adata,
                "NB",
                use_highly_variable=use_hv,
                use_layer=args.rna_layer,
                use_rep=args.rna_rep,
            )
        else:
            scglue.models.configure_dataset(
                adata,
                "NB",
                use_highly_variable=use_hv,
                use_layer=args.chrom_layer,
                use_rep=args.chrom_rep,
            )

    train_dir = out_dir / "checkpoints"
    train_dir.mkdir(parents=True, exist_ok=True)

    init_kws = {
        "latent_dim": args.latent_dim,
        "h_depth": args.h_depth,
        "h_dim": args.h_dim,
        "dropout": args.dropout,
        "shared_batches": False,
        "random_seed": args.random_seed,
    }
    compile_kws = {
        "lam_data": args.lam_data,
        "lam_kl": args.lam_kl,
        "lam_graph": args.lam_graph,
        "lam_align": args.lam_align,
        "lam_sup": args.lam_sup,
        "lr": args.learning_rate,
    }
    fit_kws = {
        "directory": str(train_dir),
        "neg_samples": args.neg_samples,
        "val_split": args.val_split,
        "data_batch_size": args.data_batch_size,
        "graph_batch_size": args.graph_batch_size,
        "align_burnin": args.align_burnin,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "reduce_lr_patience": args.reduce_lr_patience,
        "wait_n_lrs": args.wait_n_lrs,
    }

    print("Fitting joint scGLUE model...")
    glue = scglue.models.fit_SCGLUE(
        adatas,
        graph,
        init_kws=init_kws,
        compile_kws=compile_kws,
        fit_kws=fit_kws,
    )

    model_path = out_dir / "scglue_joint_model.dill"
    glue.save(model_path)

    embedding_rows = []
    modality_rows = []
    all_emb_frames = []
    for key, adata in adatas.items():
        adata.obsm["X_glue"] = glue.encode_data(key, adata, batch_size=args.encode_batch_size)
        mark = key_to_mark[key]

        emb_df = build_embedding_df(adata.obsm["X_glue"], adata.obs_names, "GLUE")
        emb_tsv = modalities_dir / f"{key}_glue_embeddings.tsv"
        emb_df.to_csv(emb_tsv, sep="\t", index_label="cell")

        out_h5ad = modalities_dir / f"{key}_with_glue.h5ad"
        adata.write_h5ad(out_h5ad, compression="gzip")

        frame = emb_df.copy()
        frame.insert(0, "cell", frame.index.astype(str))
        frame.insert(1, "modality_key", key)
        frame.insert(2, "mark", mark)
        all_emb_frames.append(frame.reset_index(drop=True))

        modality_rows.append(
            {
                "modality_key": key,
                "mark": mark,
                "h5ad": str(out_h5ad),
                "embeddings_tsv": str(emb_tsv),
                "n_cells": int(adata.n_obs),
                "n_features": int(adata.n_vars),
                "dropped_features_to_match_graph": int(dropped.get(key, 0)),
                "dropped_nonfinite_rep_rows": int(dropped_nonfinite_rep_rows.get(key, 0)),
            }
        )

    all_cells_df = pd.concat(all_emb_frames, axis=0, ignore_index=True)
    all_cells_tsv = out_dir / "all_cells_glue_embeddings.tsv"
    all_cells_df.to_csv(all_cells_tsv, sep="\t", index=False)

    modality_manifest_tsv = out_dir / "modality_outputs.tsv"
    write_tsv(
        modality_manifest_tsv,
        modality_rows,
        [
            "modality_key",
            "mark",
            "h5ad",
            "embeddings_tsv",
            "n_cells",
            "n_features",
            "dropped_features_to_match_graph",
            "dropped_nonfinite_rep_rows",
        ],
    )

    feat_emb = glue.encode_graph(graph)
    feat_df = build_embedding_df(feat_emb, glue.vertices, "GLUE")
    node_attrs = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
    node_attrs.index.name = "feature"
    feat_df.index.name = "feature"
    feat_out_df = node_attrs.join(feat_df, how="right")
    feat_tsv = out_dir / "feature_glue_embeddings.tsv"
    feat_out_df.to_csv(feat_tsv, sep="\t")

    consistency_tsv = None
    if args.compute_consistency:
        consistency_rows = []
        for key, mark in key_to_mark.items():
            if key == args.rna_key:
                continue
            pair_adatas = {args.rna_key: adatas[args.rna_key], key: adatas[key]}
            pair_nodes = set(map(str, adatas[args.rna_key].var_names)) | set(map(str, adatas[key].var_names))
            pair_graph = graph.subgraph(pair_nodes).copy()
            if pair_graph.number_of_edges() == 0:
                continue
            c = scglue.models.integration_consistency(
                glue,
                pair_adatas,
                pair_graph,
                count_layers={args.rna_key: args.rna_layer, key: args.chrom_layer},
            )
            c = c.copy()
            c.insert(0, "modality_key", key)
            c.insert(1, "mark", mark)
            consistency_rows.append(c)
        if consistency_rows:
            consistency_df = pd.concat(consistency_rows, axis=0, ignore_index=True)
            consistency_tsv = out_dir / "integration_consistency_pairwise.tsv"
            consistency_df.to_csv(consistency_tsv, sep="\t", index=False)

    summary = {
        "run_type": "joint",
        "inputs": {
            "preprocess_dir": str(preprocess_dir),
            "graph_dir": str(graph_dir),
            "graph_edges_tsv": str(edges_path),
            "graph_nodes_tsv": str(nodes_path),
        },
        "graph": {
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
        },
        "modalities": modality_rows,
        "params": vars(args),
        "runtime_compat": {
            "patched_scglue_lr_scheduler_verbose": bool(patched_scheduler),
        },
        "outputs": {
            "model_dill": str(model_path),
            "all_cells_glue_embeddings_tsv": str(all_cells_tsv),
            "modality_outputs_tsv": str(modality_manifest_tsv),
            "feature_glue_embeddings_tsv": str(feat_tsv),
            "consistency_tsv": str(consistency_tsv) if consistency_tsv else None,
            "checkpoint_dir": str(train_dir),
        },
    }
    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {all_cells_tsv}")
    print(f"Wrote: {modality_manifest_tsv}")
    print(f"Wrote: {feat_tsv}")
    if consistency_tsv:
        print(f"Wrote: {consistency_tsv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
