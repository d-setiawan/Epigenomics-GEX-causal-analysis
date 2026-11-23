#!/usr/bin/env python3
"""Train scGLUE on a pilot RNA + chromatin pair.

Expected upstream artifacts:
- pipelines/empirical/integration/runs/scglue_pilot_<MARK>/preprocess/{rna_preprocessed.h5ad, chrom_<MARK>_preprocessed.h5ad}
- pipelines/empirical/integration/runs/scglue_pilot_<MARK>/graph/{guidance_edges.tsv, guidance_nodes.tsv}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from _common import (
    build_embedding_df,
    infer_repo_root,
    load_graph,
    patch_scglue_scheduler_verbose_compat,
    resolve_path,
    subset_adata_to_graph,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Train scGLUE on pilot pair")
    p.add_argument("--mark", required=True)
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--preprocess-dir", default=None, help="Default: pipelines/empirical/integration/runs/scglue_pilot_<MARK>/preprocess")
    p.add_argument("--graph-dir", default=None, help="Default: pipelines/empirical/integration/runs/scglue_pilot_<MARK>/graph")
    p.add_argument("--out-dir", default=None, help="Default: pipelines/empirical/integration/runs/scglue_pilot_<MARK>/train")

    p.add_argument("--rna-key", default="rna")
    p.add_argument("--chrom-key", default="chromatin")
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
    p.add_argument("--max-epochs", type=int, default=100)
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
    preprocess_dir = resolve_path(repo_root, args.preprocess_dir) or repo_root / f"pipelines/empirical/integration/runs/scglue_pilot_{args.mark}/preprocess"
    graph_dir = resolve_path(repo_root, args.graph_dir) or repo_root / f"pipelines/empirical/integration/runs/scglue_pilot_{args.mark}/graph"
    out_dir = resolve_path(repo_root, args.out_dir) or repo_root / f"pipelines/empirical/integration/runs/scglue_pilot_{args.mark}/train"
    out_dir.mkdir(parents=True, exist_ok=True)

    rna_path = preprocess_dir / "rna_preprocessed.h5ad"
    chrom_path = preprocess_dir / f"chrom_{args.mark}_preprocessed.h5ad"
    edges_path = graph_dir / "guidance_edges.tsv"
    nodes_path = graph_dir / "guidance_nodes.tsv"

    if not rna_path.exists() or not chrom_path.exists():
        raise FileNotFoundError(f"Missing preprocessed files: {rna_path} or {chrom_path}")

    print(f"Loading RNA: {rna_path}")
    print(f"Loading chromatin: {chrom_path}")
    rna = ad.read_h5ad(rna_path)
    chrom = ad.read_h5ad(chrom_path)

    print(f"Loading graph from {edges_path}")
    graph = load_graph(nodes_path, edges_path)

    dropped = {"rna": 0, "chrom": 0}
    if not args.no_subset_to_graph:
        gnodes = set(map(str, graph.nodes))
        rna, dropped["rna"] = subset_adata_to_graph(rna, gnodes)
        chrom, dropped["chrom"] = subset_adata_to_graph(chrom, gnodes)

    use_hv = not args.no_use_highly_variable
    rna_use_hv = use_hv and ("highly_variable" in rna.var) and (int(rna.var["highly_variable"].sum()) > 0)
    chrom_use_hv = use_hv and ("highly_variable" in chrom.var) and (int(chrom.var["highly_variable"].sum()) > 0)

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

    scglue.models.configure_dataset(
        rna,
        "NB",
        use_highly_variable=rna_use_hv,
        use_layer=args.rna_layer,
        use_rep=args.rna_rep,
    )
    scglue.models.configure_dataset(
        chrom,
        "NB",
        use_highly_variable=chrom_use_hv,
        use_layer=args.chrom_layer,
        use_rep=args.chrom_rep,
    )

    adatas = {
        args.rna_key: rna,
        args.chrom_key: chrom,
    }

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

    print("Fitting scGLUE model...")
    glue = scglue.models.fit_SCGLUE(
        adatas,
        graph,
        init_kws=init_kws,
        compile_kws=compile_kws,
        fit_kws=fit_kws,
    )

    model_path = out_dir / "scglue_model.dill"
    glue.save(model_path)

    print("Encoding cells...")
    rna.obsm["X_glue"] = glue.encode_data(args.rna_key, rna, batch_size=args.encode_batch_size)
    chrom.obsm["X_glue"] = glue.encode_data(args.chrom_key, chrom, batch_size=args.encode_batch_size)

    rna_out = out_dir / "rna_with_glue.h5ad"
    chrom_out = out_dir / f"chrom_{args.mark}_with_glue.h5ad"
    rna.write_h5ad(rna_out, compression="gzip")
    chrom.write_h5ad(chrom_out, compression="gzip")

    rna_emb_df = build_embedding_df(rna.obsm["X_glue"], rna.obs_names, "GLUE")
    chrom_emb_df = build_embedding_df(chrom.obsm["X_glue"], chrom.obs_names, "GLUE")
    rna_emb_tsv = out_dir / "rna_glue_embeddings.tsv"
    chrom_emb_tsv = out_dir / f"chrom_{args.mark}_glue_embeddings.tsv"
    rna_emb_df.to_csv(rna_emb_tsv, sep="\t", index_label="cell")
    chrom_emb_df.to_csv(chrom_emb_tsv, sep="\t", index_label="cell")

    print("Encoding graph features...")
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
        print("Computing integration consistency...")
        consistency = scglue.models.integration_consistency(
            glue,
            adatas,
            graph,
            count_layers={args.rna_key: args.rna_layer, args.chrom_key: args.chrom_layer},
        )
        consistency_tsv = out_dir / "integration_consistency.tsv"
        consistency.to_csv(consistency_tsv, sep="\t", index=False)

    summary = {
        "mark": args.mark,
        "inputs": {
            "rna_h5ad": str(rna_path),
            "chrom_h5ad": str(chrom_path),
            "graph_edges_tsv": str(edges_path),
            "graph_nodes_tsv": str(nodes_path),
        },
        "graph": {
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
        },
        "data": {
            "rna_cells": int(rna.n_obs),
            "rna_features": int(rna.n_vars),
            "chrom_cells": int(chrom.n_obs),
            "chrom_features": int(chrom.n_vars),
            "dropped_features_to_match_graph": dropped,
            "rna_use_highly_variable": bool(rna_use_hv),
            "chrom_use_highly_variable": bool(chrom_use_hv),
        },
        "params": vars(args),
        "runtime_compat": {
            "patched_scglue_lr_scheduler_verbose": bool(patched_scheduler),
        },
        "outputs": {
            "model_dill": str(model_path),
            "rna_with_glue_h5ad": str(rna_out),
            "chrom_with_glue_h5ad": str(chrom_out),
            "rna_glue_embeddings_tsv": str(rna_emb_tsv),
            "chrom_glue_embeddings_tsv": str(chrom_emb_tsv),
            "feature_glue_embeddings_tsv": str(feat_tsv),
            "consistency_tsv": str(consistency_tsv) if consistency_tsv else None,
            "checkpoint_dir": str(train_dir),
        },
    }

    summary_path = out_dir / "train_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {rna_out}")
    print(f"Wrote: {chrom_out}")
    print(f"Wrote: {feat_tsv}")
    if consistency_tsv:
        print(f"Wrote: {consistency_tsv}")
    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
