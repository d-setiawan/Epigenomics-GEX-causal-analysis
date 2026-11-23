#!/usr/bin/env python3
"""Shared utilities for scGLUE integration scripts."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad
    import networkx as nx

# Direct script entry points put only the method directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integration_common import (
    apply_harmonization,
    build_embedding_df,
    infer_repo_root,
    load_harmonization_table,
    load_manifest_row,
    load_rna_cell_annotations,
    majority_vote,
    normalize_label_level,
    pick_rna_h5,
    read_named_column,
    resolve_path,
    strip_10x_suffix,
    write_tsv,
)


def load_graph(nodes_tsv: Path, edges_tsv: Path) -> nx.MultiDiGraph:
    import networkx as nx

    g = nx.MultiDiGraph()

    if nodes_tsv.exists():
        nodes = pd.read_csv(nodes_tsv, sep="\t")
        if "node" not in nodes.columns:
            raise ValueError(f"nodes file missing `node` column: {nodes_tsv}")
        for row in nodes.itertuples(index=False):
            g.add_node(
                str(getattr(row, "node")),
                node_type=str(getattr(row, "node_type", "")),
                modality=str(getattr(row, "modality", "")),
                mark=str(getattr(row, "mark", "")),
            )

    if not edges_tsv.exists():
        raise FileNotFoundError(f"guidance edges file not found: {edges_tsv}")

    edges = pd.read_csv(edges_tsv, sep="\t")
    required = {"source", "target", "weight", "sign"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"edges file missing columns: {sorted(missing)}")

    n_skipped = 0
    for row in edges.itertuples(index=False):
        src = str(getattr(row, "source"))
        tgt = str(getattr(row, "target"))
        weight = float(getattr(row, "weight"))
        sign = int(getattr(row, "sign"))

        if not np.isfinite(weight) or weight <= 0:
            n_skipped += 1
            continue
        if weight > 1:
            weight = 1.0
        if sign not in (-1, 1):
            sign = 1 if sign >= 0 else -1

        g.add_edge(
            src,
            tgt,
            weight=float(weight),
            sign=int(sign),
            type=str(getattr(row, "type", "")),
            mark=str(getattr(row, "mark", "")),
            distance_bp=int(getattr(row, "distance_bp", 0)),
        )

    if g.number_of_nodes() == 0:
        raise RuntimeError("Loaded graph has zero nodes")
    if g.number_of_edges() == 0:
        raise RuntimeError("Loaded graph has zero edges")
    if n_skipped:
        print(f"[WARN] Skipped {n_skipped} invalid graph edges")

    return g


def subset_adata_to_graph(adata: ad.AnnData, graph_nodes: set[str]) -> tuple[ad.AnnData, int]:
    mask = adata.var_names.isin(graph_nodes)
    dropped = int((~mask).sum())
    return adata[:, mask].copy(), dropped


def drop_nonfinite_obs_by_rep(adata: ad.AnnData, rep_key: str) -> tuple[ad.AnnData, int]:
    rep = adata.obsm.get(rep_key)
    if rep is None:
        return adata, 0
    keep = np.isfinite(rep).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        adata = adata[keep].copy()
    return adata, dropped


def patch_scglue_scheduler_verbose_compat() -> bool:
    import scglue.models.plugins as plugins
    import torch

    sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau)
    if "verbose" in sig.parameters:
        return False

    base_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

    class _CompatReduceLROnPlateau(base_cls):
        def __init__(self, *args, verbose=None, **kwargs):
            super().__init__(*args, **kwargs)

    plugins.ReduceLROnPlateau = _CompatReduceLROnPlateau
    return True
