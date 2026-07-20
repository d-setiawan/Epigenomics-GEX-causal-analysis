#!/usr/bin/env python3
"""Shared utilities for scGLUE integration scripts."""

from __future__ import annotations

import csv
import inspect
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def strip_10x_suffix(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"-\d+$", "", regex=True)


def pick_rna_h5(repo_root: Path, rna_h5: str | None) -> Path:
    if rna_h5:
        p = resolve_path(repo_root, rna_h5)
        if p is None or not p.exists():
            raise FileNotFoundError(f"RNA h5 not found: {p}")
        return p

    candidates = [
        repo_root / "Integration/outputs/workspace/data/rna/gex_feature_cell_matrix.h5",
        repo_root / "Data/raw/GexData/GEX_feature_cell_matrix_HDF5.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    gex_dir = repo_root / "Data/raw/GexData"
    for ext in ("*.h5", "*.hdf5"):
        found = sorted(gex_dir.glob(ext))
        if found:
            return found[0]
    raise FileNotFoundError("Could not find RNA h5 input. Provide --rna-h5 explicitly.")


def load_manifest_row(manifest_path: Path, mark: str) -> Dict[str, str]:
    df = pd.read_csv(manifest_path, sep="\t")
    if "mark" not in df.columns:
        raise ValueError(f"Manifest missing 'mark' column: {manifest_path}")
    hit = df.loc[df["mark"] == mark]
    if hit.empty:
        raise ValueError(f"Mark '{mark}' not found in manifest: {manifest_path}")
    return hit.iloc[0].to_dict()


def read_named_column(path: Path, preferred_col: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    if preferred_col in df.columns:
        return df[preferred_col].astype(str)
    return df.iloc[:, 0].astype(str)


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


def build_embedding_df(mat: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i+1}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=index, columns=cols)


def write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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


def load_rna_cell_annotations(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r:gz") as tf:
        member = "cell_types/cell_types.csv"
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Missing {member} in {path}")
        ann = pd.read_csv(BytesIO(f.read()))
    required = {"barcode", "coarse_cell_type", "fine_cell_type"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")
    keep_cols = ["barcode", "coarse_cell_type", "fine_cell_type"]
    if "cell_count_in_model" in ann.columns:
        keep_cols.append("cell_count_in_model")
    ann = ann[keep_cols].copy()
    ann["barcode"] = ann["barcode"].astype(str)
    ann["barcode_core"] = strip_10x_suffix(ann["barcode"])
    return ann.drop_duplicates(subset=["barcode_core"])


def normalize_label_level(level: str) -> str:
    aliases = {
        "coarse_cell_type": "coarse_cell_type",
        "fine_cell_type": "fine_cell_type",
        "harmonized_coarse": "harmonized_coarse",
        "harmonized_fine": "harmonized_fine",
    }
    if level not in aliases:
        raise ValueError(f"Unsupported label level: {level}")
    return aliases[level]


def load_harmonization_table(path: Path) -> pd.DataFrame:
    harm = pd.read_csv(path, sep="\t")
    required = {"coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine"}
    missing = required - set(harm.columns)
    if missing:
        raise ValueError(f"Harmonization table missing columns: {sorted(missing)}")
    dup = harm.duplicated(subset=["coarse_cell_type", "fine_cell_type"])
    if dup.any():
        raise ValueError("Harmonization table has duplicate coarse/fine label pairs")
    if "cell_ontology_id" not in harm.columns:
        harm["cell_ontology_id"] = pd.NA
    return harm


def apply_harmonization(ann: pd.DataFrame, harm: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    merged = ann.copy()
    if harm is None:
        merged["harmonized_coarse"] = merged["coarse_cell_type"]
        merged["harmonized_fine"] = merged["fine_cell_type"]
        merged["cell_ontology_id"] = pd.NA
        return merged, {"mode": "identity", "n_unmapped_rows": 0}

    merged = merged.merge(
        harm[["coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine", "cell_ontology_id"]],
        on=["coarse_cell_type", "fine_cell_type"],
        how="left",
    )
    unmapped = int(merged["harmonized_coarse"].isna().sum() + merged["harmonized_fine"].isna().sum())
    merged["harmonized_coarse"] = merged["harmonized_coarse"].fillna(merged["coarse_cell_type"])
    merged["harmonized_fine"] = merged["harmonized_fine"].fillna(merged["fine_cell_type"])
    return merged, {"mode": "table", "n_unmapped_rows": unmapped}


def majority_vote(labels: list[str]) -> tuple[str, float]:
    labels = [str(x) for x in labels if pd.notna(x) and str(x) != ""]
    if not labels:
        return "unknown", 0.0
    label, count = Counter(labels).most_common(1)[0]
    return label, float(count / len(labels))
