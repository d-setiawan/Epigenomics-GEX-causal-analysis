"""Train-only annotation-based gene alignment for paper-faithful scMRDR.

This module has no simulator or truth imports. Annotation channels supply a
permitted feature correspondence, not causal edges or cell correspondences.
All-zero measurements are distinct from unavailable genes in the mask.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GeneConfig:
    annotation_key: str = "gene_id"
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat_v3"
    feature_selection: str = "union_per_assay_hvg"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _counts(data):
    matrix = sparse.csr_matrix(data.X, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("Gene alignment requires finite nonnegative raw counts")
    if np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError("Gene alignment requires integer counts, not normalized X")
    return matrix


def annotation_matrix(feature_names, gene_ids, gene_order):
    """Sparse deterministic sum mapping; missing genes contribute no value."""
    if len(feature_names) != len(set(map(str, feature_names))):
        raise ValueError("Input feature IDs must be unique")
    if len(feature_names) != len(gene_ids):
        raise ValueError("Annotation length does not match input features")
    if len(gene_order) != len(set(map(str, gene_order))):
        raise ValueError("Gene order must be unique")
    index = {str(g): i for i, g in enumerate(gene_order)}
    pairs = [(i, index[str(g)]) for i, g in enumerate(gene_ids) if str(g) in index]
    row, col = zip(*pairs) if pairs else ([], [])
    return sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), (row, col)),
                             shape=(len(feature_names), len(gene_order)))


def fit_gene_features(train_paths: Mapping[str, str | Path], out_dir: str | Path,
                      config: GeneConfig = GeneConfig(),
                      exclude_features: tuple[str, ...] = (),
                      train_limit: int | None = None) -> dict:
    """Fit native HVGs separately, then freeze the three-modality gene union.

    `train_limit` uses an already randomized cached prefix, never truth-based
    selection. Exclusion is an explicit ablation applied before every sum/HVG.
    """
    import anndata as ad
    import scanpy as sc

    root = Path(out_dir)
    if root.exists():
        raise FileExistsError(f"Refusing to replace gene transform: {root}")
    if config.feature_selection != "union_per_assay_hvg" or config.n_top_genes < 1:
        raise ValueError("Unsupported or empty feature selection")
    excluded = set(exclude_features)
    by_assay, selected = {}, set()
    for assay, path in sorted(train_paths.items()):
        data = ad.read_h5ad(path)
        if not data.obs_names.is_unique:
            raise ValueError("Cell identifiers must be unique")
        if train_limit is not None:
            data = data[:train_limit].copy()
        if config.annotation_key not in data.var:
            raise ValueError(f"Missing permitted annotation {config.annotation_key}")
        names = data.var_names.astype(str).tolist()
        annotations = data.var[config.annotation_key].astype(str).tolist()
        if any(g in {"", "nan", "None"} for g in annotations):
            raise ValueError("Missing gene annotations need explicit unavailable handling")
        used = [i for i, name in enumerate(names) if name not in excluded]
        genes = sorted({annotations[i] for i in used})
        mapping = annotation_matrix([names[i] for i in used],
                                    [annotations[i] for i in used], genes)
        raw = _counts(data)[:, used] @ mapping
        positive_genes = np.asarray(raw.sum(axis=0)).ravel() > 0
        positive_cells = np.asarray(raw.sum(axis=1)).ravel() > 0
        if positive_cells.sum() < 3 or positive_genes.sum() < 2:
            raise ValueError(f"Insufficient nonzero gene counts in {assay}")
        hvg_data = ad.AnnData(raw[positive_cells][:, positive_genes].copy())
        hvg_data.var_names = np.asarray(genes)[positive_genes]
        k = min(config.n_top_genes, hvg_data.n_vars)
        if k == hvg_data.n_vars:
            assay_selected = hvg_data.var_names.astype(str).tolist()
        else:
            # No favorable fallback if the stated HVG estimator is singular.
            sc.pp.highly_variable_genes(hvg_data, n_top_genes=k,
                                        flavor=config.hvg_flavor, subset=False)
            assay_selected = hvg_data.var_names[hvg_data.var.highly_variable].astype(str).tolist()
        selected.update(assay_selected)
        by_assay[assay] = {
            "source_path": str(Path(path).resolve()), "source_sha256": _sha(Path(path)),
            "input_features": names, "input_gene_annotations": annotations,
            "used_feature_indices": used, "available_genes": genes,
            "selected_hvg_genes": assay_selected,
            "train_cell_ids": data.obs_names.astype(str).tolist(),
            "raw_zero_gene_libraries": int((~positive_cells).sum()),
        }
    manifest = {"schema_version": 1, "config": asdict(config), "train_limit": train_limit,
                "excluded_features": sorted(excluded), "assays": by_assay,
                "feature_names": sorted(selected),
                "source_sha256": _sha(Path(__file__)),
                "rule": "raw annotation-only sums; train-only assay HVG union; availability is measured, not nonzero"}
    for assay, entry in by_assay.items():
        available = set(entry["available_genes"])
        entry["availability"] = [g in available for g in manifest["feature_names"]]
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def transform_gene_features(input_path: str | Path, assay: str, manifest: dict,
                             limit: int | None = None) -> dict:
    """Apply saved feature/annotation order, without refitting on new cells."""
    import anndata as ad

    data = ad.read_h5ad(input_path)
    if limit is not None:
        data = data[:limit].copy()
    info = manifest["assays"][assay]
    names = data.var_names.astype(str).tolist()
    if names != info["input_features"]:
        raise ValueError(f"Feature order differs from fitted {assay} manifest")
    key = manifest["config"]["annotation_key"]
    annotations = data.var[key].astype(str).tolist()
    if annotations != info["input_gene_annotations"]:
        raise ValueError("Gene annotations differ from the frozen training mapping")
    used = info["used_feature_indices"]
    mapping = annotation_matrix([names[i] for i in used], [annotations[i] for i in used],
                                manifest["feature_names"])
    matrix = sparse.csr_matrix(_counts(data)[:, used] @ mapping)
    library = np.asarray(matrix.sum(axis=1)).ravel()
    keep = library > 0
    if not data.obs_names.is_unique:
        raise ValueError("Cell identifiers must be unique")
    return {"counts": matrix[keep], "availability": np.asarray(info["availability"], dtype=bool),
            "cell_ids": data.obs_names[keep].to_numpy(dtype=str),
            "dropped_cell_ids": data.obs_names[~keep].to_numpy(dtype=str),
            "feature_names": manifest["feature_names"],
            "observed_library": library[keep]}
