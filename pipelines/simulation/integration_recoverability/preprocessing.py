"""Training-only scGLUE preprocessing with persisted out-of-sample maps.

Native scGLUE LSI is reproduced exactly through TF-IDF, L1 normalization,
log1p(1e4 * X), randomized SVD's U and per-row sample standardization.  Its
right singular vectors and singular values are retained for held-out U=XVD^-1.
There is no sklearn-LSI or HVG-flavor fallback.  No function reads simulator
truth; optional feature exclusions are explicit, role-free input configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

try:
    from .calibration import ASSAYS, canonical_hash, file_sha256
except ImportError:
    from calibration import ASSAYS, canonical_hash, file_sha256


@dataclass
class PreprocessingConfig:
    seed: int = 0
    train_cells_per_assay: int | None = 2000
    discovery_cells_per_assay: int | None = 1000
    rna_min_genes: int = 200
    rna_min_cells: int = 3
    rna_top_genes: int = 3000
    rna_hvg_flavor: str = "seurat_v3"
    rna_target_sum: float = 10000.0
    rna_scale_max: float = 10.0
    n_components: int = 50
    chrom_encoder_features: int = 30000
    chrom_reconstruction_features: int = 15000
    chrom_lsi_iter: int = 15
    excluded_feature_ids: dict[str, list[str]] = field(default_factory=dict)
    guidance_excluded_feature_ids: list[str] = field(default_factory=list)
    guidance_mode: str = "normal"


def _configuration(value: PreprocessingConfig | Mapping[str, Any] | None) -> PreprocessingConfig:
    config = value if isinstance(value, PreprocessingConfig) else PreprocessingConfig(**(dict(value) if value else {}))
    if config.rna_hvg_flavor != "seurat_v3":
        raise ValueError("This pinned native adapter requires seurat_v3; no silent HVG fallback")
    if config.n_components < 2 or config.rna_min_cells < 1 or config.rna_min_genes < 0:
        raise ValueError("Invalid preprocessing dimensions or QC thresholds")
    if config.guidance_mode not in {"normal", "shuffled"}:
        raise ValueError("Guidance mode must be normal or explicitly shuffled")
    return config


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in (
        "numpy", "scipy", "anndata", "scanpy", "scglue", "scikit-learn", "scikit-misc")}


def _read_counts(path: str | Path, assay: str, excluded: list[str],
                  row_limit: int | None = None):
    import anndata as ad
    import pandas as pd

    original = ad.read_h5ad(path)
    if not original.obs_names.is_unique or not original.var_names.is_unique:
        raise ValueError(f"{assay}: nonunique cell or feature IDs")
    source_n_cells = original.n_obs
    if row_limit is not None:
        if row_limit < 1:
            raise ValueError("Cell limit must be positive")
        original = original[:min(row_limit, original.n_obs)].copy()
    present = original.var_names.isin(excluded)
    audit = {"source": str(Path(path).resolve()), "source_sha256": file_sha256(path),
             "source_cells": int(source_n_cells), "selected_prefix_cells": int(original.n_obs),
             "cell_selection": "fixed input-order prefix before QC",
             "source_features": int(original.n_vars), "excluded_present": original.var_names[present].tolist(),
             "excluded_absent": sorted(set(excluded) - set(original.var_names))}
    original = original[:, ~present]
    matrix = sparse.csr_matrix(original.X, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0) or np.any(matrix.data != np.floor(matrix.data)):
        raise ValueError(f"{assay}: X must contain original integer counts")
    if "gene_id" not in original.var:
        raise ValueError("Allowed annotation gene_id is required for guidance")
    # Rebuild from a strict public whitelist.  In particular, pre-exclusion
    # library metadata must not disclose removed focal counts in background-only.
    annotations = [x for x in ("gene_id", "channel_id", "chrom", "chromStart", "chromEnd", "strand") if x in original.var]
    obs = pd.DataFrame({"assay": assay}, index=original.obs_names.copy())
    data = ad.AnnData(X=matrix, obs=obs, var=original.var[annotations].copy())
    return data, audit


def _drop_rows(data, keep: np.ndarray, audit: dict[str, Any], reason: str):
    keep = np.asarray(keep, bool)
    audit.setdefault("dropped_cell_ids", {})[reason] = data.obs_names[~keep].tolist()
    return data[keep].copy()


def _top_feature_indices(sums: np.ndarray, n_top: int) -> np.ndarray:
    """Native empirical top-total selection, preserving original feature order."""
    n_top = min(n_top, len(sums))
    if n_top < 1:
        raise ValueError("No features to select")
    if n_top == len(sums):
        return np.arange(len(sums))
    return np.sort(np.argpartition(sums, -n_top)[-n_top:])


def _native_tfidf(counts: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0) or not np.isfinite(idf).all():
        raise ValueError("Native LSI needs nonzero rows and finite training IDF")
    tfidf = counts.multiply(1.0 / totals[:, None]).multiply(idf[None, :]).tocsr()
    transformed = normalize(tfidf, norm="l1", axis=1).tocsr()
    transformed.data = np.log1p(transformed.data * 1e4)
    return transformed


def _row_standardize(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result -= result.mean(axis=1, keepdims=True)
    denominator = result.std(axis=1, ddof=1, keepdims=True)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("Degenerate native LSI row; no imputation or fallback")
    return result / denominator


def fit_native_lsi(counts: sparse.csr_matrix, n_components: int, *,
                    seed: int = 0, n_iter: int = 15) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Native equations and backend, with factors exposed for held-out data."""
    import scglue.data
    import scglue.num

    counts = sparse.csr_matrix(counts, dtype=np.float64)
    sums = np.asarray(counts.sum(axis=0)).ravel()
    if np.any(sums <= 0):
        raise ValueError("Zero-total training columns must be explicitly filtered before LSI")
    if n_components >= min(counts.shape):
        raise ValueError("Too few cells/features for frozen requested LSI dimension")
    idf = counts.shape[0] / sums
    transformed = _native_tfidf(counts, idf)
    # Audit the first transformation against the actual installed implementation.
    native = normalize(scglue.num.tfidf(counts), norm="l1")
    native = np.log1p(native * 1e4)
    difference = sparse.csr_matrix(native - transformed)
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
        raise RuntimeError("Installed scGLUE TF-IDF differs from pinned adapter equations")
    u, singular, vt = randomized_svd(transformed, n_components=n_components,
                                      n_iter=n_iter, random_state=seed)
    if np.any(singular <= np.finfo(float).eps * singular[0]):
        raise ValueError("Rank-deficient LSI cannot be extended out of sample")
    representation = _row_standardize(u)
    projected_u = np.asarray(transformed @ vt.T) / singular[None, :]
    projected_rep = _row_standardize(projected_u)
    audit = {"backend": "native_scglue_equations_sklearn_randomized_svd_U",
             "n_iter": n_iter, "seed": seed, "row_std_ddof": 1,
             "training_u_projection_relative_error": float(np.linalg.norm(projected_u - u) / np.linalg.norm(u)),
             "training_rep_projection_rmse": float(np.sqrt(np.mean((projected_rep - representation) ** 2))),
             "finite_svd_caveat": "Randomized SVD is approximate; report training U versus XVD^-1 discrepancy rather than claiming exact numerical equality.",
             "native_lsi_source_sha256": canonical_hash(inspect.getsource(scglue.data.lsi)),
             "native_tfidf_source_sha256": canonical_hash(inspect.getsource(scglue.num.tfidf))}
    return representation, {"idf": idf, "singular_values": singular, "right_singular_vectors": vt.T}, audit


def transform_native_lsi(counts: sparse.csr_matrix, parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    transformed = _native_tfidf(sparse.csr_matrix(counts, dtype=float), parameters["idf"])
    values = np.asarray(transformed @ parameters["right_singular_vectors"]) / parameters["singular_values"][None, :]
    return _row_standardize(values)


def _raw_output(data, counts: sparse.csr_matrix, representation: np.ndarray, rep_key: str):
    import anndata as ad

    output = ad.AnnData(X=sparse.csr_matrix(counts, dtype=np.float32), obs=data.obs.copy(), var=data.var.copy())
    output.layers["counts"] = sparse.csr_matrix(counts, dtype=np.float32)
    output.var["highly_variable"] = True
    output.obsm[rep_key] = np.asarray(representation, dtype=np.float32)
    output.obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    output.obs["n_features"] = np.asarray((counts > 0).sum(axis=1)).ravel()
    return output


def _fit_rna(data, config: PreprocessingConfig, audit: dict[str, Any]):
    import scanpy as sc

    data = _drop_rows(data, np.asarray((data.X > 0).sum(axis=1)).ravel() >= config.rna_min_genes,
                       audit, "rna_min_genes")
    detected = np.asarray((data.X > 0).sum(axis=0)).ravel()
    keep = detected >= config.rna_min_cells
    audit["dropped_feature_ids"] = data.var_names[~keep].tolist()
    data = data[:, keep].copy()
    data = _drop_rows(data, np.asarray(data.X.sum(axis=1)).ravel() > 0, audit, "zero_normalization_library")
    normalization_features = data.var_names.tolist()
    if data.n_obs <= config.n_components or data.n_vars <= config.n_components:
        raise ValueError("Too few RNA cells/features for frozen requested PCA dimension")
    data.layers["counts"] = data.X.copy()
    sc.pp.highly_variable_genes(data, n_top_genes=min(config.rna_top_genes, data.n_vars),
                                flavor="seurat_v3", layer="counts")
    hvg = np.asarray(data.var["highly_variable"], bool)
    if hvg.sum() <= config.n_components:
        raise ValueError("Too few selected RNA HVGs for requested PCA dimension")
    sc.pp.normalize_total(data, target_sum=config.rna_target_sum)
    sc.pp.log1p(data)
    # Scaling each column is independent; scale only HVGs to avoid a needless
    # dense full-gene matrix, while normalization still used every retained gene.
    scaled = data[:, hvg].copy()
    sc.pp.scale(scaled, max_value=config.rna_scale_max)
    pca_center = np.asarray(scaled.X.mean(axis=0)).ravel()
    sc.tl.pca(scaled, n_comps=config.n_components, use_highly_variable=False,
               svd_solver="arpack", random_state=config.seed)
    parameters = {"scale_mean": scaled.var["mean"].to_numpy(float),
                   "scale_std": scaled.var["std"].to_numpy(float),
                   "pca_center": pca_center, "pca_loadings": np.asarray(scaled.varm["PCs"])}
    metadata = {"assay": "RNA", "normalization_feature_ids": normalization_features,
                 "encoder_feature_ids": scaled.var_names.tolist(),
                 "reconstruction_feature_ids": scaled.var_names.tolist(),
                 "rep_key": "X_pca", "normalization": "train-retained-gene total normalize, log1p, training mean/std, symmetric clipping, frozen PCA",
                 "target_sum": config.rna_target_sum, "scale_max": config.rna_scale_max,
                 "scale_ddof": 1, "pca_solver": "arpack", "hvg_flavor": "seurat_v3",
                 "pca_variance_ratio": scaled.uns["pca"]["variance_ratio"].tolist(),
                 "audit": audit}
    counts = sparse.csr_matrix(scaled.layers["counts"])
    keep = np.asarray(counts.sum(axis=1)).ravel() > 0
    audit.setdefault("dropped_cell_ids", {})["zero_reconstruction_library"] = scaled.obs_names[~keep].tolist()
    output = _raw_output(scaled[keep], counts[keep], scaled.obsm["X_pca"][keep], "X_pca")
    return output, parameters, metadata


def _fit_chrom(data, assay: str, config: PreprocessingConfig, audit: dict[str, Any]):
    data = _drop_rows(data, np.asarray(data.X.sum(axis=1)).ravel() > 0, audit, "zero_input_library")
    sums = np.asarray(data.X.sum(axis=0)).ravel()
    positive = sums > 0
    audit["zero_training_feature_ids"] = data.var_names[~positive].tolist()
    data = data[:, positive].copy()
    sums = sums[positive]
    encoder_indices = _top_feature_indices(sums, config.chrom_encoder_features)
    audit["encoder_dropped_feature_ids"] = data.var_names[np.setdiff1d(np.arange(data.n_vars), encoder_indices)].tolist()
    data = data[:, encoder_indices].copy()
    data = _drop_rows(data, np.asarray(data.X.sum(axis=1)).ravel() > 0, audit, "zero_encoder_library")
    # Dropped zero rows cannot introduce new zero columns.
    representation, parameters, lsi_audit = fit_native_lsi(data.X, config.n_components,
                                                          seed=config.seed, n_iter=config.chrom_lsi_iter)
    sums = np.asarray(data.X.sum(axis=0)).ravel()
    reconstruction_indices = _top_feature_indices(sums, config.chrom_reconstruction_features)
    reconstruction = data[:, reconstruction_indices].copy()
    keep = np.asarray(reconstruction.X.sum(axis=1)).ravel() > 0
    audit.setdefault("dropped_cell_ids", {})["zero_reconstruction_library"] = reconstruction.obs_names[~keep].tolist()
    metadata = {"assay": assay, "normalization_feature_ids": data.var_names.tolist(),
                 "encoder_feature_ids": data.var_names.tolist(),
                 "reconstruction_feature_ids": reconstruction.var_names.tolist(),
                 "rep_key": "X_lsi", "lsi": lsi_audit, "audit": audit,
                 "selection": "positive training columns, native top count-total encoder cap, separate native count-total reconstruction cap"}
    output = _raw_output(reconstruction[keep], reconstruction.X[keep], representation[keep], "X_lsi")
    return output, parameters, metadata


def _align(data, feature_ids: list[str], label: str):
    missing = sorted(set(feature_ids) - set(data.var_names))
    if missing:
        raise ValueError(f"{label}: missing frozen input features ({len(missing)}), first: {missing[:3]}")
    return data[:, feature_ids].copy()


def _transform_assay(data, assay: str, metadata: dict[str, Any], parameters: Mapping[str, np.ndarray],
                       config: PreprocessingConfig, audit: dict[str, Any]):
    if assay == "RNA":
        data = _drop_rows(data, np.asarray((data.X > 0).sum(axis=1)).ravel() >= config.rna_min_genes,
                           audit, "rna_min_genes")
    data = _align(data, metadata["normalization_feature_ids"], f"{assay} normalization")
    totals = np.asarray(data.X.sum(axis=1)).ravel()
    data = _drop_rows(data, totals > 0, audit, "zero_normalization_library")
    totals = totals[totals > 0]
    if data.n_obs == 0:
        raise ValueError(f"{assay}: no cells remain after frozen-input QC")
    if assay == "RNA":
        normalized = data.X.multiply(config.rna_target_sum / totals[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
        indices = data.var_names.get_indexer(metadata["encoder_feature_ids"])
        scaled = normalized[:, indices].toarray()
        scaled = (scaled - parameters["scale_mean"][None, :]) / parameters["scale_std"][None, :]
        scaled = np.clip(scaled, -config.rna_scale_max, config.rna_scale_max)
        representation = (scaled - parameters["pca_center"][None, :]) @ parameters["pca_loadings"]
    else:
        representation = transform_native_lsi(data.X, parameters)
    reconstruction = _align(data, metadata["reconstruction_feature_ids"], f"{assay} reconstruction")
    keep = np.asarray(reconstruction.X.sum(axis=1)).ravel() > 0
    audit.setdefault("dropped_cell_ids", {})["zero_reconstruction_library"] = reconstruction.obs_names[~keep].tolist()
    if not keep.any():
        raise ValueError(f"{assay}: no cells have a reconstruction count")
    return _raw_output(reconstruction[keep], reconstruction.X[keep], representation[keep], metadata["rep_key"])


def _save_h5ad_and_rep(data, h5ad_path: Path, rep_path: Path, rep_key: str) -> dict[str, Any]:
    h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(h5ad_path, compression="gzip")
    np.savez_compressed(rep_path, cell_ids=data.obs_names.to_numpy(str), shared=np.asarray(data.obsm[rep_key]))
    return {"h5ad": str(h5ad_path.resolve()), "h5ad_sha256": file_sha256(h5ad_path),
            "representation": str(rep_path.resolve()), "representation_sha256": file_sha256(rep_path),
            "n_cells": data.n_obs, "n_features": data.n_vars,
            "feature_ids": data.var_names.tolist(), "cell_ids": data.obs_names.tolist(), "rep_key": rep_key}


def _guidance(outputs: Mapping[str, Any], path: Path, config: PreprocessingConfig) -> dict[str, Any]:
    import networkx as nx

    graph = nx.DiGraph()
    excluded = set(config.guidance_excluded_feature_ids)
    gene_to_rna = {}
    for feature_id, row in outputs["RNA"].var.iterrows():
        if feature_id not in excluded:
            gene_to_rna.setdefault(str(row["gene_id"]), []).append(str(feature_id))
    vertices = []
    for assay in ASSAYS:
        for feature_id in outputs[assay].var_names:
            vertices.append(str(feature_id))
            graph.add_node(str(feature_id), assay=assay)
            graph.add_edge(str(feature_id), str(feature_id), weight=1.0, sign=1)
    for assay in ASSAYS[1:]:
        for feature_id, row in outputs[assay].var.iterrows():
            if feature_id in excluded:
                continue
            for rna_id in gene_to_rna.get(str(row["gene_id"]), []):
                graph.add_edge(rna_id, str(feature_id), weight=1.0, sign=1)
                graph.add_edge(str(feature_id), rna_id, weight=1.0, sign=1)
    if config.guidance_mode == "shuffled":
        # Permute RNA labels only among nonexcluded vertices.  This preserves
        # the RNA degree distribution (not each named gene's degree), mark
        # degrees, assay partitions, weights/signs and all self-loops.
        rna_ids = sorted(x for x in outputs["RNA"].var_names if x not in excluded)
        permutation = np.random.default_rng(config.seed).permutation(rna_ids)
        graph = nx.relabel_nodes(graph, dict(zip(rna_ids, permutation)), copy=True)
    nx.write_graphml(graph, path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "vertices": vertices,
            "n_nodes": graph.number_of_nodes(), "n_directed_edges": graph.number_of_edges(),
            "construction": "RNA-to-mark equal gene_id annotation, symmetric weight=1 sign=1, self-loops; no simulator graph accessed",
            "mode": config.guidance_mode, "excluded_cross_feature_ids": sorted(excluded),
            "shuffle_rule": "RNA label permutation preserving assay partitions, degree distributions, mark degrees, edge weights/signs and self-loops" if config.guidance_mode == "shuffled" else None}


def _validate_saved_outputs(manifest: Mapping[str, Any]) -> None:
    for entry in manifest["assays"].values():
        for key in ("h5ad", "representation"):
            if file_sha256(entry[key]) != entry[key + "_sha256"]:
                raise ValueError(f"Cached preprocessing artifact changed: {entry[key]}")
        if "transform_manifest" in entry:
            if file_sha256(entry["transform_manifest"]) != entry["transform_manifest_sha256"]:
                raise ValueError("Cached transform manifest changed")
            metadata = json.loads(Path(entry["transform_manifest"]).read_text())
            if file_sha256(metadata["parameter_path"]) != metadata["parameter_sha256"]:
                raise ValueError("Cached transform parameters changed")
    if "guidance" in manifest and file_sha256(manifest["guidance"]["path"]) != manifest["guidance"]["sha256"]:
        raise ValueError("Cached guidance graph changed")


def fit_preprocessing(train_by_assay: Mapping[str, str | Path], out_dir: str | Path,
                        config: PreprocessingConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit on the fixed initial training prefix only; save all transform state."""
    config = _configuration(config)
    root = Path(out_dir)
    inputs = {a: {"path": str(Path(train_by_assay[a]).resolve()), "sha256": file_sha256(train_by_assay[a])} for a in ASSAYS}
    fingerprint = canonical_hash({"inputs": inputs, "config": asdict(config),
                                   "source_sha256": file_sha256(__file__), "versions": package_versions()})
    if root.exists():
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if existing.get("fingerprint") == fingerprint:
                _validate_saved_outputs(existing)
                return existing
        raise FileExistsError("Existing preprocessing differs or is incomplete; do not silently overwrite")
    root.mkdir(parents=True)
    (root / "transforms").mkdir()
    manifest = {"status": "running", "fingerprint": fingerprint, "config": asdict(config),
                "inputs": inputs, "versions": package_versions(), "assays": {},
                "information_boundary": "Training counts and allowed annotations only; no hidden truth loaded"}
    outputs = {}
    for assay in ASSAYS:
        data, audit = _read_counts(train_by_assay[assay], assay, config.excluded_feature_ids.get(assay, []), config.train_cells_per_assay)
        print(f"Fit preprocessing {assay}: {data.n_obs} cells, {data.n_vars} allowed features", flush=True)
        output, parameters, metadata = (_fit_rna(data, config, audit) if assay == "RNA" else _fit_chrom(data, assay, config, audit))
        array_path = root / "transforms" / f"{assay}.npz"
        np.savez_compressed(array_path, **parameters)
        metadata.update(parameter_path=str(array_path.resolve()), parameter_sha256=file_sha256(array_path))
        metadata_path = root / "transforms" / f"{assay}.json"
        with metadata_path.open("x") as handle:
            json.dump(metadata, handle, sort_keys=True, indent=2, allow_nan=False)
        entries = _save_h5ad_and_rep(output, root / "train" / f"{assay}.h5ad", root / "representations/train" / f"{assay}.npz", metadata["rep_key"])
        entries.update(transform_manifest=str(metadata_path.resolve()), transform_manifest_sha256=file_sha256(metadata_path),
                        encoder_features=len(metadata["encoder_feature_ids"]), normalization_features=len(metadata["normalization_feature_ids"]))
        manifest["assays"][assay] = entries
        outputs[assay] = output
    manifest["guidance"] = _guidance(outputs, root / "guidance.graphml.gz", config)
    manifest["status"] = "complete"
    with (root / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
    return manifest


def transform_preprocessing(input_h5ad_by_assay: Mapping[str, str | Path], fitted_dir: str | Path,
                              out_dir: str | Path | None = None, cohort: str = "discovery") -> dict[str, Any]:
    root = Path(fitted_dir)
    fitted_manifest_path = root / "manifest.json"
    fitted = json.loads(fitted_manifest_path.read_text())
    if fitted["status"] != "complete":
        raise ValueError("Cannot transform using incomplete preprocessing")
    if package_versions() != fitted["versions"]:
        raise ValueError("Preprocessing package versions differ from frozen fit")
    config = _configuration(fitted["config"])
    output_root = root if out_dir is None else Path(out_dir)
    cohort_dir = output_root / "cohorts" / cohort
    inputs = {a: {"path": str(Path(input_h5ad_by_assay[a]).resolve()), "sha256": file_sha256(input_h5ad_by_assay[a])} for a in ASSAYS}
    fingerprint = canonical_hash({"inputs": inputs, "fit_manifest_sha256": file_sha256(fitted_manifest_path),
                                   "cohort": cohort, "source_sha256": file_sha256(__file__)})
    if cohort_dir.exists():
        saved = cohort_dir / "manifest.json"
        if saved.exists() and json.loads(saved.read_text()).get("fingerprint") == fingerprint:
            existing = json.loads(saved.read_text())
            _validate_saved_outputs(existing)
            return existing
        raise FileExistsError("Existing transformed cohort differs or is incomplete")
    cohort_dir.mkdir(parents=True)
    manifest = {"cohort": cohort, "status": "running", "fingerprint": fingerprint, "inputs": inputs,
                "fit_manifest_sha256": file_sha256(fitted_manifest_path), "assays": {}}
    for assay in ASSAYS:
        entry = fitted["assays"][assay]
        metadata_path = Path(entry["transform_manifest"])
        if file_sha256(metadata_path) != entry["transform_manifest_sha256"]:
            raise ValueError("Frozen transform metadata changed")
        metadata = json.loads(metadata_path.read_text())
        array_path = Path(metadata["parameter_path"])
        if file_sha256(array_path) != metadata["parameter_sha256"]:
            raise ValueError("Frozen numerical transformation changed")
        with np.load(array_path, allow_pickle=False) as archive:
            parameters = {key: archive[key] for key in archive.files}
        limit = config.discovery_cells_per_assay if cohort == "discovery" else None
        data, audit = _read_counts(input_h5ad_by_assay[assay], assay, config.excluded_feature_ids.get(assay, []), limit)
        output = _transform_assay(data, assay, metadata, parameters, config, audit)
        manifest["assays"][assay] = _save_h5ad_and_rep(output, cohort_dir / f"{assay}.h5ad",
                                                       output_root / "representations" / cohort / f"{assay}.npz", metadata["rep_key"])
        manifest["assays"][assay]["audit"] = audit
    manifest["status"] = "complete"
    with (cohort_dir / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--input-dir", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--config", type=Path)
    transform = commands.add_parser("transform")
    transform.add_argument("--input-dir", type=Path, required=True)
    transform.add_argument("--fit-dir", type=Path, required=True)
    transform.add_argument("--out", type=Path)
    transform.add_argument("--cohort", required=True)
    args = parser.parse_args()
    paths = {a: args.input_dir / f"{a}.h5ad" for a in ASSAYS}
    if args.command == "fit":
        config = json.loads(args.config.read_text()) if args.config else None
        result = fit_preprocessing(paths, args.out, config)
    else:
        result = transform_preprocessing(paths, args.fit_dir, args.out, args.cohort)
    print(json.dumps({"status": result["status"], "fingerprint": result["fingerprint"]}))


if __name__ == "__main__":
    main()
