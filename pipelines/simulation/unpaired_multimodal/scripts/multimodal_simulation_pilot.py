#!/usr/bin/env python3
r"""Closed-loop pilot simulation for unpaired RNA + two CUT&Tag experiments.

This is the smallest useful end-to-end version of the multimodal simulation:

1. Draw independent cells for RNA, H3K27ac, and H3K4me3 from a known biological
   state distribution.
2. Generate a full counterfactual molecular state for each cell, including
   planted direct, shared-driver, and null regulatory motifs.
3. Observe modality-specific sparse counts.
4. Compare an oracle biological-state embedding with a learned scGLUE embedding.
5. Transfer RNA labels, retain predicted monocytes, construct complete
   cross-modality metacells, and aggregate the focal locus.
6. Run the repository's production FCI implementation and score its edges
   against the planted truth.

The simulation intentionally separates the independent experimental unit
(``--seed``, a newly drawn biological world) from scGLUE and metacell seeds,
which are repeated measurements of the same world.

Run with the pinned ``scrna_r`` environment::

    NUMBA_CACHE_DIR=/tmp/epigenomics_numba_cache \
    MPLCONFIGDIR=/tmp/epigenomics_mpl_cache \
    XDG_CACHE_HOME=/tmp/epigenomics_xdg_cache \
    /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
      pipelines/simulation/unpaired_multimodal/scripts/multimodal_simulation_pilot.py --integration both

For a quick installation/smoke benchmark::

    /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
      pipelines/simulation/unpaired_multimodal/scripts/multimodal_simulation_pilot.py --smoke --integration both
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# scanpy/scGLUE/numba try to populate user caches during import.  Defaults under
# /tmp keep the script runnable in restricted and batch environments while still
# allowing callers to override them.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/epigenomics_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/epigenomics_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/epigenomics_xdg_cache")

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[4]
MODALITIES = ("rna", "chrom_H3K27ac", "chrom_H3K4me3")
MARKS = {"rna": "RNA", "chrom_H3K27ac": "H3K27ac", "chrom_H3K4me3": "H3K4me3"}
TARGET_LABEL = "monocyte"
N_FOCAL_PER_MARK = 6


@dataclass(frozen=True)
class PilotConfig:
    seed: int
    integration_seed: int
    metacell_seed: int
    n_cells_per_modality: int
    n_features: int
    n_metacells: int
    latent_dim: int
    h_dim: int
    max_epochs: int
    chrom_detection: float
    population_shift: float
    modality_batch_scale: float
    guidance_mode: str
    label_k: int
    min_label_confidence: float
    alpha: float
    max_depth: int
    integration: str
    smoke: bool


@dataclass
class SyntheticWorld:
    adatas: dict[str, ad.AnnData]
    true_z: dict[str, np.ndarray]
    true_labels: dict[str, np.ndarray]
    true_technical_depth: dict[str, np.ndarray]
    counterfactual_focal: dict[str, dict[str, np.ndarray]]
    observation_parameters: dict[str, pd.DataFrame]
    graph: nx.MultiDiGraph
    generative_dag: nx.DiGraph
    gene_name: str
    focal_features: dict[str, list[str]]
    truth_direct: set[tuple[str, str]]
    truth_shared_driver: set[tuple[str, str]]
    feature_semantics: list[dict[str, str]]


@dataclass
class MetacellTables:
    observed: pd.DataFrame
    ground_truth: pd.DataFrame
    membership: pd.DataFrame
    metrics: dict[str, float | int]


def canonical_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def draw_states(
    n: int,
    *,
    target_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a discrete cell type plus a continuous three-dimensional state."""
    is_target = rng.random(n) < target_fraction
    labels = np.where(is_target, TARGET_LABEL, "other")
    z = rng.normal(size=(n, 3))
    # The non-target population is separated but overlapping, so label transfer
    # is measurable rather than trivial.
    z[~is_target] += np.array([2.8, -0.9, 0.8])
    # A weak activation shift within monocytes creates continuous structure.
    z[is_target, 1] += 0.35 * z[is_target, 0]
    return z, labels


def focal_activities(z: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Generate counterfactual focal-locus activities for every latent cell.

    H3K27ac peak 0 is state-dominated and directly affects the gene; H3K4me3
    peak 0 has much larger cell-specific innovation and also directly affects
    the gene.  This creates a first recoverability contrast between a slow/state
    effect and a fast/cell-specific effect after cells are sampled independently.
    """
    n = z.shape[0]
    e = lambda scale=1.0: rng.normal(scale=scale, size=n)

    ac_slow_cause = 1.25 * z[:, 1] + e(0.25)
    ac_shared_only = 1.25 * z[:, 1] + e(0.25)
    ac_child = 0.85 * ac_slow_cause + 0.25 * z[:, 2] + e(0.35)
    ac_state_only = 0.75 * z[:, 0] + e(0.55)
    ac_null_1, ac_null_2 = e(), e()

    me3_fast_cause = 0.20 * z[:, 2] + e(1.00)
    me3_child = 0.85 * me3_fast_cause + e(0.35)
    me3_shared_a = 1.15 * z[:, 2] + e(0.30)
    me3_shared_b = 1.15 * z[:, 2] + e(0.30)
    me3_null_1, me3_null_2 = e(), e()

    gene = (
        0.90 * ac_slow_cause
        + 0.90 * me3_fast_cause
        + 0.35 * z[:, 0]
        + e(0.65)
    )
    return {
        "gene": gene,
        "H3K27ac": np.column_stack(
            [ac_slow_cause, ac_shared_only, ac_child, ac_state_only, ac_null_1, ac_null_2]
        ),
        "H3K4me3": np.column_stack(
            [me3_fast_cause, me3_child, me3_shared_a, me3_shared_b, me3_null_1, me3_null_2]
        ),
    }


def negative_binomial_counts(
    activity: np.ndarray,
    *,
    rng: np.random.Generator,
    mean_library: float = 2500.0,
    dispersion: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RNA activity into overdispersed counts with variable depth."""
    n, p = activity.shape
    baseline = rng.normal(scale=0.35, size=p)
    eta = np.clip(0.45 * activity + baseline, -8.0, 8.0)
    eta -= eta.max(axis=1, keepdims=True)
    probs = np.exp(eta)
    probs /= probs.sum(axis=1, keepdims=True)
    library = mean_library * np.exp(rng.normal(scale=0.40, size=n))
    mu = library[:, None] * probs
    nb_prob = dispersion / (dispersion + mu)
    counts = rng.negative_binomial(dispersion, np.clip(nb_prob, 1e-8, 1 - 1e-8))
    return counts.astype(np.int32, copy=False), library, baseline


def calibrate_detection_intercept(
    activity: np.ndarray,
    depth: np.ndarray,
    target: float,
    *,
    beta: float = 0.85,
) -> float:
    lo, hi = -20.0, 8.0
    signal = np.clip(beta * activity, -7.0, 7.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        rate = np.mean(1.0 - np.exp(-depth * np.exp(signal + mid)))
        if rate < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sparse_chromatin_counts(
    activity: np.ndarray,
    *,
    rng: np.random.Generator,
    median_detection: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate near-Bernoulli CUT&Tag fragment observations."""
    n, p = activity.shape
    depth = np.exp(rng.normal(scale=0.66, size=n))
    targets = np.clip(
        median_detection * np.exp(rng.normal(scale=0.60, size=p)),
        0.01,
        0.35,
    )
    counts = np.zeros((n, p), dtype=np.int8)
    intercepts = np.zeros(p, dtype=float)
    for j in range(p):
        intercept = calibrate_detection_intercept(activity[:, j], depth, float(targets[j]))
        intercepts[j] = intercept
        lam = depth * np.exp(np.clip(0.85 * activity[:, j], -7.0, 7.0) + intercept)
        counts[:, j] = (rng.poisson(lam) > 0).astype(np.int8)
    # LSI cannot represent an all-zero cell.  At pilot scale, force one random
    # background detection in the rare zero rows and record realism separately.
    zero_rows = np.flatnonzero(counts.sum(axis=1) == 0)
    for i in zero_rows:
        counts[i, int(rng.integers(N_FOCAL_PER_MARK, p))] = 1
    return counts, depth, targets, intercepts


def prepare_rna_rep(adata: ad.AnnData, *, n_components: int, seed: int) -> None:
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=float)
    lib = np.asarray(counts.sum(axis=1)).ravel()
    lib[lib <= 0] = 1.0
    norm = counts.multiply((1e4 / lib)[:, None]).toarray()
    x = np.log1p(norm)
    x = StandardScaler().fit_transform(x)
    n_comp = max(2, min(n_components, x.shape[0] - 1, x.shape[1] - 1))
    adata.obsm["X_pca"] = PCA(n_components=n_comp, random_state=seed).fit_transform(x)


def prepare_chrom_rep(adata: ad.AnnData, *, n_components: int, seed: int) -> None:
    counts = sparse.csr_matrix(adata.layers["counts"], dtype=float)
    row_sum = np.asarray(counts.sum(axis=1)).ravel()
    row_sum[row_sum <= 0] = 1.0
    tf = counts.multiply((1.0 / row_sum)[:, None])
    doc_freq = np.asarray((counts > 0).sum(axis=0)).ravel()
    doc_freq[doc_freq <= 0] = 1.0
    tfidf = tf.multiply(np.log1p(counts.shape[0] / doc_freq))
    n_comp = max(2, min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    adata.obsm["X_lsi"] = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(tfidf)


def make_feature_names(n_features: int) -> tuple[list[str], dict[str, list[str]]]:
    genes = ["SIMG"] + [f"GENE{j:04d}" for j in range(1, n_features)]
    focal: dict[str, list[str]] = {}
    for mark_i, mark in enumerate(("H3K27ac", "H3K4me3")):
        names = []
        for j in range(n_features):
            start = 1_000_000 + mark_i * 2_000_000 + j * 1_000
            names.append(f"chr1:{start}-{start + 499}__{mark}")
        focal[mark] = names[:N_FOCAL_PER_MARK]
    return genes, focal


def build_guidance_graph(
    genes: list[str],
    chrom_names: dict[str, list[str]],
    *,
    guidance_mode: str,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for gene in genes:
        graph.add_node(gene, node_type="gene", modality="rna", mark="")
    for mark, names in chrom_names.items():
        for name in names:
            graph.add_node(name, node_type="chromatin", modality="chromatin", mark=mark)

    # Background modules are the integration signal.  The focal locus is kept
    # separate so target-included versus target-held-out guidance is testable.
    for j in range(N_FOCAL_PER_MARK, len(genes)):
        for mark in ("H3K27ac", "H3K4me3"):
            peak = chrom_names[mark][j]
            graph.add_edge(genes[j], peak, weight=1.0, sign=1)
            graph.add_edge(peak, genes[j], weight=1.0, sign=1)

    if guidance_mode == "perfect":
        targets = [chrom_names["H3K27ac"][0], chrom_names["H3K4me3"][0]]
    elif guidance_mode == "holdout":
        targets = []
    elif guidance_mode == "shuffled":
        targets = [chrom_names["H3K27ac"][4], chrom_names["H3K4me3"][4]]
    else:
        raise ValueError(f"Unknown guidance mode: {guidance_mode}")
    for peak in targets:
        graph.add_edge(genes[0], peak, weight=1.0, sign=1)
        graph.add_edge(peak, genes[0], weight=1.0, sign=1)

    # scGLUE's feature graph representation expects self loops.
    for node in list(graph.nodes):
        graph.add_edge(node, node, weight=1.0, sign=1)
    return graph


def build_generative_dag(
    genes: list[str],
    chrom_names: dict[str, list[str]],
    modality_loadings: dict[str, np.ndarray],
    observation_parameters: dict[str, pd.DataFrame],
) -> nx.DiGraph:
    """Build the seed-specific structural DAG behind the simulated assays.

    Nodes represent scalar variables except for the RNA softmax normalizer, which
    is an explicit deterministic vector-valued measurement node. Independent
    innovation terms are recorded as node attributes rather than separate roots;
    this keeps the graph readable while preserving every non-noise parent.
    """
    dag = nx.DiGraph(
        schema_version="1.0",
        scope="counterfactual_single_cell_and_observation",
        independent_modality_sampling=True,
        independent_innovations_implicit=True,
        rna_feature_baseline_distribution="Normal(0, 0.35)",
        chrom_feature_detection_target_distribution="clipped_lognormal_around_configured_median",
    )

    def add_node(
        node: str,
        *,
        layer: str,
        variable_type: str,
        modality: str = "",
        observability: str = "latent",
        matrix_column: str = "",
        matrix_transform: str = "",
        description: str,
        has_independent_noise: bool = False,
        structural_equation: str = "",
        noise_distribution: str = "",
        noise_scale: float | str = "",
        observation_intercept: float | str = "",
        target_detection: float | str = "",
        dispersion: float | str = "",
    ) -> None:
        dag.add_node(
            node,
            layer=layer,
            variable_type=variable_type,
            modality=modality,
            observability=observability,
            matrix_column=matrix_column,
            matrix_transform=matrix_transform,
            description=description,
            has_independent_noise=has_independent_noise,
            structural_equation=structural_equation,
            noise_distribution=noise_distribution,
            noise_scale=noise_scale,
            observation_intercept=observation_intercept,
            target_detection=target_detection,
            dispersion=dispersion,
        )

    def add_edge(
        source: str,
        target: str,
        *,
        mechanism: str,
        coefficient: float | str = "",
    ) -> None:
        dag.add_edge(source, target, mechanism=mechanism, coefficient=coefficient)

    add_node(
        "latent__cell_type",
        layer="biological_context",
        variable_type="categorical",
        observability="oracle_only",
        description="True monocyte versus other-cell identity.",
        structural_equation="Bernoulli(target_fraction_by_modality)",
        noise_distribution="bernoulli",
    )
    for dim in range(3):
        add_node(
            f"latent__z{dim}",
            layer="biological_state",
            variable_type="continuous",
            observability="oracle_only",
            description=f"True continuous biological-state coordinate z{dim}.",
            has_independent_noise=True,
            structural_equation="cell_type_shift + Normal(0, 1)",
            noise_distribution="normal",
            noise_scale=1.0,
        )
        add_edge(
            "latent__cell_type",
            f"latent__z{dim}",
            mechanism="cell_type_specific_location_shift",
        )
    add_edge(
        "latent__z0",
        "latent__z1",
        mechanism="monocyte_specific_activation_coupling",
        coefficient="0.35_if_monocyte",
    )

    for modality in MODALITIES:
        mark = MARKS[modality]
        add_node(
            f"technical__depth__{modality}",
            layer="technical_nuisance",
            variable_type="continuous",
            modality=mark,
            observability="oracle_only",
            description="True lognormal sampling-depth factor used by the observation model.",
            has_independent_noise=True,
            structural_equation=(
                "2500 * exp(Normal(0, 0.40))"
                if modality == "rna"
                else "exp(Normal(0, 0.66))"
            ),
            noise_distribution="lognormal",
            noise_scale=0.40 if modality == "rna" else 0.66,
        )

    activity_nodes: dict[str, list[str]] = {m: [] for m in MODALITIES}
    count_nodes: dict[str, list[str]] = {m: [] for m in MODALITIES}
    rna_parameters = observation_parameters["rna"].set_index("feature")

    for feature_i, gene in enumerate(genes):
        activity = f"activity__rna__{gene}"
        count = f"count__rna__{gene}"
        activity_nodes["rna"].append(activity)
        count_nodes["rna"].append(count)
        add_node(
            activity,
            layer="molecular_activity",
            variable_type="continuous",
            modality="RNA",
            description=f"Latent RNA activity for {gene} before count sampling.",
            has_independent_noise=True,
            structural_equation=(
                "focal_regulatory_equation" if feature_i == 0 else "z @ seed_specific_loadings + epsilon"
            ),
            noise_distribution="normal",
            noise_scale=0.65 if feature_i == 0 else 0.70,
        )
        add_node(
            count,
            layer="observed_measurement",
            variable_type="count",
            modality="RNA",
            observability="raw_assay_observed",
            matrix_column="expr__SIMG_log1p" if feature_i == 0 else "",
            matrix_transform="metacell_pseudobulk_log1p_cpm" if feature_i == 0 else "",
            description=f"Observed negative-binomial RNA count for {gene}.",
            structural_equation="NegativeBinomial(mean=true_depth * softmax(0.45 * activity + baseline))",
            noise_distribution="negative_binomial",
            observation_intercept=float(rna_parameters.loc[gene, "observation_intercept"]),
            dispersion=float(rna_parameters.loc[gene, "dispersion"]),
        )
        if feature_i > 0:
            for dim in range(3):
                add_edge(
                    f"latent__z{dim}",
                    activity,
                    mechanism="linear_background_loading",
                    coefficient=float(modality_loadings["rna"][feature_i, dim]),
                )

    for modality in MODALITIES[1:]:
        mark = MARKS[modality]
        mark_parameters = observation_parameters[modality].set_index("feature")
        for feature_i, feature in enumerate(chrom_names[mark]):
            activity = f"activity__{mark}__{feature}"
            count = f"count__{mark}__{feature}"
            activity_nodes[modality].append(activity)
            count_nodes[modality].append(count)
            add_node(
                activity,
                layer="molecular_activity",
                variable_type="continuous",
                modality=mark,
                description=f"Latent {mark} activity for {feature} before detection sampling.",
                has_independent_noise=True,
                structural_equation=(
                    "focal_regulatory_equation"
                    if feature_i < N_FOCAL_PER_MARK
                    else "z @ seed_specific_loadings + epsilon"
                ),
                noise_distribution="normal",
                noise_scale=0.70 if feature_i >= N_FOCAL_PER_MARK else "see_focal_noise_scale",
            )
            add_node(
                count,
                layer="observed_measurement",
                variable_type="binary_count",
                modality=mark,
                observability="raw_assay_observed",
                matrix_column=feature if feature_i < N_FOCAL_PER_MARK else "",
                matrix_transform=(
                    "metacell_weighted_mean_per_cell_log1p_cpm"
                    if feature_i < N_FOCAL_PER_MARK
                    else ""
                ),
                description=f"Observed sparse {mark} fragment detection for {feature}.",
                structural_equation="indicator(Poisson(true_depth * exp(0.85 * activity + intercept)) > 0)",
                noise_distribution="poisson_threshold",
                observation_intercept=float(
                    mark_parameters.loc[feature, "observation_intercept"]
                ),
                target_detection=float(mark_parameters.loc[feature, "target_detection"]),
            )
            if feature_i >= N_FOCAL_PER_MARK:
                for dim in range(3):
                    add_edge(
                        f"latent__z{dim}",
                        activity,
                        mechanism="linear_background_loading",
                        coefficient=float(modality_loadings[modality][feature_i, dim]),
                    )

    ac = activity_nodes["chrom_H3K27ac"]
    me3 = activity_nodes["chrom_H3K4me3"]
    gene_activity = activity_nodes["rna"][0]
    focal_edges = [
        ("latent__z1", ac[0], "linear_state_effect", 1.25),
        ("latent__z1", ac[1], "linear_shared_driver", 1.25),
        (ac[0], ac[2], "direct_same_mark_effect", 0.85),
        ("latent__z2", ac[2], "linear_state_effect", 0.25),
        ("latent__z0", ac[3], "linear_state_effect", 0.75),
        ("latent__z2", me3[0], "weak_linear_state_effect", 0.20),
        (me3[0], me3[1], "direct_same_mark_effect", 0.85),
        ("latent__z2", me3[2], "linear_shared_driver", 1.15),
        ("latent__z2", me3[3], "linear_shared_driver", 1.15),
        (ac[0], gene_activity, "direct_cross_modal_gene_effect", 0.90),
        (me3[0], gene_activity, "direct_cross_modal_gene_effect", 0.90),
        ("latent__z0", gene_activity, "direct_state_effect", 0.35),
    ]
    for source, target, mechanism, coefficient in focal_edges:
        add_edge(source, target, mechanism=mechanism, coefficient=coefficient)

    focal_noise_scales = {
        ac[0]: 0.25,
        ac[1]: 0.25,
        ac[2]: 0.35,
        ac[3]: 0.55,
        ac[4]: 1.00,
        ac[5]: 1.00,
        me3[0]: 1.00,
        me3[1]: 0.35,
        me3[2]: 0.30,
        me3[3]: 0.30,
        me3[4]: 1.00,
        me3[5]: 1.00,
    }
    for activity, scale in focal_noise_scales.items():
        dag.nodes[activity]["noise_scale"] = scale

    add_node(
        "measurement__rna_softmax_normalizer",
        layer="observation_model",
        variable_type="deterministic_vector",
        modality="RNA",
        description="Softmax normalization coupling the RNA feature probabilities.",
        structural_equation="softmax denominator across all RNA logits",
    )
    for activity, count in zip(activity_nodes["rna"], count_nodes["rna"], strict=True):
        add_edge(
            activity,
            "measurement__rna_softmax_normalizer",
            mechanism="softmax_logit",
            coefficient=0.45,
        )
        add_edge(
            activity,
            count,
            mechanism="feature_specific_negative_binomial_mean",
            coefficient=0.45,
        )
        add_edge(
            "measurement__rna_softmax_normalizer",
            count,
            mechanism="compositional_probability_normalization",
        )
        add_edge(
            "technical__depth__rna",
            count,
            mechanism="negative_binomial_mean_scaling",
        )

    for modality in MODALITIES[1:]:
        for activity, count in zip(activity_nodes[modality], count_nodes[modality], strict=True):
            add_edge(activity, count, mechanism="bernoulli_detection_intensity", coefficient=0.85)
            add_edge(
                f"technical__depth__{modality}",
                count,
                mechanism="poisson_intensity_scaling",
            )

    for modality in MODALITIES:
        mark = MARKS[modality]
        lib_col = "libsize__rna" if modality == "rna" else f"libsize__{mark}"
        library_node = f"observed__{lib_col}"
        add_node(
            library_node,
            layer="observed_covariate",
            variable_type="derived_continuous",
            modality=mark,
            observability="realistically_observed",
            matrix_column=lib_col,
            matrix_transform="metacell_mean_observed_library_size",
            description=(
                "Observed library-size proxy derived from counts; the discovery pipeline log-transforms "
                "and constrains it as a root even though it is downstream of the measurements."
            ),
            structural_equation="sum of observed feature counts, then metacell mean",
        )
        for count in count_nodes[modality]:
            add_edge(count, library_node, mechanism="observed_count_sum")

    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("Internal error: the simulated generative graph is not a DAG")
    return dag


def generate_world(cfg: PilotConfig) -> SyntheticWorld:
    if cfg.n_features <= N_FOCAL_PER_MARK + 4:
        raise ValueError(f"--n-features must exceed {N_FOCAL_PER_MARK + 4}")
    genes, _ = make_feature_names(cfg.n_features)
    chrom_names = {
        mark: [
            f"chr1:{1_000_000 + i * 2_000_000 + j * 1_000}-"
            f"{1_000_000 + i * 2_000_000 + j * 1_000 + 499}__{mark}"
            for j in range(cfg.n_features)
        ]
        for i, mark in enumerate(("H3K27ac", "H3K4me3"))
    }
    focal = {mark: names[:N_FOCAL_PER_MARK] for mark, names in chrom_names.items()}

    master_rng = np.random.default_rng(cfg.seed)
    base_loadings = master_rng.normal(scale=0.75, size=(cfg.n_features, 3))
    adatas: dict[str, ad.AnnData] = {}
    true_z: dict[str, np.ndarray] = {}
    true_labels: dict[str, np.ndarray] = {}
    true_technical_depth: dict[str, np.ndarray] = {}
    counterfactual_focal: dict[str, dict[str, np.ndarray]] = {}
    observation_parameters: dict[str, pd.DataFrame] = {}
    modality_loadings_by_assay: dict[str, np.ndarray] = {}

    target_fractions = {
        "rna": 0.70,
        "chrom_H3K27ac": float(np.clip(0.70 + cfg.population_shift, 0.10, 0.95)),
        "chrom_H3K4me3": float(np.clip(0.70 - cfg.population_shift, 0.10, 0.95)),
    }
    seed_seq = np.random.SeedSequence(cfg.seed)
    modality_seeds = seed_seq.spawn(len(MODALITIES))

    for modality, child_seed in zip(MODALITIES, modality_seeds, strict=True):
        rng = np.random.default_rng(child_seed)
        z, labels = draw_states(
            cfg.n_cells_per_modality,
            target_fraction=target_fractions[modality],
            rng=rng,
        )
        full = focal_activities(z, rng)
        counterfactual_focal[modality] = full
        modality_loadings = base_loadings + rng.normal(
            scale=cfg.modality_batch_scale, size=base_loadings.shape
        )
        modality_loadings_by_assay[modality] = modality_loadings
        activity = z @ modality_loadings.T + rng.normal(
            scale=0.70, size=(cfg.n_cells_per_modality, cfg.n_features)
        )

        if modality == "rna":
            activity[:, 0] = full["gene"]
            counts, technical_depth, baseline = negative_binomial_counts(activity, rng=rng)
            var_names = genes
            observation_parameters[modality] = pd.DataFrame(
                {
                    "modality": modality,
                    "feature": var_names,
                    "observation_intercept": baseline,
                    "target_detection": np.nan,
                    "dispersion": 12.0,
                }
            )
        else:
            mark = MARKS[modality]
            activity[:, :N_FOCAL_PER_MARK] = full[mark]
            counts, technical_depth, detection_targets, intercepts = sparse_chromatin_counts(
                activity,
                rng=rng,
                median_detection=cfg.chrom_detection,
            )
            var_names = chrom_names[mark]
            observation_parameters[modality] = pd.DataFrame(
                {
                    "modality": modality,
                    "feature": var_names,
                    "observation_intercept": intercepts,
                    "target_detection": detection_targets,
                    "dispersion": np.nan,
                }
            )

        observed_library = counts.sum(axis=1).astype(float)

        obs = pd.DataFrame(
            {
                "modality_key": modality,
                "mark": MARKS[modality],
                "true_cell_type": labels,
                "true_z0": z[:, 0],
                "true_z1": z[:, 1],
                "true_z2": z[:, 2],
                "true_technical_depth": technical_depth,
                "observed_library_size": observed_library,
            },
            index=[f"{modality}_cell_{i:05d}" for i in range(cfg.n_cells_per_modality)],
        )
        var = pd.DataFrame(index=pd.Index(var_names, name="feature"))
        var["highly_variable"] = True
        adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
        adata.layers["counts"] = sparse.csr_matrix(counts)
        if modality == "rna":
            prepare_rna_rep(adata, n_components=min(30, cfg.n_features - 1), seed=cfg.seed)
        else:
            prepare_chrom_rep(adata, n_components=min(30, cfg.n_features - 1), seed=cfg.seed)
        adatas[modality] = adata
        true_z[modality] = z
        true_labels[modality] = labels
        true_technical_depth[modality] = technical_depth

    graph = build_guidance_graph(genes, chrom_names, guidance_mode=cfg.guidance_mode)
    generative_dag = build_generative_dag(
        genes,
        chrom_names,
        modality_loadings_by_assay,
        observation_parameters,
    )
    expr_name = "expr__SIMG_log1p"
    direct = {
        canonical_edge(focal["H3K27ac"][0], expr_name),
        canonical_edge(focal["H3K4me3"][0], expr_name),
        canonical_edge(focal["H3K27ac"][0], focal["H3K27ac"][2]),
        canonical_edge(focal["H3K4me3"][0], focal["H3K4me3"][1]),
    }
    shared = {
        canonical_edge(focal["H3K27ac"][0], focal["H3K27ac"][1]),
        canonical_edge(focal["H3K4me3"][2], focal["H3K4me3"][3]),
    }
    semantics = [
        {"node": focal["H3K27ac"][0], "role": "slow_state_dominated_direct_gene_cause"},
        {"node": focal["H3K27ac"][1], "role": "shared_driver_peer_no_direct_gene_edge"},
        {"node": focal["H3K27ac"][2], "role": "same_mark_child_of_peak0"},
        {"node": focal["H3K27ac"][3], "role": "state_associated_not_direct"},
        {"node": focal["H3K27ac"][4], "role": "null"},
        {"node": focal["H3K27ac"][5], "role": "null"},
        {"node": focal["H3K4me3"][0], "role": "fast_residual_dominated_direct_gene_cause"},
        {"node": focal["H3K4me3"][1], "role": "same_mark_child_of_peak0"},
        {"node": focal["H3K4me3"][2], "role": "shared_driver_member_a"},
        {"node": focal["H3K4me3"][3], "role": "shared_driver_member_b"},
        {"node": focal["H3K4me3"][4], "role": "null"},
        {"node": focal["H3K4me3"][5], "role": "null"},
        {"node": expr_name, "role": "target_gene"},
    ]
    return SyntheticWorld(
        adatas=adatas,
        true_z=true_z,
        true_labels=true_labels,
        true_technical_depth=true_technical_depth,
        counterfactual_focal=counterfactual_focal,
        observation_parameters=observation_parameters,
        graph=graph,
        generative_dag=generative_dag,
        gene_name=expr_name,
        focal_features=focal,
        truth_direct=direct,
        truth_shared_driver=shared,
        feature_semantics=semantics,
    )


def oracle_embeddings(world: SyntheticWorld) -> dict[str, np.ndarray]:
    all_z = np.vstack([world.true_z[m] for m in MODALITIES])
    scaler = StandardScaler().fit(all_z)
    return {m: scaler.transform(world.true_z[m]) for m in MODALITIES}


def guidance_aligned_pca_embeddings(
    world: SyntheticWorld,
    *,
    n_components: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Integrate assays through observed, guidance-linked background features.

    The simulator's non-focal gene/peak triplets are linked by the guidance
    graph and share seed-specific biological loadings.  Treat each linked
    triplet as one common feature, normalize within assay, remove assay-specific
    marginal scale, and fit one PCA over all cells.  Focal columns are excluded,
    so target recovery cannot be copied from focal guidance edges.

    This is deliberately a simple, auditable integration baseline: in real data
    its analogue is a gene-activity matrix constructed from a held-out guidance
    graph, followed by batch-standardized PCA.
    """
    aligned: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        counts = sparse.csr_matrix(world.adatas[modality].layers["counts"], dtype=float)
        background = counts[:, N_FOCAL_PER_MARK:]
        library = np.asarray(counts.sum(axis=1)).ravel()
        library[library <= 0] = 1.0
        dense_background = background.toarray()
        expected = (
            (library[:, None] / max(float(library.sum()), 1.0))
            * dense_background.sum(axis=0, keepdims=True)
        )
        # Pearson residuals put dense NB RNA counts and sparse Bernoulli-like
        # chromatin detections on a common signal-to-noise scale before the
        # feature-loading alignment.
        normalized = (dense_background - expected) / np.sqrt(expected + 1.0)
        # Equalize assay-specific count scale and detection rate feature by
        # feature while retaining each cell's multivariate biological pattern.
        aligned[modality] = StandardScaler().fit_transform(normalized)

    # PCA axes are only identifiable up to rotation.  Fit each assay separately
    # and align its feature-loading basis to RNA using the guidance-linked feature
    # coordinates.  Three dimensions match the simulator's declared biological
    # state rank; a real-data analogue would select this rank by held-out loading
    # agreement rather than knowing latent truth.
    shared_dim = max(
        2,
        min(3, n_components, aligned["rna"].shape[1] - 1),
    )
    scores: dict[str, np.ndarray] = {}
    loadings: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        pca = PCA(n_components=shared_dim, random_state=seed)
        scores[modality] = pca.fit_transform(aligned[modality])
        loadings[modality] = pca.components_.T

    reference = loadings["rna"]
    output = {"rna": StandardScaler().fit_transform(scores["rna"])}
    for modality in MODALITIES[1:]:
        rotation, _ = orthogonal_procrustes(loadings[modality], reference)
        output[modality] = StandardScaler().fit_transform(scores[modality] @ rotation)
    return output


def fit_scglue_embeddings(
    world: SyntheticWorld,
    cfg: PilotConfig,
    *,
    checkpoint_dir: Path,
) -> dict[str, np.ndarray]:
    import scglue
    import scglue.models.plugins as plugins
    import torch

    # Keep this compatibility shim local.  Both pipelines/empirical/integration/ and
    # pipelines/empirical/analysis/ contain a module named ``_common``; importing the former
    # into sys.modules here would make the later production-FCI import resolve
    # the wrong helper module.
    if "verbose" not in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
        base_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

        class _CompatReduceLROnPlateau(base_cls):
            def __init__(self, *args, verbose=None, **kwargs):
                super().__init__(*args, **kwargs)

        plugins.ReduceLROnPlateau = _CompatReduceLROnPlateau
    scglue.config.CPU_ONLY = True
    adatas = {key: value.copy() for key, value in world.adatas.items()}
    for key, adata in adatas.items():
        scglue.models.configure_dataset(
            adata,
            "NB",
            use_highly_variable=True,
            use_layer="counts",
            use_rep="X_pca" if key == "rna" else "X_lsi",
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    glue = scglue.models.fit_SCGLUE(
        adatas,
        world.graph,
        init_kws={
            "latent_dim": cfg.latent_dim,
            "h_depth": 1,
            "h_dim": cfg.h_dim,
            "dropout": 0.15,
            "shared_batches": False,
            "random_seed": cfg.integration_seed,
        },
        compile_kws={
            "lam_data": 1.0,
            "lam_kl": 1.0,
            "lam_graph": 0.02,
            "lam_align": 0.08,
            "lam_sup": 0.02,
            "lr": 2e-3,
        },
        fit_kws={
            "directory": str(checkpoint_dir),
            "neg_samples": 5,
            "val_split": 0.1,
            "data_batch_size": min(128, cfg.n_cells_per_modality),
            "graph_batch_size": min(512, max(64, world.graph.number_of_edges())),
            "align_burnin": max(1, min(3, cfg.max_epochs // 4)),
            "max_epochs": cfg.max_epochs,
            "patience": max(2, min(4, cfg.max_epochs)),
            "reduce_lr_patience": 2,
            "wait_n_lrs": 1,
        },
    )
    return {
        key: np.asarray(glue.encode_data(key, adata, batch_size=256), dtype=float)
        for key, adata in adatas.items()
    }


def standardize_embeddings(embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    stacked = np.vstack([embeddings[m] for m in MODALITIES])
    scaler = StandardScaler().fit(stacked)
    return {m: scaler.transform(embeddings[m]) for m in MODALITIES}


def integration_metrics(
    world: SyntheticWorld,
    embeddings: dict[str, np.ndarray],
    *,
    k: int = 15,
) -> dict[str, float]:
    embeddings = standardize_embeddings(embeddings)
    x = np.vstack([embeddings[m] for m in MODALITIES])
    z = np.vstack([world.true_z[m] for m in MODALITIES])
    mods = np.concatenate([[m] * embeddings[m].shape[0] for m in MODALITIES])
    n_neighbors = min(k + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    same_fraction = float(np.mean(mods[idx] == mods[:, None]))

    state_errors = []
    oracle_errors = []
    for modality in MODALITIES:
        other = [m for m in MODALITIES if m != modality]
        other_emb = np.vstack([embeddings[m] for m in other])
        other_z = np.vstack([world.true_z[m] for m in other])
        learned_idx = NearestNeighbors(n_neighbors=1).fit(other_emb).kneighbors(
            embeddings[modality], return_distance=False
        )[:, 0]
        oracle_idx = NearestNeighbors(n_neighbors=1).fit(other_z).kneighbors(
            world.true_z[modality], return_distance=False
        )[:, 0]
        state_errors.extend(
            np.linalg.norm(world.true_z[modality] - other_z[learned_idx], axis=1).tolist()
        )
        oracle_errors.extend(
            np.linalg.norm(world.true_z[modality] - other_z[oracle_idx], axis=1).tolist()
        )

    silhouette = float(silhouette_score(x, mods)) if len(set(mods)) > 1 else float("nan")
    mean_state_error = float(np.mean(state_errors))
    oracle_floor = float(np.mean(oracle_errors))
    return {
        "same_modality_neighbor_fraction": same_fraction,
        "silhouette_by_modality": silhouette,
        "crossmodal_true_state_error": mean_state_error,
        "oracle_crossmodal_state_floor": oracle_floor,
        "state_error_over_oracle": mean_state_error / max(oracle_floor, 1e-12),
    }


def transfer_rna_labels(
    world: SyntheticWorld,
    embeddings: dict[str, np.ndarray],
    *,
    k: int,
    min_confidence: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    embeddings = standardize_embeddings(embeddings)
    rna_x = embeddings["rna"]
    rna_is_target = world.true_labels["rna"] == TARGET_LABEL
    k = min(k, rna_x.shape[0])
    knn = NearestNeighbors(n_neighbors=k).fit(rna_x)
    keep = {"rna": rna_is_target.copy()}
    metrics: dict[str, float] = {}
    for modality in MODALITIES[1:]:
        neighbors = knn.kneighbors(embeddings[modality], return_distance=False)
        confidence = rna_is_target[neighbors].mean(axis=1)
        predicted = confidence >= 0.5
        keep[modality] = predicted & (np.maximum(confidence, 1.0 - confidence) >= min_confidence)
        truth = world.true_labels[modality] == TARGET_LABEL
        metrics[f"{modality}_label_accuracy"] = float(np.mean(predicted == truth))
        metrics[f"{modality}_retained_fraction"] = float(np.mean(keep[modality]))
    metrics["rna_retained_fraction"] = float(np.mean(keep["rna"]))
    return keep, metrics


def aggregate_complete_metacells(
    world: SyntheticWorld,
    embeddings: dict[str, np.ndarray],
    keep: dict[str, np.ndarray],
    *,
    n_metacells: int,
    seed: int,
) -> MetacellTables:
    embeddings = standardize_embeddings(embeddings)
    frames = []
    x_parts = []
    for modality in MODALITIES:
        idx = np.flatnonzero(keep[modality])
        frame = pd.DataFrame({"modality": modality, "local_index": idx})
        frames.append(frame)
        x_parts.append(embeddings[modality][idx])
    meta = pd.concat(frames, ignore_index=True)
    x = np.vstack(x_parts)
    n_metacells = min(n_metacells, max(2, x.shape[0] // len(MODALITIES)))
    meta["metacell"] = KMeans(
        n_clusters=n_metacells,
        random_state=seed,
        n_init=10,
    ).fit_predict(x)

    complete = []
    for cluster, sub in meta.groupby("metacell"):
        if set(sub["modality"]) == set(MODALITIES):
            complete.append(int(cluster))

    counts = {
        m: sparse.csr_matrix(world.adatas[m].layers["counts"], dtype=float) for m in MODALITIES
    }
    libraries = {m: np.asarray(counts[m].sum(axis=1)).ravel() for m in MODALITIES}
    # Leave the focal locus out of nuisance-depth summaries.  The full library
    # size is a descendant of every focal count and treating it as a causal root
    # can condition on a collider.  Background depth is still fully observable
    # in a real assay, but does not mechanically contain the variables searched.
    background_libraries = {
        "rna": np.asarray(counts["rna"][:, 1:].sum(axis=1)).ravel(),
        "chrom_H3K27ac": np.asarray(
            counts["chrom_H3K27ac"][:, N_FOCAL_PER_MARK:].sum(axis=1)
        ).ravel(),
        "chrom_H3K4me3": np.asarray(
            counts["chrom_H3K4me3"][:, N_FOCAL_PER_MARK:].sum(axis=1)
        ).ravel(),
    }
    observed_rows: list[dict[str, float | str]] = []
    truth_rows: list[dict[str, float | str | int]] = []
    membership_rows: list[dict[str, float | str | int | bool]] = []
    mismatch = []
    for out_i, cluster in enumerate(sorted(complete)):
        sub = meta.loc[meta["metacell"] == cluster]
        members = {
            m: sub.loc[sub["modality"] == m, "local_index"].to_numpy(dtype=int)
            for m in MODALITIES
        }
        sample_id = f"MC_{out_i + 1:04d}"
        observed_row: dict[str, float | str] = {"sample_id": sample_id}
        truth_row: dict[str, float | str | int] = {
            "sample_id": sample_id,
            "truth__source_cluster": int(cluster),
        }

        rna_idx = members["rna"]
        rna_depth = float(libraries["rna"][rna_idx].sum())
        gene_count = float(counts["rna"][rna_idx, 0].sum())
        observed_row[world.gene_name] = np.log1p(1e4 * gene_count / max(rna_depth, 1.0))
        observed_row["libsize__rna"] = float(np.mean(libraries["rna"][rna_idx]))
        observed_row["libsize_background__rna"] = float(
            np.mean(background_libraries["rna"][rna_idx])
        )

        for modality in MODALITIES[1:]:
            mark = MARKS[modality]
            idx = members[modality]
            depth = libraries[modality][idx]
            safe_depth = np.where(depth <= 0, 1.0, depth)
            for feature_i, feature in enumerate(world.focal_features[mark]):
                values = counts[modality][idx, feature_i].toarray().ravel()
                observed_row[feature] = float(np.log1p(values * (1e4 / safe_depth)).mean())
            observed_row[f"libsize__{mark}"] = float(np.mean(depth))
            observed_row[f"libsize_background__{mark}"] = float(
                np.mean(background_libraries[modality][idx])
            )

        centroids = []
        for modality in MODALITIES:
            label = "rna" if modality == "rna" else MARKS[modality]
            idx = members[modality]
            centroid = world.true_z[modality][idx].mean(axis=0)
            centroids.append(centroid)
            truth_row[f"truth__n_cells__{label}"] = int(idx.size)
            truth_row[f"truth__target_fraction__{label}"] = float(
                np.mean(world.true_labels[modality][idx] == TARGET_LABEL)
            )
            truth_row[f"truth__technical_depth_mean__{label}"] = float(
                np.mean(world.true_technical_depth[modality][idx])
            )
            truth_row[f"truth__observed_library_mean__{label}"] = float(
                np.mean(libraries[modality][idx])
            )
            for dim in range(3):
                truth_row[f"truth__state_mean__{label}__z{dim}"] = float(centroid[dim])

            for local_index in idx:
                membership_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_cluster": int(cluster),
                        "modality": modality,
                        "cell": str(world.adatas[modality].obs_names[local_index]),
                        "local_index": int(local_index),
                        "true_cell_type": str(world.true_labels[modality][local_index]),
                        "true_is_target": bool(
                            world.true_labels[modality][local_index] == TARGET_LABEL
                        ),
                        "true_z0": float(world.true_z[modality][local_index, 0]),
                        "true_z1": float(world.true_z[modality][local_index, 1]),
                        "true_z2": float(world.true_z[modality][local_index, 2]),
                        "true_technical_depth": float(
                            world.true_technical_depth[modality][local_index]
                        ),
                        "observed_library_size": float(libraries[modality][local_index]),
                    }
                )

        consensus = np.mean(centroids, axis=0)
        for dim in range(3):
            truth_row[f"truth__state_consensus__z{dim}"] = float(consensus[dim])
        pair_dist = [
            float(np.linalg.norm(centroids[i] - centroids[j]))
            for i in range(len(centroids))
            for j in range(i + 1, len(centroids))
        ]
        truth_row["truth__state_mismatch"] = float(np.mean(pair_dist))
        mismatch.append(float(np.mean(pair_dist)))
        observed_rows.append(observed_row)
        truth_rows.append(truth_row)

    observed_columns = [
        "sample_id",
        world.gene_name,
        "libsize__rna",
        "libsize_background__rna",
    ]
    for mark in ("H3K27ac", "H3K4me3"):
        observed_columns.extend(world.focal_features[mark])
        observed_columns.append(f"libsize__{mark}")
        observed_columns.append(f"libsize_background__{mark}")
    observed = pd.DataFrame(observed_rows, columns=observed_columns)
    ground_truth = pd.DataFrame(truth_rows)
    if ground_truth.empty:
        ground_truth = pd.DataFrame(columns=["sample_id"])
    membership_columns = [
        "sample_id",
        "source_cluster",
        "modality",
        "cell",
        "local_index",
        "true_cell_type",
        "true_is_target",
        "true_z0",
        "true_z1",
        "true_z2",
        "true_technical_depth",
        "observed_library_size",
    ]
    membership = pd.DataFrame(membership_rows, columns=membership_columns)
    metrics = {
        "n_metacells_requested": int(n_metacells),
        "n_complete_metacells": int(len(complete)),
        "complete_fraction": float(len(complete) / max(n_metacells, 1)),
        "mean_true_state_mismatch": float(np.mean(mismatch)) if mismatch else float("nan"),
    }
    return MetacellTables(
        observed=observed,
        ground_truth=ground_truth,
        membership=membership,
        metrics=metrics,
    )


def validate_metacell_tables(tables: MetacellTables) -> None:
    """Protect the realistic matrix from accidental ground-truth leakage."""
    required_observed = {
        "sample_id",
        "libsize__rna",
        "libsize__H3K27ac",
        "libsize__H3K4me3",
    }
    missing = required_observed - set(tables.observed.columns)
    if missing:
        raise RuntimeError(f"Observed metacell matrix is missing columns: {sorted(missing)}")
    leaked = [
        col
        for col in tables.observed.columns
        if col.startswith(("true_", "truth__", "latent__", "technical__", "conf__"))
    ]
    if leaked:
        raise RuntimeError(f"Ground-truth or analysis-only columns leaked into observed matrix: {leaked}")
    observed_ids = tables.observed["sample_id"].astype(str).tolist()
    truth_ids = tables.ground_truth["sample_id"].astype(str).tolist()
    if observed_ids != truth_ids:
        raise RuntimeError("Observed and ground-truth metacell rows are not aligned")
    if not tables.membership.empty:
        member_ids = set(tables.membership["sample_id"].astype(str))
        if member_ids != set(observed_ids):
            raise RuntimeError("Metacell membership does not cover exactly the complete metacells")


def run_discovery(
    observed_matrix: pd.DataFrame,
    world: SyntheticWorld,
    *,
    alpha: float,
    max_depth: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    command_dir = REPO / "pipelines/empirical/analysis/scripts/commands"
    sys.path.insert(0, str(command_dir))
    from run_pc_causallearn import (
        apply_transform,
        build_background_knowledge,
        build_confounder_columns,
        edge_to_record,
        run_fci_search,
    )

    analysis_matrix = observed_matrix.copy()
    confounders = build_confounder_columns(analysis_matrix, "depth_permark")
    biological = [world.gene_name]
    for mark in ("H3K27ac", "H3K4me3"):
        biological.extend(world.focal_features[mark])
    columns = confounders + [c for c in biological if c in analysis_matrix.columns]
    analysis_matrix = analysis_matrix[["sample_id", *columns]].copy()

    if analysis_matrix.shape[0] < 8:
        return (
            pd.DataFrame(),
            {"error": "fewer_than_8_complete_metacells", "confounder_columns": confounders},
            analysis_matrix,
        )
    live = [
        c for c in columns if analysis_matrix[c].astype(float).std(ddof=0) > 1e-10
    ]
    dropped = sorted(set(columns) - set(live))
    if len(live) < 4:
        return (
            pd.DataFrame(),
            {
                "error": "fewer_than_4_live_variables",
                "n_dropped": len(dropped),
                "confounder_columns": confounders,
            },
            analysis_matrix,
        )
    model = apply_transform(analysis_matrix[live].astype(float), "rank_gaussian")
    model = (model - model.mean()) / model.std(ddof=0).replace(0, 1)
    bk, _ = build_background_knowledge(live, "minimal_expr_sink")
    _, edges = run_fci_search(
        model.to_numpy(),
        alpha=alpha,
        indep_test="fisherz",
        background_knowledge=bk,
        show_progress=False,
        node_names=live,
        max_depth=max_depth,
    )
    records = pd.DataFrame([edge_to_record(edge) for edge in edges])
    estimated = set()
    if not records.empty:
        for rec in records.to_dict(orient="records"):
            a, b = str(rec["node1"]), str(rec["node2"])
            if a.startswith("conf__") or b.startswith("conf__"):
                continue
            estimated.add(canonical_edge(a, b))
    direct = world.truth_direct
    shared = world.truth_shared_driver
    metrics: dict[str, Any] = {
        "n_rows": int(analysis_matrix.shape[0]),
        "confounder_columns": confounders,
        "n_live_variables": int(len(live)),
        "n_dropped_variables": int(len(dropped)),
        "n_estimated_biological_edges": int(len(estimated)),
        "direct_recall": float(len(estimated & direct) / len(direct)),
        "direct_precision": float(len(estimated & direct) / len(estimated)) if estimated else 0.0,
        "shared_driver_recall": float(len(estimated & shared) / len(shared)),
        "slow_gene_edge_recovered": int(
            canonical_edge(world.focal_features["H3K27ac"][0], world.gene_name) in estimated
        ),
        "fast_gene_edge_recovered": int(
            canonical_edge(world.focal_features["H3K4me3"][0], world.gene_name) in estimated
        ),
    }
    return records, metrics, analysis_matrix


def save_truth(world: SyntheticWorld, out_dir: Path) -> None:
    rows = []
    for edge in sorted(world.truth_direct):
        rows.append({"node1": edge[0], "node2": edge[1], "truth_type": "direct"})
    for edge in sorted(world.truth_shared_driver):
        rows.append({"node1": edge[0], "node2": edge[1], "truth_type": "shared_driver"})
    pd.DataFrame(rows).to_csv(out_dir / "truth_edges.tsv", sep="\t", index=False)
    pd.DataFrame(world.feature_semantics).to_csv(
        out_dir / "focal_feature_semantics.tsv", sep="\t", index=False
    )

    dag_nodes = [
        {"node": node, **world.generative_dag.nodes[node]}
        for node in nx.topological_sort(world.generative_dag)
    ]
    dag_edges = [
        {"source": source, "target": target, **attrs}
        for source, target, attrs in world.generative_dag.edges(data=True)
    ]
    pd.DataFrame(dag_nodes).to_csv(out_dir / "generative_dag_nodes.tsv", sep="\t", index=False)
    pd.DataFrame(dag_edges).to_csv(out_dir / "generative_dag_edges.tsv", sep="\t", index=False)
    dag_payload = {
        "graph": dict(world.generative_dag.graph),
        "nodes": dag_nodes,
        "edges": dag_edges,
    }
    (out_dir / "generative_dag.json").write_text(
        json.dumps(json_ready(dag_payload), indent=2, allow_nan=False)
    )
    pd.concat(
        [world.observation_parameters[modality] for modality in MODALITIES],
        ignore_index=True,
    ).to_csv(out_dir / "observation_parameters.tsv", sep="\t", index=False)

    cell_rows = []
    for modality in MODALITIES:
        observed_library = np.asarray(
            world.adatas[modality].layers["counts"].sum(axis=1)
        ).ravel()
        for i, cell in enumerate(world.adatas[modality].obs_names):
            row = {
                "cell": str(cell),
                "modality": modality,
                "true_cell_type": str(world.true_labels[modality][i]),
                "true_z0": float(world.true_z[modality][i, 0]),
                "true_z1": float(world.true_z[modality][i, 1]),
                "true_z2": float(world.true_z[modality][i, 2]),
                "true_technical_depth": float(world.true_technical_depth[modality][i]),
                "observed_library_size": float(observed_library[i]),
                f"true_activity__{world.gene_name}": float(
                    world.counterfactual_focal[modality]["gene"][i]
                ),
            }
            for mark in ("H3K27ac", "H3K4me3"):
                for feature_i, feature in enumerate(world.focal_features[mark]):
                    row[f"true_activity__{feature}"] = float(
                        world.counterfactual_focal[modality][mark][i, feature_i]
                    )
            cell_rows.append(row)
    pd.DataFrame(cell_rows).to_csv(out_dir / "cell_ground_truth.tsv", sep="\t", index=False)


def save_embeddings(
    world: SyntheticWorld,
    embeddings: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    rows = []
    for modality in MODALITIES:
        for i in range(embeddings[modality].shape[0]):
            row = {
                "modality": modality,
                "cell": world.adatas[modality].obs_names[i],
                "true_cell_type": world.true_labels[modality][i],
                "true_z0": world.true_z[modality][i, 0],
                "true_z1": world.true_z[modality][i, 1],
                "true_z2": world.true_z[modality][i, 2],
            }
            row.update({f"embedding_{j}": v for j, v in enumerate(embeddings[modality][i])})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def save_observed_cells(world: SyntheticWorld, out_dir: Path) -> list[str]:
    """Write focal single-cell assay tables with no latent or truth columns."""
    outputs: list[str] = []
    for modality in MODALITIES:
        counts = sparse.csr_matrix(world.adatas[modality].layers["counts"])
        full_library = np.asarray(counts.sum(axis=1)).ravel()
        focal_width = 1 if modality == "rna" else N_FOCAL_PER_MARK
        background_library = np.asarray(counts[:, focal_width:].sum(axis=1)).ravel()
        frame = pd.DataFrame(
            {
                "cell": world.adatas[modality].obs_names.astype(str),
                "observed_library_size": full_library,
                "observed_background_library_size": background_library,
            }
        )
        if modality == "rna":
            frame[world.gene_name] = counts[:, 0].toarray().ravel().astype(int)
        else:
            mark = MARKS[modality]
            focal = counts[:, :N_FOCAL_PER_MARK].toarray().astype(int)
            for feature_i, feature in enumerate(world.focal_features[mark]):
                frame[feature] = focal[:, feature_i]
        filename = f"observed_cells__{modality}.tsv"
        frame.to_csv(out_dir / filename, sep="\t", index=False)
        outputs.append(filename)
    return outputs


def save_analysis_embeddings(
    world: SyntheticWorld,
    embeddings: dict[str, np.ndarray],
    keep: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Write analysis-eligible integration coordinates without oracle labels."""
    standardized = standardize_embeddings(embeddings)
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for i, cell in enumerate(world.adatas[modality].obs_names):
            row: dict[str, Any] = {
                "modality": modality,
                "cell": str(cell),
                "retained_target": bool(keep[modality][i]),
            }
            row.update(
                {
                    f"integration__dim{j}": float(value)
                    for j, value in enumerate(standardized[modality][i])
                }
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--integration",
        choices=["oracle", "scglue", "gapca", "both", "all"],
        default="both",
        help=(
            "gapca = guidance-aligned PCA on held-out background features; "
            "both = oracle + scGLUE (backward compatible); all = all three arms"
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Independent synthetic-world seed")
    p.add_argument("--integration-seed", type=int, default=0)
    p.add_argument("--metacell-seed", type=int, default=0)
    p.add_argument("--n-cells-per-modality", type=int, default=600)
    p.add_argument("--n-features", type=int, default=128)
    p.add_argument("--n-metacells", type=int, default=36)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--h-dim", type=int, default=64)
    p.add_argument(
        "--max-epochs",
        type=int,
        default=40,
        help="Maximum scGLUE epochs per phase; early stopping is enabled (default: 40)",
    )
    p.add_argument("--chrom-detection", type=float, default=0.07)
    p.add_argument("--population-shift", type=float, default=0.0)
    p.add_argument("--modality-batch-scale", type=float, default=0.20)
    p.add_argument("--guidance-mode", choices=["perfect", "holdout", "shuffled"], default="perfect")
    p.add_argument("--label-k", type=int, default=25)
    p.add_argument("--min-label-confidence", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny CPU smoke preset")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.n_cells_per_modality = 180
        args.n_features = 48
        args.n_metacells = 12
        args.latent_dim = 4
        args.h_dim = 32
        args.max_epochs = 3

    cfg = PilotConfig(
        seed=args.seed,
        integration_seed=args.integration_seed,
        metacell_seed=args.metacell_seed,
        n_cells_per_modality=args.n_cells_per_modality,
        n_features=args.n_features,
        n_metacells=args.n_metacells,
        latent_dim=args.latent_dim,
        h_dim=args.h_dim,
        max_epochs=args.max_epochs,
        chrom_detection=args.chrom_detection,
        population_shift=args.population_shift,
        modality_batch_scale=args.modality_batch_scale,
        guidance_mode=args.guidance_mode,
        label_k=args.label_k,
        min_label_confidence=args.min_label_confidence,
        alpha=args.alpha,
        max_depth=args.max_depth,
        integration=args.integration,
        smoke=args.smoke,
    )
    run_id = args.run_id or (
        f"pilot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_seed{cfg.seed}"
    )
    out_dir = args.out_dir or REPO / "pipelines/simulation/unpaired_multimodal/outputs/multimodal_simulation_pilot" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    start = time.perf_counter()
    world = generate_world(cfg)
    timings["generate_world_seconds"] = time.perf_counter() - start
    save_truth(world, out_dir)
    observed_cell_outputs = save_observed_cells(world, out_dir)

    realism = {}
    for modality in MODALITIES:
        counts = sparse.csr_matrix(world.adatas[modality].layers["counts"])
        realism[modality] = {
            "n_cells": int(counts.shape[0]),
            "n_features": int(counts.shape[1]),
            "median_library_size": float(np.median(np.asarray(counts.sum(axis=1)).ravel())),
            "median_true_technical_depth": float(
                np.median(world.true_technical_depth[modality])
            ),
            "matrix_density": float(counts.nnz / np.prod(counts.shape)),
            "target_fraction": float(np.mean(world.true_labels[modality] == TARGET_LABEL)),
        }

    arm_embeddings: dict[str, dict[str, np.ndarray]] = {}
    if cfg.integration in {"oracle", "both", "all"}:
        start = time.perf_counter()
        arm_embeddings["oracle"] = oracle_embeddings(world)
        timings["oracle_embedding_seconds"] = time.perf_counter() - start
    if cfg.integration in {"gapca", "all"}:
        start = time.perf_counter()
        arm_embeddings["gapca"] = guidance_aligned_pca_embeddings(
            world,
            n_components=cfg.latent_dim,
            seed=cfg.integration_seed,
        )
        timings["gapca_embedding_seconds"] = time.perf_counter() - start
    if cfg.integration in {"scglue", "both", "all"}:
        start = time.perf_counter()
        arm_embeddings["scglue"] = fit_scglue_embeddings(
            world,
            cfg,
            checkpoint_dir=out_dir / "scglue_checkpoints",
        )
        timings["scglue_fit_encode_seconds"] = time.perf_counter() - start

    arms: dict[str, Any] = {}
    for arm, embeddings in arm_embeddings.items():
        arm_start = time.perf_counter()
        i_metrics = integration_metrics(world, embeddings)
        keep, label_metrics = transfer_rna_labels(
            world,
            embeddings,
            k=cfg.label_k,
            min_confidence=cfg.min_label_confidence,
        )
        save_analysis_embeddings(
            world,
            embeddings,
            keep,
            out_dir / f"{arm}_analysis_embeddings.tsv",
        )
        tables = aggregate_complete_metacells(
            world,
            embeddings,
            keep,
            n_metacells=cfg.n_metacells,
            seed=cfg.metacell_seed,
        )
        validate_metacell_tables(tables)
        tables.observed.to_csv(
            out_dir / f"{arm}_observed_metacell_matrix.tsv", sep="\t", index=False
        )
        # Keep the original filename as an alias for downstream scripts written
        # against the first pilot. It now contains the leakage-safe observed table.
        tables.observed.to_csv(out_dir / f"{arm}_metacell_matrix.tsv", sep="\t", index=False)
        tables.ground_truth.to_csv(
            out_dir / f"{arm}_metacell_ground_truth.tsv", sep="\t", index=False
        )
        tables.membership.to_csv(
            out_dir / f"{arm}_metacell_membership.tsv", sep="\t", index=False
        )
        discovery_start = time.perf_counter()
        edges, discovery_metrics, analysis_matrix = run_discovery(
            tables.observed,
            world,
            alpha=cfg.alpha,
            max_depth=cfg.max_depth,
        )
        timings[f"{arm}_discovery_seconds"] = time.perf_counter() - discovery_start
        analysis_matrix.to_csv(
            out_dir / f"{arm}_fci_analysis_matrix.tsv", sep="\t", index=False
        )
        edges.to_csv(out_dir / f"{arm}_fci_edges.tsv", sep="\t", index=False)
        if args.save_embeddings:
            save_embeddings(world, embeddings, out_dir / f"{arm}_cell_embeddings.tsv")
        timings[f"{arm}_postintegration_seconds"] = time.perf_counter() - arm_start
        arms[arm] = {
            "integration": i_metrics,
            "label_transfer": label_metrics,
            "metacells": tables.metrics,
            "discovery": discovery_metrics,
        }

    timings["total_seconds"] = time.perf_counter() - total_start
    summary = {
        "run_id": run_id,
        "config": asdict(cfg),
        "realism": realism,
        "truth": {
            "n_direct_edges": len(world.truth_direct),
            "n_shared_driver_pairs": len(world.truth_shared_driver),
            "generative_dag_nodes": world.generative_dag.number_of_nodes(),
            "generative_dag_edges": world.generative_dag.number_of_edges(),
            "generative_dag_is_acyclic": nx.is_directed_acyclic_graph(world.generative_dag),
        },
        "arms": arms,
        "timings_seconds": timings,
        "outputs": {
            "out_dir": str(out_dir),
            "global_ground_truth": [
                "generative_dag.json",
                "generative_dag_nodes.tsv",
                "generative_dag_edges.tsv",
                "observation_parameters.tsv",
                "cell_ground_truth.tsv",
                "truth_edges.tsv",
                "focal_feature_semantics.tsv",
            ],
            "observed_single_cell_assays": observed_cell_outputs,
            "per_arm_patterns": [
                "{arm}_observed_metacell_matrix.tsv",
                "{arm}_analysis_embeddings.tsv",
                "{arm}_metacell_ground_truth.tsv",
                "{arm}_metacell_membership.tsv",
                "{arm}_fci_analysis_matrix.tsv",
                "{arm}_fci_edges.tsv",
            ],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, allow_nan=True))
    (out_dir / "timings.json").write_text(json.dumps(json_ready(timings), indent=2))

    print("\n=== MULTIMODAL SIMULATION PILOT ===")
    print(f"output: {out_dir}")
    print(f"generated in {timings['generate_world_seconds']:.2f}s")
    if "scglue_fit_encode_seconds" in timings:
        print(f"scGLUE fit + encode: {timings['scglue_fit_encode_seconds']:.2f}s")
    for arm, result in arms.items():
        integ = result["integration"]
        mc = result["metacells"]
        disc = result["discovery"]
        print(
            f"{arm:>7s}: state-error/oracle={integ['state_error_over_oracle']:.2f}  "
            f"complete-MC={mc['n_complete_metacells']}/{mc['n_metacells_requested']}  "
            f"direct-recall={disc.get('direct_recall', float('nan')):.2f}  "
            f"shared-recall={disc.get('shared_driver_recall', float('nan')):.2f}"
        )
    print(f"total: {timings['total_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
