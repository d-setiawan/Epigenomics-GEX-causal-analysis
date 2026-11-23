#!/usr/bin/env python3
"""Channel-aware multi-resolution BOSS for independently sampled modalities.

The ordinary pipeline gives one matched-metacell matrix to one causal search.
That discards two important facts: variables from the same assay were observed
in the same cells, while variables from different assays were not.  This
candidate therefore estimates two explicitly different layers:

* a matched-state layer from the cross-modality metacell matrix, using
  rank-Gaussian SEM-BIC BOSS plus calibrated marginal state associations;
* a within-assay layer from the original sparse single-cell peak detections,
  using discrete BDeu BOSS and an optional sparse contingency-table screen.

The merged graph is a typed adjacency graph.  It intentionally abstains from
peak-to-peak orientation.  A chromatin-to-expression arrowhead can be emitted
only as an explicitly labelled prior constraint, never as a learned direction.

Run in the pinned ``causaldisco`` environment with JDK 21::

    JAVA_HOME=/home/dgsetiawan/anaconda3/envs/causaldisco \
      /home/dgsetiawan/anaconda3/envs/causaldisco/bin/python \
      pipelines/simulation/unpaired_multimodal/scripts/channel_aware_boss.py \
        --simulation-dir pipelines/simulation/unpaired_multimodal/outputs/multimodal_simulation_pilot/RUN \
        --arm oracle --out-dir pipelines/simulation/unpaired_multimodal/outputs/channel_aware_boss/RUN

Only observed assay tables and analysis embeddings are read.  Truth, DAG,
semantics, membership, and ground-truth metacell files are never opened.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm, pearsonr, rankdata
from sklearn.cluster import KMeans


REPO = Path(__file__).resolve().parents[4]
TETRAD_BRIDGE = REPO / "pipelines/empirical/analysis/scripts"
if str(TETRAD_BRIDGE) not in sys.path:
    sys.path.insert(0, str(TETRAD_BRIDGE))
from run_gfci_tetrad import parse_tetrad_edges  # noqa: E402


BIOLOGICAL_PREFIXES = ("expr__", "chr")
ENDPOINT_ORDER = {"TAIL": 0, "CIRCLE": 1, "ARROW": 2}


@dataclass(frozen=True)
class ChannelAwareConfig:
    state_algorithm: str = "boss"
    state_penalty: float = 1.0
    state_starts: int = 20
    state_use_bes: bool = True
    state_bootstraps: int = 0
    state_bootstrap_fraction: float = 0.8
    state_boss_support: float = 0.5
    state_boss_min_abs_r: float = 0.20
    state_boss_alpha: float = 0.05
    state_marginal_min_abs_r: float = 0.20
    state_marginal_alpha: float = 0.05
    depth_residualize: bool = True
    within_sample_prior: float = 10.0
    within_structure_prior: float = 0.0
    within_starts: int = 20
    within_use_bes: bool = True
    within_order_repeats: int = 1
    within_boss_support: float = 0.5
    sparse_min_odds_ratio: float = 1.5
    sparse_alpha: float = 0.20
    within_profile_clusters: int = 0
    within_profile_min_abs_r: float = 0.60
    within_profile_alpha: float = 0.05
    retained_cells_only: bool = False
    impose_gene_sink: bool = True
    seed: int = 0


def canonical_edge(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def is_biological(column: str) -> bool:
    return column.startswith(BIOLOGICAL_PREFIXES)


def node_channel(node: str) -> str:
    if node.startswith("expr__"):
        return "RNA"
    if "__" in node:
        return node.rsplit("__", 1)[-1]
    return "unknown"


def rank_gaussian_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = rankdata(values, method="average")
    return norm.ppf((ranks - 0.5) / len(ranks))


def rank_gaussian_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {column: rank_gaussian_vector(frame[column].to_numpy()) for column in frame.columns},
        index=frame.index,
    )


def prepare_state_data(
    observed: pd.DataFrame,
    *,
    depth_residualize: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    biological = [column for column in observed.columns if is_biological(column)]
    if len(biological) < 2:
        raise ValueError("The metacell matrix contains fewer than two biological columns")
    numeric = observed[biological].apply(pd.to_numeric, errors="coerce")
    live = [column for column in biological if numeric[column].std(ddof=0) > 1e-12]
    dropped = sorted(set(biological) - set(live))
    numeric = numeric[live]

    depth_columns = [
        column
        for column in observed.columns
        if column.startswith("libsize_background__")
        and pd.api.types.is_numeric_dtype(observed[column])
    ]
    complete = numeric.notna().all(axis=1)
    if depth_residualize and depth_columns:
        complete &= observed[depth_columns].notna().all(axis=1)
    numeric = numeric.loc[complete].reset_index(drop=True)
    transformed = rank_gaussian_frame(numeric)

    if depth_residualize and depth_columns:
        depth = np.column_stack(
            [
                rank_gaussian_vector(np.log1p(observed.loc[complete, column].to_numpy(dtype=float)))
                for column in depth_columns
            ]
        )
        design = np.column_stack([np.ones(len(depth)), depth])
        values = transformed.to_numpy(dtype=float, copy=True)
        values -= design @ np.linalg.lstsq(design, values, rcond=None)[0]
        scale = values.std(axis=0, ddof=0)
        scale[scale <= 1e-12] = 1.0
        transformed = pd.DataFrame(values / scale, columns=transformed.columns)

    return transformed, {
        "n_rows": int(len(transformed)),
        "n_biological_columns": int(len(live)),
        "dropped_biological_columns": dropped,
        "background_depth_columns": depth_columns,
        "depth_residualized": bool(depth_residualize and depth_columns),
    }


def _map_safe_edges(
    records: Iterable[dict[str, Any]],
    inverse: dict[str, str],
) -> list[dict[str, str]]:
    mapped: list[dict[str, str]] = []
    for record in records:
        mapped.append(
            {
                "node1": inverse.get(str(record["node1"]), str(record["node1"])),
                "endpoint1": str(record["endpoint1"]),
                "node2": inverse.get(str(record["node2"]), str(record["node2"])),
                "endpoint2": str(record["endpoint2"]),
            }
        )
    return mapped


def run_continuous_boss(
    frame: pd.DataFrame,
    *,
    algorithm: str,
    penalty: float,
    starts: int,
    use_bes: bool,
    seed: int,
) -> list[dict[str, str]]:
    from pytetrad.tools import TetradSearch as ts

    rng = np.random.default_rng(seed)
    columns = list(frame.columns)
    ordered = [columns[i] for i in rng.permutation(len(columns))]
    safe = {column: f"V{i}" for i, column in enumerate(ordered)}
    inverse = {value: key for key, value in safe.items()}
    data = frame[ordered].rename(columns=safe).astype(np.float64)
    search = ts.TetradSearch(data)
    search.use_sem_bic(penalty_discount=float(penalty))
    search.use_fisher_z(alpha=0.05)
    if algorithm == "boss":
        search.run_boss(
            num_starts=int(starts),
            use_bes=bool(use_bes),
            use_data_order=True,
            output_cpdag=True,
        )
    elif algorithm == "boss_fci":
        search.run_boss_fci(
            depth=2,
            max_disc_path_length=4,
            complete_rule_set_used=True,
            guarantee_pag=True,
        )
    elif algorithm == "lvh":
        search.run_lv_heuristic(
            num_starts=int(starts),
            use_bes=bool(use_bes),
            use_data_order=True,
            output_cpdag=True,
            complete_rule_set_used=True,
        )
    else:
        raise ValueError(f"Unknown state algorithm: {algorithm}")
    parsed = parse_tetrad_edges(str(search.get_string()))
    return _map_safe_edges(parsed, inverse)


def state_layer(
    frame: pd.DataFrame,
    cfg: ChannelAwareConfig,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    repetitions = max(1, cfg.state_bootstraps)
    adjacency_counts: Counter[tuple[str, str]] = Counter()
    endpoint_votes: dict[tuple[str, str], Counter[tuple[str, str]]] = defaultdict(Counter)
    for repetition in range(repetitions):
        if cfg.state_bootstraps > 0:
            n_take = max(8, int(math.ceil(cfg.state_bootstrap_fraction * len(frame))))
            indices = rng.choice(len(frame), size=min(n_take, len(frame)), replace=False)
            sample = frame.iloc[indices].reset_index(drop=True)
        else:
            sample = frame
        records = run_continuous_boss(
            sample,
            algorithm=cfg.state_algorithm,
            penalty=cfg.state_penalty,
            starts=cfg.state_starts,
            use_bes=cfg.state_use_bes,
            seed=cfg.seed + repetition,
        )
        for record in records:
            a, b = str(record["node1"]), str(record["node2"])
            key = canonical_edge(a, b)
            endpoints = (
                (str(record["endpoint1"]), str(record["endpoint2"]))
                if (a, b) == key
                else (str(record["endpoint2"]), str(record["endpoint1"]))
            )
            adjacency_counts[key] += 1
            endpoint_votes[key][endpoints] += 1

    correlations: dict[tuple[str, str], tuple[float, float]] = {}
    for left, right in combinations(frame.columns, 2):
        r_value, p_value = pearsonr(frame[left].to_numpy(), frame[right].to_numpy())
        correlations[canonical_edge(left, right)] = (float(r_value), float(p_value))

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for key, (r_value, p_value) in correlations.items():
        boss_support = adjacency_counts[key] / repetitions
        layers: list[str] = []
        if (
            boss_support >= cfg.state_boss_support
            and abs(r_value) >= cfg.state_boss_min_abs_r
            and p_value <= cfg.state_boss_alpha
        ):
            layers.append("state_boss")
        if (
            abs(r_value) >= cfg.state_marginal_min_abs_r
            and p_value <= cfg.state_marginal_alpha
        ):
            layers.append("state_marginal")
        if layers:
            edges[key] = {
                "layers": layers,
                "state_r": r_value,
                "state_p": p_value,
                "state_boss_support": boss_support,
                "state_endpoint_votes": dict(endpoint_votes[key]),
            }
    return edges, {
        "n_search_repetitions": repetitions,
        "n_state_boss_edges_at_support": int(
            sum(value / repetitions >= cfg.state_boss_support for value in adjacency_counts.values())
        ),
        "n_state_layer_edges": int(len(edges)),
    }


def run_discrete_boss(
    frame: pd.DataFrame,
    *,
    sample_prior: float,
    structure_prior: float,
    starts: int,
    use_bes: bool,
    seed: int,
) -> list[dict[str, str]]:
    from pytetrad.tools import TetradSearch as ts

    rng = np.random.default_rng(seed)
    columns = list(frame.columns)
    ordered = [columns[i] for i in rng.permutation(len(columns))]
    safe = {column: f"V{i}" for i, column in enumerate(ordered)}
    inverse = {value: key for key, value in safe.items()}
    data = frame[ordered].rename(columns=safe).astype(int)
    search = ts.TetradSearch(data)
    search.use_bdeu(
        sample_prior=float(sample_prior),
        structure_prior=float(structure_prior),
    )
    search.run_boss(
        num_starts=int(starts),
        use_bes=bool(use_bes),
        use_data_order=True,
        output_cpdag=True,
    )
    return _map_safe_edges(parse_tetrad_edges(str(search.get_string())), inverse)


def load_cell_table(
    path: Path,
    *,
    embeddings: pd.DataFrame | None,
    retained_cells_only: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t")
    features = [column for column in frame.columns if column.startswith("chr")]
    if embeddings is not None and retained_cells_only:
        retained = embeddings.loc[embeddings["retained_target"].astype(bool), ["cell"]]
        frame = frame.merge(retained, on="cell", how="inner")
    live = [column for column in features if frame[column].nunique(dropna=True) > 1]
    return frame[live].astype(int), {
        "path": str(path),
        "n_cells": int(len(frame)),
        "n_features": int(len(live)),
        "dropped_features": sorted(set(features) - set(live)),
    }


def within_state_profile_statistics(
    path: Path,
    *,
    embeddings: pd.DataFrame | None,
    retained_cells_only: bool,
    n_clusters: int,
    seed: int,
) -> tuple[dict[tuple[str, str], tuple[float, float]], int]:
    """Estimate pairwise similarity of peak detection profiles over microstates."""
    if embeddings is None or n_clusters < 2:
        return {}, 0
    frame = pd.read_csv(path, sep="\t")
    embedding_columns = [
        column for column in embeddings.columns if column.startswith("integration__")
    ]
    if not embedding_columns:
        return {}, 0
    embedding_frame = embeddings[["cell", "retained_target", *embedding_columns]].copy()
    if retained_cells_only:
        embedding_frame = embedding_frame.loc[embedding_frame["retained_target"].astype(bool)]
    frame = frame.merge(embedding_frame, on="cell", how="inner")
    features = [column for column in frame.columns if column.startswith("chr")]
    live = [column for column in features if frame[column].nunique(dropna=True) > 1]
    if len(live) < 2 or len(frame) < 20:
        return {}, 0
    actual_clusters = min(int(n_clusters), max(2, len(frame) // 20))
    labels = KMeans(
        n_clusters=actual_clusters,
        random_state=seed,
        n_init=10,
    ).fit_predict(frame[embedding_columns].to_numpy(dtype=float))
    rates: list[np.ndarray] = []
    depths: list[float] = []
    for cluster in range(actual_clusters):
        mask = labels == cluster
        rates.append(frame.loc[mask, live].mean(axis=0).to_numpy(dtype=float))
        depths.append(
            float(np.log1p(frame.loc[mask, "observed_background_library_size"]).mean())
        )
    rate_matrix = np.asarray(rates)
    design = np.column_stack([np.ones(actual_clusters), np.asarray(depths)])
    rate_matrix -= design @ np.linalg.lstsq(design, rate_matrix, rcond=None)[0]
    statistics: dict[tuple[str, str], tuple[float, float]] = {}
    for left_i, right_i in combinations(range(len(live)), 2):
        if (
            rate_matrix[:, left_i].std(ddof=0) <= 1e-12
            or rate_matrix[:, right_i].std(ddof=0) <= 1e-12
        ):
            continue
        r_value, p_value = pearsonr(rate_matrix[:, left_i], rate_matrix[:, right_i])
        statistics[canonical_edge(live[left_i], live[right_i])] = (
            float(r_value),
            float(p_value),
        )
    return statistics, actual_clusters


def within_assay_layer(
    cell_tables: list[Path],
    *,
    embeddings: pd.DataFrame | None,
    cfg: ChannelAwareConfig,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    table_summaries: list[dict[str, Any]] = []
    for table_i, path in enumerate(cell_tables):
        frame, table_summary = load_cell_table(
            path,
            embeddings=embeddings,
            retained_cells_only=cfg.retained_cells_only,
        )
        profile_stats, actual_profile_clusters = within_state_profile_statistics(
            path,
            embeddings=embeddings,
            retained_cells_only=cfg.retained_cells_only,
            n_clusters=cfg.within_profile_clusters,
            seed=cfg.seed + 2000 * (table_i + 1),
        )
        table_summary["within_profile_clusters"] = actual_profile_clusters
        table_summaries.append(table_summary)
        if frame.shape[1] < 2 or frame.shape[0] < 8:
            continue

        adjacency_counts: Counter[tuple[str, str]] = Counter()
        repeats = max(1, cfg.within_order_repeats)
        for repetition in range(repeats):
            records = run_discrete_boss(
                frame,
                sample_prior=cfg.within_sample_prior,
                structure_prior=cfg.within_structure_prior,
                starts=cfg.within_starts,
                use_bes=cfg.within_use_bes,
                seed=cfg.seed + 1000 * (table_i + 1) + repetition,
            )
            adjacency_counts.update(
                canonical_edge(str(record["node1"]), str(record["node2"]))
                for record in records
            )

        for left, right in combinations(frame.columns, 2):
            key = canonical_edge(left, right)
            support = adjacency_counts[key] / repeats
            values_left = frame[left].to_numpy(dtype=int)
            values_right = frame[right].to_numpy(dtype=int)
            contingency = np.asarray(
                [
                    [
                        np.sum((values_left == left_value) & (values_right == right_value))
                        for right_value in (0, 1)
                    ]
                    for left_value in (0, 1)
                ],
                dtype=int,
            )
            odds_ratio, p_value = fisher_exact(contingency)
            profile_r, profile_p = profile_stats.get(key, (float("nan"), float("nan")))
            layers: list[str] = []
            if support >= cfg.within_boss_support:
                layers.append("within_bdeu_boss")
            if (
                cfg.sparse_alpha > 0
                and np.isfinite(odds_ratio)
                and odds_ratio >= cfg.sparse_min_odds_ratio
                and p_value <= cfg.sparse_alpha
            ):
                layers.append("within_sparse_association")
            if (
                actual_profile_clusters >= 2
                and np.isfinite(profile_r)
                and abs(profile_r) >= cfg.within_profile_min_abs_r
                and profile_p <= cfg.within_profile_alpha
            ):
                layers.append("within_state_profile")
            if not layers:
                continue
            record = edges.setdefault(key, {"layers": []})
            record["layers"] = sorted(set(record["layers"]) | set(layers))
            record["within_boss_support"] = support
            record["within_odds_ratio"] = float(odds_ratio)
            record["within_p"] = float(p_value)
            record["within_n_cells"] = int(len(frame))
            record["within_profile_r"] = profile_r
            record["within_profile_p"] = profile_p
            record["within_profile_clusters"] = actual_profile_clusters
    return edges, table_summaries


def merge_layers(
    state_edges: dict[tuple[str, str], dict[str, Any]],
    within_edges: dict[tuple[str, str], dict[str, Any]],
    *,
    impose_gene_sink: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(state_edges) | set(within_edges)):
        left, right = key
        state = state_edges.get(key, {})
        within = within_edges.get(key, {})
        layers = sorted(set(state.get("layers", [])) | set(within.get("layers", [])))
        endpoint_left = "CIRCLE"
        endpoint_right = "CIRCLE"
        orientation_source = "abstained"
        if impose_gene_sink and (left.startswith("expr__") ^ right.startswith("expr__")):
            if left.startswith("expr__"):
                endpoint_left = "ARROW"
            else:
                endpoint_right = "ARROW"
            orientation_source = "gene_sink_prior"
        channels = {node_channel(left), node_channel(right)}
        cross_channel = len(channels) > 1
        interpretation = (
            "matched_state_adjacency"
            if cross_channel or not within
            else "within_assay_adjacency"
        )
        row: dict[str, Any] = {
            "node1": left,
            "endpoint1": endpoint_left,
            "node2": right,
            "endpoint2": endpoint_right,
            "edge_text": f"{left} {'o->' if endpoint_right == 'ARROW' else '<-o' if endpoint_left == 'ARROW' else 'o-o'} {right}",
            "evidence_layers": ",".join(layers),
            "interpretation": interpretation,
            "cross_channel": cross_channel,
            "orientation_source": orientation_source,
        }
        for source in (state, within):
            for field, value in source.items():
                if field not in {"layers", "state_endpoint_votes"}:
                    row[field] = value
        rows.append(row)
    return pd.DataFrame(rows)


def discover(
    metacell_matrix: Path,
    cell_tables: list[Path],
    embedding_table: Path | None,
    cfg: ChannelAwareConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed = pd.read_csv(metacell_matrix, sep="\t")
    state_data, preprocessing = prepare_state_data(
        observed,
        depth_residualize=cfg.depth_residualize,
    )
    state_edges, state_summary = state_layer(state_data, cfg)
    embeddings = pd.read_csv(embedding_table, sep="\t") if embedding_table else None
    within_edges, within_summaries = within_assay_layer(
        cell_tables,
        embeddings=embeddings,
        cfg=cfg,
    )
    edges = merge_layers(
        state_edges,
        within_edges,
        impose_gene_sink=cfg.impose_gene_sink,
    )
    summary = {
        "method": "channel_aware_multiresolution_boss",
        "config": asdict(cfg),
        "inputs": {
            "metacell_matrix": str(metacell_matrix),
            "cell_tables": [str(path) for path in cell_tables],
            "embedding_table": str(embedding_table) if embedding_table else None,
        },
        "preprocessing": preprocessing,
        "state_layer": state_summary,
        "within_assay_tables": within_summaries,
        "n_merged_edges": int(len(edges)),
        "n_cross_channel_edges": int(edges.get("cross_channel", pd.Series(dtype=bool)).sum()),
        "contract": {
            "cross_channel": "matched-state adjacency only; no learned direct orientation",
            "within_channel": "within-assay adjacency; peak-to-peak orientation abstained",
            "gene_sink": "arrowhead is imposed prior when enabled",
        },
    }
    return edges, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-dir", type=Path)
    parser.add_argument("--arm", choices=["oracle", "scglue", "gapca"], default="oracle")
    parser.add_argument("--metacell-matrix", type=Path)
    parser.add_argument("--cell-table", type=Path, action="append", default=[])
    parser.add_argument("--embedding-table", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--state-algorithm", choices=["boss", "boss_fci", "lvh"], default="boss")
    parser.add_argument("--state-penalty", type=float, default=1.0)
    parser.add_argument("--state-starts", type=int, default=20)
    parser.add_argument("--state-bootstraps", type=int, default=0)
    parser.add_argument("--state-boss-support", type=float, default=0.5)
    parser.add_argument("--state-boss-min-abs-r", type=float, default=0.20)
    parser.add_argument("--state-boss-alpha", type=float, default=0.05)
    parser.add_argument("--state-marginal-min-abs-r", type=float, default=0.20)
    parser.add_argument("--state-marginal-alpha", type=float, default=0.05)
    parser.add_argument("--within-sample-prior", type=float, default=10.0)
    parser.add_argument("--within-starts", type=int, default=20)
    parser.add_argument("--within-order-repeats", type=int, default=1)
    parser.add_argument("--within-boss-support", type=float, default=0.5)
    parser.add_argument("--sparse-min-odds-ratio", type=float, default=1.5)
    parser.add_argument("--sparse-alpha", type=float, default=0.20)
    parser.add_argument("--within-profile-clusters", type=int, default=0)
    parser.add_argument("--within-profile-min-abs-r", type=float, default=0.60)
    parser.add_argument("--within-profile-alpha", type=float, default=0.05)
    parser.add_argument("--retained-cells-only", action="store_true")
    parser.add_argument("--no-depth-residualize", action="store_true")
    parser.add_argument("--no-bes", action="store_true")
    parser.add_argument("--no-gene-sink", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metacell_matrix = args.metacell_matrix
    cell_tables = list(args.cell_table)
    embedding_table = args.embedding_table
    if args.simulation_dir:
        metacell_matrix = metacell_matrix or (
            args.simulation_dir / f"{args.arm}_observed_metacell_matrix.tsv"
        )
        if not cell_tables:
            cell_tables = sorted(args.simulation_dir.glob("observed_cells__chrom_*.tsv"))
        embedding_table = embedding_table or (
            args.simulation_dir / f"{args.arm}_analysis_embeddings.tsv"
        )
    if metacell_matrix is None:
        raise SystemExit("Provide --metacell-matrix or --simulation-dir")
    if embedding_table is not None and not embedding_table.exists():
        embedding_table = None
    missing = [path for path in [metacell_matrix, *cell_tables] if not path.exists()]
    if missing:
        raise SystemExit(f"Missing input files: {missing}")

    cfg = ChannelAwareConfig(
        state_algorithm=args.state_algorithm,
        state_penalty=args.state_penalty,
        state_starts=args.state_starts,
        state_use_bes=not args.no_bes,
        state_bootstraps=args.state_bootstraps,
        state_boss_support=args.state_boss_support,
        state_boss_min_abs_r=args.state_boss_min_abs_r,
        state_boss_alpha=args.state_boss_alpha,
        state_marginal_min_abs_r=args.state_marginal_min_abs_r,
        state_marginal_alpha=args.state_marginal_alpha,
        depth_residualize=not args.no_depth_residualize,
        within_sample_prior=args.within_sample_prior,
        within_starts=args.within_starts,
        within_use_bes=not args.no_bes,
        within_order_repeats=args.within_order_repeats,
        within_boss_support=args.within_boss_support,
        sparse_min_odds_ratio=args.sparse_min_odds_ratio,
        sparse_alpha=args.sparse_alpha,
        within_profile_clusters=args.within_profile_clusters,
        within_profile_min_abs_r=args.within_profile_min_abs_r,
        within_profile_alpha=args.within_profile_alpha,
        retained_cells_only=args.retained_cells_only,
        impose_gene_sink=not args.no_gene_sink,
        seed=args.seed,
    )
    edges, summary = discover(metacell_matrix, cell_tables, embedding_table, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(args.out_dir / "channel_aware_edges.tsv", sep="\t", index=False)
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(
        f"[channel-aware-boss] {metacell_matrix.name}: "
        f"{summary['preprocessing']['n_rows']} rows -> {len(edges)} typed adjacencies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
