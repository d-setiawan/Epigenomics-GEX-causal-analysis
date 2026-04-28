#!/usr/bin/env python3
"""Single entrypoint for the current CausalDiscovery workflow."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_causal_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def sanitize_token(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def window_suffix(window_bp: int) -> str:
    if window_bp == 10000:
        return ""
    if window_bp % 1000 == 0:
        return f"_{window_bp // 1000}kb"
    return f"_{window_bp}bp"


def default_gene_panel(project_root: Path) -> Path:
    return project_root / "CausalDiscovery" / "configs" / "gene_panels" / "monocyte_cuttag_peak_genes.tsv"


def default_matching_dir(project_root: Path, run_id: str) -> Path:
    return (
        project_root
        / "CausalDiscovery"
        / "outputs"
        / "scglue_pairings"
        / sanitize_token(run_id)
        / "harmonized_coarse__monocyte__rna_anchor"
    )


def default_dataset_root(project_root: Path, run_id: str, gene_panel: Path, window_bp: int) -> Path:
    panel_token = sanitize_token(gene_panel.stem) + window_suffix(window_bp)
    return project_root / "CausalDiscovery" / "outputs" / "datasets" / sanitize_token(run_id) / panel_token


def commands_dir() -> Path:
    return Path(__file__).resolve().parent / "commands"


def run_python(pybin: str, script_name: str, args: list[str]) -> None:
    script_path = commands_dir() / script_name
    cmd = [pybin, str(script_path), *args]
    print(f"[causal-cli] {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def optional_arg(args: list[str], flag: str, value) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def load_panel_genes(panel_path: Path) -> list[str]:
    with panel_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        genes = [row["gene"].strip() for row in reader if row.get("gene", "").strip()]
    if not genes:
        raise ValueError(f"No genes found in panel: {panel_path}")
    return genes


def build_match_args(project_root: Path, args) -> list[str]:
    out: list[str] = [
        "--repo-root",
        str(project_root),
        "--run-id",
        args.run_id,
        "--cell-type",
        args.cell_type,
        "--label-column",
        args.label_column,
        "--min-label-confidence",
        str(args.min_label_confidence),
        "--anchor-modality",
        args.anchor_modality,
        "--marks",
        args.marks,
    ]
    optional_arg(out, "--train-dir", args.train_dir)
    optional_arg(out, "--label-tsv", args.label_tsv)
    optional_arg(out, "--out-dir", args.out_dir)
    optional_arg(out, "--n-pairs", args.n_pairs)
    return out


def build_dataset_args(project_root: Path, args) -> list[str]:
    out: list[str] = [
        "--repo-root",
        str(project_root),
        "--run-id",
        args.run_id,
        "--gene-panel",
        str(args.gene_panel),
        "--window-bp",
        str(args.window_bp),
        "--marks",
        args.marks,
        "--target-sum",
        str(args.target_sum),
        "--quant-mode",
        args.quant_mode,
    ]
    optional_arg(out, "--matching-dir", args.matching_dir)
    optional_arg(out, "--manifest", args.manifest)
    optional_arg(out, "--rna-h5ad", args.rna_h5ad)
    optional_arg(out, "--out-root", args.out_root)
    return out


def build_graph_args(project_root: Path, args) -> list[str]:
    out: list[str] = [
        "--repo-root",
        str(project_root),
        "--matrix-tsv",
        str(args.matrix_tsv),
        "--method",
        args.method,
        "--columns",
        args.columns,
        "--alpha",
        str(args.alpha),
        "--indep-test",
        args.indep_test,
        "--transform",
        args.transform,
        "--background-mode",
        args.background_mode,
        "--min-unique",
        str(args.min_unique),
        "--min-std",
        str(args.min_std),
        "--max-depth",
        str(args.max_depth),
        "--plot-layout",
        args.plot_layout,
        "--plot-max-edges",
        str(args.plot_max_edges),
        "--plot-peaks-per-row",
        str(args.plot_peaks_per_row),
        "--plot-layout-seed",
        str(args.plot_layout_seed),
        "--dagma-loss-type",
        getattr(args, "dagma_loss_type", "l2"),
        "--dagma-family-config",
        getattr(args, "dagma_family_config", "gaussian_gaussian"),
        "--dagma-lambda1",
        str(getattr(args, "dagma_lambda1", 0.03)),
        "--dagma-w-threshold",
        str(getattr(args, "dagma_w_threshold", 0.3)),
        "--dagma-T",
        str(getattr(args, "dagma_T", 5)),
        "--dagma-peak-binary-threshold",
        str(getattr(args, "dagma_peak_binary_threshold", 0.0)),
    ]
    if args.stable:
        out.append("--stable")
    else:
        out.append("--unstable")
    if args.standardize:
        out.append("--standardize")
    else:
        out.append("--no-standardize")
    if args.dropna_rows:
        out.append("--dropna-rows")
    else:
        out.append("--keep-na-rows")
    if args.plot:
        out.append("--plot")
    else:
        out.append("--no-plot")
    if getattr(args, "dagma_verbose", False):
        out.append("--dagma-verbose")
    optional_arg(out, "--out-dir", args.out_dir)
    optional_arg(out, "--plot-title", args.plot_title)
    optional_arg(out, "--plot-out-prefix", args.plot_out_prefix)
    return out


def run_sweep(project_root: Path, args) -> None:
    genes = load_panel_genes(args.gene_panel)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unsupported = [method for method in methods if method not in {"pc", "fci"}]
    if unsupported:
        raise SystemExit(
            "DAGMA is graph-only in v1 and is not part of the depth-based sweep. "
            f"Unsupported sweep methods: {unsupported}"
        )
    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    total_runs = len(genes) * len(methods) * len(depths)
    completed = 0
    sweep_start = time.monotonic()

    print(
        "[causal-cli] sweep starting: "
        f"{len(genes)} genes x {len(methods)} methods x {len(depths)} depths = {total_runs} runs",
        flush=True,
    )

    for gene in genes:
        matrix_path = args.out_root / sanitize_token(gene) / "nearby_peaks" / f"{gene}_nearby_peak_matrix.tsv"
        if not matrix_path.exists():
            raise FileNotFoundError(
                "Missing nearby peak matrix for sweep: "
                f"{matrix_path}\n"
                "Run dataset generation first, for example:\n"
                f"python3 {infer_causal_dir() / 'scripts' / 'causal_cli.py'} dataset "
                f"--run-id {args.run_id} --gene-panel {args.gene_panel} --window-bp {args.window_bp}"
            )
        for method in methods:
            for depth in depths:
                completed += 1
                run_start = time.monotonic()
                print(
                    "[causal-cli] "
                    f"run {completed}/{total_runs}: gene={gene} method={method} depth={depth}",
                    flush=True,
                )
                graph_ns = argparse.Namespace(
                    matrix_tsv=matrix_path,
                    out_dir=None,
                    method=method,
                    columns="auto",
                    alpha=args.alpha,
                    indep_test=args.indep_test,
                    stable=args.stable,
                    transform=args.transform,
                    standardize=args.standardize,
                    background_mode=args.background_mode,
                    min_unique=args.min_unique,
                    min_std=args.min_std,
                    max_depth=depth,
                    dropna_rows=args.dropna_rows,
                    plot=args.plot,
                    plot_layout=args.plot_layout,
                    plot_max_edges=args.plot_max_edges,
                    plot_peaks_per_row=args.plot_peaks_per_row,
                    plot_layout_seed=args.plot_layout_seed,
                    plot_title=None,
                    plot_out_prefix=None,
                )
                run_python(args.python_bin, "run_pc_causallearn.py", build_graph_args(project_root, graph_ns))
                run_elapsed = time.monotonic() - run_start
                total_elapsed = time.monotonic() - sweep_start
                avg_elapsed = total_elapsed / completed
                remaining = total_runs - completed
                eta_seconds = avg_elapsed * remaining
                print(
                    "[causal-cli] "
                    f"completed {completed}/{total_runs} "
                    f"(last {run_elapsed:.1f}s, total {total_elapsed/60:.1f}m, eta {eta_seconds/60:.1f}m)",
                    flush=True,
                )

    total_elapsed = time.monotonic() - sweep_start
    print(
        "[causal-cli] sweep finished: "
        f"{total_runs} runs in {total_elapsed/60:.1f} minutes",
        flush=True,
    )


def run_dagma_sweep(project_root: Path, args) -> None:
    genes = load_panel_genes(args.gene_panel)
    family_configs = [config.strip() for config in args.family_configs.split(",") if config.strip()]
    lambda1_values = [float(v.strip()) for v in args.lambda1_values.split(",") if v.strip()]
    w_threshold_values = [float(v.strip()) for v in args.w_threshold_values.split(",") if v.strip()]
    T_values = [int(v.strip()) for v in args.T_values.split(",") if v.strip()]

    total_runs = len(genes) * len(family_configs) * len(lambda1_values) * len(w_threshold_values) * len(T_values)
    completed = 0
    sweep_start = time.monotonic()

    print(
        "[causal-cli] dagma sweep starting: "
        f"{len(genes)} genes x {len(family_configs)} family configs x "
        f"{len(lambda1_values)} lambda1 values x {len(w_threshold_values)} thresholds x {len(T_values)} T values = {total_runs} runs",
        flush=True,
    )

    for gene in genes:
        matrix_path = args.out_root / sanitize_token(gene) / "nearby_peaks" / f"{gene}_nearby_peak_matrix.tsv"
        if not matrix_path.exists():
            raise FileNotFoundError(
                "Missing nearby peak matrix for DAGMA sweep: "
                f"{matrix_path}\n"
                "Run dataset generation first, for example:\n"
                f"python3 {infer_causal_dir() / 'scripts' / 'causal_cli.py'} dataset "
                f"--run-id {args.run_id} --gene-panel {args.gene_panel} --window-bp {args.window_bp}"
            )

        for family_config in family_configs:
            for lambda1 in lambda1_values:
                for w_threshold in w_threshold_values:
                    for dagma_T in T_values:
                        completed += 1
                        run_start = time.monotonic()
                        print(
                            "[causal-cli] "
                            f"run {completed}/{total_runs}: gene={gene} family={family_config} "
                            f"lambda1={lambda1} w_threshold={w_threshold} T={dagma_T}",
                            flush=True,
                        )
                        graph_ns = argparse.Namespace(
                            matrix_tsv=matrix_path,
                            out_dir=None,
                            method="dagma",
                            columns="auto",
                            alpha=args.alpha,
                            indep_test=args.indep_test,
                            stable=args.stable,
                            transform=args.transform,
                            standardize=args.standardize,
                            background_mode=args.background_mode,
                            min_unique=args.min_unique,
                            min_std=args.min_std,
                            max_depth=-1,
                            dropna_rows=args.dropna_rows,
                            plot=args.plot,
                            plot_layout=args.plot_layout,
                            plot_max_edges=args.plot_max_edges,
                            plot_peaks_per_row=args.plot_peaks_per_row,
                            plot_layout_seed=args.plot_layout_seed,
                            plot_title=None,
                            plot_out_prefix=None,
                            dagma_loss_type=args.dagma_loss_type,
                            dagma_family_config=family_config,
                            dagma_lambda1=lambda1,
                            dagma_w_threshold=w_threshold,
                            dagma_T=dagma_T,
                            dagma_peak_binary_threshold=args.dagma_peak_binary_threshold,
                            dagma_verbose=args.dagma_verbose,
                        )
                        run_python(args.python_bin, "run_pc_causallearn.py", build_graph_args(project_root, graph_ns))
                        run_elapsed = time.monotonic() - run_start
                        total_elapsed = time.monotonic() - sweep_start
                        avg_elapsed = total_elapsed / completed
                        remaining = total_runs - completed
                        eta_seconds = avg_elapsed * remaining
                        print(
                            "[causal-cli] "
                            f"completed {completed}/{total_runs} "
                            f"(last {run_elapsed:.1f}s, total {total_elapsed/60:.1f}m, eta {eta_seconds/60:.1f}m)",
                            flush=True,
                        )

    total_elapsed = time.monotonic() - sweep_start
    print(
        "[causal-cli] dagma sweep finished: "
        f"{total_runs} runs in {total_elapsed/60:.1f} minutes",
        flush=True,
    )


def add_shared_graph_args(p, *, include_method: bool = True, include_max_depth: bool = True) -> None:
    if include_method:
        p.add_argument("--method", default="pc", choices=["pc", "fci", "dagma"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--indep-test", default="kci")
    p.add_argument("--stable", action="store_true", default=True)
    p.add_argument("--unstable", action="store_false", dest="stable")
    p.add_argument("--transform", default="none", choices=["none", "rank_gaussian"])
    p.add_argument("--standardize", action="store_true", default=True)
    p.add_argument("--no-standardize", action="store_false", dest="standardize")
    p.add_argument("--background-mode", default="tiered_distal_promoter_expr", choices=["none", "minimal_expr_sink", "tiered_distal_promoter_expr"])
    p.add_argument("--min-unique", type=int, default=2)
    p.add_argument("--min-std", type=float, default=1e-8)
    if include_max_depth:
        p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--dropna-rows", action="store_true", default=True)
    p.add_argument("--keep-na-rows", action="store_false", dest="dropna_rows")
    p.add_argument("--plot", action="store_true", default=True)
    p.add_argument("--no-plot", action="store_false", dest="plot")
    p.add_argument("--plot-layout", default="auto", choices=["auto", "local", "spring"])
    p.add_argument("--plot-max-edges", type=int, default=30)
    p.add_argument("--plot-peaks-per-row", type=int, default=18)
    p.add_argument("--plot-layout-seed", type=int, default=13)
    p.add_argument("--plot-title", default=None)
    p.add_argument("--plot-out-prefix", default=None)


def add_dagma_args(p, *, include_family_config: bool = True, include_search_params: bool = True) -> None:
    p.add_argument("--dagma-loss-type", default="l2", choices=["l2", "logistic"])
    if include_family_config:
        p.add_argument(
            "--dagma-family-config",
            "--dagma-family-preset",
            dest="dagma_family_config",
            default="gaussian_gaussian",
            choices=["gaussian_gaussian", "bernoulli_peak_nb_gene", "gaussian_nb", "nb_nb"],
        )
    if include_search_params:
        p.add_argument("--dagma-lambda1", type=float, default=0.03)
        p.add_argument("--dagma-w-threshold", type=float, default=0.3)
        p.add_argument("--dagma-T", type=int, default=5)
    p.add_argument("--dagma-peak-binary-threshold", type=float, default=0.0)
    p.add_argument("--dagma-verbose", action="store_true", default=False)


def main() -> int:
    project_root = infer_project_root()
    causal_dir = infer_causal_dir()
    parser = argparse.ArgumentParser(
        description="Single entrypoint for the active CausalDiscovery workflow. Start here.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    sub = parser.add_subparsers(dest="command", required=True)

    p_match = sub.add_parser("match", help="Build one-to-one monocyte pseudo-pairs from scGLUE.")
    p_match.add_argument("--run-id", default="joint_v2")
    p_match.add_argument("--train-dir", default=None)
    p_match.add_argument("--label-tsv", default=None)
    p_match.add_argument("--out-dir", default=None)
    p_match.add_argument("--label-column", default="harmonized_coarse")
    p_match.add_argument("--cell-type", default="monocyte")
    p_match.add_argument("--min-label-confidence", type=float, default=0.5)
    p_match.add_argument("--anchor-modality", default="rna")
    p_match.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p_match.add_argument("--n-pairs", type=int, default=None)

    p_dataset = sub.add_parser("dataset", help="Generate nearby CUT&Tag peak matrices for a gene panel.")
    p_dataset.add_argument("--run-id", default="joint_v2")
    p_dataset.add_argument("--matching-dir", default=None)
    p_dataset.add_argument("--gene-panel", type=Path, default=default_gene_panel(project_root))
    p_dataset.add_argument("--manifest", default="integration/manifests/scglue_input_manifest.tsv")
    p_dataset.add_argument("--rna-h5ad", default=None)
    p_dataset.add_argument("--window-bp", type=int, default=10000)
    p_dataset.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p_dataset.add_argument("--target-sum", type=float, default=1e4)
    p_dataset.add_argument("--quant-mode", default="log1p_norm", choices=["log1p_norm", "raw_counts"])
    p_dataset.add_argument("--out-root", type=Path, default=None)

    p_graph = sub.add_parser("graph", help="Run one PC, FCI, or DAGMA graph discovery job on a matrix.")
    p_graph.add_argument("--matrix-tsv", type=Path, required=True)
    p_graph.add_argument("--out-dir", default=None)
    p_graph.add_argument("--columns", default="auto")
    add_shared_graph_args(p_graph)
    add_dagma_args(p_graph)

    p_sweep = sub.add_parser("sweep", help="Run PC/FCI across an existing nearby-peak dataset root.")
    p_sweep.add_argument("--run-id", default="joint_v2")
    p_sweep.add_argument("--gene-panel", type=Path, default=default_gene_panel(project_root))
    p_sweep.add_argument("--window-bp", type=int, default=300000)
    p_sweep.add_argument("--out-root", "--dataset-root", dest="out_root", type=Path, default=None)
    p_sweep.add_argument("--methods", default="pc,fci")
    p_sweep.add_argument("--depths", default="1,2,3,4,5")
    add_shared_graph_args(p_sweep, include_method=False, include_max_depth=False)

    p_dagma_sweep = sub.add_parser("dagma-sweep", help="Run DAGMA across an existing nearby-peak dataset root.")
    p_dagma_sweep.add_argument("--run-id", default="joint_v2")
    p_dagma_sweep.add_argument("--gene-panel", type=Path, default=default_gene_panel(project_root))
    p_dagma_sweep.add_argument("--window-bp", type=int, default=300000)
    p_dagma_sweep.add_argument("--out-root", "--dataset-root", dest="out_root", type=Path, default=None)
    p_dagma_sweep.add_argument(
        "--family-configs",
        "--family-presets",
        dest="family_configs",
        default="gaussian_gaussian,bernoulli_peak_nb_gene,gaussian_nb,nb_nb",
        help="Comma-separated DAGMA family configs to evaluate.",
    )
    p_dagma_sweep.add_argument("--lambda1-values", default="0.03")
    p_dagma_sweep.add_argument("--w-threshold-values", default="0.3")
    p_dagma_sweep.add_argument("--T-values", default="5")
    add_shared_graph_args(p_dagma_sweep, include_method=False, include_max_depth=False)
    add_dagma_args(p_dagma_sweep, include_family_config=False, include_search_params=False)

    p_plot = sub.add_parser("plot", help="Plot a saved graph directory.")
    p_plot.add_argument("--graph-dir", type=Path, required=True)
    p_plot.add_argument("--out-prefix", default=None)
    p_plot.add_argument("--node-support-tsv", default=None)
    p_plot.add_argument("--layout-seed", type=int, default=13)
    p_plot.add_argument("--title", default=None)
    p_plot.add_argument("--layout", default="auto", choices=["auto", "local", "spring"])
    p_plot.add_argument("--max-edges", type=int, default=30)
    p_plot.add_argument("--peaks-per-row", type=int, default=18)

    p_support = sub.add_parser("support", help="Build node support annotations for a graph directory.")
    p_support.add_argument("--graph-dir", type=Path, required=True)
    p_support.add_argument("--locus-config", default=None)
    p_support.add_argument("--out-tsv", default=None)
    p_support.add_argument("--matches-tsv", default=None)
    p_support.add_argument("--eperturbdb-tsv", default=None)
    p_support.add_argument("--encode-screen-tsv", default=None)

    args = parser.parse_args()

    if args.command == "match":
        run_python(args.python_bin, "build_scglue_one_to_one_matches.py", build_match_args(project_root, args))
        return 0

    if args.command == "dataset":
        if args.out_root is None:
            args.out_root = default_dataset_root(project_root, args.run_id, args.gene_panel, args.window_bp)
        run_python(args.python_bin, "generate_monocyte_cuttag_peak_datasets.py", build_dataset_args(project_root, args))
        return 0

    if args.command == "graph":
        run_python(args.python_bin, "run_pc_causallearn.py", build_graph_args(project_root, args))
        return 0

    if args.command == "sweep":
        if args.out_root is None:
            args.out_root = default_dataset_root(project_root, args.run_id, args.gene_panel, args.window_bp)
        run_sweep(project_root, args)
        return 0

    if args.command == "dagma-sweep":
        if args.out_root is None:
            args.out_root = default_dataset_root(project_root, args.run_id, args.gene_panel, args.window_bp)
        run_dagma_sweep(project_root, args)
        return 0

    if args.command == "plot":
        plot_args = [
            "--repo-root",
            str(project_root),
            "--graph-dir",
            str(args.graph_dir),
            "--layout-seed",
            str(args.layout_seed),
            "--layout",
            args.layout,
            "--max-edges",
            str(args.max_edges),
            "--peaks-per-row",
            str(args.peaks_per_row),
        ]
        optional_arg(plot_args, "--out-prefix", args.out_prefix)
        optional_arg(plot_args, "--node-support-tsv", args.node_support_tsv)
        optional_arg(plot_args, "--title", args.title)
        run_python(args.python_bin, "plot_pc_graph.py", plot_args)
        return 0

    if args.command == "support":
        support_args = [
            "--repo-root",
            str(project_root),
            "--pc-dir",
            str(args.graph_dir),
        ]
        optional_arg(support_args, "--locus-config", args.locus_config)
        optional_arg(support_args, "--out-tsv", args.out_tsv)
        optional_arg(support_args, "--matches-tsv", args.matches_tsv)
        optional_arg(support_args, "--eperturbdb-tsv", args.eperturbdb_tsv)
        optional_arg(support_args, "--encode-screen-tsv", args.encode_screen_tsv)
        run_python(args.python_bin, "build_node_support_table.py", support_args)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
