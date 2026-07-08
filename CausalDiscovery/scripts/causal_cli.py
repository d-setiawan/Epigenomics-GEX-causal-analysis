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


def default_elastic_net_out_root(causal_dir: Path, top_k: int, graph_split_fraction: float = 0.0) -> Path:
    name = f"monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10_elasticnet_top{int(top_k)}_crossfit"
    if graph_split_fraction and graph_split_fraction > 0.0:
        name += f"_graphsplit{int(round(graph_split_fraction * 100))}"
    return causal_dir / "outputs" / "datasets" / "joint_v2" / name


def resolve_project_path(project_root: Path, value) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def default_graph_out_dir_name(args) -> str:
    if args.method == "dagma":
        return (
            f"dagma_{sanitize_token(args.dagma_family_config)}"
            f"_lambda1_{sanitize_token(str(args.dagma_lambda1))}"
            f"_wthr_{sanitize_token(str(args.dagma_w_threshold))}"
            f"_T_{sanitize_token(str(args.dagma_T))}"
            + ("" if args.transform == "none" else f"_xform_{sanitize_token(args.transform)}")
            + ("" if args.background_mode == "none" else f"_bg_{sanitize_token(args.background_mode)}")
        )
    return (
        f"{sanitize_token(args.method)}_{sanitize_token(args.indep_test)}_alpha_{sanitize_token(str(args.alpha))}"
        + ("" if args.max_depth < 0 else f"_depth_{args.max_depth}")
        + ("" if args.transform == "none" else f"_xform_{sanitize_token(args.transform)}")
        + ("" if args.background_mode == "none" else f"_bg_{sanitize_token(args.background_mode)}")
    )


def infer_graph_output_dir(project_root: Path, args) -> Path:
    explicit = resolve_project_path(project_root, args.out_dir)
    if explicit is not None:
        return explicit

    matrix_path = resolve_project_path(project_root, args.matrix_tsv)
    assert matrix_path is not None
    outputs_root = project_root / "CausalDiscovery" / "outputs"
    methods_root = outputs_root / "methods" / sanitize_token(args.method)
    run_dir_name = default_graph_out_dir_name(args)
    try:
        rel = matrix_path.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        return matrix_path.parent / run_dir_name

    parts = rel.parts
    if len(parts) >= 5 and parts[0] == "datasets":
        run_id = parts[1]
        dataset_name = parts[2]
        gene_name = parts[3]
        dataset_id = sanitize_token(f"{run_id}__{dataset_name}")
        if sanitize_token(args.method) == "dagma":
            return methods_root / dataset_id / sanitize_token(args.dagma_family_config) / gene_name / run_dir_name
        return methods_root / dataset_id / gene_name / run_dir_name

    if len(parts) >= 4 and parts[0] in {"locus_matrices_matched", "locus_matrices"}:
        dataset_family = parts[0]
        run_id = parts[1]
        locus_name = parts[2]
        dataset_id = sanitize_token(f"{run_id}__{dataset_family}")
        if sanitize_token(args.method) == "dagma":
            return methods_root / dataset_id / sanitize_token(args.dagma_family_config) / locus_name / run_dir_name
        return methods_root / dataset_id / locus_name / run_dir_name

    return matrix_path.parent / run_dir_name


def commands_dir() -> Path:
    return Path(__file__).resolve().parent / "commands"


def run_python(pybin: str, script_name: str, args: list[str]) -> None:
    if not str(pybin).strip():
        raise SystemExit(
            "--python-bin resolved to an empty string. "
            "If you used --python-bin \"$PYTHON_BIN\", set it first, for example: "
            "export PYTHON_BIN=/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python"
        )
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


def build_match_average_args(project_root: Path, args) -> list[str]:
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
        "--n-neighbors",
        str(args.n_neighbors),
        "--neighbor-weight-mode",
        args.neighbor_weight_mode,
    ]
    optional_arg(out, "--train-dir", args.train_dir)
    optional_arg(out, "--label-tsv", args.label_tsv)
    optional_arg(out, "--out-dir", args.out_dir)
    optional_arg(out, "--max-anchors", args.max_anchors)
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


def build_filter_dataset_args(project_root: Path, args) -> list[str]:
    return [
        "--repo-root",
        str(project_root),
        "--dataset-root",
        str(args.dataset_root),
        "--out-root",
        str(args.out_root),
        "--gene-panel",
        str(args.gene_panel),
        "--min-nonzero-fraction",
        str(args.min_nonzero_fraction),
    ]


def build_elastic_net_screen_args(project_root: Path, args) -> list[str]:
    out: list[str] = [
        "--repo-root",
        str(project_root),
        "--dataset-root",
        str(args.dataset_root),
        "--out-root",
        str(args.out_root),
        "--gene-panel",
        str(args.gene_panel),
        "--top-k",
        str(args.top_k),
        "--n-splits",
        str(args.n_splits),
        "--n-repeats",
        str(args.n_repeats),
        "--random-state",
        str(args.random_state),
        "--l1-ratios",
        args.l1_ratios,
        "--alpha-min-exp",
        str(args.alpha_min_exp),
        "--alpha-max-exp",
        str(args.alpha_max_exp),
        "--n-alphas",
        str(args.n_alphas),
        "--max-iter",
        str(args.max_iter),
        "--tol",
        str(args.tol),
        "--coef-tol",
        str(args.coef_tol),
        "--graph-split-fraction",
        str(args.graph_split_fraction),
        "--split-random-state",
        str(args.split_random_state),
        "--min-graph-rows",
        str(args.min_graph_rows),
    ]
    optional_arg(out, "--report-root", args.report_root)
    optional_arg(out, "--genes", args.genes)
    return out


def build_organize_args(project_root: Path, args) -> list[str]:
    out: list[str] = [
        "--repo-root",
        str(project_root),
    ]
    if args.dry_run:
        out.append("--dry-run")
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
        "--fastkci-k",
        str(getattr(args, "fastkci_k", 3)),
        "--fastkci-j",
        str(getattr(args, "fastkci_j", 4)),
        "--fastkci-alpha-param",
        str(getattr(args, "fastkci_alpha_param", 500.0)),
        "--fastkci-fallback",
        getattr(args, "fastkci_fallback", "rcit"),
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
        "--fci-possible-dsep-depth",
        str(getattr(args, "fci_possible_dsep_depth", -2)),
        "--fci-max-path-length",
        str(getattr(args, "fci_max_path_length", -2)),
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
    if getattr(args, "fastkci_use_gp", False):
        out.append("--fastkci-use-gp")
    optional_arg(out, "--out-dir", args.out_dir)
    optional_arg(out, "--plot-title", args.plot_title)
    optional_arg(out, "--plot-out-prefix", args.plot_out_prefix)
    return out


def build_support_command_args(project_root: Path, graph_dir: Path, args, *, include_output_paths: bool = False) -> list[str]:
    out = [
        "--repo-root",
        str(project_root),
        "--pc-dir",
        str(graph_dir),
    ]
    optional_arg(out, "--locus-config", getattr(args, "locus_config", None))
    if include_output_paths:
        optional_arg(out, "--out-tsv", getattr(args, "out_tsv", None))
        optional_arg(out, "--matches-tsv", getattr(args, "matches_tsv", None))
        optional_arg(out, "--edge-support-tsv", getattr(args, "edge_support_tsv", None))
        optional_arg(out, "--report-md", getattr(args, "report_md", None))
    optional_arg(out, "--eperturbdb-tsv", getattr(args, "eperturbdb_tsv", None))
    optional_arg(out, "--eperturbdb-dir", getattr(args, "eperturbdb_dir", None))
    optional_arg(out, "--eperturbdb-database-tsv", getattr(args, "eperturbdb_database_tsv", None))
    optional_arg(out, "--eperturbdb-enhancer-gene-bed", getattr(args, "eperturbdb_enhancer_gene_bed", None))
    optional_arg(out, "--eperturbdb-tf-overlap-bed", getattr(args, "eperturbdb_tf_overlap_bed", None))
    optional_arg(out, "--max-eperturbdb-records", getattr(args, "max_eperturbdb_records", None))
    optional_arg(out, "--encode-screen-tsv", getattr(args, "encode_screen_tsv", None))
    optional_arg(out, "--eperturbdb-url", getattr(args, "eperturbdb_url", None))
    optional_arg(out, "--external-cache-dir", getattr(args, "external_cache_dir", None))
    optional_arg(out, "--min-overlap-bp", getattr(args, "min_overlap_bp", None))
    optional_arg(out, "--max-online-records", getattr(args, "max_online_records", None))
    optional_arg(out, "--encode-assembly", getattr(args, "encode_assembly", None))
    optional_arg(out, "--max-encode-files", getattr(args, "max_encode_files", None))
    optional_arg(out, "--jaspar-collection", getattr(args, "jaspar_collection", None))
    optional_arg(out, "--jaspar-tax-group", getattr(args, "jaspar_tax_group", None))
    optional_arg(out, "--jaspar-genome", getattr(args, "jaspar_genome", None))
    optional_arg(out, "--max-jaspar-motifs", getattr(args, "max_jaspar_motifs", None))
    optional_arg(out, "--jaspar-min-score", getattr(args, "jaspar_min_score", None))
    optional_arg(out, "--jaspar-max-peak-bp", getattr(args, "jaspar_max_peak_bp", None))
    for url in getattr(args, "encode_screen_url", []):
        out.extend(["--encode-screen-url", url])
    for term in getattr(args, "encode_biosample_term", []):
        out.extend(["--encode-biosample-term", term])
    for term in getattr(args, "encode_assay_term", []):
        out.extend(["--encode-assay-term", term])
    for matrix_id in getattr(args, "jaspar_matrix_id", []):
        out.extend(["--jaspar-matrix-id", matrix_id])
    for url in getattr(args, "external_support_url", []):
        out.extend(["--external-support-url", url])
    if getattr(args, "fetch_encode_screen", False):
        out.append("--fetch-encode-screen")
    if getattr(args, "fetch_encode_pbmc", False):
        out.append("--fetch-encode-pbmc")
    if getattr(args, "fetch_jaspar", False):
        out.append("--fetch-jaspar")
    return out


def build_plot_command_args(project_root: Path, graph_dir: Path, args) -> list[str]:
    out = [
        "--repo-root",
        str(project_root),
        "--graph-dir",
        str(graph_dir),
        "--layout-seed",
        str(getattr(args, "layout_seed", getattr(args, "plot_layout_seed", 13))),
        "--layout",
        getattr(args, "layout", getattr(args, "plot_layout", "auto")),
        "--max-edges",
        str(getattr(args, "max_edges", getattr(args, "plot_max_edges", 30))),
        "--peaks-per-row",
        str(getattr(args, "peaks_per_row", getattr(args, "plot_peaks_per_row", 18))),
    ]
    optional_arg(out, "--out-prefix", getattr(args, "out_prefix", getattr(args, "plot_out_prefix", None)))
    optional_arg(out, "--node-support-tsv", getattr(args, "node_support_tsv", None))
    optional_arg(out, "--edge-support-tsv", getattr(args, "edge_support_tsv", None))
    optional_arg(out, "--title", getattr(args, "title", getattr(args, "plot_title", None)))
    return out


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def apply_support_preset(args) -> None:
    preset = getattr(args, "support_preset", "none")
    if preset in {None, "none"}:
        return

    if preset == "pbmc_encode":
        args.fetch_encode_pbmc = True
        args.fetch_encode_screen = True
        append_unique(args.encode_biosample_term, "peripheral blood mononuclear cell")
        return

    if preset == "monocyte_encode":
        args.fetch_encode_pbmc = True
        args.fetch_encode_screen = True
        append_unique(args.encode_biosample_term, "CD14-positive monocyte")
        append_unique(args.encode_assay_term, "DNase-seq")
        return

    if preset == "monocyte_regulatory":
        args.fetch_encode_pbmc = True
        args.fetch_encode_screen = True
        args.fetch_jaspar = True
        append_unique(args.encode_biosample_term, "CD14-positive monocyte")
        append_unique(args.encode_assay_term, "DNase-seq")
        append_unique(args.jaspar_matrix_id, "MA0139.1")
        if float(args.jaspar_min_score) == 0.85:
            args.jaspar_min_score = 0.75
        return

    if preset == "local_eperturbdb":
        if not getattr(args, "eperturbdb_dir", None):
            args.eperturbdb_dir = "Data/external/eperturbdb"
        return

    raise SystemExit(f"Unsupported support preset: {preset}")


def has_external_support_source(args) -> bool:
    return any(
        [
            bool(getattr(args, "fetch_encode_screen", False)),
            bool(getattr(args, "fetch_encode_pbmc", False)),
            bool(getattr(args, "fetch_jaspar", False)),
            bool(getattr(args, "eperturbdb_url", None)),
            bool(getattr(args, "eperturbdb_tsv", None)),
            bool(getattr(args, "eperturbdb_dir", None)),
            bool(getattr(args, "eperturbdb_database_tsv", None)),
            bool(getattr(args, "eperturbdb_enhancer_gene_bed", None)),
            bool(getattr(args, "eperturbdb_tf_overlap_bed", None)),
            bool(getattr(args, "encode_screen_tsv", None)),
            bool(getattr(args, "external_support_url", [])),
        ]
    )


def require_external_support_source(args) -> None:
    if not has_external_support_source(args):
        raise SystemExit(
            "--online-support requires at least one evidence source, for example "
            "--fetch-encode-pbmc, --fetch-encode-screen, --fetch-jaspar, --eperturbdb-url, "
            "--eperturbdb-tsv, --eperturbdb-dir, --encode-screen-tsv, or --external-support-url."
        )


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
                    fastkci_k=args.fastkci_k,
                    fastkci_j=args.fastkci_j,
                    fastkci_alpha_param=args.fastkci_alpha_param,
                    fastkci_use_gp=args.fastkci_use_gp,
                    fastkci_fallback=args.fastkci_fallback,
                    stable=args.stable,
                    transform=args.transform,
                    standardize=args.standardize,
                    background_mode=args.background_mode,
                    min_unique=args.min_unique,
                    min_std=args.min_std,
                    max_depth=depth,
                    fci_possible_dsep_depth=args.fci_possible_dsep_depth,
                    fci_max_path_length=args.fci_max_path_length,
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
                            fastkci_k=args.fastkci_k,
                            fastkci_j=args.fastkci_j,
                            fastkci_alpha_param=args.fastkci_alpha_param,
                            fastkci_use_gp=args.fastkci_use_gp,
                            fastkci_fallback=args.fastkci_fallback,
                            stable=args.stable,
                            transform=args.transform,
                            standardize=args.standardize,
                            background_mode=args.background_mode,
                            min_unique=args.min_unique,
                            min_std=args.min_std,
                            max_depth=-1,
                            fci_possible_dsep_depth=-2,
                            fci_max_path_length=-2,
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
    p.add_argument("--fastkci-k", type=int, default=3)
    p.add_argument("--fastkci-j", type=int, default=4)
    p.add_argument("--fastkci-alpha-param", type=float, default=500.0)
    p.add_argument("--fastkci-use-gp", action="store_true", default=False)
    p.add_argument("--fastkci-fallback", default="rcit", choices=["rcit", "fisherz", "kci"])
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
    p.add_argument(
        "--fci-possible-dsep-depth",
        type=int,
        default=-2,
        help="FCI Possible-D-Sep conditioning cap. -2 reuses --max-depth; -1 uses causal-learn unlimited behavior.",
    )
    p.add_argument(
        "--fci-max-path-length",
        type=int,
        default=-2,
        help="FCI discriminating-path cap. -2 reuses --max-depth; -1 uses causal-learn unlimited behavior.",
    )
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


def add_external_support_args(p, *, include_online_support_flag: bool = False, online_help: str | None = None) -> None:
    if include_online_support_flag:
        p.add_argument("--online-support", action="store_true", help=online_help or "Fetch/build external support before plotting.")
    p.add_argument(
        "--support-preset",
        default="none",
        choices=["none", "pbmc_encode", "monocyte_encode", "monocyte_regulatory", "local_eperturbdb"],
        help="Convenience bundle of external support sources. Explicit source flags can still be combined with a preset.",
    )
    p.add_argument("--locus-config", default=None, help="Optional locus config TSV for online/manual support annotation.")
    p.add_argument("--fetch-encode-screen", action="store_true")
    p.add_argument("--encode-screen-url", action="append", default=[])
    p.add_argument("--fetch-encode-pbmc", action="store_true")
    p.add_argument("--encode-biosample-term", action="append", default=[])
    p.add_argument("--encode-assay-term", action="append", default=[])
    p.add_argument("--encode-assembly", default="GRCh38")
    p.add_argument("--max-encode-files", type=int, default=25)
    p.add_argument("--fetch-jaspar", action="store_true")
    p.add_argument("--jaspar-matrix-id", action="append", default=[])
    p.add_argument("--jaspar-collection", default="CORE")
    p.add_argument("--jaspar-tax-group", default="vertebrates")
    p.add_argument("--jaspar-genome", default="hg38")
    p.add_argument("--max-jaspar-motifs", type=int, default=25)
    p.add_argument("--jaspar-min-score", type=float, default=0.85)
    p.add_argument("--jaspar-max-peak-bp", type=int, default=2000)
    p.add_argument("--eperturbdb-url", default=None)
    p.add_argument("--eperturbdb-tsv", default=None)
    p.add_argument("--eperturbdb-dir", default=None)
    p.add_argument("--eperturbdb-database-tsv", default=None)
    p.add_argument("--eperturbdb-enhancer-gene-bed", default=None)
    p.add_argument("--eperturbdb-tf-overlap-bed", default=None)
    p.add_argument("--max-eperturbdb-records", type=int, default=None)
    p.add_argument("--encode-screen-tsv", default=None)
    p.add_argument("--external-support-url", action="append", default=[])
    p.add_argument("--external-cache-dir", default="CausalDiscovery/evaluation/external/cache")
    p.add_argument("--max-online-records", type=int, default=None)
    p.add_argument("--min-overlap-bp", type=int, default=1)


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

    p_match_avg = sub.add_parser(
        "match-average",
        help="Build RNA-anchor kNN neighbor tables for averaged chromatin modality profiles.",
    )
    p_match_avg.add_argument("--run-id", default="joint_v2")
    p_match_avg.add_argument("--train-dir", default=None)
    p_match_avg.add_argument("--label-tsv", default=None)
    p_match_avg.add_argument("--out-dir", default=None)
    p_match_avg.add_argument("--label-column", default="harmonized_coarse")
    p_match_avg.add_argument("--cell-type", default="monocyte")
    p_match_avg.add_argument("--min-label-confidence", type=float, default=0.5)
    p_match_avg.add_argument("--anchor-modality", default="rna")
    p_match_avg.add_argument("--marks", default="H3K27ac,H3K27me3,H3K4me1,H3K4me2,H3K4me3,H3K9me3")
    p_match_avg.add_argument("--n-neighbors", type=int, default=10)
    p_match_avg.add_argument("--neighbor-weight-mode", default="uniform", choices=["uniform", "inverse_distance"])
    p_match_avg.add_argument("--max-anchors", type=int, default=None)

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

    p_filter_dataset = sub.add_parser(
        "filter-dataset",
        help="Create a filtered nearby-peak dataset root from an existing dataset root.",
    )
    p_filter_dataset.add_argument("--dataset-root", type=Path, required=True)
    p_filter_dataset.add_argument("--out-root", type=Path, required=True)
    p_filter_dataset.add_argument("--gene-panel", type=Path, default=default_gene_panel(project_root))
    p_filter_dataset.add_argument("--min-nonzero-fraction", type=float, default=0.01)

    p_elastic = sub.add_parser(
        "elastic-net-screen",
        help="Create an elastic-net screened nearby-peak dataset root from an existing dataset root.",
    )
    p_elastic.add_argument(
        "--dataset-root",
        type=Path,
        default=causal_dir / "outputs" / "datasets" / "joint_v2" / "monocyte_cuttag_peak_genes_1mb_rna_anchor_average_k10",
    )
    p_elastic.add_argument(
        "--out-root",
        type=Path,
        default=None,
    )
    p_elastic.add_argument("--report-root", type=Path, default=None)
    p_elastic.add_argument("--gene-panel", type=Path, default=default_gene_panel(project_root))
    p_elastic.add_argument("--genes", default=None)
    p_elastic.add_argument("--top-k", type=int, default=30)
    p_elastic.add_argument("--n-splits", type=int, default=5)
    p_elastic.add_argument("--n-repeats", type=int, default=20)
    p_elastic.add_argument("--random-state", type=int, default=13)
    p_elastic.add_argument("--l1-ratios", default="0.1,0.25,0.5,0.75,0.9,0.95,1.0")
    p_elastic.add_argument("--alpha-min-exp", type=float, default=-4.0)
    p_elastic.add_argument("--alpha-max-exp", type=float, default=1.0)
    p_elastic.add_argument("--n-alphas", type=int, default=60)
    p_elastic.add_argument("--max-iter", type=int, default=20000)
    p_elastic.add_argument("--tol", type=float, default=1e-4)
    p_elastic.add_argument("--coef-tol", type=float, default=1e-10)
    p_elastic.add_argument(
        "--graph-split-fraction",
        type=float,
        default=0.0,
        help="Hold out this fraction of rows as a disjoint graph split so peak selection "
        "and graph discovery do not share samples (clean post-selection). 0.0 = legacy shared-sample.",
    )
    p_elastic.add_argument("--split-random-state", type=int, default=17)
    p_elastic.add_argument("--min-graph-rows", type=int, default=50)

    p_organize = sub.add_parser(
        "organize-outputs",
        help="Move graph run directories into outputs/methods/<method>/<dataset>/<gene-or-locus>/<result>.",
    )
    p_organize.add_argument("--dry-run", action="store_true", default=False)

    p_graph = sub.add_parser("graph", help="Run one PC, FCI, or DAGMA graph discovery job on a matrix.")
    p_graph.add_argument("--matrix-tsv", type=Path, required=True)
    p_graph.add_argument("--out-dir", default=None)
    p_graph.add_argument("--columns", default="auto")
    add_shared_graph_args(p_graph)
    add_dagma_args(p_graph)
    add_external_support_args(
        p_graph,
        include_online_support_flag=True,
        online_help="After graph discovery, fetch/build external support and render the final plot with validated edges highlighted.",
    )

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
    p_plot.add_argument("--edge-support-tsv", default=None)
    p_plot.add_argument("--layout-seed", type=int, default=13)
    p_plot.add_argument("--title", default=None)
    p_plot.add_argument("--layout", default="auto", choices=["auto", "local", "spring"])
    p_plot.add_argument("--max-edges", type=int, default=30)
    p_plot.add_argument("--peaks-per-row", type=int, default=18)
    add_external_support_args(p_plot, include_online_support_flag=True)

    p_support = sub.add_parser("support", help="Build node support annotations for a graph directory.")
    p_support.add_argument("--graph-dir", type=Path, required=True)
    p_support.add_argument("--out-tsv", default=None)
    p_support.add_argument("--matches-tsv", default=None)
    p_support.add_argument("--edge-support-tsv", default=None)
    p_support.add_argument("--report-md", default=None)
    add_external_support_args(p_support)

    p_evaluate = sub.add_parser("evaluate", help="Evaluate edge stability and support across saved graph runs.")
    p_evaluate.add_argument("--methods-root", default="CausalDiscovery/outputs/methods")
    p_evaluate.add_argument("--graph-dir", action="append", default=[])
    p_evaluate.add_argument("--out-dir", default="CausalDiscovery/outputs/evaluation/graph_stability/latest")
    p_evaluate.add_argument("--group-cols", default="dataset,gene,method,family_config")
    p_evaluate.add_argument("--methods", default=None)
    p_evaluate.add_argument("--datasets", default=None)
    p_evaluate.add_argument("--genes", default=None)
    p_evaluate.add_argument("--family-configs", default=None)
    p_evaluate.add_argument("--stable-threshold", type=float, default=0.5)
    p_evaluate.add_argument("--core-threshold", type=float, default=0.8)

    args = parser.parse_args()
    apply_support_preset(args)

    if args.command == "match":
        run_python(args.python_bin, "build_scglue_one_to_one_matches.py", build_match_args(project_root, args))
        return 0

    if args.command == "match-average":
        run_python(
            args.python_bin,
            "build_scglue_anchor_neighbor_averages.py",
            build_match_average_args(project_root, args),
        )
        return 0

    if args.command == "dataset":
        if args.out_root is None:
            args.out_root = default_dataset_root(project_root, args.run_id, args.gene_panel, args.window_bp)
        run_python(args.python_bin, "generate_monocyte_cuttag_peak_datasets.py", build_dataset_args(project_root, args))
        return 0

    if args.command == "filter-dataset":
        run_python(args.python_bin, "filter_nearby_peak_dataset.py", build_filter_dataset_args(project_root, args))
        return 0

    if args.command == "elastic-net-screen":
        if args.out_root is None:
            args.out_root = default_elastic_net_out_root(causal_dir, args.top_k, args.graph_split_fraction)
        run_python(args.python_bin, "screen_nearby_peaks_elastic_net.py", build_elastic_net_screen_args(project_root, args))
        return 0

    if args.command == "organize-outputs":
        run_python(args.python_bin, "reorganize_graph_outputs.py", build_organize_args(project_root, args))
        return 0

    if args.command == "graph":
        if args.online_support:
            require_external_support_source(args)
        graph_dir = infer_graph_output_dir(project_root, args)
        original_plot = bool(args.plot)
        if args.online_support and original_plot:
            args.plot = False
        run_python(args.python_bin, "run_pc_causallearn.py", build_graph_args(project_root, args))
        args.plot = original_plot
        if args.online_support:
            run_python(args.python_bin, "build_node_support_table.py", build_support_command_args(project_root, graph_dir, args))
            if original_plot:
                run_python(args.python_bin, "plot_pc_graph.py", build_plot_command_args(project_root, graph_dir, args))
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
        if args.online_support:
            require_external_support_source(args)
            run_python(args.python_bin, "build_node_support_table.py", build_support_command_args(project_root, args.graph_dir, args))

        run_python(args.python_bin, "plot_pc_graph.py", build_plot_command_args(project_root, args.graph_dir, args))
        return 0

    if args.command == "support":
        run_python(
            args.python_bin,
            "build_node_support_table.py",
            build_support_command_args(project_root, args.graph_dir, args, include_output_paths=True),
        )
        return 0

    if args.command == "evaluate":
        evaluate_args = [
            "--repo-root",
            str(project_root),
            "--methods-root",
            str(args.methods_root),
            "--out-dir",
            str(args.out_dir),
            "--group-cols",
            args.group_cols,
            "--stable-threshold",
            str(args.stable_threshold),
            "--core-threshold",
            str(args.core_threshold),
        ]
        for graph_dir in args.graph_dir:
            evaluate_args.extend(["--graph-dir", str(graph_dir)])
        optional_arg(evaluate_args, "--methods", args.methods)
        optional_arg(evaluate_args, "--datasets", args.datasets)
        optional_arg(evaluate_args, "--genes", args.genes)
        optional_arg(evaluate_args, "--family-configs", args.family_configs)
        run_python(args.python_bin, "evaluate_graph_stability.py", evaluate_args)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
