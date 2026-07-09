#!/usr/bin/env python3
"""End-to-end per-gene causal-discovery + edge-validation pipeline.

Given ONE target gene and an existing nearby-peak dataset root, this driver:

  1. discovery   -- runs each requested causal algorithm (pc / fci / dagma) on
                    the gene's nearby-peak matrix via `causal_cli.py graph`;
  2. validation  -- runs the external-support "evaluation suite" on every graph
                    via `causal_cli.py support` (target-gene-aware sources:
                    ENCODE-rE2G prediction + BLUEPRINT PCHi-C physical, both
                    local; optional online ENCODE/SCREEN node context);
  3. consensus   -- aggregates edges ACROSS algorithms and classifies each edge
                    with an *axis-aware* validation rule, so we can answer the
                    project question: "which edges from different causal
                    algorithms are actually supported by the evaluation suite?"

It intentionally shells out to the existing `causal_cli.py` sub-commands rather
than re-implementing any stage -- every stage stays the single, tested code path.

Key honesty rule (formalizes ROADMAP Phase B "axis-aware edge validation"):
  * A peak->gene edge is only "validated" when a support match ties the PEAK to
    the EDGE's TARGET GENE (rule `region_interval_with_matching_target_gene`).
  * peak-peak "validation" via `region_interval_on_either_endpoint` is merely
    node-existence (both intervals appear in some database, gene-agnostic); it is
    NOT counted as a validated causal edge until a target-aware peak-peak
    validator (Phase B: ChIP-Atlas / Hi-C loops) lands.

Usage (defaults reproduce the CD14 kNN-avg 300kb baseline):
  python CausalDiscovery/scripts/run_gene_pipeline.py --gene CD14 \
      --python-bin /home/dgsetiawan/anaconda3/envs/scrna_r/bin/python
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# --- source name -> triangulation axis (peak->gene). eqtl slots in at Phase B. -
SOURCE_AXIS = {
    "re2g": "prediction",
    "pchic": "physical",
    "eqtl": "genetics",
}
# Sources that are cell-type + target-gene aware (count toward real edge validation).
TARGET_AWARE_SOURCES = set(SOURCE_AXIS)
# validation_rule values that carry a genuine target-gene-aware verdict.
TARGET_AWARE_RULES = {"region_interval_with_matching_target_gene"}


def sh(cmd: list[str]) -> None:
    """Run a subprocess, streaming output; raise on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(cmd)}")


def resolve_matrix(dataset_root: Path, gene: str) -> Path:
    """Locate <root>/<gene_lower>/nearby_peaks/<GENE>_nearby_peak_matrix.tsv."""
    cand = dataset_root / gene.lower() / "nearby_peaks" / f"{gene.upper()}_nearby_peak_matrix.tsv"
    if cand.exists():
        return cand
    # fall back: any *_nearby_peak_matrix.tsv under the gene dir
    gene_dir = dataset_root / gene.lower() / "nearby_peaks"
    if gene_dir.is_dir():
        hits = sorted(gene_dir.glob("*_nearby_peak_matrix.tsv"))
        if hits:
            return hits[0]
    raise SystemExit(f"no nearby-peak matrix for gene={gene} under {dataset_root}")


def graph_cmd(cli: Path, python_bin: str, method: str, matrix: Path,
              out_dir: Path, a: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(cli), "--python-bin", python_bin, "graph",
        "--matrix-tsv", str(matrix),
        "--method", method,
        "--alpha", str(a.alpha),
        "--transform", a.transform,
        "--background-mode", a.background_mode,
        "--out-dir", str(out_dir),
        "--no-plot",
    ]
    if method in ("pc", "fci"):
        cmd += ["--indep-test", a.indep_test, "--max-depth", str(a.max_depth)]
    if method == "dagma":
        # Q3 decision: keep only the continuous Gaussian DAGMA as a sensitivity check
        # (count-family presets were dropped).
        cmd += ["--dagma-family-config", "gaussian_gaussian",
                "--dagma-w-threshold", str(a.dagma_w_threshold)]
    return cmd


def support_cmd(cli: Path, python_bin: str, graph_dir: Path,
                a: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(cli), "--python-bin", python_bin, "support",
        "--graph-dir", str(graph_dir),
        "--min-overlap-bp", str(a.min_overlap_bp),
    ]
    if a.re2g_tsv:
        cmd += ["--re2g-tsv", str(a.re2g_tsv)]
    if a.pchic_tsv:
        cmd += ["--external-support-tsv", f"pchic={a.pchic_tsv}"]
    if a.eqtl_tsv:
        cmd += ["--external-support-tsv", f"eqtl={a.eqtl_tsv}"]
    if a.node_context_preset != "none":
        cmd += ["--support-preset", a.node_context_preset]
    return cmd


def edge_kind(node1: str, node2: str) -> str:
    e1, e2 = node1.startswith("expr__"), node2.startswith("expr__")
    if e1 and e2:
        return "gene_gene"
    if e1 or e2:
        return "peak_gene"
    return "peak_peak"


def parse_axes(sources_field: str) -> tuple[list[str], list[str]]:
    """Split a '|'-joined sources string into (target_aware_sources, axes)."""
    srcs = [s for s in str(sources_field or "").split("|") if s]
    aware = [s for s in srcs if s in TARGET_AWARE_SOURCES]
    axes = sorted({SOURCE_AXIS[s] for s in aware})
    return aware, axes


def classify_edge(rule: str, sources: str, kind: str) -> dict:
    """Axis-aware verdict for a single method's edge row."""
    aware, axes = parse_axes(sources)
    target_aware_rule = rule in TARGET_AWARE_RULES
    # A causal edge is validated ONLY under a target-aware rule with >=1 axis.
    validated = bool(target_aware_rule and axes)
    if kind == "peak_gene":
        tier = "target_aware" if target_aware_rule else "weak"
    elif kind == "peak_peak":
        # No target-aware peak-peak validator exists yet (Phase B pending).
        tier = "no_target_validator"
    else:
        tier = "gene_gene"
    return {
        "validated": validated,
        "tier": tier,
        "target_aware_sources": aware,
        "axes": axes,
    }


def run_gene(a: argparse.Namespace) -> dict:
    cli = a.repo_root / "CausalDiscovery" / "scripts" / "causal_cli.py"
    matrix = resolve_matrix(a.dataset_root, a.gene)
    gene_out = a.out_root / a.gene.upper()
    gene_out.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] gene={a.gene}  matrix={matrix}")
    print(f"[pipeline] methods={a.methods}  out={gene_out}")

    per_method: dict[str, dict] = {}
    edge_rows: list[dict] = []

    for method in a.methods:
        mdir = gene_out / method
        t0 = time.time()
        sh(graph_cmd(cli, a.python_bin, method, matrix, mdir, a))
        sh(support_cmd(cli, a.python_bin, mdir, a))
        dt = time.time() - t0

        summ = json.loads((mdir / "run_summary.json").read_text())
        esp = mdir / "edge_support.tsv"
        edf = pd.read_csv(esp, sep="\t") if esp.exists() else pd.DataFrame()
        per_method[method] = {
            "n_edges": int(summ.get("n_edges", len(edf))),
            "n_columns_selected": int(summ.get("n_columns_selected", 0)),
            "seconds": round(dt, 1),
            "graph_dir": str(mdir),
        }
        for _, r in edf.iterrows():
            n1, n2 = str(r["node1"]), str(r["node2"])
            kind = edge_kind(n1, n2)
            cls = classify_edge(str(r.get("validation_rule", "")),
                                str(r.get("sources", "")), kind)
            edge_rows.append({
                "method": method,
                "node1": n1, "node2": n2,
                "edge_text": str(r.get("edge_text", "")),
                "edge_kind": kind,
                "validation_rule": str(r.get("validation_rule", "")),
                "raw_sources": str(r.get("sources", "")),
                **cls,
            })

    consensus = build_consensus(a.gene, edge_rows)
    write_outputs(a, gene_out, matrix, per_method, edge_rows, consensus)
    return {"gene": a.gene.upper(), "per_method": per_method,
            "consensus": consensus, "out_dir": str(gene_out)}


def build_consensus(gene: str, edge_rows: list[dict]) -> pd.DataFrame:
    """Collapse per-method edges to one row per undirected adjacency."""
    by_key: dict[tuple, dict] = {}
    for e in edge_rows:
        key = tuple(sorted([e["node1"], e["node2"]]))
        agg = by_key.setdefault(key, {
            "gene": gene.upper(),
            "node1": key[0], "node2": key[1],
            "edge_kind": e["edge_kind"],
            "methods": set(), "orientations": set(),
            "target_aware_validated": False,
            "axes": set(), "target_aware_sources": set(),
            "node_context_sources": set(),
            "tier": e["tier"],
        })
        agg["methods"].add(e["method"])
        agg["orientations"].add(f"{e['method']}:{e['edge_text']}")
        if e["validated"]:
            agg["target_aware_validated"] = True
        agg["axes"].update(e["axes"])
        agg["target_aware_sources"].update(e["target_aware_sources"])
        # gene-agnostic node context = any raw source that is not target-aware
        for s in str(e["raw_sources"]).split("|"):
            if s and s not in TARGET_AWARE_SOURCES:
                agg["node_context_sources"].add(s)

    out = []
    for _, agg in sorted(by_key.items()):
        n_methods = len(agg["methods"])
        n_axes = len(agg["axes"])
        out.append({
            "gene": agg["gene"],
            "node1": agg["node1"], "node2": agg["node2"],
            "edge_kind": agg["edge_kind"],
            "n_methods": n_methods,
            "methods": "|".join(sorted(agg["methods"])),
            "n_axes": n_axes,
            "axes": "|".join(sorted(agg["axes"])),
            "target_aware_validated": agg["target_aware_validated"],
            "target_aware_sources": "|".join(sorted(agg["target_aware_sources"])),
            "node_context_sources": "|".join(sorted(agg["node_context_sources"])),
            "validation_tier": agg["tier"],
            "verdict": verdict(agg["edge_kind"], n_methods,
                               agg["target_aware_validated"], n_axes),
            "orientations": " ; ".join(sorted(agg["orientations"])),
        })
    cols = ["gene", "node1", "node2", "edge_kind", "n_methods", "methods",
            "n_axes", "axes", "target_aware_validated", "target_aware_sources",
            "node_context_sources", "validation_tier", "verdict", "orientations"]
    df = pd.DataFrame(out, columns=cols)
    if not df.empty:
        df = df.sort_values(
            ["target_aware_validated", "n_axes", "n_methods", "edge_kind"],
            ascending=[False, False, False, True]).reset_index(drop=True)
    return df


def verdict(kind: str, n_methods: int, validated: bool, n_axes: int) -> str:
    if kind == "peak_gene":
        if validated and n_methods >= 2:
            return "VALIDATED_MULTI_ALGO"      # strongest: real support + reproducible
        if validated:
            return "VALIDATED"                 # target-aware support, single algorithm
        if n_methods >= 2:
            return "recurrent_unvalidated"     # reproducible but no target-aware support
        return "unvalidated"
    if kind == "peak_peak":
        return "no_target_validator"           # cannot be validated yet (Phase B pending)
    return "gene_gene"


def write_outputs(a, gene_out: Path, matrix: Path, per_method: dict,
                  edge_rows: list[dict], consensus: pd.DataFrame) -> None:
    consensus.to_csv(gene_out / "edges_consensus.tsv", sep="\t", index=False)
    pd.DataFrame(edge_rows).to_csv(gene_out / "edges_per_method.tsv", sep="\t", index=False)

    manifest = {
        "gene": a.gene.upper(),
        "matrix_tsv": str(matrix),
        "dataset_root": str(a.dataset_root),
        "methods": a.methods,
        "params": {
            "alpha": a.alpha, "indep_test": a.indep_test, "transform": a.transform,
            "max_depth": a.max_depth, "background_mode": a.background_mode,
            "dagma_w_threshold": a.dagma_w_threshold, "min_overlap_bp": a.min_overlap_bp,
        },
        "sources": {
            "re2g_tsv": str(a.re2g_tsv) if a.re2g_tsv else None,
            "pchic_tsv": str(a.pchic_tsv) if a.pchic_tsv else None,
            "eqtl_tsv": str(a.eqtl_tsv) if a.eqtl_tsv else None,
            "node_context_preset": a.node_context_preset,
        },
        "per_method": per_method,
    }
    (gene_out / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2))

    write_report(a, gene_out, matrix, per_method, consensus)


def write_report(a, gene_out: Path, matrix: Path, per_method: dict,
                 consensus: pd.DataFrame) -> None:
    pg = consensus[consensus["edge_kind"] == "peak_gene"] if not consensus.empty else consensus
    validated = pg[pg["target_aware_validated"]] if not pg.empty else pg
    lines = [
        f"# Causal pipeline report — {a.gene.upper()}",
        "",
        f"- Matrix: `{matrix}`",
        f"- Algorithms: {', '.join(a.methods)}",
        f"- CI test: `{a.indep_test}` · transform: `{a.transform}` · alpha: {a.alpha} · max-depth: {a.max_depth}",
        f"- Edge-validation sources (target-aware): "
        + ", ".join([s for s, v in [("re2g", a.re2g_tsv), ("pchic", a.pchic_tsv), ("eqtl", a.eqtl_tsv)] if v] or ["(none)"]),
        f"- Node-context preset (gene-agnostic): `{a.node_context_preset}`",
        "",
        "## Per-algorithm discovery",
        "",
        "| method | edges | vars | seconds |",
        "| --- | --- | --- | --- |",
    ]
    for m in a.methods:
        d = per_method.get(m, {})
        lines.append(f"| {m} | {d.get('n_edges','?')} | {d.get('n_columns_selected','?')} | {d.get('seconds','?')} |")

    total = len(consensus)
    n_pg = len(pg)
    n_val = len(validated)
    n_multi = int((pg["n_methods"] >= 2).sum()) if not pg.empty else 0
    lines += [
        "",
        "## Edge-validation summary (axis-aware)",
        "",
        f"- Unique adjacencies across algorithms: **{total}**",
        f"- peak→gene edges: **{n_pg}** (these are the biologically interpretable ones)",
        f"- peak→gene **target-aware validated**: **{n_val}** "
        f"(support ties the peak to *{a.gene.upper()}* specifically)",
        f"- peak→gene found by ≥2 algorithms: **{n_multi}**",
        "",
        "> peak-peak edges are reported but *cannot* be validated yet — the suite has no",
        "> target-aware peak-peak validator (Phase B: ChIP-Atlas co-binding / Hi-C loops).",
        "> Their gene-agnostic 'either-endpoint' overlap is node-existence, not edge support.",
        "",
        "## peak→gene edges (ranked)",
        "",
    ]
    if pg.empty:
        lines.append("_No peak→gene edges discovered._")
    else:
        lines += ["| peak | gene | verdict | algos | axes | sources |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for _, r in pg.iterrows():
            peak = r["node1"] if r["node1"] != f"expr__{a.gene.upper()}" and not r["node1"].startswith("expr__") else r["node2"]
            gene_node = r["node2"] if peak == r["node1"] else r["node1"]
            lines.append(
                f"| `{peak}` | {gene_node} | **{r['verdict']}** | {r['methods']} "
                f"| {r['axes'] or '—'} | {r['target_aware_sources'] or '—'} |")
    (gene_out / "pipeline_report.md").write_text("\n".join(lines) + "\n")


def default_python_bin() -> str:
    cand = Path("/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python")
    return str(cand) if cand.exists() else sys.executable


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene", required=True, help="Target gene symbol (e.g. CD14).")
    p.add_argument("--repo-root", type=Path, default=repo_root)
    p.add_argument("--dataset-root", type=Path,
                   default=repo_root / "CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb",
                   help="Nearby-peak dataset root (default: 300kb kNN-avg baseline).")
    p.add_argument("--methods", default="fci,pc",
                   help="Comma-separated algorithms: any of pc,fci,dagma.")
    p.add_argument("--out-root", type=Path,
                   default=repo_root / "CausalDiscovery/outputs/pipeline")
    p.add_argument("--python-bin", default=default_python_bin(),
                   help="Python with causal-learn/dagma (default: scrna_r env).")
    # discovery params (Q3 defaults)
    p.add_argument("--indep-test", default="fisherz")
    p.add_argument("--transform", default="rank_gaussian", choices=["none", "rank_gaussian"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--background-mode", default="tiered_distal_promoter_expr")
    p.add_argument("--dagma-w-threshold", type=float, default=0.3)
    # validation sources
    p.add_argument("--re2g-tsv", type=Path,
                   default=_default_if_exists(repo_root / "Data/external/re2g/re2g_cd14_monocyte_gene_links.tsv"))
    p.add_argument("--pchic-tsv", type=Path,
                   default=_default_if_exists(repo_root / "Data/external/pchic/pchic_mon_gene_links.tsv"))
    p.add_argument("--eqtl-tsv", type=Path, default=None,
                   help="eQTL Catalogue monocyte gene-link table (Phase B genetics axis).")
    p.add_argument("--node-context-preset", default="none",
                   choices=["none", "monocyte_encode", "monocyte_regulatory", "pbmc_encode"],
                   help="Optional gene-agnostic node context (online ENCODE/SCREEN). Default off (fast/offline).")
    p.add_argument("--min-overlap-bp", type=int, default=1)
    a = p.parse_args()
    a.methods = [m.strip().lower() for m in a.methods.split(",") if m.strip()]

    result = run_gene(a)

    print("\n" + "=" * 64)
    print(f"DONE {result['gene']}  ->  {result['out_dir']}")
    c = result["consensus"]
    if not c.empty:
        val = c[(c["edge_kind"] == "peak_gene") & (c["target_aware_validated"])]
        print(f"  peak→gene edges: {(c['edge_kind']=='peak_gene').sum()}"
              f"  | target-aware validated: {len(val)}")
        for _, r in val.iterrows():
            print(f"    ✓ {r['verdict']:20s} {r['methods']:8s} axes={r['axes']}"
                  f"  {r['node1']} — {r['node2']}")
    print("=" * 64)
    return 0


def _default_if_exists(path: Path):
    return path if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
