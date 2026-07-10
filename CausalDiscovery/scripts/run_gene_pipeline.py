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
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Discovery methods that route through causal_cli.py graph (vs. the GFCI/Tetrad path).
CLI_METHODS = {"pc", "fci", "dagma"}

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


def sh(cmd: list[str], env: dict | None = None) -> None:
    """Run a subprocess, streaming output; raise on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False, env=env)
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
        cmd += ["--indep-test", a.indep_test, "--max-depth", str(a.max_depth),
                "--confounders", a.confounders]
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


def gfci_cmd(script: Path, gfci_python_bin: str, matrix: Path,
             out_dir: Path, a: argparse.Namespace) -> list[str]:
    """GFCI runs in the causaldisco env (JDK 21) via run_gfci_tetrad.py."""
    return [
        gfci_python_bin, str(script),
        "--matrix-tsv", str(matrix),
        "--out-dir", str(out_dir),
        "--alpha", str(a.gfci_alpha),
        "--penalty-discount", str(a.gfci_penalty_discount),
        "--transform", a.transform,
    ]


def gfci_env(a: argparse.Namespace) -> dict:
    """Env for the GFCI subprocess: force JAVA_HOME to the JDK-21 env."""
    env = os.environ.copy()
    env["JAVA_HOME"] = str(a.java_home)
    return env


def resolve_stability_method(methods: list[str], requested: str | None) -> str:
    """Stability re-runs discovery via causal_cli graph, which only supports
    pc/fci/dagma. Default to the primary CLI discovery method (first such in
    --methods), falling back to fci if only GFCI was requested."""
    if requested:
        return requested
    for m in methods:
        if m in CLI_METHODS:
            return m
    return "fci"


def stability_cmd(script: Path, python_bin: str, matrix: Path, out_dir: Path,
                  label: str, method: str, a: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(script),
        "--matrix-tsv", str(matrix),
        "--label", label,
        "--out-dir", str(out_dir),
        "--n-boot", str(a.n_boot),
        "--subsample-frac", str(a.subsample_frac),
        "--stable-threshold", str(a.stable_threshold),
        "--python-bin", python_bin,
        "--method", method,
        "--indep-test", a.indep_test,
        "--transform", a.transform,
        "--alpha", str(a.alpha),
        "--max-depth", str(a.max_depth),
        "--background-mode", a.background_mode,
        "--confounders", a.confounders,
    ]


def run_stability(a: argparse.Namespace, gene_out: Path, matrix: Path) -> dict:
    """Bootstrap edge-stability selection on the gene's matrix (headline step)."""
    script = a.repo_root / "CausalDiscovery" / "scripts" / "bootstrap_stability.py"
    method = resolve_stability_method(a.methods, a.stability_method)
    out_dir = gene_out / "stability"
    label = a.gene.upper()
    sh(stability_cmd(script, a.python_bin, matrix, out_dir, label, method, a))

    freq: dict[tuple, float] = {}
    stab_tsv = out_dir / f"edge_stability_{label}.tsv"
    if stab_tsv.exists():
        sdf = pd.read_csv(stab_tsv, sep="\t")
        for _, r in sdf.iterrows():
            key = tuple(sorted([str(r["node1"]), str(r["node2"])]))
            freq[key] = float(r["frequency"])
    summ_json = out_dir / f"stability_summary_{label}.json"
    summary = json.loads(summ_json.read_text()) if summ_json.exists() else {}
    return {
        "method": method,
        "out_dir": str(out_dir),
        "freq": freq,
        "summary": summary,
        "stable_threshold": a.stable_threshold,
    }


def merge_stability(consensus: pd.DataFrame, freq: dict[tuple, float],
                    threshold: float) -> pd.DataFrame:
    """Add per-edge stability frequency + stable/top-tier flags to the consensus."""
    if consensus.empty:
        for col in ("stability", "stable", "top_tier"):
            consensus[col] = pd.Series(dtype="float64" if col == "stability" else "bool")
        return consensus
    stab, stable, top = [], [], []
    for _, r in consensus.iterrows():
        key = tuple(sorted([str(r["node1"]), str(r["node2"])]))
        f = freq.get(key, 0.0)
        is_stable = f >= threshold
        stab.append(f)
        stable.append(is_stable)
        top.append(bool(is_stable and r.get("target_aware_validated", False)))
    consensus = consensus.copy()
    consensus["stability"] = stab
    consensus["stable"] = stable
    consensus["top_tier"] = top
    return consensus


def edge_kind(node1: str, node2: str) -> str:
    # Observed-confounder nodes (conf__*) are modeling covariates, not regulatory elements:
    # any edge touching one is a "confounder" edge and is NOT validated or counted as a
    # peak->gene / peak<->peak causal edge.
    if node1.startswith("conf__") or node2.startswith("conf__"):
        return "confounder"
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
        if method == "gfci":
            script = a.repo_root / "CausalDiscovery" / "scripts" / "run_gfci_tetrad.py"
            sh(gfci_cmd(script, a.gfci_python_bin, matrix, mdir, a), env=gfci_env(a))
        else:
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

    stability = None
    if a.stability:
        stability = run_stability(a, gene_out, matrix)
        consensus = merge_stability(consensus, stability["freq"], a.stable_threshold)
    else:
        consensus = merge_stability(consensus, {}, a.stable_threshold)

    write_outputs(a, gene_out, matrix, per_method, edge_rows, consensus, stability)
    return {"gene": a.gene.upper(), "per_method": per_method,
            "consensus": consensus, "out_dir": str(gene_out),
            "stability": stability}


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
                  edge_rows: list[dict], consensus: pd.DataFrame,
                  stability: dict | None = None) -> None:
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
            "gfci_alpha": a.gfci_alpha, "gfci_penalty_discount": a.gfci_penalty_discount,
        },
        "sources": {
            "re2g_tsv": str(a.re2g_tsv) if a.re2g_tsv else None,
            "pchic_tsv": str(a.pchic_tsv) if a.pchic_tsv else None,
            "eqtl_tsv": str(a.eqtl_tsv) if a.eqtl_tsv else None,
            "node_context_preset": a.node_context_preset,
        },
        "stability": {
            "enabled": bool(stability),
            "method": stability["method"] if stability else None,
            "n_boot": a.n_boot, "subsample_frac": a.subsample_frac,
            "stable_threshold": a.stable_threshold,
            "summary": stability["summary"] if stability else None,
            "out_dir": stability["out_dir"] if stability else None,
        },
        "per_method": per_method,
    }
    (gene_out / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2))

    write_report(a, gene_out, matrix, per_method, consensus, stability)


def _stability_headline(a, consensus: pd.DataFrame, stability: dict | None) -> list[str]:
    """The headline 'Stability-selected + validated edges' section."""
    lines = ["## Stability-selected + validated edges (headline)", ""]
    if not a.stability or stability is None:
        lines += ["_Stability selection disabled (`--no-stability`)._", ""]
        return lines
    summ = stability.get("summary", {})
    thr = a.stable_threshold
    lines += [
        f"- Bootstrap stability: method `{stability['method']}` · "
        f"{a.n_boot} resamples · subsample {a.subsample_frac} · stable ≥ {thr}",
        f"- Edges ever seen: **{summ.get('n_edges_seen', '?')}** · "
        f"stable (≥ {thr}): **{summ.get('n_stable', '?')}** "
        f"(peak→gene {summ.get('n_stable_peak_gene', '?')})",
        "",
        "**Top tier = edges that are BOTH bootstrap-stable AND target-aware validated:**",
        "",
    ]
    pg = consensus[consensus["edge_kind"] == "peak_gene"] if not consensus.empty else consensus
    top = pg[pg.get("top_tier", False) == True] if not pg.empty else pg  # noqa: E712
    if top is None or top.empty:
        lines += ["_No edge is both stable and target-aware validated at the current threshold._"]
        # still surface the best of each so the section is informative
        if not pg.empty:
            stab_only = pg[pg["stable"] == True].sort_values("stability", ascending=False)  # noqa: E712
            val_only = pg[pg["target_aware_validated"] == True]  # noqa: E712
            if not stab_only.empty:
                lines += ["", f"- stable peak→gene edges (freq ≥ {thr}): "
                          + ", ".join(f"`{_peak_of(r, a)}` ({r['stability']:.2f})"
                                      for _, r in stab_only.head(6).iterrows())]
            if not val_only.empty:
                lines += [f"- target-aware validated peak→gene edges: "
                          + ", ".join(f"`{_peak_of(r, a)}` [{r['axes'] or '—'}]"
                                      for _, r in val_only.iterrows())]
    else:
        lines += ["| peak | gene | stability | axes | sources | algos | verdict |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in top.sort_values("stability", ascending=False).iterrows():
            lines.append(
                f"| `{_peak_of(r, a)}` | {a.gene.upper()} | **{r['stability']:.2f}** "
                f"| {r['axes'] or '—'} | {r['target_aware_sources'] or '—'} "
                f"| {str(r['methods']).replace('|', ', ')} | {r['verdict']} |")
    lines += [""]
    return lines


def _peak_of(r, a) -> str:
    """Return the peak endpoint of a peak→gene consensus row."""
    n1, n2 = str(r["node1"]), str(r["node2"])
    return n2 if n1.startswith("expr__") else n1


def write_report(a, gene_out: Path, matrix: Path, per_method: dict,
                 consensus: pd.DataFrame, stability: dict | None = None) -> None:
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
        *_stability_headline(a, consensus, stability),
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
                   help="Comma-separated algorithms: any of pc,fci,dagma,gfci.")
    p.add_argument("--out-root", type=Path,
                   default=repo_root / "CausalDiscovery/outputs/pipeline")
    p.add_argument("--python-bin", default=default_python_bin(),
                   help="Python with causal-learn/dagma (default: scrna_r env).")
    # GFCI (Tetrad / pytetrad) runs in a separate JDK-21 env.
    p.add_argument("--gfci-python-bin",
                   default="/home/dgsetiawan/anaconda3/envs/causaldisco/bin/python",
                   help="Python with pytetrad (causaldisco env) for the GFCI method.")
    p.add_argument("--java-home",
                   default="/home/dgsetiawan/anaconda3/envs/causaldisco",
                   help="JAVA_HOME (JDK 21) exported for the GFCI subprocess.")
    p.add_argument("--gfci-alpha", type=float, default=0.01,
                   help="Fisher-Z alpha for GFCI (independent of PC/FCI --alpha).")
    p.add_argument("--gfci-penalty-discount", type=float, default=2.0,
                   help="SEM-BIC penalty discount for GFCI.")
    # Stability-selection (headline) step.
    p.add_argument("--stability", action="store_true", default=True,
                   help="Run bootstrap edge-stability selection (default: on).")
    p.add_argument("--no-stability", dest="stability", action="store_false",
                   help="Disable the stability-selection step.")
    p.add_argument("--stability-method", default=None,
                   help="Discovery method for stability re-runs (pc/fci/dagma). "
                        "Default: primary CLI method in --methods.")
    p.add_argument("--n-boot", type=int, default=50,
                   help="Bootstrap resamples for stability selection.")
    p.add_argument("--subsample-frac", type=float, default=0.7,
                   help="Row subsample fraction per bootstrap.")
    p.add_argument("--stable-threshold", type=float, default=0.5,
                   help="Recurrence frequency at/above which an edge is 'stable'.")
    # discovery params (Q3 defaults)
    p.add_argument("--indep-test", default="fisherz")
    p.add_argument("--transform", default="rank_gaussian", choices=["none", "rank_gaussian"])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--background-mode", default="tiered_distal_promoter_expr")
    p.add_argument("--confounders", default="depth",
                   help="Observed confounders added as ROOT nodes (pc/fci): 'none' | 'depth' "
                        "(per-metacell library depths, on by default) | 'depth,cellcycle' | 'depth,donor'.")
    p.add_argument("--dagma-w-threshold", type=float, default=0.3)
    # validation sources
    p.add_argument("--re2g-tsv", type=Path,
                   default=_default_if_exists(repo_root / "Data/external/re2g/re2g_cd14_monocyte_gene_links.tsv"))
    p.add_argument("--pchic-tsv", type=Path,
                   default=_default_if_exists(repo_root / "Data/external/pchic/pchic_mon_gene_links.tsv"))
    p.add_argument("--eqtl-tsv", type=Path,
                   default=_default_if_exists(repo_root / "Data/external/eqtl/eqtl_mon_gene_links.tsv"),
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
        if "top_tier" in c.columns:
            top = c[(c["edge_kind"] == "peak_gene") & (c["top_tier"] == True)]  # noqa: E712
            print(f"  STABLE + VALIDATED (top tier): {len(top)}")
            for _, r in top.iterrows():
                print(f"    ★ stability={r['stability']:.2f}  {_peak_of(r, a)} → {a.gene.upper()}"
                      f"  axes={r['axes']}")
    print("=" * 64)
    return 0


def _default_if_exists(path: Path):
    return path if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
