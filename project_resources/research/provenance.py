#!/usr/bin/env python3
"""Backtrack any result to the raw data it came from.

Walks the recorded manifests backwards through the pipeline:

    pipelines/empirical/evaluation result
      -> pipeline run        (pipeline_manifest.json: matrix, params, external sources)
      -> dataset             (panel_summary.json: matching_dir, gene panel, scGLUE run_id)
      -> pairing / metacells (matched_samples.tsv, metacell_summary.json)
      -> integration run     (pipelines/empirical/integration/runs/scglue_<run_id>)
      -> per-mark matrices   (pipelines/empirical/integration/manifests/scglue_input_manifest.tsv)
      -> raw CUT&Tag + GEX   (pipelines/empirical/data/raw/...)

Every link is labelled by how it is known:

    [recorded]   read from a manifest written by the run itself  — trustworthy
    [convention] inferred from the layout/naming rules           — verify before citing

That distinction is the point: a lineage claim is only as good as its weakest link,
so the tool never lets an inferred hop masquerade as a recorded one.

Usage
  python project_resources/research/provenance.py --gene CD14
  python project_resources/research/provenance.py --gene CD14 --panel pipeline_globalact
  python project_resources/research/provenance.py --list
  python project_resources/research/provenance.py --gene CD14 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REC, CONV = "recorded", "convention"


def rel(p) -> str:
    """Repo-relative path, tolerant of the pre-2026-07-20 absolute paths in old manifests."""
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except (ValueError, OSError):
        s = str(p)
        marker = "Epigenomics-GEX-causal-analysis/"
        return s.split(marker, 1)[1] if marker in s else s


def remap_legacy(path: str) -> str:
    """Old manifests reference the pre-restructure layout; point them at today's tree."""
    for old, new in (("Data/external/", "pipelines/empirical/data/raw/external/"),
                     ("Data/", "pipelines/empirical/data/"),
                     ("CausalDiscovery/evaluation/", "pipelines/empirical/evaluation/outputs/"),
                     ("CausalDiscovery/", "pipelines/empirical/analysis/"),
                     ("Evaluation/", "pipelines/empirical/evaluation/"),
                     ("Integration/outputs/", "pipelines/empirical/integration/runs/"),
                     ("Integration/", "pipelines/empirical/integration/"),
                     ("pipelines/empirical/data/external/", "pipelines/empirical/data/raw/external/"),
                     ("pipelines/empirical/analysis/evaluation/", "pipelines/empirical/evaluation/outputs/"),
                     ("integration/", "pipelines/empirical/integration/")):
        if path.startswith(old):
            return new + path[len(old):]
    return path


def exists(path: str) -> bool:
    return (REPO / path).exists()


class Trace:
    def __init__(self) -> None:
        self.stages: list[dict] = []

    def add(self, name: str, how: str, items: dict, note: str = "") -> None:
        self.stages.append({"stage": name, "known": how, "items": items, "note": note})


def trace_gene(gene: str, panel: str) -> Trace:
    t = Trace()
    g = gene.upper()

    # --- 1. the pipeline run -------------------------------------------------------
    man_p = REPO / "pipelines/empirical/analysis/outputs" / panel / g / "pipeline_manifest.json"
    if not man_p.exists():
        raise SystemExit(f"no pipeline manifest: {rel(man_p)}\n"
                         f"(try --list to see available panels/genes)")
    man = json.loads(man_p.read_text())
    matrix = remap_legacy(rel(man["matrix_tsv"]))
    droot = remap_legacy(rel(man["dataset_root"]))
    sources = {k: remap_legacy(rel(v)) for k, v in man.get("sources", {}).items()
               if isinstance(v, str) and "/" in str(v)}

    t.add("result / pipeline run", REC, {
        "manifest": rel(man_p),
        "gene": man.get("gene", g),
        "methods": ",".join(man.get("methods", [])),
        "edges": rel(man_p.parent / "edges_consensus.tsv"),
        "report": rel(man_p.parent / "pipeline_report.md"),
    }, note=f"params: {json.dumps(man.get('params', {}), separators=(',', '='))[:110]}")

    if sources:
        t.add("external validation evidence", REC, sources,
              note="the 3 peak->gene axes; builders live in pipelines/empirical/evaluation/scripts/")

    t.add("input matrix", REC, {
        "matrix": matrix,
        "run_summary": rel(Path(matrix).parent / "run_summary.json"),
    })

    # --- 2. the dataset ------------------------------------------------------------
    rs_p = REPO / Path(matrix).parent / "run_summary.json"
    if rs_p.exists():
        rs = json.loads(rs_p.read_text())
        t.add("dataset construction", REC, {
            "window_bp": rs.get("window_bp"),
            "locus": f"{rs.get('chrom')}:{rs.get('window_start')}-{rs.get('window_end')}",
            "n_samples (rows)": rs.get("n_samples"),
            "n_peak_variables": rs.get("n_nearby_peak_variables"),
            "strategy": rs.get("matrix_construction_strategy"),
            "quant_mode": rs.get("quant_mode"),
        })

    ps_p = REPO / droot / "panel_summary.json"
    if not ps_p.exists():
        t.add("dataset panel", CONV, {"dataset_root": droot},
              note="panel_summary.json absent — upstream chain unverifiable from here")
        return t
    ps = json.loads(ps_p.read_text())
    matching_dir = remap_legacy(rel(ps["matching_dir"]))
    run_id = ps.get("run_id", "?")

    t.add("dataset panel", REC, {
        "panel_summary": rel(ps_p),
        "gene_panel": remap_legacy(rel(ps["gene_panel"])),
        "marks": ",".join(ps.get("marks", [])),
        "integration run_id": run_id,
    })

    # --- 3. pairing / metacells ----------------------------------------------------
    t.add("cell pairing", REC, {
        "matching_dir": matching_dir,
        "matched_samples": f"{matching_dir}/matched_samples.tsv",
        "neighbor_assignments": remap_legacy(rel(ps.get("neighbor_assignments_tsv", ""))),
    })

    mc = f"pipelines/empirical/analysis/outputs/metacells/{run_id}"
    if exists(mc):
        t.add("metacell assignment", REC if exists(f"{mc}/metacell_summary.json") else CONV, {
            "metacell_dir": mc,
            "assignments": f"{mc}/cell_metacell_assignments.tsv",
            "composition": f"{mc}/metacell_composition.tsv",
        }, note="non-overlapping metacells over the scGLUE latent (honest n)")

    # --- 4. integration ------------------------------------------------------------
    integ = f"pipelines/empirical/integration/runs/scglue_{run_id}"
    t.add("integration (scGLUE)", REC if exists(integ) else CONV, {
        "run_dir": integ,
        "embeddings": f"{integ}/train/all_cells_glue_embeddings.tsv",
        "validation": f"{integ}/train/validation/validation_metrics.json",
    }, note=f"run_id '{run_id}' from panel_summary; directory located by layout convention")

    # --- 5. per-mark inputs --------------------------------------------------------
    inman = REPO / "pipelines/empirical/integration/manifests/scglue_input_manifest.tsv"
    if inman.exists():
        rows = [l.split("\t") for l in inman.read_text().splitlines()[1:] if l.strip()]
        hdr = inman.read_text().splitlines()[0].split("\t")
        items = {}
        for r in rows:
            d = dict(zip(hdr, [c.strip() for c in r]))
            items[d.get("mark", "?")] = d.get("bin_mtx", "")
        t.add("per-mark chromatin matrices", REC, items,
              note=f"{rel(inman)} — regenerate with pipelines/empirical/integration/scripts/setup_integration_workspace.py")

    # --- 6. raw ---------------------------------------------------------------------
    marks = ps.get("marks", [])
    raw = {m: f"pipelines/empirical/data/raw/{m}/" for m in marks if exists(f"pipelines/empirical/data/raw/{m}")}
    raw["GEX"] = "pipelines/empirical/data/raw/GexData/"
    t.add("raw data", CONV, raw,
          note="pipelines/empirical/data/outputs/<MARK>/ is built from pipelines/empirical/data/raw/<MARK>/ by pipelines/empirical/data/scripts/ — "
               "the preprocessing step writes no manifest, so this hop is by layout only")
    return t


def render(t: Trace, gene: str, panel: str) -> None:
    print(f"\nProvenance — {gene}  (panel: {panel})")
    print("=" * 72)
    print("reading upward: each stage is the input to the one above it\n")
    ordered = list(reversed(t.stages))
    for i, s in enumerate(ordered):
        last = i == len(ordered) - 1
        tag = "recorded  " if s["known"] == REC else "CONVENTION"
        rail = " " if last else "│"
        print(f"{'└─' if last else '├─'} [{tag}] {s['stage']}")
        for k, v in s["items"].items():
            if v in (None, ""):
                continue
            missing = " (MISSING)" if isinstance(v, str) and "/" in v and not exists(v) else ""
            print(f"{rail}    {k:<24} {v}{missing}")
        if s["note"]:
            print(f"{rail}    · {s['note']}")
        if not last:
            print("│")
    print("\nLegend: [recorded] = read from a manifest the run wrote."
          "\n        [CONVENTION] = inferred from layout/naming — verify before citing.\n")


def list_available() -> None:
    root = REPO / "pipelines/empirical/analysis/outputs"
    print("Available panels and genes:\n")
    for panel in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("pipeline")):
        genes = sorted(g.name for g in panel.iterdir()
                       if g.is_dir() and (g / "pipeline_manifest.json").exists())
        if genes:
            print(f"  {panel.name}\n      {' '.join(genes)}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene")
    ap.add_argument("--panel", default="pipeline_confounder_panel",
                    help="pipeline output dir under pipelines/empirical/analysis/outputs (default: %(default)s)")
    ap.add_argument("--list", action="store_true", help="list panels/genes that can be traced")
    ap.add_argument("--json", action="store_true", help="emit machine-readable lineage")
    a = ap.parse_args()

    if a.list or not a.gene:
        list_available()
        return 0
    t = trace_gene(a.gene, a.panel)
    if a.json:
        print(json.dumps({"gene": a.gene.upper(), "panel": a.panel, "stages": t.stages}, indent=2))
    else:
        render(t, a.gene.upper(), a.panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
