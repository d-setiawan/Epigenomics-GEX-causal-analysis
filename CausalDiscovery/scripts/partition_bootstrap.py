#!/usr/bin/env python3
"""Partition-level bootstrap for metacell causal discovery.

WHY. Discovery runs on ONE metacell partition (one Leiden seed). We showed that re-seeding changes
which validated edges FCI recovers (9/11 flip) even though aggregate counts are stable
(Evaluation/results/partition_stability_findings.md). Within-partition bootstrap cannot see this — it
resamples rows inside a fixed partition. The honest stability of an edge is the fraction of
*partitions* in which it is recovered. The survey
(CausalDiscovery/results/aggregation_partition_dependence_survey.md) finds this is a textbook
Modifiable Areal Unit Problem and that partition-bootstrap for single-cell causal discovery appears
undone.

WHAT. For each of K Leiden seeds: rebuild metacells -> matching dir -> dataset -> run the panel. Then
report, per edge, its recovery frequency across the K partitions, and per-partition summary stats
(bidirected count, same-mark rate). Each stage is skipped if its output already exists, so the run is
resumable and the expensive builds happen once.

    # pilot: 3 seeds, 6 genes
    python CausalDiscovery/scripts/partition_bootstrap.py --seeds 0,1,2 \
        --genes CD14,IL4R,LSP1,PLCL1,CD52,CD151
    # re-report without rebuilding
    python CausalDiscovery/scripts/partition_bootstrap.py --seeds 0,1,2 --genes ... --analyze-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def sh(cmd: list[str], log: Path) -> None:
    """Run a build step, streaming to a log; raise on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"step failed (see {log}): {' '.join(str(c) for c in cmd)}")


def build_partition(seed: int, genes: list[str], target_mc: int, window_bp: int,
                    confounders: str, methods: str, out_root: Path, log: Path) -> Path:
    """Build metacells->matching->dataset->panel for one seed. Idempotent."""
    win_k = window_bp // 1000
    mc_dir = REPO / f"CausalDiscovery/outputs/metacells/joint_v2_pb{seed}"
    match_dir = (REPO / "CausalDiscovery/outputs/scglue_pairings/joint_v2"
                 / f"harmonized_coarse__monocyte__metacell_pb{seed}")
    ds_root = (REPO / "CausalDiscovery/outputs/datasets/joint_v2"
               / f"monocyte_cuttag_peak_genes_{win_k}kb_metacell_pb{seed}")
    panel = out_root / f"pb_seed{seed}"

    if not (mc_dir / "metacell_summary.json").exists():
        sh([PY, "CausalDiscovery/scripts/metacell_assign.py",
            "--target-metacells", str(target_mc), "--seed", str(seed),
            "--out-dir", str(mc_dir)], log)
    if not (match_dir / "metacell_rna.h5ad").exists():
        sh([PY, "CausalDiscovery/scripts/build_metacell_matching_dir.py",
            "--assignments", str(mc_dir / "cell_metacell_assignments.tsv"),
            "--out-dir", str(match_dir)], log)
    if not (ds_root / "panel_summary.json").exists():
        sh([PY, "CausalDiscovery/scripts/commands/generate_monocyte_cuttag_peak_datasets.py",
            "--matching-dir", str(match_dir), "--rna-h5ad", str(match_dir / "metacell_rna.h5ad"),
            "--gene-panel", "CausalDiscovery/configs/gene_panels/monocyte_expanded_peak_genes.tsv",
            "--window-bp", str(window_bp), "--out-root", str(ds_root)], log)
    for g in genes:
        # guard: a bad token (e.g. a stray dir name) would otherwise crash the whole run
        if not (ds_root / g.lower()).is_dir():
            print(f"[partition_bootstrap] skip {g!r} — no dataset dir under {ds_root.name}", flush=True)
            continue
        if (panel / g / "edges_consensus.tsv").exists():
            continue
        sh([PY, "CausalDiscovery/scripts/run_gene_pipeline.py", "--gene", g,
            "--dataset-root", str(ds_root), "--out-root", str(panel),
            "--methods", methods, "--indep-test", "fisherz", "--transform", "rank_gaussian",
            "--confounders", confounders, "--no-stability"], log)
    return panel


def _is_directed(orient: str) -> bool:
    s = str(orient)
    return (("-->" in s) or ("<--" in s)) and not any(b in s for b in ("o-", "-o", "<->", "---"))


def load_edges(panel: Path, genes: list[str]) -> pd.DataFrame:
    rows = []
    for g in genes:
        f = panel / g / "edges_consensus.tsv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t")
        d["gene"] = g
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    # canonical undirected edge key (peaks unordered)
    d["edge"] = d.apply(lambda r: g_key(r["gene"], r["node1"], r["node2"]), axis=1)
    d["validated"] = d["verdict"].astype(str).str.startswith("VALIDATED")
    d["directed"] = d["orientations"].map(_is_directed)
    d["bidirected"] = d["orientations"].astype(str).str.contains("<->")
    return d


def g_key(gene, n1, n2) -> str:
    a, b = sorted([str(n1), str(n2)])
    return f"{gene}|{a}|{b}"


def same_mark(edge_key: str) -> int:
    parts = edge_key.split("|")
    if len(parts) != 3:
        return 0
    m1 = parts[1].split("__")[-1]
    m2 = parts[2].split("__")[-1]
    return int(m1 == m2 and parts[1].startswith("chr") and parts[2].startswith("chr"))


def analyze(seeds: list[int], genes: list[str], out_root: Path) -> None:
    panels = {s: load_edges(out_root / f"pb_seed{s}", genes) for s in seeds}
    panels = {s: d for s, d in panels.items() if not d.empty}
    K = len(panels)
    if K == 0:
        raise SystemExit("no panels found — build first (drop --analyze-only)")

    # per-edge recovery across partitions
    seen: dict[str, dict] = {}
    for s, d in panels.items():
        for _, r in d.iterrows():
            e = seen.setdefault(r["edge"], {"gene": r["gene"], "kind": r["edge_kind"],
                                            "seeds": set(), "seeds_validated": set(),
                                            "seeds_directed": set(), "seeds_bidirected": set()})
            e["seeds"].add(s)
            if r["validated"]:
                e["seeds_validated"].add(s)
            if r["directed"]:
                e["seeds_directed"].add(s)
            if r["bidirected"]:
                e["seeds_bidirected"].add(s)

    rows = []
    for edge, e in seen.items():
        rows.append({
            "edge": edge, "gene": e["gene"], "kind": e["kind"],
            "n_seeds": len(e["seeds"]), "frequency": len(e["seeds"]) / K,
            "n_validated": len(e["seeds_validated"]),
            "n_directed": len(e["seeds_directed"]),
            "n_bidirected": len(e["seeds_bidirected"]),
        })
    freq = pd.DataFrame(rows).sort_values(["kind", "frequency"], ascending=[True, False])
    out_root.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out_root / "edge_frequency.tsv", sep="\t", index=False)

    pg = freq[freq.kind == "peak_gene"]
    pp = freq[freq.kind == "peak_peak"]

    print(f"\n{'='*66}\nPARTITION BOOTSTRAP — {K} partitions, {len(genes)} genes\n{'='*66}")
    print(f"\nper-partition panel summary:")
    print(f"{'seed':>5} {'peak_gene':>10} {'validated':>10} {'peak_peak':>10} {'bidirected':>11} {'sm_rate':>8}")
    for s, d in panels.items():
        pgd = d[d.edge_kind == "peak_gene"]; ppd = d[d.edge_kind == "peak_peak"]
        bd = ppd[ppd.bidirected]
        smr = bd["edge"].map(same_mark).mean() if len(bd) else float("nan")
        print(f"{s:>5} {len(pgd):>10} {int(pgd.validated.sum()):>10} {len(ppd):>10} "
              f"{len(bd):>11} {smr:>8.3f}")

    def tier(sub, label):
        if sub.empty:
            print(f"  {label}: (none)"); return
        dist = sub.frequency.value_counts().sort_index(ascending=False)
        print(f"  {label}: {len(sub)} distinct edges")
        for fq, n in dist.items():
            print(f"     found in {fq*K:.0f}/{K} partitions ({fq:.0%}): {n} edges")

    print(f"\npeak->gene edge stability across partitions:")
    tier(pg, "all peak->gene")
    tier(pg[pg.n_validated > 0], "validated in >=1 partition")

    core = pg[pg.frequency == 1.0]
    corev = core[core.n_validated == K]
    print(f"\n  CORE (found in ALL {K} partitions): {len(core)} peak->gene edges")
    print(f"  CORE + validated in all: {len(corev)}")
    for _, r in core.sort_values("n_validated", ascending=False).iterrows():
        print(f"     [{r.n_seeds}/{K}, val {r.n_validated}/{K}] {r.edge}")

    (out_root / "partition_bootstrap_summary.json").write_text(json.dumps({
        "n_partitions": K, "seeds": list(panels), "genes": genes,
        "n_peak_gene_edges_union": int(len(pg)),
        "n_peak_gene_core_allpartitions": int(len(core)),
        "n_peak_gene_core_validated": int(len(corev)),
        "peak_gene_frequency_hist": pg.frequency.value_counts().sort_index().to_dict(),
    }, indent=2))
    print(f"\nwrote {out_root/'edge_frequency.tsv'} and summary.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated Leiden seeds (= partitions)")
    ap.add_argument("--genes", default="CD14,IL4R,LSP1,PLCL1,CD52,CD151")
    ap.add_argument("--target-metacells", type=int, default=150)
    ap.add_argument("--window-bp", type=int, default=300000)
    ap.add_argument("--confounders", default="depth_permark")
    ap.add_argument("--methods", default="fci,pc")
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "CausalDiscovery/outputs/partition_bootstrap")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip() != ""]
    genes = [g.strip() for g in a.genes.split(",") if g.strip()]

    if not a.analyze_only:
        for s in seeds:
            log = a.out_root / "build_logs" / f"pb_seed{s}.log"
            print(f"[partition_bootstrap] building/using partition seed {s} ...", flush=True)
            build_partition(s, genes, a.target_metacells, a.window_bp,
                            a.confounders, a.methods, a.out_root, log)
    analyze(seeds, genes, a.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
