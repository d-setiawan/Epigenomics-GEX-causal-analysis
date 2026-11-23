#!/usr/bin/env python3
"""Partition-bootstrap with a SCORE-BASED Tetrad algorithm (BOSS-FCI etc.), with Markov verification.

WHY (professor's SuggestionsToTryCausalDiscovery + project_resources/research/DECISIONS): constraint-based FCI on metacells
is partition-unstable (partition_bootstrap_results.md) and — the Markov Checker shows — its dense graph
actually MISFITS the data. Score-based search (BOSS/BOSS-FCI/GRaSP-FCI) is far sparser, fits the data
better, and on CD14 keeps exactly the partition-robust biological edges. If those edges are also
partition-STABLE, the biology-first framing is back.

This runs a chosen Tetrad algorithm on every gene of every existing partition dataset
(monocyte_cuttag_peak_genes_300kb_metacell_pb{0..9}) and reports each peak->gene edge's recovery
frequency across partitions, plus the Markov-check goodness-of-fit per graph. Compare its
edge_frequency table to the FCI one (partition_bootstrap/edge_frequency.tsv).

Runs in the causaldisco env (JAVA_HOME set):
    JAVA_HOME=/home/dgsetiawan/anaconda3/envs/causaldisco \
      /home/dgsetiawan/anaconda3/envs/causaldisco/bin/python \
      pipelines/empirical/analysis/scripts/partition_bootstrap_tetrad.py --algorithm boss_fci
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "pipelines/empirical/analysis/scripts"))
import run_gfci_tetrad as G  # noqa: E402


def run_algo(sub: pd.DataFrame, algorithm: str, alpha: float, penalty: float,
             forbidden, do_markov: bool):
    """Safe-name rename (Tetrad chokes on ':'/'-'), run `algorithm`, map names back."""
    from pytetrad.tools import TetradSearch as ts
    cols = list(sub.columns)
    safe = {c: f"V{i}" for i, c in enumerate(cols)}
    inv = {v: k for k, v in safe.items()}
    s = ts.TetradSearch(sub.rename(columns=safe))
    s.use_sem_bic(penalty_discount=penalty)
    s.use_fisher_z(alpha=alpha)                      # search test
    if do_markov:
        s.use_fisher_z(alpha=alpha, use_for_mc=True)  # markov-check test
    for frm, to in (forbidden or []):
        if frm in safe and to in safe:
            s.set_forbidden(safe[frm], safe[to])
    getattr(s, "run_" + algorithm)()
    g = str(s.get_string())
    if inv:
        pat = re.compile(r"\b(" + "|".join(re.escape(v) for v in
                         sorted(inv, key=len, reverse=True)) + r")\b")
        g = pat.sub(lambda m: inv[m.group(0)], g)
    mk = None
    if do_markov:
        try:
            r = s.markov_check(s.java)
            mk = {"ks_ind": float(r[2]), "frac_dep_ind": float(r[6])}
        except Exception:
            mk = None
    return g, mk


def discover(matrix_tsv: Path, algorithm: str, confounders: str, alpha: float,
             penalty: float, do_markov: bool):
    df = pd.read_csv(matrix_tsv, sep="\t")
    peaks = [c for c in df.columns if str(c).startswith("chr")]
    expr = [c for c in df.columns if str(c).startswith("expr__") and c.endswith("_log1p")]
    conf = G.build_confounder_columns(df, confounders)          # adds conf__* in place
    selected = peaks + expr + conf
    sub = df[selected].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[1] < 3 or len(sub) < 20:
        return [], None
    sub = G.rank_gaussian(sub)
    sub = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1)
    forbidden = G.build_forbidden_pairs(selected, "tiered_distal_promoter_expr")
    graph_str, mk = run_algo(sub, algorithm, alpha, penalty, forbidden, do_markov)
    return G.parse_tetrad_edges(graph_str), mk


def is_gene_edge(e, gene_col):
    return gene_col in (e["node1"], e["node2"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", default="boss_fci",
                    help="TetradSearch run_<algorithm>: boss_fci | grasp_fci | boss | gfci | pc_max | fci")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--genes", default="CD14,IL4R,LSP1,PLCL1,CD52,CD151")
    ap.add_argument("--confounders", default="depth_permark")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--penalty-discount", type=float, default=2.0)
    ap.add_argument("--window-kb", type=int, default=300)
    ap.add_argument("--markov", action="store_true", default=True)
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/partition_bootstrap")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",")]
    genes = [g.strip() for g in a.genes.split(",")]
    ds = REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2"

    # per (gene, canonical undirected edge) -> partitions found; also per-partition markov + counts
    seen: dict[tuple, set] = {}
    directed: dict[tuple, set] = {}
    part_markov: dict[int, list] = {}
    part_counts: dict[int, dict] = {}

    for s in seeds:
        root = ds / f"monocyte_cuttag_peak_genes_{a.window_kb}kb_metacell_pb{s}"
        pg = 0
        for gene in genes:
            mtx = root / gene.lower() / "nearby_peaks" / f"{gene}_nearby_peak_matrix.tsv"
            if not mtx.exists():
                continue
            gene_col = f"expr__{gene}_log1p"
            edges, mk = discover(mtx, a.algorithm, a.confounders, a.alpha,
                                 a.penalty_discount, a.markov)
            if mk:
                part_markov.setdefault(s, []).append(mk)
            for e in edges:
                if not is_gene_edge(e, gene_col):
                    continue
                peak = e["node1"] if e["node1"] != gene_col else e["node2"]
                if not str(peak).startswith("chr"):
                    continue
                key = (gene, peak)
                seen.setdefault(key, set()).add(s)
                pg += 1
                # directed = arrow at gene, tail at peak
                peak_ep = e["endpoint1"] if e["node1"] == peak else e["endpoint2"]
                gene_ep = e["endpoint2"] if e["node1"] == peak else e["endpoint1"]
                if str(gene_ep).upper().startswith("ARROW") and str(peak_ep).upper().startswith("TAIL"):
                    directed.setdefault(key, set()).add(s)
            print(f"  seed {s} {gene}: {sum(1 for e in edges if is_gene_edge(e, gene_col))} peak->gene",
                  flush=True)
        part_counts[s] = {"peak_gene": pg}

    K = len(seeds)
    rows = []
    for (gene, peak), parts in seen.items():
        rows.append({"gene": gene, "peak": peak, "edge": f"{gene}|{peak}",
                     "n_partitions": len(parts), "frequency": len(parts) / K,
                     "n_directed": len(directed.get((gene, peak), set()))})
    freq = pd.DataFrame(rows).sort_values("frequency", ascending=False)
    out = a.out_root / f"edge_frequency_{a.algorithm}.tsv"
    a.out_root.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out, sep="\t", index=False)

    print(f"\n{'='*64}\n{a.algorithm.upper()} partition bootstrap — {K} partitions, {len(genes)} genes\n{'='*64}")
    print("\nper-partition: peak->gene edges + Markov goodness-of-fit (mean over genes)")
    print(f"{'seed':>5} {'peak_gene':>10} {'ks_ind':>8} {'frac_dep_ind':>13}")
    for s in seeds:
        mk = part_markov.get(s, [])
        ks = np.mean([m["ks_ind"] for m in mk]) if mk else float("nan")
        fd = np.mean([m["frac_dep_ind"] for m in mk]) if mk else float("nan")
        print(f"{s:>5} {part_counts.get(s, {}).get('peak_gene', 0):>10} {ks:>8.3f} {fd:>13.3f}")

    print("\npeak->gene edge stability across partitions:")
    for thr in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
        n = int((freq.frequency >= thr).sum())
        print(f"  >= {int(thr*100):>3}% of partitions: {n} edges")
    print(f"\n  total distinct peak->gene edges (union): {len(freq)}")
    print(f"  recovered in ALL {K}: {int((freq.frequency == 1.0).sum())}")
    if len(freq):
        print("\n  top edges by partition frequency:")
        for _, r in freq.head(10).iterrows():
            print(f"     {r.n_partitions:>2}/{K}  (directed {r.n_directed}/{r.n_partitions})  {r.edge}")

    (a.out_root / f"summary_{a.algorithm}.json").write_text(json.dumps({
        "algorithm": a.algorithm, "n_partitions": K, "genes": genes,
        "n_edges_union": len(freq),
        "n_core_all_partitions": int((freq.frequency == 1.0).sum()),
        "freq_ge_50pct": int((freq.frequency >= 0.5).sum()),
        "freq_ge_30pct": int((freq.frequency >= 0.3).sum()),
        "per_partition_markov": {str(s): part_markov.get(s, []) for s in seeds},
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
