#!/usr/bin/env python3
"""Validate peak<->peak (element<->element) edges against a biological signature,
with a distance-matched null.

THE CLAIM (MISSION.md): a bidirected `<->` edge in the confounder-conditioned PAG
means the two regulatory elements share a LATENT COMMON CAUSE (a shared TF or a
shared chromatin domain), not a direct link. This is the novel object no
co-accessibility method (Cicero/JRIM/Ziyani) can separate. To make it falsifiable
we test: do `<->` peak pairs carry a shared-driver signature ABOVE a
distance-matched null?

Signature axes (MVP = pchic_hub; others are plug-in modules):
  * pchic_hub  -- two peaks loop to a shared PCHi-C bait (same promoter hub /
                  physical domain). Available offline from the local PCHi-C table.
                  Independent of discovery (PCHi-C is not used to build peak<->peak
                  edges) and of the peak->gene validation (which is target-gene-aware).
  * [TODO] motif_shared  -- shared JASPAR TF motif (needs the online motif scan)
  * [TODO] chip_shared   -- co-bound by the same TF in monocyte ChIP (ChIP-Atlas)
  * [TODO] coaccess       -- Cicero co-accessibility contrast (R/monocle3)

Distance is THE confound (near peaks share everything by chance), so every test is
distance-matched: we bin candidate pairs by genomic distance and permute the
edge-class label WITHIN bins to get the null.

Input: a pipeline out-root with <GENE>/edges_consensus.tsv (+ <GENE>/fci/
run_summary.json for the full peak universe). Run the panel first with
run_gene_pipeline.py --confounders depth.
"""
from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_right
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PEAK_RE = re.compile(r"^(chr[^:]+):(\d+)-(\d+)__")
BAIT_RE = re.compile(r"bait(\d+)")
OE_RE = re.compile(r"oe(\d+)")


def parse_peak(node: str):
    """(chrom, start, end, midpoint) from a peak node name, or None."""
    m = PEAK_RE.match(str(node))
    if not m:
        return None
    chrom, s, e = m.group(1), int(m.group(2)), int(m.group(3))
    return chrom, s, e, (s + e) // 2


def _marks(orient: str) -> list[str]:
    """Edge marks (e.g. '<->','-->','o-o') from the consensus orientations string,
    one per method."""
    toks = [t.split(":", 1)[1] if ":" in t else t for t in str(orient).split(" ; ")]
    out = []
    for t in toks:
        for sym in ("<->", "-->", "<--", "o->", "<-o", "o-o", "---"):
            if sym in t:
                out.append(sym)
                break
    return out


def edge_class(orient: str) -> str:
    marks = _marks(orient)
    if not marks:
        return "other_adjacent"
    if any(m == "<->" for m in marks):
        return "bidirected"
    if all(m in ("-->", "<--") for m in marks):
        return "directed"
    return "other_adjacent"


# ---------------------------------------------------------------- PCHi-C hub index
class PchicHub:
    """Map a peak interval -> set of PCHi-C bait fragment IDs it participates in.
    Two peaks share a hub iff their bait-ID sets intersect (both loop to a common
    promoter bait, or one is that bait)."""

    def __init__(self, pchic_tsv: Path):
        df = pd.read_csv(pchic_tsv, sep="\t")
        # per chrom: sorted starts + (end, baitid) for overlap lookup
        self.by_chrom: dict[str, list] = defaultdict(list)
        for _, r in df.iterrows():
            eid = str(r.get("external_id", ""))
            bait = BAIT_RE.search(eid)
            oe = OE_RE.search(eid)
            # hub key = the bait id (promoter anchor); every end of this loop shares it,
            # so two enhancers looping to a common bait get the same hub key.
            hub = ("b" + bait.group(1)) if bait else ("o" + (oe.group(1) if oe else eid))
            self.by_chrom[str(r["chrom"])].append((int(r["start"]), int(r["end"]), hub))
        for c in self.by_chrom:
            self.by_chrom[c].sort()
        self._starts = {c: [x[0] for x in v] for c, v in self.by_chrom.items()}

    def hubs(self, chrom: str, start: int, end: int) -> set[str]:
        rows = self.by_chrom.get(chrom, [])
        if not rows:
            return set()
        starts = self._starts[chrom]
        # all fragments whose start <= end; filter those whose end >= start (overlap)
        hi = bisect_right(starts, end)
        out = set()
        for i in range(hi):
            s, e, hub = rows[i]
            if e >= start:
                out.add(hub)
        return out


# ------------------------------------------------------------------- pair building
def gene_peaks(out_root: Path, gene: str) -> list[str]:
    """Full peak universe for a gene = chr* selected_columns of its fci run."""
    rs = out_root / gene / "fci" / "run_summary.json"
    if rs.exists():
        cols = json.loads(rs.read_text()).get("selected_columns", [])
        peaks = [c for c in cols if str(c).startswith("chr")]
        if peaks:
            return peaks
    # fallback: peaks appearing in consensus
    ce = out_root / gene / "edges_consensus.tsv"
    if not ce.exists():
        return []
    df = pd.read_csv(ce, sep="\t")
    s = set()
    for _, r in df.iterrows():
        for n in (r["node1"], r["node2"]):
            if str(n).startswith("chr"):
                s.add(str(n))
    return sorted(s)


def load_motif_sets(motif_tsv: Path, min_score: float) -> dict[str, frozenset]:
    """peak -> frozenset(TF matrix_id) for hits at/above min_score."""
    if not motif_tsv or not Path(motif_tsv).exists():
        return {}
    df = pd.read_csv(motif_tsv, sep="\t")
    df = df[df["score"] >= min_score]
    return {p: frozenset(g["matrix_id"].astype(str))
            for p, g in df.groupby("peak")}


def load_chip_sets(chip_tsv: Path, min_exp: int) -> dict[str, frozenset]:
    """peak -> frozenset(bound TF) from ChIP-Atlas occupancy, requiring a TF to be
    bound in >= min_exp experiments (guards against single-experiment noise)."""
    if not chip_tsv or not Path(chip_tsv).exists():
        return {}
    df = pd.read_csv(chip_tsv, sep="\t")
    df = df[df["n_experiments"] >= min_exp]
    return {p: frozenset(g["tf"].astype(str)) for p, g in df.groupby("peak")}


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def build_pairs(out_root: Path, genes: list[str], hub: PchicHub,
                motif_sets: dict[str, frozenset] | None = None,
                chip_sets: dict[str, frozenset] | None = None) -> pd.DataFrame:
    motif_sets = motif_sets or {}
    chip_sets = chip_sets or {}
    rows = []
    for gene in genes:
        peaks = gene_peaks(out_root, gene)
        if len(peaks) < 2:
            continue
        coords = {p: parse_peak(p) for p in peaks}
        peaks = [p for p in peaks if coords[p]]
        # edge class per adjacent pair from the consensus
        cls: dict[frozenset, str] = {}
        ce = out_root / gene / "edges_consensus.tsv"
        if ce.exists():
            cdf = pd.read_csv(ce, sep="\t")
            for _, r in cdf[cdf["edge_kind"] == "peak_peak"].iterrows():
                cls[frozenset((str(r["node1"]), str(r["node2"])))] = edge_class(r["orientations"])
        # PAG degree per peak = how many adjacent partners it has in the consensus graph.
        # Hub peaks are adjacent to many others AND tend to carry more of any signature, so
        # degree is a confound of the same kind as distance — see --degree-bins.
        degree: dict[str, int] = defaultdict(int)
        for key in cls:
            for node in key:
                degree[str(node)] += 1

        # precompute hub sets
        hubs = {p: hub.hubs(coords[p][0], coords[p][1], coords[p][2]) for p in peaks}
        for a, b in combinations(peaks, 2):
            ca, cb = coords[a], coords[b]
            if ca[0] != cb[0]:
                continue  # intra-chromosome only (cis)
            dist = abs(ca[3] - cb[3])
            klass = cls.get(frozenset((a, b)), "non_adjacent")
            ta, tb = motif_sets.get(a, frozenset()), motif_sets.get(b, frozenset())
            cha, chb = chip_sets.get(a, frozenset()), chip_sets.get(b, frozenset())
            da, db = degree[a], degree[b]
            rows.append({"gene": gene, "peak_a": a, "peak_b": b, "distance": dist,
                         "edge_class": klass,
                         "deg_a": da, "deg_b": db,
                         "deg_min": min(da, db), "deg_max": max(da, db), "deg_sum": da + db,
                         "pchic_hub": int(bool(hubs[a] & hubs[b])),
                         "motif_shared": int(bool(ta & tb)),
                         "motif_jaccard": _jaccard(ta, tb),
                         "motif_n_shared": len(ta & tb),
                         "both_have_motifs": int(bool(ta) and bool(tb)),
                         "chip_shared": int(bool(cha & chb)),
                         "chip_jaccard": _jaccard(cha, chb),
                         "chip_n_shared": len(cha & chb),
                         "both_have_chip": int(bool(cha) and bool(chb))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------- distance-matched test
def matched_permutation_test(pairs: pd.DataFrame, pos_class: str, sig: str,
                             n_bins: int, n_perm: int, seed: int,
                             degree_bins: int = 0, degree_col: str = "deg_min") -> dict:
    """Within matched bins, compare signature rate of `pos_class` vs non_adjacent.
    Statistic = signature-rate(pos) - signature-rate(matched null). Null: permute the
    pos/neg label within each bin.

    Bins are quantiles of genomic distance — the dominant confound, since nearby peaks
    share everything by proximity alone. With `degree_bins > 0` the bins become the cross
    product of distance x PAG degree, controlling the second confound: hub peaks are both
    more likely to be adjacent and more likely to carry any given signature. Matching on
    degree asks the sharper question -- among pairs of equal separation AND equal
    connectivity, do bidirected ones differ?"""
    sub = pairs[pairs["edge_class"].isin([pos_class, "non_adjacent"])].copy()
    if sub.empty or (sub["edge_class"] == pos_class).sum() == 0:
        return {"n_pos": 0}
    # distance bins from the pooled distribution (quantile edges)
    q = np.unique(np.quantile(sub["distance"], np.linspace(0, 1, n_bins + 1)))
    sub["bin"] = np.clip(np.searchsorted(q, sub["distance"], side="right") - 1, 0, len(q) - 2)
    if degree_bins > 0 and degree_col in sub.columns:
        dq = np.unique(np.quantile(sub[degree_col], np.linspace(0, 1, degree_bins + 1)))
        if len(dq) > 2:  # degree has spread worth stratifying on
            dbin = np.clip(np.searchsorted(dq, sub[degree_col], side="right") - 1, 0, len(dq) - 2)
            sub["bin"] = sub["bin"].to_numpy() * len(dq) + dbin  # joint distance x degree cell
    is_pos = (sub["edge_class"] == pos_class).to_numpy()
    sigv = sub[sig].to_numpy().astype(float)
    binv = sub["bin"].to_numpy()

    def stat(mask_pos):
        # weight each bin by its number of positives; rate diff pos vs neg per bin
        diffs, wts = [], []
        for b in np.unique(binv):
            m = binv == b
            p = mask_pos & m
            n = (~mask_pos) & m
            if p.sum() == 0 or n.sum() == 0:
                continue
            diffs.append(sigv[p].mean() - sigv[n].mean())
            wts.append(p.sum())
        if not wts:
            return np.nan
        return float(np.average(diffs, weights=wts))

    obs = stat(is_pos)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = is_pos.copy()
        for b in np.unique(binv):
            idx = np.where(binv == b)[0]
            rng.shuffle(perm[idx])  # shuffle labels within bin
        null[i] = stat(perm)
    null = null[~np.isnan(null)]
    p = float((1 + np.sum(null >= obs)) / (1 + len(null))) if len(null) else np.nan
    return {
        "n_pos": int(is_pos.sum()), "n_neg": int((~is_pos).sum()),
        "pos_sig_rate": float(sigv[is_pos].mean()),
        "neg_sig_rate": float(sigv[~is_pos].mean()),
        "matched_rate_diff": obs, "perm_p": p, "n_perm_valid": int(len(null)),
    }


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path,
                    default=repo / "CausalDiscovery/outputs/pipeline_confounder_panel")
    ap.add_argument("--pchic-tsv", type=Path,
                    default=repo / "Data/raw/external/pchic/pchic_mon_gene_links.tsv")
    ap.add_argument("--genes", default="",
                    help="Comma-separated; default = every subdir with edges_consensus.tsv.")
    ap.add_argument("--report-dir", type=Path,
                    default=repo / "Evaluation/outputs")
    ap.add_argument("--motif-tsv", type=Path,
                    default=repo / "Evaluation/outputs/peak_tf_motifs.tsv",
                    help="peak->TF motif hits from scan_peak_motifs.py (shared-TF axis).")
    ap.add_argument("--motif-score", type=float, default=0.90,
                    help="Min JASPAR relative score to count a TF motif hit.")
    ap.add_argument("--chip-tsv", type=Path,
                    default=repo / "Evaluation/outputs/peak_tf_chip.tsv",
                    help="peak->bound-TF occupancy from build_chip_occupancy.py (shared-TF axis).")
    ap.add_argument("--chip-min-exp", type=int, default=1,
                    help="Min ChIP experiments binding a TF at a peak to count it bound.")
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--degree-bins", type=int, default=3,
                    help="Also match the null on PAG degree (0 = distance-only, the pre-2026-07 "
                         "behaviour). Hub peaks are both more adjacent and more signature-rich, so "
                         "degree is a confound of the same kind as distance.")
    ap.add_argument("--degree-col", default="deg_min", choices=["deg_min", "deg_max", "deg_sum"])
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    if a.genes.strip():
        genes = [g.strip().upper() for g in a.genes.split(",") if g.strip()]
    else:
        genes = sorted(p.name for p in a.out_root.iterdir()
                       if (p / "edges_consensus.tsv").exists())
    motif_sets = load_motif_sets(a.motif_tsv, a.motif_score)
    chip_sets = load_chip_sets(a.chip_tsv, a.chip_min_exp)
    print(f"[ppval] genes={len(genes)}  pchic={a.pchic_tsv.name}  "
          f"motif={'yes ('+str(len(motif_sets))+' peaks)' if motif_sets else 'none'}  "
          f"chip={'yes ('+str(len(chip_sets))+' peaks)' if chip_sets else 'none'}")
    hub = PchicHub(a.pchic_tsv)
    pairs = build_pairs(a.out_root, genes, hub, motif_sets, chip_sets)
    if pairs.empty:
        raise SystemExit("no candidate pairs built")
    a.report_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(a.report_dir / "peak_peak_pairs.tsv", sep="\t", index=False)

    POS = ("bidirected", "directed", "other_adjacent")
    # (signature column, subset-mask, human label)
    sigs = [("pchic_hub", None, "PCHi-C shared hub (physical domain)")]
    if motif_sets:
        sigs += [("motif_shared", "both_have_motifs", "shared TF motif: >=1 common"),
                 ("motif_jaccard", "both_have_motifs", "shared TF motif: Jaccard")]
    if chip_sets:
        sigs += [("chip_shared", "both_have_chip", "shared TF ChIP: >=1 co-bound"),
                 ("chip_jaccard", "both_have_chip", "shared TF ChIP: Jaccard")]

    results = {}
    for sig, submask, label in sigs:
        pp = pairs[pairs[submask] == 1] if submask else pairs
        raw = (pp.groupby("edge_class")[sig].agg(["mean", "size"])
               .rename(columns={"mean": "rate", "size": "n_pairs"}))
        tests = {c: matched_permutation_test(pp, c, sig, a.n_bins, a.n_perm, a.seed,
                                             degree_bins=a.degree_bins, degree_col=a.degree_col)
                 for c in POS}
        # distance-only, for side-by-side comparison with the published numbers
        tests_dist = {c: matched_permutation_test(pp, c, sig, a.n_bins, a.n_perm, a.seed,
                                                  degree_bins=0)
                      for c in POS}
        results[sig] = {"label": label, "n_pairs_used": int(len(pp)),
                        "raw_rates": raw.reset_index().to_dict("records"),
                        "matched_tests": tests,
                        "matched_tests_distance_only": tests_dist}
        print(f"\n=== {label}  [{sig}]  (n_pairs={len(pp)}) ===")
        print(raw.to_string())
        for c, t in tests.items():
            if t.get("n_pos", 0) == 0:
                print(f"  {c:15s}: no pairs"); continue
            td = tests_dist.get(c, {})
            print(f"  {c:15s}: n={t['n_pos']:4d}  pos {t['pos_sig_rate']:.4f} vs null "
                  f"{t['neg_sig_rate']:.4f}")
            print(f"  {'':15s}  dist-only      diff={td.get('matched_rate_diff', float('nan')):+.4f}  "
                  f"p={td.get('perm_p', float('nan')):.4f}")
            print(f"  {'':15s}  dist+degree    diff={t['matched_rate_diff']:+.4f}  "
                  f"p={t['perm_p']:.4f}")

    summary = {"n_genes": len(genes), "n_pairs": int(len(pairs)),
               "motif_score_threshold": a.motif_score,
               "degree_bins": a.degree_bins, "degree_col": a.degree_col,
               "signatures": results}
    (a.report_dir / "peak_peak_validation_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
