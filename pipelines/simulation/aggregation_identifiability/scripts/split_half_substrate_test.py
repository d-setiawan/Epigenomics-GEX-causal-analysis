#!/usr/bin/env python3
"""Split-half test: is the same-mark bidirected enrichment shared *cell substrate* or biology?

The finding (same_mark_finding.md): bidirected `<->` pairs are enriched for being measured on the
SAME histone mark, and it survives conditioning on per-mark sequencing depth. Two live explanations:

  (biology)   same-mark peaks share a chromatin STATE that genuinely co-varies across metacells.
  (substrate) same-mark peaks are quantified from the SAME cells within a metacell (a peak of mark M
              uses that metacell's mark-M cells), so they share per-cell technical noise (capture /
              dropout) that induces correlation with no biological content.

Discriminator. Within each metacell, split that metacell's mark-M cells into two DISJOINT halves H1/H2
and quantify every peak from each half. For a peak pair (i, j):

    rho_same  = corr( X1_i , X1_j )   # both from H1 -> SHARE cells    -> substrate + biology
    rho_cross = corr( X1_i , X2_j )   # i from H1, j from H2 -> DISJOINT cells -> biology only

  * substrate  => rho_cross << rho_same  (breaking the shared cells kills the correlation)
  * biology    => rho_cross ~= rho_same  (both halves estimate the same metacell state)

Cross-mark pairs are a built-in negative control: they never share cells (different marks -> different
cells), so their rho_same ~= rho_cross regardless.

Headline: does the SAME-MARK EXCESS in correlation (same-mark minus cross-mark) survive going
cross-half? If the excess collapses cross-half, the same-mark signal is substrate (technical).

Reuses the pipeline's own per-cell quantifier (stream_raw_region_aggregates), so peak values are
identical to production; only the aggregation cells differ.

    python pipelines/simulation/aggregation_identifiability/scripts/split_half_substrate_test.py            # 15-gene panel, seed 0
    python pipelines/simulation/aggregation_identifiability/scripts/split_half_substrate_test.py --seed 1   # repeat with a different split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "pipelines/empirical/analysis/scripts/commands"))
from generate_monocyte_cuttag_peak_datasets import build_nearby_peak_regions, read_peaks_bed  # noqa: E402
from export_locus_matrix_scglue_matches import stream_raw_region_aggregates  # noqa: E402

sys.path.insert(0, str(REPO / "pipelines/empirical/evaluation/scripts"))
import peak_peak_validator as V  # noqa: E402


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / p


def split_halves(cells: list[str], rng: np.random.Generator) -> tuple[list[str], list[str]]:
    c = list(cells)
    rng.shuffle(c)
    h = len(c) // 2
    return c[:h], c[h:]


def half_aggregate(per_cell: pd.DataFrame, region_cols: list[str],
                   cells: list[str]) -> np.ndarray:
    """Mean peak value over `cells` (rows of per_cell indexed by barcode). NaN if no cells present."""
    sub = per_cell.reindex(cells).dropna(how="all")
    if sub.empty:
        return np.full(len(region_cols), np.nan)
    return sub[region_cols].to_numpy(dtype=float).mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching-dir", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/scglue_pairings/joint_v2/"
                                   "harmonized_coarse__monocyte__metacell")
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "pipelines/empirical/integration/manifests/scglue_input_manifest.tsv")
    ap.add_argument("--panel", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/pipeline_confounder_panel",
                    help="pipeline out-root, for edge classes (bidirected vs not).")
    ap.add_argument("--dataset-root", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/datasets/joint_v2/"
                                   "monocyte_cuttag_peak_genes_300kb_metacell_expanded")
    ap.add_argument("--window-bp", type=int, default=300000)
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=20,
                    help="Random half-splits to average per gene/mark (streams matrices once). "
                         "Averaging kills the per-split noise that dominates at small n.")
    ap.add_argument("--out-json", type=Path,
                    default=REPO / "pipelines/simulation/aggregation_identifiability/outputs/split_half_substrate.json")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    manifest = pd.read_csv(a.manifest, sep="\t")
    assign = pd.read_csv(a.matching_dir / "neighbor_assignments.tsv", sep="\t")
    assign["sample_id"] = assign["sample_id"].astype(str)
    assign["matched_cell"] = assign["matched_cell"].astype(str)
    metacells = sorted(assign["sample_id"].unique())

    genes = sorted(p.name for p in a.dataset_root.iterdir()
                   if p.is_dir() and (p / "nearby_peaks").is_dir())
    panel_genes = {p.name.upper() for p in a.panel.iterdir()
                   if p.is_dir() and (p / "edges_consensus.tsv").exists()}

    all_pairs = []
    for gene in genes:
        GU = gene.upper()
        if GU not in panel_genes:
            continue
        gene_dir = a.dataset_root / gene / "nearby_peaks"
        # gene TSS/strand from the dataset run_summary
        rs = json.loads((gene_dir / "run_summary.json").read_text())
        gene_row = pd.Series({"gene": GU, "chrom": rs["chrom"], "tss": rs["tss"],
                              "strand": rs["strand"]})

        # Stream each mark's per-cell peak values ONCE; keep them + the metacell->cell lists,
        # then do many random splits without re-streaming.
        per_cell_by_mark: dict[str, pd.DataFrame] = {}
        region_cols_by_mark: dict[str, list[str]] = {}
        by_mc_by_mark: dict[str, dict[str, list[str]]] = {}
        mark_of: dict[str, str] = {}
        for mrow in manifest.to_dict("records"):
            mark = str(mrow["mark"])
            peaks_df = read_peaks_bed(resolve(str(mrow["peaks_bed"])), mark=mark)
            regions = build_nearby_peak_regions(gene_row, pd.DataFrame(columns=["chrom", "start", "end", "region_name"]),
                                                peaks_df, mark=mark, window_bp=a.window_bp)
            if regions.empty:
                continue
            region_cols = regions["region_name"].tolist()
            marka = assign[assign["target_mark"].astype(str) == mark]
            sel = marka["matched_cell"].drop_duplicates().tolist()
            per_cell_df, _, _, _ = stream_raw_region_aggregates(
                matrix_path=resolve(str(mrow["bin_mtx"])),
                barcode_path=resolve(str(mrow["bin_barcodes"])),
                feature_path=resolve(str(mrow["bin_features"])),
                selected_barcodes=sel, regions=regions,
                target_sum=a.target_sum, quant_mode="log1p_norm")
            per_cell_by_mark[mark] = per_cell_df.rename(columns={"barcode": "matched_cell"}).set_index("matched_cell")
            region_cols_by_mark[mark] = region_cols
            by_mc_by_mark[mark] = {mc: g["matched_cell"].tolist() for mc, g in marka.groupby("sample_id")}
            for col in region_cols:
                mark_of[col] = mark

        cls: dict[frozenset, str] = {}
        ce = a.panel / GU / "edges_consensus.tsv"
        if ce.exists():
            cdf = pd.read_csv(ce, sep="\t")
            for _, r in cdf[cdf["edge_kind"] == "peak_peak"].iterrows():
                cls[frozenset((str(r["node1"]), str(r["node2"])))] = V.edge_class(r["orientations"])

        def corr(x, y):
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 10:
                return np.nan
            xs, ys = x[m], y[m]
            if xs.std() == 0 or ys.std() == 0:
                return np.nan
            return abs(np.corrcoef(xs, ys)[0, 1])

        # accumulate over K splits: pair -> [sum_same, sum_cross, n]
        acc: dict[tuple, list] = {}
        for _s in range(a.n_splits):
            X1, X2 = {}, {}
            for mark, per_cell in per_cell_by_mark.items():
                rcols = region_cols_by_mark[mark]
                by_mc = by_mc_by_mark[mark]
                m1 = np.full((len(metacells), len(rcols)), np.nan)
                m2 = np.full((len(metacells), len(rcols)), np.nan)
                for r, mc in enumerate(metacells):
                    cells = by_mc.get(mc, [])
                    if len(cells) < 2:
                        continue
                    h1, h2 = split_halves(cells, rng)
                    m1[r] = half_aggregate(per_cell, rcols, h1)
                    m2[r] = half_aggregate(per_cell, rcols, h2)
                for c, col in enumerate(rcols):
                    X1[col] = m1[:, c]; X2[col] = m2[:, c]
            peaks = [p for p in X1 if V.parse_peak(p)]
            for i in range(len(peaks)):
                for j in range(i + 1, len(peaks)):
                    pi, pj = peaks[i], peaks[j]
                    if V.parse_peak(pi)[0] != V.parse_peak(pj)[0]:
                        continue
                    rs_ = np.nanmean([corr(X1[pi], X1[pj]), corr(X2[pi], X2[pj])])
                    rc_ = np.nanmean([corr(X1[pi], X2[pj]), corr(X2[pi], X1[pj])])
                    if np.isnan(rs_) or np.isnan(rc_):
                        continue
                    key = (pi, pj)
                    if key not in acc:
                        acc[key] = [0.0, 0.0, 0]
                    acc[key][0] += rs_; acc[key][1] += rc_; acc[key][2] += 1

        for (pi, pj), (ssum, csum, n) in acc.items():
            if n == 0:
                continue
            all_pairs.append({
                "gene": GU, "same_mark": int(mark_of[pi] == mark_of[pj]),
                "edge_class": cls.get(frozenset((pi, pj)), "non_adjacent"),
                "rho_same": ssum / n, "rho_cross": csum / n,
            })

    df = pd.DataFrame(all_pairs).dropna(subset=["rho_same", "rho_cross"])
    print(f"\n{len(df)} peak pairs with valid split-half correlations "
          f"({df.same_mark.sum()} same-mark)\n")

    def block(sub, name):
        if sub.empty:
            print(f"{name:26s} (no pairs)"); return None
        rs, rc = sub.rho_same.mean(), sub.rho_cross.mean()
        print(f"{name:26s} n={len(sub):4d}  rho_same={rs:.3f}  rho_cross={rc:.3f}  "
              f"drop={rs-rc:+.3f} ({100*(rs-rc)/rs if rs else 0:+.0f}%)")
        return rs, rc

    print("=== ALL adjacent+nonadjacent pairs, by same/cross mark ===")
    sm = block(df[df.same_mark == 1], "same-mark (shares cells)")
    xm = block(df[df.same_mark == 0], "cross-mark (control)")

    print("\n=== BIDIRECTED pairs only ===")
    b = df[df.edge_class == "bidirected"]
    bsm = block(b[b.same_mark == 1], "bidirected same-mark")
    bxm = block(b[b.same_mark == 0], "bidirected cross-mark")

    # Headline is about the BIDIRECTED class specifically (the finding), and the excess is measured
    # over the non-adjacent SAME-MARK baseline -- NOT the all-pairs cross/same difference (which is
    # ~0 and noise-dominated). Baseline: same-mark non-adjacent pairs, which share cells but have no
    # edge; their same-half vs cross-half difference is the pure floor.
    verdict = {}
    nb = df[(df.edge_class == "non_adjacent") & (df.same_mark == 1)]
    base_same = nb.rho_same.mean() if len(nb) else float("nan")
    base_cross = nb.rho_cross.mean() if len(nb) else float("nan")
    bsm_df = df[(df.edge_class == "bidirected") & (df.same_mark == 1)]
    if len(bsm_df) and not np.isnan(base_same):
        rs, rc = bsm_df.rho_same.mean(), bsm_df.rho_cross.mean()
        ex_same = rs - base_same          # excess correlation, peaks share cells
        ex_cross = rc - base_cross        # excess correlation, peaks from DISJOINT cells
        kept = ex_cross / ex_same if ex_same > 1e-6 else float("nan")
        print("\n=== HEADLINE: bidirected same-mark correlation, excess over same-mark non-adjacent baseline ===")
        print(f"  baseline (non-adj same-mark):  same-half {base_same:.3f}  cross-half {base_cross:.3f}")
        print(f"  bidirected same-mark:          same-half {rs:.3f}  cross-half {rc:.3f}   (n={len(bsm_df)})")
        print(f"  excess over baseline:          same-half {ex_same:+.3f}  cross-half {ex_cross:+.3f}")
        print(f"  fraction of excess surviving disjoint cells: {kept:.0%}")
        if kept < 0.35:
            v = "SUBSTRATE — most of the same-mark correlation is shared per-cell noise (technical)."
        elif kept > 0.65:
            v = "MOSTLY BIOLOGICAL — the same-mark correlation is largely genuine metacell-level state."
        else:
            v = "MIXED — a substantial substrate component plus a real biological component."
        print(f"  -> {v}   (n={len(bsm_df)} bidirected same-mark pairs; interpret with n in mind)")
        verdict = {"baseline_same": base_same, "baseline_cross": base_cross,
                   "bidir_same_mark_rho_same": rs, "bidir_same_mark_rho_cross": rc,
                   "excess_same_half": ex_same, "excess_cross_half": ex_cross,
                   "fraction_surviving": kept, "n_bidir_same_mark": int(len(bsm_df)),
                   "verdict": v}

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps({
        "seed": a.seed, "n_pairs": len(df), "n_same_mark": int(df.same_mark.sum()),
        "by_class": df.groupby(["edge_class", "same_mark"])[["rho_same", "rho_cross"]]
                      .mean().reset_index().to_dict("records"),
        **verdict}, indent=2))
    print(f"\nwrote {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
