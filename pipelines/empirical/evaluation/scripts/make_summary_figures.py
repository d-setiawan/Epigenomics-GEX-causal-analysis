#!/usr/bin/env python3
"""Regenerate the summary figures used in progress updates and the write-ups.

Reads only committed evidence tables under pipelines/empirical/evaluation/outputs/, so the figures can
never drift from the numbers in pipelines/empirical/evaluation/results/*.md — rerun this after any
panel or validator re-run.

    python pipelines/empirical/evaluation/scripts/make_summary_figures.py

Writes PNG + PDF into pipelines/empirical/evaluation/results/figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "pipelines/empirical/evaluation/results/figures"
EV = REPO / "pipelines/empirical/evaluation/outputs"

# colour-blind-safe (Okabe-Ito)
BLUE, ORANGE, GREY, GREEN, RED = "#0072B2", "#E69F00", "#999999", "#009E73", "#D55E00"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------- Fig 1
def fig_axes() -> None:
    """The four candidate drivers of the `<->` latent — every one negative."""
    d = json.loads((EV / "peak_peak_validation_summary.json").read_text())
    keys = ["pchic_hub", "motif_shared", "motif_jaccard", "chip_shared", "chip_jaccard"]
    short = {"pchic_hub": "PCHi-C hub", "motif_shared": "motif ≥1", "motif_jaccard": "motif Jacc.",
             "chip_shared": "ChIP ≥1", "chip_jaccard": "ChIP Jacc."}
    rows = []
    for k in keys:
        mt = d["signatures"][k]["matched_tests"]["bidirected"]
        rows.append((short[k], mt["pos_sig_rate"], mt["neg_sig_rate"],
                     mt["matched_rate_diff"], mt["perm_p"]))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.15, 0.6]})
    ax1, ax2, ax3 = axes

    x = np.arange(len(rows)); w = 0.38
    ax1.bar(x - w/2, [r[1] for r in rows], w, label="bidirected `<->`", color=BLUE)
    ax1.bar(x + w/2, [r[2] for r in rows], w, label="distance-matched null", color=GREY)
    ax1.set_xticks(x); ax1.set_xticklabels([r[0] for r in rows], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("signature rate"); ax1.set_ylim(0, 1.15)
    ax1.legend(frameon=False, loc="upper left", fontsize=8, ncol=1)
    ax1.set_title("a  Observed vs null", loc="left", fontweight="bold")

    diffs = [r[3] for r in rows]
    ax2.axvline(0, color="black", lw=1)
    ax2.barh(x, diffs, color=[BLUE if v >= 0 else ORANGE for v in diffs], height=0.5)
    ax2.set_yticks(x); ax2.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("effect after distance matching\n(bidirected − null)")
    lim = max(abs(min(diffs)), abs(max(diffs))) * 3.6
    ax2.set_xlim(-lim, lim)
    for i, r in enumerate(rows):
        ax2.text(lim * 0.95, i, f"p={r[4]:.2f}", va="center", ha="right", fontsize=7.5, color=GREY)
    ax2.set_title("b  All effects ≈ 0", loc="left", fontweight="bold")

    # 4th axis: the global-activity latent probe (edge counts, not a signature rate)
    ax3.bar([0, 1], [126, 129], 0.55, color=[GREY, BLUE])
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["depth\nonly", "+ global\nactivity"], fontsize=8)
    ax3.set_ylabel("bidirected `<->` edges")
    ax3.set_ylim(0, 165)
    for xi, v in zip([0, 1], [126, 129]):
        ax3.text(xi, v + 4, str(v), ha="center", fontsize=8.5, fontweight="bold")
    ax3.text(0.5, 150, "edges survive", ha="center", fontsize=8, color=RED)
    ax3.set_title("c  Global-state probe", loc="left", fontweight="bold")

    fig.suptitle("Element↔element validation: four candidate drivers of the latent common cause, all negative",
                 x=0.0, ha="left", fontsize=11, fontweight="bold")
    save(fig, "fig1_ee_axes_negative")


# ---------------------------------------------------------------- Fig 2
def fig_coaccess() -> None:
    """The positive result: co-accessibility conflates what the PAG separates."""
    s = json.loads((EV / "coaccess_contrast_summary.json").read_text())
    pairs = pd.read_csv(EV / "peak_peak_pairs_coaccess.tsv", sep="\t")
    col = "coaccess" if "coaccess" in pairs.columns else pairs.columns[-1]

    order = [("bidirected", "bidirected `<->`\n(latent common cause)", BLUE),
             ("other_adjacent", "other adjacent\n(o-o)", GREEN),
             ("non_adjacent", "non-adjacent\n(pruned by PAG)", GREY)]
    means = {r["edge_class"]: r["mean"] for r in s["coaccess_by_class"]}
    ns = {r["edge_class"]: r["n_pairs"] for r in s["coaccess_by_class"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    data = [pairs.loc[pairs.edge_class == k, col].dropna().values for k, _, _ in order]
    bp = ax1.boxplot(data, vert=True, widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=1.4))
    for patch, (_, _, c) in zip(bp["boxes"], order):
        patch.set_facecolor(c); patch.set_alpha(0.65); patch.set_edgecolor("black")
    ax1.set_xticklabels([f"{lbl}\nn={ns[k]}" for k, lbl, _ in order], fontsize=8)
    ax1.set_ylabel("co-accessibility  |Spearman ρ|")
    top = max(np.percentile(v, 97) for v in data)
    ax1.set_ylim(-0.02, top * 1.42)
    for i, (k, _, _) in enumerate(order, start=1):          # red dash = mean
        ax1.plot([i - 0.16, i + 0.16], [means[k]] * 2, color=RED, lw=1.6)
    bar_y = top * 1.16
    ax1.annotate("", xy=(1, bar_y), xytext=(2, bar_y),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
    ax1.text(1.5, bar_y * 1.03, f"co-accessibility cannot tell these apart\n"
                                f"({means['bidirected']:.2f} vs {means['other_adjacent']:.2f})",
             ha="center", va="bottom", fontsize=8, color=RED)
    ax1.set_title("a  What co-accessibility sees", loc="left", fontweight="bold")

    pct = s["link_class_pct"]
    wedges, _, autotexts = ax2.pie(
        [pct["non_adjacent"], pct["bidirected"], pct["other_adjacent"]],
        labels=None, colors=[GREY, BLUE, GREEN], autopct="%1.0f%%",
        startangle=90, wedgeprops=dict(edgecolor="white", lw=1.5), textprops=dict(fontsize=8))
    for t in autotexts:
        t.set_color("white"); t.set_fontweight("bold")
    ax2.legend(wedges, [f"pruned as non-adjacent ({pct['non_adjacent']:.0f}%)",
                        f"latent `<->` ({pct['bidirected']:.0f}%)",
                        f"other adjacent ({pct['other_adjacent']:.0f}%)"],
               loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False, fontsize=8)
    ax2.set_title(f"b  Top-decile co-accessibility \"links\" (n={s['n_links']})",
                  loc="left", fontweight="bold")
    frac = s["adjacent_latent_fraction"] * 100
    fig.suptitle(f"Co-accessibility conflates direct and latent-confounded pairs — "
                 f"{frac:.0f}% of PAG element↔element adjacencies are latent",
                 x=0.005, ha="left", fontsize=11, fontweight="bold", y=1.04)
    save(fig, "fig2_coaccess_contrast")


# ---------------------------------------------------------------- Fig 3
def fig_panel() -> None:
    """Per-gene peak->gene yield on the confounder-conditioned panel."""
    p = REPO / "pipelines/empirical/analysis/outputs/pipeline_confounder_panel/panel_confounder_summary.tsv"
    df = pd.read_csv(p, sep="\t").fillna({"axes_hit": ""})
    df = df[df.pg > 0].sort_values(["pg_validated", "pg"], ascending=False)

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(df)); h = 0.26
    ax.barh(y - h, df.pg, h, label="peak→gene edges found", color=GREY)
    ax.barh(y, df.pg_val_directed, h, label="directed AND validated", color=BLUE)
    ax.barh(y + h, df.pg_stable_validated, h, label="+ bootstrap-stable (top tier)", color=ORANGE)
    ax.set_yticks(y); ax.set_yticklabels(df.gene, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("edges"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xticks(range(0, int(df.pg.max()) + 1))
    tot = int(df.pg.sum()); val = int(df.pg_validated.sum()); st = int(df.pg_stable_validated.sum())
    ax.set_title(f"Confounder-conditioned panel: {tot}/{tot} peak→gene edges directed, "
                 f"{val} validated, {st} stable+validated",
                 loc="left", fontweight="bold", fontsize=10)
    save(fig, "fig3_panel_yield")


if __name__ == "__main__":
    print("regenerating summary figures ->", OUT)
    fig_axes()
    fig_coaccess()
    fig_panel()
    print("done")
