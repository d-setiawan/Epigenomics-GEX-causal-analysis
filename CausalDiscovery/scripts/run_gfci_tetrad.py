#!/usr/bin/env python3
"""Run GFCI (Tetrad / pytetrad) on a nearby-peak matrix.

GFCI (Greedy Fast Causal Inference) is a hybrid score+constraint FCI variant that,
like FCI, returns a PAG (does NOT assume causal sufficiency) but uses a GES-style
score search to seed the skeleton, giving more stable adjacencies at small n.

This driver writes a graph directory that is DROP-IN COMPATIBLE with
`causal_cli.py support` -- it emits the exact same files/columns the FCI path does
(`gfci_edges.tsv` with node1/endpoint1/node2/endpoint2/edge_text, `run_summary.json`
with `selected_columns`, and `selected_matrix.tsv`). Support keys on node names, so
we keep the FCI node naming convention verbatim (e.g. `expr__CD14_log1p`); the
support step already strips the `_log1p` suffix when it matches the peak to the
target gene.

Requires the `causaldisco` env (JDK 21). Before running you MUST:
    export JAVA_HOME=/home/dgsetiawan/anaconda3/envs/causaldisco
    /home/dgsetiawan/anaconda3/envs/causaldisco/bin/python run_gfci_tetrad.py ...
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

# Endpoint char -> canonical name, matching the FCI convention in fci_edges.tsv.
_ENDPOINT = {">": "ARROW", "<": "ARROW", "o": "CIRCLE", "-": "TAIL"}
# Non-feature metadata columns that must never enter the search.
_META_DROP = {
    "sample_id", "cell_rna", "anchor_rank", "mapping_strategy", "n_neighbors",
    "neighbor_weight_mode", "harmonized_coarse", "harmonized_fine", "label_source",
    "coarse_confidence", "fine_confidence", "cell_ontology_id", "metacell_id",
    "row_id",
}


def is_feature_column(col: str) -> bool:
    """Auto feature rule: confounder roots (conf__*) + gene expression (log1p) +
    all chr* peak columns; drop raw_counts, libsize__*, support_*, harmonized_*,
    *_confidence, and metadata. conf__* columns are synthesized by
    build_confounder_columns() before selection, so they only appear when
    --confounders is on."""
    if col in _META_DROP:
        return False
    if col.startswith("conf__"):
        return True
    if col.startswith("chr"):
        return True
    if col.startswith("expr__") and col.endswith("_log1p"):
        return True
    return False


def select_columns(df: pd.DataFrame, columns_arg: str) -> list[str]:
    if columns_arg and columns_arg != "auto":
        wanted = [c.strip() for c in columns_arg.split(",") if c.strip()]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise SystemExit(f"--columns not found in matrix: {missing}")
        return wanted
    cols = [c for c in df.columns if is_feature_column(c)]
    # keep expr node(s) first, matching the FCI selected_columns ordering
    expr = [c for c in cols if c.startswith("expr__")]
    peaks = [c for c in cols if not c.startswith("expr__")]
    return expr + peaks


# --- Confounder roots + mechanistic orientation, mirrored 1:1 from the FCI worker
# (commands/run_pc_causallearn.py) so GFCI honors the SAME tiered background
# knowledge. Tetrad `set_forbidden(FROM, TO)` == forbid FROM->TO, the same
# direction convention as causal-learn `add_forbidden_by_pattern`.
CONFOUNDER_MARKS = ["H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3"]


def node_role(node_name: str) -> str:
    if node_name.startswith("conf__"):
        return "confounder"
    if node_name.startswith("expr__"):
        return "expr"
    if "__" not in node_name or "__H3" not in node_name:
        return "other"
    region = node_name.split("__", 1)[0].lower()
    if "promoter" in region:
        return "promoter"
    if "enhancer" in region or "fire" in region or region.startswith("ure"):
        return "distal"
    return "other"


def build_confounder_columns(df: pd.DataFrame, mode: str) -> list[str]:
    """Synthesize conf__* root columns from libsize__* (mirror of the FCI worker):
    conf__rna_depth = log1p(libsize__rna); conf__chrom_depth = log1p(mean mark depth)."""
    tokens = {t.strip() for t in str(mode).split(",") if t.strip() and t.strip() != "none"}
    added: list[str] = []
    if "depth" in tokens:
        if "libsize__rna" in df.columns:
            df["conf__rna_depth"] = np.log1p(df["libsize__rna"].astype(float))
            added.append("conf__rna_depth")
        mark_cols = [f"libsize__{m}" for m in CONFOUNDER_MARKS if f"libsize__{m}" in df.columns]
        if mark_cols:
            df["conf__chrom_depth"] = np.log1p(df[mark_cols].astype(float).mean(axis=1))
            added.append("conf__chrom_depth")
    if "activity" in tokens and "activity__global" in df.columns:
        # per-metacell GLOBAL chromatin-activity latent (leave-one-locus-out) as a
        # common-cause root; mirrors the FCI worker.
        v = df["activity__global"].astype(float)
        sd = v.std(ddof=0) or 1.0
        df["conf__global_activity"] = (v - v.mean()) / sd
        added.append("conf__global_activity")
    for extra in ("cellcycle", "donor"):
        if extra in tokens:
            for col in [c for c in df.columns
                        if c.startswith(f"{extra[:4]}__") or c.startswith(f"{extra}__")]:
                cc = f"conf__{col.split('__', 1)[-1]}"
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[cc] = df[col].astype(float)
                    added.append(cc)
    return added


def build_forbidden_pairs(selected: list[str], background_mode: str) -> list[tuple[str, str]]:
    """Return (from, to) edges to FORBID, mirroring FCI build_background_knowledge:
      * expr is a sink: forbid expr -> histone  (chromatin upstream of expression)
      * tiered: also forbid promoter -> distal
      * confounders are roots: forbid <anything> -> conf
    """
    conf_nodes = [c for c in selected if node_role(c) == "confounder"]
    expr_nodes = [c for c in selected if node_role(c) == "expr"]
    promoter_nodes = [c for c in selected if node_role(c) == "promoter"]
    distal_nodes = [c for c in selected if node_role(c) == "distal"]
    histone_nodes = (promoter_nodes + distal_nodes
                     + [c for c in selected if node_role(c) == "other" and "__H3" in c])
    pairs: list[tuple[str, str]] = []
    if background_mode in {"minimal_expr_sink", "tiered_distal_promoter_expr"}:
        for expr in expr_nodes:
            for hist in histone_nodes:
                pairs.append((expr, hist))
    if background_mode == "tiered_distal_promoter_expr":
        for promoter in promoter_nodes:
            for distal in distal_nodes:
                pairs.append((promoter, distal))
    if conf_nodes:
        non_conf = [c for c in selected if c not in set(conf_nodes)]
        for conf in conf_nodes:
            for other in non_conf:
                pairs.append((other, conf))
    return pairs


def rank_gaussian(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        x = out[c].to_numpy(dtype=float)
        r = rankdata(x)
        out[c] = norm.ppf((r - 0.5) / len(r))
    return out


_EDGE_LINE = re.compile(r"^\s*\d+\.\s+(.*)$")


def parse_tetrad_edges(graph_str: str) -> list[dict]:
    """Parse the 'Graph Edges:' block of TetradSearch.get_string().

    Each edge line looks like '3. A <-> B' (node names have no spaces). We take the
    first token as node1, the arrow token as the endpoint pair, the next as node2.
    """
    lines = graph_str.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "Graph Edges:")
    except StopIteration:
        return []
    rows: list[dict] = []
    for ln in lines[start + 1:]:
        s = ln.strip()
        if not s:
            break  # blank line terminates the edges block
        m = _EDGE_LINE.match(ln)
        body = m.group(1) if m else s
        toks = body.split()
        if len(toks) < 3:
            continue
        node1, symbol, node2 = toks[0], toks[1], toks[2]
        if len(symbol) != 3 or symbol[1] != "-" or symbol[0] not in _ENDPOINT \
                or symbol[2] not in _ENDPOINT:
            continue  # not an edge symbol we recognise
        endpoint1 = _ENDPOINT[symbol[0]]
        endpoint2 = _ENDPOINT[symbol[2]]
        rows.append({
            "node1": node1, "endpoint1": endpoint1,
            "node2": node2, "endpoint2": endpoint2,
            "edge_text": f"{node1} {symbol} {node2}",
        })
    return rows


def run_gfci(sub_transformed: pd.DataFrame, alpha: float, penalty: float,
             forbidden_pairs: list[tuple[str, str]] | None = None) -> str:
    """Run GFCI with background knowledge.

    CRITICAL gotcha (verified 2026-07-15): Tetrad `Knowledge` matches variables by
    a name spec that SILENTLY FAILS on our node names, which contain ':' and '-'
    (e.g. `chr5:140638860-140639949__H3K4me3`). With such names, `set_forbidden` /
    `add_to_tier` register but bind to nothing, so the search runs as if there were
    no knowledge at all. Fix: rename every variable to a knowledge-safe token
    (`V0`, `V1`, ...) before the search, then translate node names back in the
    output graph string. With safe names the knowledge binds (isForbidden == True)
    and the FGES skeleton becomes confounder-aware.

    NOTE on orientation: in this Tetrad build `Gfci` consumes knowledge for the
    (score-based) skeleton but does NOT propagate it to PAG *endpoint* orientation,
    so cross-tier edges stay `o-o` even though the forbidden directions are
    registered. (Tetrad's constraint-based FCI *does* orient them to `o->`.) The
    directed chromatin->expression claims therefore come from the causal-learn FCI
    primary; GFCI here contributes a confounder-aware, knowledge-constrained
    *adjacency* set as a score-based sensitivity check.
    """
    import re
    from pytetrad.tools import TetradSearch as ts
    pairs = forbidden_pairs or []
    cols = list(sub_transformed.columns)
    safe = {c: f"V{i}" for i, c in enumerate(cols)}
    inv = {v: k for k, v in safe.items()}

    search = ts.TetradSearch(sub_transformed.rename(columns=safe))
    search.use_sem_bic(penalty_discount=penalty)
    search.use_fisher_z(alpha=alpha)
    # set_forbidden(FROM, TO) forbids the directed edge FROM->TO (adjacency may
    # still appear). Names must be the safe tokens so Tetrad actually matches them.
    for frm, to in pairs:
        if frm in safe and to in safe:
            search.set_forbidden(safe[frm], safe[to])
    search.run_gfci()

    graph_str = str(search.get_string())
    if inv:  # map V<i> tokens back to original node names (single word-boundary pass)
        pat = re.compile(r"\b(" + "|".join(re.escape(v) for v in
                         sorted(inv, key=len, reverse=True)) + r")\b")
        graph_str = pat.sub(lambda m: inv[m.group(0)], graph_str)
    return graph_str


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-tsv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--penalty-discount", type=float, default=2.0)
    p.add_argument("--transform", default="rank_gaussian",
                   choices=["none", "rank_gaussian"])
    p.add_argument("--columns", default="auto",
                   help="'auto' = expr__<GENE>_log1p + all chr* peaks; or a "
                        "comma-separated explicit list.")
    p.add_argument("--confounders", default="none",
                   help="Observed confounders as ROOT nodes: 'none' | 'depth' "
                        "(per-metacell library depths from libsize__*) | 'depth,cellcycle' "
                        "Synthesized as conf__* and added to the search. NB: donor is UNAVAILABLE (no shared donor structure across modalities; see docs/LIMITATIONS.md L1).")
    p.add_argument("--background-mode", default="none",
                   choices=["none", "minimal_expr_sink", "tiered_distal_promoter_expr"],
                   help="Mechanistic orientation prior: forbid expr->chromatin (and, when "
                        "tiered, promoter->distal). Mirrors the FCI worker.")
    a = p.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.matrix_tsv, sep="\t")
    # Synthesize confounder roots (conf__*) from libsize__* BEFORE column selection
    # so the auto rule (is_feature_column) picks them up as search variables.
    confounder_cols = build_confounder_columns(df, a.confounders) if a.confounders != "none" else []
    selected = select_columns(df, a.columns)
    # keep confounder roots first, matching the FCI selected_columns ordering
    selected = [c for c in confounder_cols if c in selected] + \
               [c for c in selected if c not in set(confounder_cols)]
    if len(selected) < 2:
        raise SystemExit(f"need >=2 feature columns, selected {selected}")

    forbidden_pairs = build_forbidden_pairs(selected, a.background_mode)

    sub_raw = df[selected].apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)
    sub_input = rank_gaussian(sub_raw) if a.transform == "rank_gaussian" else sub_raw.copy()

    graph_str = run_gfci(sub_input, a.alpha, a.penalty_discount, forbidden_pairs)
    edges = parse_tetrad_edges(graph_str)
    edges_df = pd.DataFrame(edges, columns=["node1", "endpoint1", "node2", "endpoint2", "edge_text"])

    edges_path = a.out_dir / "gfci_edges.tsv"
    edges_df.to_csv(edges_path, sep="\t", index=False)

    # selected_matrix.tsv: raw selected values, one column per node (support only
    # reads the header for node names; run_summary.selected_columns is primary).
    sel_matrix = sub_raw.copy()
    if "sample_id" in df.columns:
        sel_matrix.insert(0, "sample_id", df.loc[sub_raw.index, "sample_id"].values
                          if len(df) == len(sub_raw) else df["sample_id"].iloc[:len(sub_raw)].values)
    sel_path = a.out_dir / "selected_matrix.tsv"
    sel_matrix.to_csv(sel_path, sep="\t", index=False)

    # standardized copy (parity with the FCI dir; z-scored raw features)
    std = sub_raw.copy()
    std = (std - std.mean()) / std.std(ddof=0).replace(0, 1)
    std_path = a.out_dir / "selected_matrix_standardized.tsv"
    std.to_csv(std_path, sep="\t", index=False)

    (a.out_dir / "gfci_graph.txt").write_text(graph_str)

    summary = {
        "method": "gfci",
        "n_edges": int(len(edges_df)),
        "n_columns_selected": int(len(selected)),
        "selected_columns": selected,
        "inputs": {"matrix_tsv": str(a.matrix_tsv)},
        "params": {
            "alpha": a.alpha,
            "penalty_discount": a.penalty_discount,
            "transform": a.transform,
            "columns": a.columns,
            "confounders": a.confounders,
            "background_mode": a.background_mode,
            "n_confounder_nodes": len(confounder_cols),
            "n_forbidden_pairs": len(forbidden_pairs),
        },
        "n_rows_before_dropna": int(len(df)),
        "n_rows_after_dropna": int(len(sub_raw)),
        "outputs": {
            "gfci_edges_tsv": str(edges_path),
            "selected_matrix_tsv": str(sel_path),
            "selected_matrix_standardized_tsv": str(std_path),
            "gfci_graph_txt": str(a.out_dir / "gfci_graph.txt"),
        },
    }
    (a.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[gfci] {a.matrix_tsv.name}: {len(selected)} vars, {len(sub_raw)} rows "
          f"-> {len(edges_df)} edges  (alpha={a.alpha} penalty={a.penalty_discount})")
    for _, r in edges_df.iterrows():
        print(f"    {r['edge_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
