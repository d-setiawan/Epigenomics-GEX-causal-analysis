#!/usr/bin/env python3
"""Build node-level external-support annotations for graph plotting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


COORD_NODE_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def load_selected_nodes(pc_dir: Path) -> list[str]:
    summary_path = pc_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cols = summary.get("selected_columns")
        if cols:
            return [str(c) for c in cols]
    selected_path = pc_dir / "selected_matrix.tsv"
    if selected_path.exists():
        df = pd.read_csv(selected_path, sep="\t", nrows=1)
        return [c for c in df.columns if c not in {"metacell_id", "sample_id", "row_id"}]
    raise FileNotFoundError(f"Could not infer selected nodes from {pc_dir}")


def parse_graph_node(node_name: str, region_lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    if node_name.startswith("expr__"):
        gene = node_name.split("__", 1)[1]
        return {
            "node_name": node_name,
            "node_type": "expression",
            "gene": gene,
            "region_name": "",
            "mark": "",
            "chrom": "",
            "start": "",
            "end": "",
            "interval_source": "",
        }

    if "__" in node_name:
        region_name, mark = node_name.split("__", 1)
    else:
        region_name, mark = node_name, ""

    coord_match = COORD_NODE_RE.match(region_name)
    if coord_match:
        return {
            "node_name": node_name,
            "node_type": "region",
            "gene": "",
            "region_name": region_name,
            "mark": mark,
            "chrom": coord_match.group("chrom"),
            "start": int(coord_match.group("start")),
            "end": int(coord_match.group("end")),
            "interval_source": "node_name",
        }

    region_row = region_lookup.get(region_name)
    if region_row is None:
        return {
            "node_name": node_name,
            "node_type": "region",
            "gene": "",
            "region_name": region_name,
            "mark": mark,
            "chrom": "",
            "start": "",
            "end": "",
            "interval_source": "",
        }

    return {
        "node_name": node_name,
        "node_type": "region",
        "gene": str(region_row.get("gene", "")),
        "region_name": region_name,
        "mark": mark,
        "chrom": str(region_row.get("chrom", "")),
        "start": int(region_row["start"]),
        "end": int(region_row["end"]),
        "interval_source": "locus_config",
    }


def normalize_external_table(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"chrom", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_name} TSV missing required columns {sorted(missing)}: {path}")

    normalized = df.copy()
    normalized["source"] = source_name
    normalized["chrom"] = normalized["chrom"].astype(str)
    normalized["start"] = normalized["start"].astype(int)
    normalized["end"] = normalized["end"].astype(int)
    for col in ["gene", "support_label", "external_id", "biosample", "notes"]:
        if col not in normalized.columns:
            normalized[col] = ""
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized


def overlaps(start_a: int, end_a: int, start_b: int, end_b: int, min_overlap_bp: int) -> bool:
    return min(end_a, end_b) - max(start_a, start_b) >= min_overlap_bp


def build_region_lookup(locus_config_path: Path | None) -> dict[str, dict[str, object]]:
    if locus_config_path is None:
        return {}
    df = pd.read_csv(locus_config_path, sep="\t")
    required = {"region_name", "chrom", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Locus config missing required columns {sorted(missing)}: {locus_config_path}")
    return {
        str(row["region_name"]): row
        for row in df.to_dict(orient="records")
    }


def highlight_class_for_sources(sources: set[str]) -> str:
    if {"eperturbdb", "encode_screen"}.issubset(sources):
        return "both"
    if "eperturbdb" in sources:
        return "eperturbdb_only"
    if "encode_screen" in sources:
        return "encode_screen_only"
    return "none"


SUMMARY_COLUMNS = [
    "node_name",
    "node_type",
    "gene",
    "region_name",
    "mark",
    "chrom",
    "start",
    "end",
    "interval_source",
    "sources",
    "source_count",
    "highlight_class",
    "matched_records",
    "support_labels",
    "external_ids",
    "biosamples",
    "notes",
]


MATCH_COLUMNS = [
    "node_name",
    "node_type",
    "match_type",
    "source",
    "support_label",
    "external_id",
    "biosample",
    "external_gene",
    "external_chrom",
    "external_start",
    "external_end",
    "notes",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Build node-level support annotations for a saved graph directory")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--pc-dir", required=True, help="Graph output directory containing run_summary.json / selected_matrix.tsv")
    p.add_argument("--locus-config", default=None, help="Optional locus config TSV for named-region nodes")
    p.add_argument("--eperturbdb-tsv", default=None, help="TSV extract from ePerturbDB using the evaluation template columns")
    p.add_argument("--encode-screen-tsv", default=None, help="TSV extract from ENCODE SCREEN using the evaluation template columns")
    p.add_argument("--min-overlap-bp", type=int, default=1)
    p.add_argument("--out-tsv", default=None, help="Default: <pc-dir>/node_support.tsv")
    p.add_argument("--matches-tsv", default=None, help="Default: <pc-dir>/node_support_matches.tsv")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    pc_dir = resolve_path(repo_root, args.pc_dir)
    if pc_dir is None or not pc_dir.exists():
        raise FileNotFoundError(f"Missing PC directory: {args.pc_dir}")

    locus_config_path = resolve_path(repo_root, args.locus_config)
    if locus_config_path is not None and not locus_config_path.exists():
        raise FileNotFoundError(f"Missing locus config TSV: {args.locus_config}")

    source_tables: list[pd.DataFrame] = []
    if args.eperturbdb_tsv:
        eperturbdb_path = resolve_path(repo_root, args.eperturbdb_tsv)
        if eperturbdb_path is None or not eperturbdb_path.exists():
            raise FileNotFoundError(f"Missing ePerturbDB TSV: {args.eperturbdb_tsv}")
        source_tables.append(normalize_external_table(eperturbdb_path, "eperturbdb"))
    if args.encode_screen_tsv:
        encode_path = resolve_path(repo_root, args.encode_screen_tsv)
        if encode_path is None or not encode_path.exists():
            raise FileNotFoundError(f"Missing ENCODE SCREEN TSV: {args.encode_screen_tsv}")
        source_tables.append(normalize_external_table(encode_path, "encode_screen"))

    if not source_tables:
        raise ValueError("At least one external support TSV is required")

    region_lookup = build_region_lookup(locus_config_path)
    nodes = load_selected_nodes(pc_dir)
    parsed_nodes = [parse_graph_node(node, region_lookup) for node in nodes]

    external_df = pd.concat(source_tables, ignore_index=True)
    match_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for node in parsed_nodes:
        node_sources: set[str] = set()
        node_labels: list[str] = []
        node_external_ids: list[str] = []
        node_biosamples: list[str] = []
        node_notes: list[str] = []

        for ext in external_df.to_dict(orient="records"):
            gene_match = False
            interval_match = False

            if str(node["node_type"]) == "expression":
                if str(node["gene"]).upper() and str(ext["gene"]).upper() == str(node["gene"]).upper():
                    gene_match = True
            elif node["chrom"] != "":
                if str(ext["chrom"]) == str(node["chrom"]):
                    interval_match = overlaps(
                        int(node["start"]),
                        int(node["end"]),
                        int(ext["start"]),
                        int(ext["end"]),
                        args.min_overlap_bp,
                    )

            if not gene_match and not interval_match:
                continue

            node_sources.add(str(ext["source"]))
            if str(ext["support_label"]):
                node_labels.append(str(ext["support_label"]))
            if str(ext["external_id"]):
                node_external_ids.append(str(ext["external_id"]))
            if str(ext["biosample"]):
                node_biosamples.append(str(ext["biosample"]))
            if str(ext["notes"]):
                node_notes.append(str(ext["notes"]))

            match_type = "gene" if gene_match else "interval"
            match_records.append(
                {
                    "node_name": node["node_name"],
                    "node_type": node["node_type"],
                    "match_type": match_type,
                    "source": ext["source"],
                    "support_label": ext["support_label"],
                    "external_id": ext["external_id"],
                    "biosample": ext["biosample"],
                    "external_gene": ext["gene"],
                    "external_chrom": ext["chrom"],
                    "external_start": ext["start"],
                    "external_end": ext["end"],
                    "notes": ext["notes"],
                }
            )

        summary_records.append(
            {
                "node_name": node["node_name"],
                "node_type": node["node_type"],
                "gene": node["gene"],
                "region_name": node["region_name"],
                "mark": node["mark"],
                "chrom": node["chrom"],
                "start": node["start"],
                "end": node["end"],
                "interval_source": node["interval_source"],
                "sources": "|".join(sorted(node_sources)),
                "source_count": len(node_sources),
                "highlight_class": highlight_class_for_sources(node_sources),
                "matched_records": len(
                    [m for m in match_records if m["node_name"] == node["node_name"]]
                ),
                "support_labels": "|".join(sorted(set(node_labels))),
                "external_ids": "|".join(sorted(set(node_external_ids))),
                "biosamples": "|".join(sorted(set(node_biosamples))),
                "notes": " | ".join(sorted(set(node_notes))),
            }
        )

    out_tsv = (
        resolve_path(repo_root, args.out_tsv)
        if args.out_tsv
        else pc_dir / "node_support.tsv"
    )
    matches_tsv = (
        resolve_path(repo_root, args.matches_tsv)
        if args.matches_tsv
        else pc_dir / "node_support_matches.tsv"
    )
    assert out_tsv is not None
    assert matches_tsv is not None
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    matches_tsv.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS)
    matches_df = pd.DataFrame(match_records, columns=MATCH_COLUMNS)
    summary_df.to_csv(out_tsv, sep="\t", index=False)
    matches_df.to_csv(matches_tsv, sep="\t", index=False)

    print(f"Wrote: {out_tsv}")
    print(f"Wrote: {matches_tsv}")
    print(f"Nodes: {len(summary_records)}")
    print(f"Supported nodes: {int((summary_df['source_count'] > 0).sum())}")
    print(f"Match records: {len(match_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
