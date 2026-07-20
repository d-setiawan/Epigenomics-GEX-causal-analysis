#!/usr/bin/env python3
"""Build a cell-type-aligned, target-gene-aware enhancer/promoter -> gene support
table from ENCODE-rE2G predictions.

ENCODE-rE2G (Element -> Gene) predictions are native GRCh38, cell-type-specific,
and carry an explicit TargetGene per regulatory element. That makes them the one
source that can validate peak->gene edges for our monocyte panel, which the
existing evaluation sources (ENCODE peaks / SCREEN / JASPAR / ePerturbDB) cannot:
those are interval/motif-only or have no monocyte / target-gene coverage.

This tool queries the ENCODE portal for rE2G annotations matching a biosample
term (default: CD14-positive monocyte), downloads the "thresholded element gene
links" BED for each donor, merges them (max score per element-gene pair, with a
donor count), and writes the support-template TSV consumed by the evaluation
suite:

    chrom  start  end  gene  support_label  external_id  biosample  notes

Feed the output to the support layer with:

    causal_cli.py support --graph-dir <dir> --re2g-tsv <this_output.tsv>   # first-class 're2g' source

Because the template `gene` column enables exact target-gene matching, a peak
node overlapping an rE2G element whose TargetGene matches the expression node
will validate that peak->gene edge under the suite's existing edge rule.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ENCODE_BASE = "https://www.encodeproject.org"
DEFAULT_BIOSAMPLE = "CD14-positive monocyte"
DEFAULT_LINK_TYPE = "thresholded element gene links"
# rE2G thresholded-link BED columns (bed3+ with a header line starting '#').
COL = {
    "chr": 0, "start": 1, "end": 2, "name": 3, "class": 4,
    "TargetGene": 5, "TargetGeneEnsemblID": 6, "TargetGeneTSS": 7,
    "isSelfPromoter": 8, "CellType": 9, "Score": 18,
}


def http_json(url: str, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def find_rE2G_datasets(biosample: str, assembly: str, timeout: int) -> list[str]:
    q = {
        "type": "Annotation",
        "annotation_type": "element gene regulatory interaction predictions",
        "assembly": assembly,
        "biosample_ontology.term_name": biosample,
        "format": "json",
        "limit": "all",
        "field": "accession",
    }
    url = f"{ENCODE_BASE}/search/?{urllib.parse.urlencode(q)}"
    data = http_json(url, timeout)
    return [g["accession"] for g in data.get("@graph", []) if g.get("accession")]


def pick_link_file(accession: str, link_type: str, assembly: str, timeout: int) -> str | None:
    url = f"{ENCODE_BASE}/annotations/{accession}/?format=json&frame=embedded"
    data = http_json(url, timeout)
    for f in data.get("files", []):
        if not isinstance(f, dict):
            continue
        if (
            f.get("output_type") == link_type
            and str(f.get("file_format")) == "bed"
            and f.get("assembly") == assembly
            and f.get("status") == "released"
            and f.get("href")
        ):
            return f["href"]
    return None


def download_gz_text(href: str, timeout: int) -> list[str]:
    url = href if href.startswith("http") else f"{ENCODE_BASE}{href}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    with gzip.open(io.BytesIO(raw), "rt") as fh:
        return fh.read().splitlines()


# Header column-name -> canonical field. Aliases reconcile the two rE2G BED
# schema versions seen on the ENCODE portal: the 19-col legacy layout (CD14
# monocyte) whose indices match COL above, and the 30-col current layout
# (e.g. naive CD4/CD8 T cell) where TargetGene/ENSG/Score sit at different
# positions. Column order therefore cannot be assumed; resolve it per file.
_HEADER_ALIASES = {
    "#chr": "chr", "chr": "chr",
    "start": "start", "end": "end", "name": "name", "class": "class",
    "TargetGene": "TargetGene",
    "TargetGeneEnsemblID": "TargetGeneEnsemblID",
    "TargetGeneEnsembl_ID": "TargetGeneEnsemblID",
    "TargetGeneTSS": "TargetGeneTSS",
    "isSelfPromoter": "isSelfPromoter",
    "CellType": "CellType",
    "Score": "Score",
}


def resolve_columns(lines: list[str]) -> dict[str, int]:
    """Map field name -> column index from the BED '#'-header if present and
    complete, else fall back to the hardcoded legacy COL layout."""
    for line in lines:
        if line.startswith("#"):
            idx: dict[str, int] = {}
            for i, raw in enumerate(line.split("\t")):
                field = _HEADER_ALIASES.get(raw)
                if field is not None and field not in idx:
                    idx[field] = i
            if all(k in idx for k in COL):
                return idx
            break
        if line:  # first non-comment line and no usable header -> legacy layout
            break
    return dict(COL)


def parse_bed_lines(lines: list[str], source_tag: str) -> list[dict[str, Any]]:
    col = resolve_columns(lines)
    out: list[dict[str, Any]] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) <= col["Score"]:
            continue
        try:
            score = float(p[col["Score"]])
        except ValueError:
            continue
        out.append({
            "chrom": p[col["chr"]],
            "start": int(p[col["start"]]),
            "end": int(p[col["end"]]),
            "gene": p[col["TargetGene"]].strip().upper(),
            "ensg": p[col["TargetGeneEnsemblID"]],
            "elem_class": p[col["class"]],
            "self_promoter": p[col["isSelfPromoter"]],
            "cell_type": p[col["CellType"]],
            "element_id": p[col["name"]],
            "score": score,
            "source_dataset": source_tag,
        })
    return out


def merge_records(records: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    merged: dict[tuple, dict[str, Any]] = {}
    for r in records:
        key = (r["chrom"], r["start"], r["end"], r["gene"])
        cur = merged.get(key)
        if cur is None:
            r = dict(r)
            r["n_donors"] = 1
            r["datasets"] = {r["source_dataset"]}
            merged[key] = r
        else:
            cur["datasets"].add(r["source_dataset"])
            cur["n_donors"] = len(cur["datasets"])
            if r["score"] > cur["score"]:
                cur["score"] = r["score"]
                cur["elem_class"] = r["elem_class"]
                cur["self_promoter"] = r["self_promoter"]
    return merged


def write_template(merged: dict[tuple, dict[str, Any]], out_tsv: Path,
                   biosample: str, min_score: float, genes: set[str] | None) -> int:
    header = ["chrom", "start", "end", "gene", "support_label", "external_id", "biosample", "notes"]
    rows = []
    for r in merged.values():
        if r["score"] < min_score:
            continue
        if genes is not None and r["gene"] not in genes:
            continue
        label = f"rE2G_{r['elem_class']}"
        notes = (
            f"ENCODE-rE2G thresholded element-gene link; score={r['score']:.4f}; "
            f"isSelfPromoter={r['self_promoter']}; ENSG={r['ensg']}; "
            f"n_donors={r['n_donors']}; datasets={','.join(sorted(r['datasets']))}"
        )
        rows.append([
            r["chrom"], str(r["start"]), str(r["end"]), r["gene"], label,
            r["element_id"], f"ENCODE {biosample}", notes,
        ])
    rows.sort(key=lambda x: (x[0], int(x[1]), int(x[2]), x[3]))
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(row) + "\n")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Build ENCODE-rE2G monocyte gene-link support template")
    p.add_argument("--biosample-term", default=DEFAULT_BIOSAMPLE)
    p.add_argument("--assembly", default="GRCh38")
    p.add_argument("--link-type", default=DEFAULT_LINK_TYPE)
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Drop links below this rE2G score (thresholded file already pre-filters).")
    p.add_argument("--max-datasets", type=int, default=0, help="0 = all donors.")
    p.add_argument("--genes", default=None, help="Optional comma-separated TargetGene filter.")
    p.add_argument("--local-bed", action="append", default=[],
                   help="Use local rE2G .bed.gz file(s) instead of fetching (repeatable).")
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    genes = {g.strip().upper() for g in args.genes.split(",")} if args.genes else None
    all_records: list[dict[str, Any]] = []

    if args.local_bed:
        for path in args.local_bed:
            lines = gzip.open(path, "rt").read().splitlines() if path.endswith(".gz") else Path(path).read_text().splitlines()
            all_records += parse_bed_lines(lines, Path(path).stem)
            print(f"[local] {path}: parsed {len(all_records)} cumulative rows", flush=True)
    else:
        datasets = find_rE2G_datasets(args.biosample_term, args.assembly, args.timeout)
        if args.max_datasets > 0:
            datasets = datasets[: args.max_datasets]
        print(f"Found {len(datasets)} rE2G dataset(s) for '{args.biosample_term}': {', '.join(datasets)}", flush=True)
        for acc in datasets:
            href = pick_link_file(acc, args.link_type, args.assembly, args.timeout)
            if not href:
                print(f"  [{acc}] no '{args.link_type}' BED found; skipping", flush=True)
                continue
            lines = download_gz_text(href, args.timeout)
            recs = parse_bed_lines(lines, acc)
            all_records += recs
            print(f"  [{acc}] {len(recs)} links", flush=True)

    if not all_records:
        print("No rE2G records parsed.", file=sys.stderr)
        return 1

    merged = merge_records(all_records)
    n = write_template(merged, Path(args.out_tsv), args.biosample_term, args.min_score, genes)
    print(f"Wrote {n} merged element-gene links -> {args.out_tsv}")
    print(f"(from {len(all_records)} raw links across donors; {len(merged)} unique element-gene pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
