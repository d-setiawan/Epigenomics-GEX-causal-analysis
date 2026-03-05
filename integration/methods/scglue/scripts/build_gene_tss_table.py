#!/usr/bin/env python3
"""Build a simple gene TSS table from a GTF for scGLUE guidance graph construction.

Output columns:
- chrom
- tss
- gene_name
- gene_id
- strand

Uses gene records by default; can fall back to transcript records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from pathlib import Path
from typing import Dict, Iterator, Optional


ATTR_RE = re.compile(r"\s*([^\s]+)\s+\"([^\"]*)\"\s*;")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("r")


def parse_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for m in ATTR_RE.finditer(attr_text):
        attrs[m.group(1)] = m.group(2)
    return attrs


def iter_gtf(path: Path) -> Iterator[dict]:
    with open_text(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = parts[:9]
            yield {
                "chrom": chrom,
                "feature": feature,
                "start": int(start),
                "end": int(end),
                "strand": strand,
                "attrs": parse_attrs(attrs),
            }


def normalize_chrom(chrom: str, mode: str) -> str:
    if mode == "keep":
        return chrom
    if mode == "add_chr" and not chrom.startswith("chr"):
        return f"chr{chrom}"
    if mode == "remove_chr" and chrom.startswith("chr"):
        return chrom[3:]
    return chrom


def main() -> int:
    p = argparse.ArgumentParser(description="Build gene TSS table from GTF")
    p.add_argument("--gtf", required=True, help="Path to GTF or GTF.GZ")
    p.add_argument("--out", required=True, help="Output TSV path")
    p.add_argument(
        "--feature-priority",
        default="gene,transcript",
        help="Comma-separated feature types to consider in order",
    )
    p.add_argument(
        "--chrom-mode",
        default="keep",
        choices=["keep", "add_chr", "remove_chr"],
        help="Chromosome naming normalization",
    )

    args = p.parse_args()

    gtf = Path(args.gtf)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    wanted = [x.strip() for x in args.feature_priority.split(",") if x.strip()]
    if not wanted:
        raise ValueError("--feature-priority must include at least one feature type")

    # We keep first-seen entry per gene_id to avoid duplicates.
    records_by_gene: Dict[str, dict] = {}
    fallback_gene_name_missing = 0

    for rec in iter_gtf(gtf):
        if rec["feature"] not in wanted:
            continue
        attrs = rec["attrs"]
        gene_id = attrs.get("gene_id", "")
        gene_name = attrs.get("gene_name", "")
        if not gene_id and not gene_name:
            continue
        if not gene_name:
            fallback_gene_name_missing += 1
            gene_name = gene_id
        key = gene_id or gene_name
        # keep first to preserve feature priority if input sorted
        if key in records_by_gene:
            continue
        chrom = normalize_chrom(rec["chrom"], args.chrom_mode)
        strand = rec["strand"]
        # GTF is 1-based inclusive. Convert TSS to 0-based coordinate.
        tss = rec["start"] - 1 if strand != "-" else rec["end"] - 1
        records_by_gene[key] = {
            "chrom": chrom,
            "tss": int(tss),
            "gene_name": gene_name,
            "gene_id": gene_id,
            "strand": strand,
        }

    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["chrom", "tss", "gene_name", "gene_id", "strand"],
            delimiter="\t",
        )
        w.writeheader()
        for rec in sorted(records_by_gene.values(), key=lambda r: (r["chrom"], r["tss"], r["gene_name"])):
            w.writerow(rec)

    print(f"Wrote {len(records_by_gene)} genes to {out}")
    if fallback_gene_name_missing:
        print(f"Note: {fallback_gene_name_missing} rows missing gene_name; used gene_id as fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
