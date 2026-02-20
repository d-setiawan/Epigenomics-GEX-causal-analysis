#!/usr/bin/env python3
import argparse
import csv
import gzip
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path


def normalize_barcode(bc: str) -> str:
    bc = bc.strip()
    if "." in bc and bc.rsplit(".", 1)[-1].isdigit():
        bc = bc.rsplit(".", 1)[0]
    if "-" in bc and bc.rsplit("-", 1)[-1].isdigit():
        bc = bc.rsplit("-", 1)[0]
    return bc


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a clean scCUT&Tag cell-by-chromatin sparse matrix from fragments, peaks, "
            "and clean barcode whitelist."
        )
    )
    p.add_argument("fragments_tsv_gz", help="fragments.tsv or fragments.tsv.gz")
    p.add_argument("peaks_bed", help="BED file with peaks (chr start end [name])")
    p.add_argument("clean_barcodes_tsv", help="TSV with barcode column (from step 03)")
    p.add_argument("out_prefix", help="Output prefix")
    return p.parse_args()


def load_barcodes(path: Path):
    barcodes = []
    seen = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "barcode" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'barcode' column in {path}")
        for row in reader:
            bc = normalize_barcode(row["barcode"])
            if bc and bc not in seen:
                seen.add(bc)
                barcodes.append(bc)
    return barcodes


def load_peaks(path: Path):
    peaks_by_chr = defaultdict(list)
    peak_names = []
    with path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            if end <= start:
                continue
            name = parts[3] if len(parts) >= 4 and parts[3] else f"{chrom}:{start}-{end}"
            peak_idx = len(peak_names)
            peak_names.append(name)
            peaks_by_chr[chrom].append((start, end, peak_idx))

    starts_by_chr = {}
    entries_by_chr = {}
    for chrom, entries in peaks_by_chr.items():
        entries_sorted = sorted(entries, key=lambda x: x[0])
        starts_by_chr[chrom] = [x[0] for x in entries_sorted]
        entries_by_chr[chrom] = entries_sorted

    return peak_names, starts_by_chr, entries_by_chr


def overlapping_peak_indices(chrom, frag_start, frag_end, starts_by_chr, entries_by_chr):
    starts = starts_by_chr.get(chrom)
    entries = entries_by_chr.get(chrom)
    if starts is None or entries is None:
        return []

    # Peaks with start < frag_end can overlap. Walk backward and stop when peak end <= frag_start
    right = bisect_left(starts, frag_end)
    hits = []
    i = right - 1
    while i >= 0:
        p_start, p_end, p_idx = entries[i]
        if p_end <= frag_start:
            break
        if p_start < frag_end and p_end > frag_start:
            hits.append(p_idx)
        i -= 1
    return hits


def main():
    args = parse_args()

    barcodes = load_barcodes(Path(args.clean_barcodes_tsv))
    if not barcodes:
        raise ValueError("No clean barcodes found.")

    barcode_to_col = {bc: i for i, bc in enumerate(barcodes)}

    peak_names, starts_by_chr, entries_by_chr = load_peaks(Path(args.peaks_bed))
    if not peak_names:
        raise ValueError("No peaks found in peaks BED.")

    counts = defaultdict(int)  # key=(row_idx_cell, col_idx_peak)

    opener = gzip.open if args.fragments_tsv_gz.endswith(".gz") else open
    with opener(args.fragments_tsv_gz, "rt") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            chrom = parts[0]
            try:
                frag_start = int(parts[1])
                frag_end = int(parts[2])
            except ValueError:
                continue
            if frag_end <= frag_start:
                continue

            bc = normalize_barcode(parts[3])
            cell_idx = barcode_to_col.get(bc)
            if cell_idx is None:
                continue

            for peak_idx in overlapping_peak_indices(chrom, frag_start, frag_end, starts_by_chr, entries_by_chr):
                counts[(cell_idx, peak_idx)] += 1

    out_prefix = Path(args.out_prefix)
    mtx_out = Path(f"{out_prefix}_chromatin_clean.mtx")
    cells_out = Path(f"{out_prefix}_chromatin_clean_barcodes.tsv")
    peaks_out = Path(f"{out_prefix}_chromatin_clean_features.tsv")

    n_rows = len(barcodes)
    n_cols = len(peak_names)
    nnz = len(counts)

    with mtx_out.open("w") as w:
        w.write("%%MatrixMarket matrix coordinate integer general\n")
        w.write("% rows=cells cols=chromatin_features\n")
        w.write(f"{n_rows} {n_cols} {nnz}\n")
        # MatrixMarket uses 1-based indexing.
        for (cell_idx, peak_idx), v in sorted(counts.items()):
            w.write(f"{cell_idx + 1} {peak_idx + 1} {v}\n")

    with cells_out.open("w") as w:
        w.write("barcode\n")
        for bc in barcodes:
            w.write(f"{bc}\n")

    with peaks_out.open("w") as w:
        w.write("feature\n")
        for p in peak_names:
            w.write(f"{p}\n")

    print(f"Wrote {mtx_out} ({n_rows} cells x {n_cols} features, nnz={nnz})")
    print(f"Wrote {cells_out}")
    print(f"Wrote {peaks_out}")


if __name__ == "__main__":
    main()
