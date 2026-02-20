#!/usr/bin/env python3
import argparse
import csv
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
        description="Filter a feature-by-cell TSV matrix using a clean barcode whitelist."
    )
    p.add_argument("matrix_tsv", help="Input matrix TSV (row1 header: feature + cell barcodes)")
    p.add_argument("barcode_tsv", help="TSV with a barcode column (e.g. *_clean_barcodes.tsv)")
    p.add_argument("out_tsv", help="Filtered output matrix TSV")
    return p.parse_args()


def load_whitelist(path: Path):
    keep = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "barcode" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'barcode' column in {path}")
        for row in reader:
            bc = normalize_barcode(row["barcode"])
            if bc:
                keep.add(bc)
    return keep


def main():
    args = parse_args()

    keep = load_whitelist(Path(args.barcode_tsv))

    with open(args.matrix_tsv, "r", newline="") as in_f, open(args.out_tsv, "w", newline="") as out_f:
        reader = csv.reader(in_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t")

        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError("Input matrix must have at least two columns (feature + >=1 barcode).")

        cell_headers = header[1:]
        cell_norm = [normalize_barcode(c) for c in cell_headers]
        keep_idx = [i for i, bc in enumerate(cell_norm) if bc in keep]

        out_header = [header[0]] + [cell_headers[i] for i in keep_idx]
        writer.writerow(out_header)

        for row in reader:
            if not row:
                continue
            if len(row) < len(header):
                row = row + ["0"] * (len(header) - len(row))
            out_row = [row[0]] + [row[i + 1] for i in keep_idx]
            writer.writerow(out_row)

    print(f"Wrote {args.out_tsv} with {len(keep_idx)} selected barcodes.")


if __name__ == "__main__":
    main()
