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


def detect_delimiter(line: str):
    if "\t" in line:
        return "tab"
    if "," in line:
        return "comma"
    return "whitespace"


def split_line(line: str, mode: str):
    s = line.rstrip("\n")
    if mode == "tab":
        return s.split("\t")
    if mode == "comma":
        return s.split(",")
    # Generic whitespace fallback (handles one-or-more spaces)
    return s.split()


def main():
    args = parse_args()

    keep = load_whitelist(Path(args.barcode_tsv))

    with open(args.matrix_tsv, "r", newline="") as in_f, open(args.out_tsv, "w", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t", lineterminator="\n")

        header_line = in_f.readline()
        if not header_line:
            raise ValueError("Input matrix is empty.")

        mode = detect_delimiter(header_line)
        header = split_line(header_line, mode)
        if len(header) < 1:
            raise ValueError("Input matrix header is malformed.")

        first_data = None
        for line in in_f:
            if not line.strip():
                continue
            first_data = split_line(line, mode)
            break
        if first_data is None:
            raise ValueError("Input matrix has no data rows.")

        # Support both common matrix styles:
        # 1) feature label present in header: feature,bc1,bc2,...
        # 2) feature label absent in header (rownames-style): bc1,bc2,... with rows feature,val1,val2,...
        if len(first_data) == len(header) + 1:
            feature_header = "feature"
            cell_headers = header
        elif len(first_data) >= len(header):
            feature_header = header[0]
            cell_headers = header[1:]
        else:
            raise ValueError(
                "Could not reconcile header/data widths. "
                f"header_fields={len(header)}, first_data_fields={len(first_data)}"
            )

        if not cell_headers:
            raise ValueError("No cell barcode columns detected in matrix header.")

        cell_norm = [normalize_barcode(c) for c in cell_headers]
        keep_idx = [i for i, bc in enumerate(cell_norm) if bc in keep]
        out_header = [feature_header] + [cell_headers[i] for i in keep_idx]
        writer.writerow(out_header)

        expected_len = len(cell_headers) + 1

        def write_row(row):
            if len(row) < expected_len:
                row = row + ["0"] * (expected_len - len(row))
            out_row = [row[0]] + [row[i + 1] for i in keep_idx]
            writer.writerow(out_row)

        write_row(first_data)
        for line in in_f:
            if not line.strip():
                continue
            write_row(split_line(line, mode))

    print(f"Wrote {args.out_tsv} with {len(keep_idx)} selected barcodes.")


if __name__ == "__main__":
    main()
