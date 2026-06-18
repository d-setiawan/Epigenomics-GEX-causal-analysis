#!/usr/bin/env python3
import argparse
import gzip
from pathlib import Path

from _common import load_barcode_whitelist, normalize_barcode


def parse_args():
    p = argparse.ArgumentParser(description="Filter fragments file by clean barcode whitelist.")
    p.add_argument("fragments_tsv_gz", help="Input fragments.tsv or fragments.tsv.gz")
    p.add_argument("clean_barcodes_tsv", help="TSV with barcode column")
    p.add_argument("out_tsv", help="Output filtered fragments.tsv (uncompressed)")
    return p.parse_args()


def main():
    args = parse_args()
    keep = load_barcode_whitelist(Path(args.clean_barcodes_tsv))

    opener = gzip.open if args.fragments_tsv_gz.endswith(".gz") else open
    n_in = 0
    n_out = 0

    with opener(args.fragments_tsv_gz, "rt") as r, open(args.out_tsv, "w") as w:
        for line in r:
            if not line or line[0] == "#":
                continue
            n_in += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            bc = normalize_barcode(parts[3])
            if bc in keep:
                w.write(line)
                n_out += 1

    print(f"Wrote {args.out_tsv}: kept {n_out}/{n_in} fragment rows across {len(keep)} clean barcodes.")


if __name__ == "__main__":
    main()
