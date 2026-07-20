#!/usr/bin/env python3
import gzip
import sys
from collections import Counter

def count_fragments(fragments_tsv_gz: str) -> Counter:
    """
    fragments.tsv(.gz) format (used by scCUT&Tag/ATAC-like pipelines):
      chr  start  end  barcode  duplicate_count(optional)
    We count lines per barcode (each line is a fragment).
    """
    counts = Counter()
    opener = gzip.open if fragments_tsv_gz.endswith(".gz") else open
    with opener(fragments_tsv_gz, "rt") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            bc = parts[3]
            counts[bc] += 1
    return counts

def main():
    if len(sys.argv) < 3:
        print("Usage: 02_count_fragments.py <fragments.tsv.gz> <out.tsv>", file=sys.stderr)
        sys.exit(1)

    frag_path = sys.argv[1]
    out_path = sys.argv[2]

    c = count_fragments(frag_path)
    with open(out_path, "w") as w:
        w.write("barcode\tn_cuttag_fragments\n")
        for bc in sorted(c):
            w.write(f"{bc}\t{c[bc]}\n")

    print(f"Wrote {out_path} with {len(c)} barcodes.", file=sys.stderr)

if __name__ == "__main__":
    main()
