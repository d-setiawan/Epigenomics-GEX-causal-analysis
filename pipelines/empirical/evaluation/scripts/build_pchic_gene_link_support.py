#!/usr/bin/env python3
"""Build a monocyte physical enhancer/promoter -> gene support table from the
BLUEPRINT promoter-capture Hi-C peak matrix (Javierre et al., Cell 2016).

This is the PHYSICAL peak->gene axis that complements ENCODE-rE2G (prediction) and
eQTL (genetics). Each PCHi-C interaction links a BAIT fragment (a gene promoter,
`baitName`) to an "other end" fragment (`oe*`, a distal region) with a per-cell-type
CHiCAGO score. The monocyte column is `Mon`; a score >= 5 is the standard significance
threshold. So a monocyte-significant interaction whose bait promoter is our target gene
means the OE region physically contacts that gene's promoter in monocytes.

The peak matrix is hg19 and chromosomes are unprefixed ("1"); this tool lifts coordinates
to hg38 (via pyliftover + the UCSC hg19->hg38 chain, auto-staged under pipelines/empirical/data/raw/external/chains)
and writes the support-template TSV consumed by the evaluation suite:

    chrom  start  end  gene  support_label  external_id  biosample  notes

Feed it as the local named source `pchic`:

    causal_cli.py support --graph-dir <dir> --external-support-tsv pchic=<this_output.tsv>

Both orientations are emitted per target gene G:
  - G in baitName  -> the OE fragment is a distal element contacting G's promoter (label pchic_oe)
  - G in oeName    -> the BAIT fragment is a distal element contacting G's promoter (label pchic_bait)
"""

from __future__ import annotations

import argparse
import gzip
import io
import urllib.request
from pathlib import Path

OSF_URL = "https://osf.io/63hh4/download"  # PCHiC_peak_matrix_cutoff5.txt.gz (Javierre 2016)
CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
DEFAULT_CHAIN = Path("pipelines/empirical/data/raw/external/chains/hg19ToHg38.over.chain.gz")
BIOSAMPLE = "BLUEPRINT Monocyte (Javierre 2016 PCHi-C)"


def ensure_chain(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pchic] downloading chain {CHAIN_URL}", flush=True)
        with urllib.request.urlopen(CHAIN_URL, timeout=120) as r, path.open("wb") as fh:
            fh.write(r.read())
    return path


def open_matrix(local: str | None, url: str, timeout: int):
    if local:
        raw = Path(local).read_bytes()
    else:
        print(f"[pchic] downloading peak matrix {url}", flush=True)
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="utf-8")
    return io.StringIO(raw.decode("utf-8"))


def split_genes(name: str) -> list[str]:
    return [g.strip().upper() for g in str(name).replace(",", ";").split(";") if g.strip() and g.strip() != "."]


def main() -> int:
    p = argparse.ArgumentParser(description="Build BLUEPRINT monocyte PCHi-C gene-link support template")
    p.add_argument("--genes", default="CSF1R,CD14,IL1B", help="Comma-separated target gene symbols.")
    p.add_argument("--cell-type-col", default="Mon", help="CHiCAGO score column (default Mon = monocyte).")
    p.add_argument("--min-score", type=float, default=5.0, help="Keep interactions with score >= this (CHiCAGO 5).")
    p.add_argument("--out-tsv", required=True)
    p.add_argument("--local-file", default=None, help="Local PCHiC_peak_matrix_cutoff5.txt(.gz); else download from OSF.")
    p.add_argument("--url", default=OSF_URL)
    p.add_argument("--chain-file", default=str(DEFAULT_CHAIN))
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    from pyliftover import LiftOver  # type: ignore
    lo = LiftOver(str(ensure_chain(Path(args.chain_file))))

    targets = {g.strip().upper() for g in args.genes.split(",") if g.strip()}

    def lift(chrom_noprefix: str, start: int, end: int):
        chrom = "chr" + str(chrom_noprefix)
        a = lo.convert_coordinate(chrom, start)
        b = lo.convert_coordinate(chrom, max(end - 1, start))
        if not a or not b or a[0][0] != b[0][0]:
            return None
        s, e = a[0][1], b[0][1] + 1
        return a[0][0], min(s, e), max(s, e)

    fh = open_matrix(args.local_file, args.url, args.timeout)
    header = fh.readline().rstrip("\n").split("\t")
    col = {name: i for i, name in enumerate(header)}
    for req in ["baitChr", "baitStart", "baitEnd", "baitID", "baitName",
                "oeChr", "oeStart", "oeEnd", "oeID", "oeName", "dist", args.cell_type_col]:
        if req not in col:
            raise ValueError(f"PCHi-C matrix missing column {req!r}; header={header[:15]}...")
    sci = col[args.cell_type_col]

    rows = {}  # (chrom,start,end,gene,orientation) -> best score record
    n_seen = n_kept = 0
    for line in fh:
        p_ = line.rstrip("\n").split("\t")
        if len(p_) <= sci:
            continue
        n_seen += 1
        try:
            score = float(p_[sci])
        except ValueError:
            continue
        if score < args.min_score:
            continue
        bait_genes = set(split_genes(p_[col["baitName"]]))
        oe_genes = set(split_genes(p_[col["oeName"]]))
        dist = p_[col["dist"]]
        for orientation, gene_set, chr_i, s_i, e_i, id_bait, id_oe in [
            ("oe", bait_genes, col["oeChr"], col["oeStart"], col["oeEnd"], p_[col["baitID"]], p_[col["oeID"]]),
            ("bait", oe_genes, col["baitChr"], col["baitStart"], col["baitEnd"], p_[col["baitID"]], p_[col["oeID"]]),
        ]:
            hit = gene_set & targets
            if not hit:
                continue
            lifted = lift(p_[chr_i], int(p_[s_i]), int(p_[e_i]))
            if lifted is None:
                continue
            chrom, start, end = lifted
            for gene in hit:
                key = (chrom, start, end, gene, orientation)
                prev = rows.get(key)
                if prev is None or score > prev["score"]:
                    rows[key] = {
                        "chrom": chrom, "start": start, "end": end, "gene": gene,
                        "orientation": orientation, "score": score, "dist": dist,
                        "external_id": f"bait{id_bait}_oe{id_oe}",
                    }
                    n_kept += 1

    header_out = ["chrom", "start", "end", "gene", "support_label", "external_id", "biosample", "notes"]
    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (r["chrom"], r["start"], r["end"], r["gene"]))
    with out.open("w") as w:
        w.write("\t".join(header_out) + "\n")
        for r in ordered:
            label = f"pchic_{r['orientation']}"
            notes = (f"CHiCAGO {args.cell_type_col}={r['score']:.2f}; orientation={r['orientation']}; "
                     f"dist={r['dist']}; hg19->hg38 lifted")
            w.write("\t".join([r["chrom"], str(r["start"]), str(r["end"]), r["gene"],
                               label, r["external_id"], BIOSAMPLE, notes]) + "\n")
    print(f"[pchic] scanned {n_seen} interactions; wrote {len(ordered)} monocyte gene-links "
          f"(score>={args.min_score}) for {sorted(targets)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
