#!/usr/bin/env python3
"""Coordinate liftOver between genome builds (default hg19 <-> hg38).

Several external validation databases (Javierre PCHi-C, DICE, Roadmap E029, ABC, EpiMap,
FANTOM5, Hi-C) are hg19, while our data and the support layer are hg38/GRCh38. This helper
converts a BED-like interval file (or the support-template TSV) between builds so those
sources can be matched against our peaks.

Backends, tried in order:
  1. pyliftover (pure Python; `pip install pyliftover`) -- preferred, no external binary.
  2. UCSC `liftOver` binary on PATH -- fallback.

Chain files are auto-downloaded from UCSC on first use into --chain-dir (default
pipelines/empirical/data/raw/external/chains, gitignored) and reused thereafter:
  hg19->hg38 : https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
  hg38->hg19 : https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz

Usage:
  # BED (chrom,start,end[,...]) -- keeps extra columns, remaps col1-3
  python3 pipelines/empirical/evaluation/scripts/liftover.py --in in.hg19.bed --out out.hg38.bed --from hg19 --to hg38
  # support-template TSV (has a header with chrom/start/end columns)
  python3 pipelines/empirical/evaluation/scripts/liftover.py --in src.tsv --out lifted.tsv --from hg19 --to hg38 --tsv

Unmapped intervals are dropped and counted (written to <out>.unmapped unless --no-unmapped).
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

CHAIN_URLS = {
    ("hg19", "hg38"): "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
    ("hg38", "hg19"): "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz",
}
DEFAULT_CHAIN_DIR = Path("pipelines/empirical/data/raw/external/chains")


def chain_path(from_build: str, to_build: str, chain_dir: Path) -> Path:
    url = CHAIN_URLS.get((from_build, to_build))
    if url is None:
        raise ValueError(f"No chain URL registered for {from_build}->{to_build}; add it to CHAIN_URLS.")
    dest = chain_dir / url.rsplit("/", 1)[-1]
    if not dest.exists():
        chain_dir.mkdir(parents=True, exist_ok=True)
        print(f"[liftover] downloading chain {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    return dest


def _open(path: str):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def find_col_indices(header: list[str]) -> tuple[int, int, int]:
    lower = [c.strip().lower() for c in header]
    def idx(names):
        for n in names:
            if n in lower:
                return lower.index(n)
        return None
    c = idx(["chrom", "chr", "#chr", "chromosome"])
    s = idx(["start", "start_38", "start_19", "chromstart"])
    e = idx(["end", "end_38", "end_19", "chromend"])
    if None in (c, s, e):
        raise ValueError(f"Could not find chrom/start/end columns in header: {header}")
    return c, s, e


def lift_rows_pyliftover(rows, ci, si, ei, from_build, to_build, chain_file):
    from pyliftover import LiftOver  # type: ignore
    lo = LiftOver(str(chain_file))
    kept, unmapped = [], []
    for parts in rows:
        chrom = parts[ci] if parts[ci].startswith("chr") else "chr" + parts[ci]
        try:
            start, end = int(parts[si]), int(parts[ei])
        except ValueError:
            unmapped.append(parts); continue
        a = lo.convert_coordinate(chrom, start)
        b = lo.convert_coordinate(chrom, max(end - 1, start))
        if not a or not b or a[0][0] != b[0][0]:
            unmapped.append(parts); continue
        new = list(parts)
        new[ci], new[si], new[ei] = a[0][0], str(a[0][1]), str(b[0][1] + 1)
        if int(new[si]) > int(new[ei]):
            new[si], new[ei] = new[ei], new[si]
        kept.append(new)
    return kept, unmapped


def have_pyliftover() -> bool:
    try:
        import pyliftover  # noqa: F401
        return True
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="liftOver interval coordinates between genome builds")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--from", dest="from_build", default="hg19")
    p.add_argument("--to", dest="to_build", default="hg38")
    p.add_argument("--chain-dir", default=str(DEFAULT_CHAIN_DIR))
    p.add_argument("--chain-file", default=None, help="Explicit chain file (skips download).")
    p.add_argument("--tsv", action="store_true", help="Input has a header; find chrom/start/end by name.")
    p.add_argument("--no-unmapped", action="store_true", help="Do not write <out>.unmapped.")
    args = p.parse_args()

    chain = Path(args.chain_file) if args.chain_file else chain_path(
        args.from_build, args.to_build, Path(args.chain_dir))

    with _open(args.in_path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        print("[liftover] empty input", file=sys.stderr); return 1

    header = None
    if args.tsv:
        header = lines[0].split("\t")
        ci, si, ei = find_col_indices(header)
        body = [ln.split("\t") for ln in lines[1:]]
    else:
        first = lines[0].split("\t")
        if not first[0].startswith(("chr", "#")) and len(first) >= 3:
            ci, si, ei = 0, 1, 2  # headerless BED
            body = [ln.split("\t") for ln in lines]
        else:
            ci, si, ei = 0, 1, 2
            body = [ln.split("\t") for ln in lines if not ln.startswith("#")]

    if not have_pyliftover() and not shutil.which("liftOver"):
        print(
            "[liftover] ERROR: neither pyliftover nor the UCSC liftOver binary is available.\n"
            "  Install one:  pip install pyliftover   (pure Python, recommended)\n"
            "  or put the UCSC 'liftOver' binary on PATH.\n"
            f"  Chain file is staged at: {chain}",
            file=sys.stderr,
        )
        return 2

    if have_pyliftover():
        kept, unmapped = lift_rows_pyliftover(body, ci, si, ei, args.from_build, args.to_build, chain)
    else:
        # UCSC binary fallback: write a temp BED3+, run liftOver, splice coords back is non-trivial for
        # arbitrary column layouts; for the binary path we support BED (col1-3) only.
        tmp_in = Path(args.out_path + ".tmp.in.bed")
        tmp_out = Path(args.out_path + ".tmp.out.bed")
        tmp_un = Path(args.out_path + ".tmp.unmapped.bed")
        with tmp_in.open("w") as fh:
            for parts in body:
                fh.write("\t".join([parts[ci], parts[si], parts[ei], *parts]) + "\n")
        subprocess.run(["liftOver", str(tmp_in), str(chain), str(tmp_out), str(tmp_un)], check=True)
        kept = []
        with tmp_out.open() as fh:
            for ln in fh:
                q = ln.rstrip("\n").split("\t")
                new = q[3:]
                new[ci], new[si], new[ei] = q[0], q[1], q[2]
                kept.append(new)
        unmapped = []
        for f in (tmp_in, tmp_out, tmp_un):
            f.unlink(missing_ok=True)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        if header is not None:
            fh.write("\t".join(header) + "\n")
        for parts in kept:
            fh.write("\t".join(parts) + "\n")
    if unmapped and not args.no_unmapped:
        with open(args.out_path + ".unmapped", "w") as fh:
            for parts in unmapped:
                fh.write("\t".join(parts) + "\n")

    print(f"[liftover] {args.from_build}->{args.to_build}: {len(kept)} lifted, {len(unmapped)} unmapped -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

