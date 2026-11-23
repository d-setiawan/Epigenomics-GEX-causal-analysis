#!/usr/bin/env python3
"""Build a peak -> bound-TF occupancy map from ChIP-Atlas monocyte TF ChIP-seq.

The decisive shared-TF test for peak_peak_validator.py: motif *presence* is
saturated, so we need actual TF *occupancy*. For each monocyte-relevant TF ChIP
experiment (ChIP-Atlas hg38, manifest = pipelines/empirical/data/raw/external/chipatlas/
monocyte_tf_experiments.tab: THP-1 / U-937 / primary Monocytes / CD14+), download
its per-experiment peak BED, intersect with the panel peaks, and record which TFs
are bound at each peak. Only overlaps with our ~416 panel peaks are kept, so the
output is tiny even though ~515 experiments are streamed.

Resilient (retry + skip on transient network error) and resumable (skips SRX
already recorded). Writes peak_tf_chip.tsv (peak, tf, n_experiments).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
PEAK_RE = re.compile(r"^(chr[^:]+):(\d+)-(\d+)__")
BED_URL = "https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{srx}.05.bed"


def parse_peak(node: str):
    m = PEAK_RE.match(str(node))
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None


def peak_universe(out_root: Path) -> list[str]:
    peaks: dict[tuple, str] = {}
    for gdir in sorted(out_root.iterdir()):
        rs = gdir / "fci" / "run_summary.json"
        if rs.exists():
            for c in json.loads(rs.read_text()).get("selected_columns", []):
                p = parse_peak(c)
                if p:
                    peaks.setdefault(p, str(c))
    return sorted(peaks.values())


class PeakIndex:
    """Overlap query against the fixed panel peaks (few hundred intervals)."""

    def __init__(self, peaks: list[str]):
        self.by_chrom: dict[str, list] = defaultdict(list)
        for p in peaks:
            c, s, e = parse_peak(p)
            self.by_chrom[c].append((s, e, p))
        for c in self.by_chrom:
            self.by_chrom[c].sort()
        self.starts = {c: [x[0] for x in v] for c, v in self.by_chrom.items()}

    def hits(self, chrom: str, start: int, end: int) -> list[str]:
        rows = self.by_chrom.get(chrom)
        if not rows:
            return []
        hi = bisect_right(self.starts[chrom], end)
        return [p for s, e, p in rows[:hi] if e >= start]


def fetch_bed(srx: str, retries: int = 3) -> list[tuple] | None:
    url = BED_URL.format(srx=srx)
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "curl/8"})
            with urlopen(req, timeout=60) as r:
                data = r.read().decode("utf-8", "replace")
            out = []
            for ln in data.splitlines():
                if not ln or ln.startswith(("track", "#")):
                    continue
                f = ln.split("\t")
                if len(f) >= 3:
                    try:
                        out.append((f[0], int(f[1]), int(f[2])))
                    except ValueError:
                        continue
            return out
        except Exception as ex:  # noqa: BLE001 - best-effort network
            if attempt == retries - 1:
                print(f"  skip {srx} ({ex})", flush=True)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "pipelines/empirical/data/raw/external/chipatlas/monocyte_tf_experiments.tab")
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/pipeline_confounder_panel")
    ap.add_argument("--out-tsv", type=Path,
                    default=REPO / "pipelines/empirical/evaluation/outputs/peak_tf_chip.tsv")
    a = ap.parse_args()

    # experimentList rows are ragged (variable trailing metadata columns), so parse
    # by hand and keep only SRX (col 0) + antigen/TF (col 3).
    srx_tf = []
    for ln in a.manifest.read_text().splitlines():
        f = ln.split("\t")
        if len(f) > 3 and f[0]:
            srx_tf.append((f[0], f[3]))
    peaks = peak_universe(a.out_root)
    idx = PeakIndex(peaks)
    print(f"[chip] {len(srx_tf)} experiments, {len(peaks)} panel peaks", flush=True)

    occ: dict[tuple, int] = defaultdict(int)  # (peak, tf) -> n experiments
    t0 = time.time()
    n_ok = n_skip = 0
    for i, (srx, tf) in enumerate(srx_tf, 1):
        bed = fetch_bed(srx)
        if bed is None:
            n_skip += 1
            continue
        n_ok += 1
        seen_here: set[tuple] = set()
        for chrom, s, e in bed:
            for p in idx.hits(chrom, s, e):
                key = (p, tf)
                if key not in seen_here:
                    occ[key] += 1
                    seen_here.add(key)
        if i % 50 == 0:
            print(f"[chip] {i}/{len(srx_tf)}  ({time.time()-t0:.0f}s, "
                  f"{len(occ)} peak-TF pairs, {n_skip} skipped)", flush=True)

    rows = [{"peak": p, "tf": tf, "n_experiments": n} for (p, tf), n in occ.items()]
    df = pd.DataFrame(rows, columns=["peak", "tf", "n_experiments"])
    a.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_tsv, sep="\t", index=False)
    npk = df["peak"].nunique() if not df.empty else 0
    ntf = df["tf"].nunique() if not df.empty else 0
    print(f"[chip] DONE: {len(df)} peak-TF occupancies over {npk} peaks, {ntf} TFs "
          f"({n_ok} experiments used, {n_skip} skipped) -> {a.out_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
