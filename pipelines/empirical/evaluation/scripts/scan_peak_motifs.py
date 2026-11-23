#!/usr/bin/env python3
"""Scan every panel peak for JASPAR TF motif hits -> peak_tf_motifs.tsv.

Feeds the shared-TF signature of peak_peak_validator.py (the direct test of the
MISSION hypothesis: a bidirected `<->` edge = two elements share a latent TRANS
regulator). Reuses the tested JASPAR/UCSC helpers in build_node_support_table.py
(fetch_jaspar_motifs / fetch_ucsc_sequence / best_pwm_hit); sequences + PWMs are
cached, so re-runs are cheap. Stores ALL hits >= a permissive floor with the score,
so the validator can threshold post-hoc without re-scanning.

Online (JASPAR + UCSC). Peak universe = chr* selected_columns across every panel
gene's fci run. Run once; then peak_peak_validator.py --motif-tsv <this>.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "pipelines/empirical/analysis" / "scripts" / "commands"))
import build_node_support_table as B  # noqa: E402

PEAK_RE = __import__("re").compile(r"^(chr[^:]+):(\d+)-(\d+)__")


def parse_peak(node: str):
    m = PEAK_RE.match(str(node))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def peak_universe(out_root: Path) -> list[str]:
    peaks: dict[tuple, str] = {}
    for gdir in sorted(out_root.iterdir()):
        rs = gdir / "fci" / "run_summary.json"
        if not rs.exists():
            continue
        for c in json.loads(rs.read_text()).get("selected_columns", []):
            p = parse_peak(c)
            if p:
                peaks.setdefault(p, str(c))  # dedupe by coordinate
    return sorted(peaks.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "pipelines/empirical/analysis/outputs/pipeline_confounder_panel")
    ap.add_argument("--motif-tsv", type=Path,
                    default=REPO / "pipelines/empirical/evaluation/outputs/peak_tf_motifs.tsv")
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "pipelines/empirical/evaluation/outputs/external/cache")
    ap.add_argument("--genome", default="hg38")
    ap.add_argument("--collection", default="CORE")
    ap.add_argument("--tax-group", default="vertebrates")
    ap.add_argument("--max-motifs", type=int, default=150)
    ap.add_argument("--score-floor", type=float, default=0.80,
                    help="Store every PWM hit >= this; threshold higher in the validator.")
    ap.add_argument("--max-peak-bp", type=int, default=3000)
    a = ap.parse_args()

    a.cache_dir.mkdir(parents=True, exist_ok=True)
    peaks = peak_universe(a.out_root)
    print(f"[scan] {len(peaks)} distinct peaks; fetching JASPAR {a.collection} "
          f"{a.tax_group} (<= {a.max_motifs} motifs) ...", flush=True)
    motifs = B.fetch_jaspar_motifs(matrix_ids=[], max_motifs=a.max_motifs,
                                   collection=a.collection, tax_group=a.tax_group,
                                   cache_dir=a.cache_dir)
    print(f"[scan] {len(motifs)} motifs; scanning peaks ...", flush=True)

    def safe_seq(chrom, start, end, retries=4):
        """fetch_ucsc_sequence raises on transient DNS/HTTP errors; retry w/ backoff
        then skip, so one network blip does not kill the whole scan."""
        for attempt in range(retries):
            try:
                return B.fetch_ucsc_sequence(genome=a.genome, chrom=chrom, start=start,
                                             end=end, cache_dir=a.cache_dir)
            except Exception as ex:  # noqa: BLE001 - network is best-effort
                if attempt == retries - 1:
                    print(f"[scan] skip {chrom}:{start}-{end} after {retries} tries ({ex})",
                          flush=True)
                    return None
                time.sleep(2 * (attempt + 1))
        return None

    rows = []
    n_skipped = 0
    t0 = time.time()
    for i, node in enumerate(peaks, 1):
        chrom, start, end = parse_peak(node)
        if end - start > a.max_peak_bp:
            continue
        seq = safe_seq(chrom, start, end)
        if not seq:
            n_skipped += 1
            continue
        for m in motifs:
            hit = B.best_pwm_hit(seq, m)
            if hit is None or hit[0] < a.score_floor:
                continue
            rows.append({"peak": node, "matrix_id": str(m.get("matrix_id", "")),
                         "tf_name": str(m.get("name", "")), "score": round(float(hit[0]), 4)})
        if i % 50 == 0:
            print(f"[scan] {i}/{len(peaks)} peaks  ({time.time()-t0:.0f}s, "
                  f"{len(rows)} hits)", flush=True)

    df = pd.DataFrame(rows, columns=["peak", "matrix_id", "tf_name", "score"])
    a.motif_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.motif_tsv, sep="\t", index=False)
    npk = df["peak"].nunique() if not df.empty else 0
    print(f"[scan] DONE: {len(df)} hits over {npk} peaks "
          f"({n_skipped} peaks skipped on fetch) -> {a.motif_tsv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
