# Monocyte vs T-cell — power head-to-head

**Date:** 2026-07-09. Same 15-gene panel, same peaks, same 300 kb window, same pipeline.
Only the cell subset (and thus metacell count + signal) differs: **monocyte n=133 metacells,
T-cell n=245** (~1.8×). Tools: `null_calibration.py` (α=0.01, 12 perms) + `bootstrap_stability.py`
(30 boots, subsample 0.7, `--permute-null`).

## Results (4 representative genes)

| gene | cell | n | single-run real vs null p95 | calib ratio* | stable (≥0.5) | null stable | core (≥0.85) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CD14 | mono | 133 | 16 vs 14 | 1.0 | 13 | 0 | **3** |
| CD14 | **T** | 245 | 13 vs 18 | 1.5 | 17 | 0 | 0 |
| LSP1 | mono | 133 | 6 vs 5 | 1.7 | **10** | 0 | 1 |
| LSP1 | **T** | 245 | 4 vs 7 | 1.6 | 5 | 0 | 1 |
| IL4R | mono | 133 | 18 vs 14 | 2.2 | 21 | 0 | 0 |
| IL4R | **T** | 245 | 16 vs 17 | 1.6 | 21 | 0 | **3** |
| PLCL1 | mono | 133 | 1 vs 2 | 1.5 | 2 | 0 | 0 |
| PLCL1 | **T** | 245 | 4 vs 4 | 1.9 | **6** | 0 | **4** |

\* calib ratio = null-mean / (α×pairs); 1.0 = perfectly calibrated.

## Interpretation — more cells ≠ uniformly more power

**The null stays 0 in every cell/gene** → bootstrap stability cleanly separates signal from noise
in *both* cell types (both are valid). But the power gain from T-cell's larger n is **gene- and
cell-type-dependent, not uniform:**

- **T-cell wins where the gene has T-cell signal:** PLCL1 (6 stable / 4 core vs 2 / 0) and IL4R
  (3 core vs 0) — the extra n buys high-confidence (core) edges.
- **Monocyte wins for myeloid-biased genes:** LSP1 (10 stable vs 5) and CD14 core (3 vs 0) —
  CD14 is a myeloid marker, weakly expressed in T cells, so more cells can't compensate for a
  smaller effect size.
- **Net on the strictest bar:** T-cell yields **8 core edges vs monocyte's 4** across these 4
  genes — a modest edge, but driven by 2 genes.

**Key confound (important):** these 15 genes were selected for *monocyte* validation richness and
monocyte expression, so the comparison is **biased toward monocyte**. That monocyte still wins on
LSP1/CD14 despite the T-cell n advantage reflects the gene selection, not a fair cell-type test.

## Conclusion

- T-cell's larger n (245 vs 133) is real and **does help for genes with T-cell-specific signal**
  (more core edges), so switching is a legitimate option — but **raw n is not the whole story;
  cell-type-matched effect size matters as much or more.**
- A *fair* evaluation of "should we switch to T cell" requires **redoing gene selection for
  T-cell validation coverage** (T-cell rE2G / eQTL / PCHi-C) and comparing on T-cell-relevant
  genes — not the monocyte-selected panel used here.
- **Recommendation:** monocyte remains a fine default (best integration confidence, and the panel
  is monocyte-tuned). Pursue T-cell only if we commit to a T-cell-specific gene selection +
  T-cell validation DBs; then it plausibly gives more high-confidence edges thanks to n. This was
  a power probe, not the validated-edge comparison (which needs T-cell validation databases).

## Caveats
- Single-run null-calibration is noise in both (expected); stability is the meaningful metric.
- Validated-edge count not computed for T-cell (needs T-cell eQTL/rE2G/PCHi-C — separate effort).
- Genes monocyte-biased (see confound above).
