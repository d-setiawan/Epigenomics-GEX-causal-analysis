# Null-calibration findings — the graphs are currently noise-dominated

**Date:** 2026-07-09 · Tool: [`scripts/null_calibration.py`](../scripts/null_calibration.py) ·
Config: FCI · fisherz · rank_gaussian · α=0.05 · depth 2 · `tiered_distal_promoter_expr` · 20 permutations.

## What was tested

For each gene's 300 kb kNN-avg matrix: run FCI on the real matrix, then on 20 matrices
where **every feature column is independently permuted across rows** (destroys all
cross-column structure → the true graph is empty → every surviving edge is a false
positive). Compare the real edge count to the null distribution.

## Result — real ≈ noise for all three genes

| gene | vars | pairs | REAL edges | NULL mean | NULL p95 | REAL peak→gene | NULL peak→gene/run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CD14 | 44 | 946 | 56 | 51.7 | 61 | 2 | 1.6 (65% of runs) |
| CSF1R | 20 | 190 | 10 | 9.2 | 13 | 0 | 0.7 (50%) |
| IL1B | 25 | 300 | 6 | 12.4 | 17 | 1 | 1.05 (75%) |

**The real edge counts fall inside (CD14, CSF1R) or below (IL1B) the null band.** The
number of edges tracks `α × pairs` (CD14 0.05×946≈47; CSF1R ≈9.5; IL1B ≈15), i.e. the
multiple-testing floor, not signal. The peak→gene edges — the ones we care about — occur
just as often under pure noise as in the real data.

## Interpretation — this reframes the aggregation question

The dominant problem is **not** kNN row-dependence alone. It is that discovery runs
**uncorrected at α=0.05 across hundreds of variable pairs**, so the output is a
multiple-testing artifact. Two distinct pathologies, both pushing the same way:

1. **Multiple testing (what this test measures):** α×pairs false edges are expected and
   observed. No correction is applied, so ~47 of CD14's 56 edges are the noise floor.
2. **Effective-sample-size inflation (measured separately):** kNN-averaging reuses each
   chromatin cell 11–17× (see ROADMAP §Q2), so the fisherz `n=1561` is optimistic — the
   real p-values are even more anti-conservative than the permutation null (which removes
   the dependence) can show. So the true situation is at least this bad.

## What actually fixes noise-dominated graphs (revised plan)

Aggregation to honest `n` is necessary for *valid* inference but **insufficient alone** —
it lowers `n` (less power) without touching the α×pairs floor. The levers that recover
signal, in priority order:

1. **Stability selection / bootstrap** *(primary)* — keep only edges that recur across
   resamples. Real edges recur; the ~47 noise edges flicker. This is the single biggest
   separator and is already ROADMAP Q3. *(Row-subsampling — donor blocks are unavailable, [L1](../../../../project_resources/research/LIMITATIONS.md#l1-donor-is-not-an-available-variable).)*
2. **Multiple-testing control** — FDR or a much stricter α on the CI tests; directly cuts
   the α×pairs floor.
3. **Fewer variables** — tighter window / elastic-net screen → fewer pairs → lower floor
   (CSF1R's 20 vars already far cleaner than CD14's 44).
4. **Honest `n` via non-overlapping metacells** — makes the per-test p-values valid.

**Yardstick:** every one of these changes should be re-run through `null_calibration.py`.
A config is only worth using when **REAL edges rise clearly above the NULL p95** — that is
the objective bar the current pipeline fails.

## Caveat on the test

Column-permutation removes row-dependence, so this null measures the multiple-testing
floor at `n=n_rows`; it does not itself exhibit the n-inflation (that needs the `n_eff`
analysis). Both diagnostics agree the graphs are not yet interpretable.
