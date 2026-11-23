# Sparse-count CI and QC-selection audit

_Integration recoverability benchmark; completed simple seed-0 audit, 2026-10-04._

---

## Verdict and scope

No evaluator/API misuse was identified, but the sparse-count null calculation is unreliable in this completed run. Results remain useful as **end-to-end pipeline diagnostics**, not calibrated causal evidence or an isolated measure of integration quality. A completed execution is not evidence of successful recovery.

The audit inspected the [combined simple report](runs/simple/realistic/evaluation_scglue_scmrdr_seeds0/report.json), its saved readouts and query receipts, [evaluate.py](evaluate.py), and the installed causal-learn implementation. One existing query was deterministically reproduced for audit only; no alternative tests, thresholds, seeds, or datasets were tried. This note preserves that already-completed work without rerunning it. See [RESULTS.md](RESULTS.md) for the broader results summary.

## Configuration and implementation checks

The recorded configuration is `causal-learn==0.1.4.5`, `test="rcit"`, `rcit=true`, `approximation="hbe"`, `num_f=100`, `num_f2=5`, `alpha=0.01`, `seed=0`, and `min_cells=20`. The report records these settings and backend provenance under `inputs.configuration` and `inputs.backend_provenance`.

- Original integer counts receive `log1p` exactly once. Native RCIT/RIT subsequently standardizes its inputs and random features; no duplicated count transformation was found.
- `rcit=true` reaches the conditional backend, which augments the tested Y representation with Z. Empty conditioning sets correctly use native RIT, whose five marginal random features match this configuration.
- HBE is explicitly selected and called as an upper-tail probability. No silent alternate approximation is used. Query-specific seeds are deterministic, NumPy state is restored, and caches are local to the test object.
- Exact constant columns and nonfinite results are rejected. Near-constant columns are not excluded. Native bandwidths use positive distances among the first 500 standardized rows; the inspected readouts have detections in those prefixes, so an empty bandwidth subset does not explain these calls.
- Conditional testing uses a `1e-10` ridge on the 100-feature conditioning covariance. Low-support count variables make that covariance rank-deficient and potentially numerically fragile. However, the false cross-assay calls already reject **marginal** independence, before this calculation.

Installed source hashes matched the report during the audit:

```text
evaluate.py
f2c8536be44fe07081ed582cc1e79deb55fd1d962319e9968931799ee13eb171
causallearn/utils/cit.py
0d3db11083985b7cb0edbaa44662307c28f27bc135053a65751cd52f451ddc8f
causallearn/utils/RCIT/RCIT.py
6dd2e8f0275efbc2fa5f6253a9ca71c7329d84788d68eeae7b4607ffc5b75e06
```

The installed backend is the implementation audited here; this does not claim byte-for-byte equivalence to the authors' original R implementation.[^2]

## Observed failures and exact reproduction

Paired counts returned two false cross-assay adjacencies relative to the source count-measurement PAG: `RNA_sentinel–H3K4me3_core` (marginal p-value `2.0791934085551134e-7`) and `H3K27ac_core–H3K4me3_sentinel` (`0.001924835258493962`). Random matching returned four cross-assay calls. All returned paired/random count edges had zero co-detections and small negative log-count correlations.

The single audit-only reproduction used [the existing random readout](runs/simple/realistic/evaluation_scglue_scmrdr_seeds0/readouts/random_reference_5c5f1d77.npz), its recorded seed, and unchanged native RIT/HBE:

| Quantity | Recorded or reproduced value |
| --- | ---: |
| Query | `H3K27ac_core–H3K4me3_core` |
| Conditioning set | Empty |
| Query seed | 1332069278 |
| Cells | 2,000 |
| Detections, X / Y | 17 / 18 |
| Co-detections | 0 |
| Expected co-detections under independence | 0.153 |
| Pearson correlation of log-counts | -0.008823634926043466 |
| Saved and reproduced p-value | 0.002902283768577707 |
| Difference from saved p-value | 0.0 |
| Native test statistic | 3.8928266654046824 |
| Native bandwidth, X / Y | 10.890193013765773 / 10.586034798275904 |
| Random-feature rank, X / Y | 1 / 1 |
| Product-covariance rank | 1 |
| Dominant covariance eigenvalue | 0.4389794030700144 |
| Sum of positive eigenvalues | 0.43897940307001454 |
| Statistic / dominant eigenvalue | 8.867902772157631 |
| Effective HBE degrees of freedom | 1.0000000000000002 |

The readout SHA-256 recorded during the audit was:

```text
d3255cc8aa7f8d755dc3821ef9ff082b3f215de19ab92b2059ee64f434c11dd7
```

The interpretation is a sparse finite-sample **plug-in null-moment failure**. With rare binary detections, the five standardized random features contain only one independent direction. The empirical covariance of their products is estimated from a sample with no joint-positive observations. That estimate can substantially understate the variability relevant under independence, turning a tiny negative association into a small p-value. Zero co-detection is unsurprising when its expected count is only 0.153.

This identifies a mechanism in the inspected query, not a measured population-wide false-positive rate. In this rank-one case HBE reduces to approximately a scaled one-degree-of-freedom chi-square calculation; the evidence does not support blaming an incorrect HBE call or conditional ridge for the marginal rejection. The RCIT paper's approximate, large-sample results do not establish calibration for this particular sparse regime.[^1]

## Additional seed-2 failure: undefined sparse bandwidth

The completed [three-seed evaluation](runs/simple/realistic/evaluation_suite_all_seeds/report.json)
records a separate numerical failure for scGLUE seed 2, not another small
p-value. The marginal query is RNA core versus H3K27ac sentinel, query seed
1466141879. All eight sentinel detections occur after the first 500 readout
rows. Native RIT estimates its bandwidth from the median of nonzero pairwise
distances within that prefix, so this input produces an empty distance set and
a NaN bandwidth. The source handles zero or missing bandwidths, but not NaN;
nonfinite random features then reach an eigendecomposition before HBE.

Two reviewers checked the saved query, assignment row order, readout detections,
and matching backend source without rerunning CI. This is distinct from the
seed-0 plug-in null-moment problem above. The integration fit succeeded; count
discovery failed with `LinAlgError: Eigenvalues did not converge`. Its graph
metrics remain unavailable. [RESULTS.md](RESULTS.md) records the evidence and
interpretation. No row reshuffling, bandwidth repair, test substitution, or
model refit was used to replace the frozen failure.

## QC selection and interpretation boundary

[Preprocessing](preprocessing.py) filters RNA by detected-gene count and filters assays by positive normalization-library totals. The evaluator intersects retained IDs across supplied methods, then equalizes pool sizes. Consequently, extended evaluations with dropped cells concern distributions conditional on QC/common eligibility. Their oracle PAG still comes from the unselected source measurement DAG, without explicit selection nodes.

QC can change dependence relationships. Source-PAG disagreement may therefore reflect selection, measurement, testing, matching, and integration together; it cannot automatically be attributed to the integration model. Current evaluator limitation text mentions matching-induced coupling and latent assay exposure but does not explicitly state this QC-selection caveat. Simple seed 0 retained all 2,000 cells per assay, so this particular mismatch does not explain its sparse-count false calls.

Keep paired, random, and state-reference controls visible alongside learned integrations. Treat their results as observed pipeline behavior, not achieved nominal error control. No settings, source code, tests, models, oracle/scorer definitions, or datasets were changed by this audit. No GPU work was performed, and preserving this note added no new CPU experiments.

## References

[^1]: Strobl, E. V., Zhang, K., and Visweswaran, S. (2019). “Approximate Kernel-Based Conditional Independence Tests for Fast Non-Parametric Causal Discovery.” *Journal of Causal Inference*, 7(1), 20180017. [Published article](https://doi.org/10.1515/jci-2018-0017); [author preprint](https://arxiv.org/abs/1702.03877).

[^2]: Strobl, E. V. Original RCIT implementation, including RIT dispatch, random-feature conditioning, and HBE null approximation. [Authors' RCIT.R source](https://github.com/ericstrobl/RCIT/blob/master/R/RCIT.R).
