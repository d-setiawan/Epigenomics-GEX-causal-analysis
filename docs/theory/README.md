# Theory: what metacell aggregation does to causal discovery

A derivation companion. The empirical results live in `Evaluation/results/`; this folder
explains **why** they came out that way, and connects each measured number to a line of algebra
that predicted it.

## Read in order

| | |
|---|---|
| [01 — The aggregation algebra](01_aggregation_algebra.md) | The generative model, the covariance the search actually receives, attenuation, and why partition instability is a *corollary* rather than a separate problem |
| [02 — Identifiability under matched pairing](02_identifiability.md) | What averaging over disjoint cells does to the target graph; why same-mark keeps orientations and cross-mark cannot; the rank bound on the between-metacell covariance |
| [03 — What each algorithm actually does](03_algorithms.md) | PC / FCI / GFCI / BOSS / BFCI / LV-Heuristic in equations, the score and CI test they share, and why two "FCI" implementations disagree 5.3× |
| [04 — The simulation](04_simulation.md) | Design, the controlled contrasts, and every result matched against the prediction it was built to test |
| [05 — Synthesis](05_synthesis.md) | Why four repairs failed, what survives, and the estimator the final claim rests on |

## The story in one page

Six histone marks are six **separate assays on separate cells**. A metacell holds some cells from
each, and variable $j$ is averaged over that metacell's channel-$\mathrm{ch}(j)$ cells only. Write
the analysis matrix as a population metacell mean plus a sampling deviation, $X_{cj} = \mu_{cj} +
\xi_{cj}$. Because two variables of the same mark are averaged over the *same* cells and two
variables of different marks over *disjoint* cells,

$$\operatorname{Cov}(X_j, X_k) \;=\; \Sigma^{B}_{jk} \;+\; \mathbf{1}\!\left[\mathrm{ch}(j)=\mathrm{ch}(k)\right]\cdot \mathbb{E}_c\!\left[\frac{\Sigma^{W}_{jk}}{n_{c}}\right].$$

Everything else follows from that one line.

1. **Same-mark and cross-mark entries are not on the same scale.** Same-mark carries a term
   cross-mark does not. This is a property of the assay design, not of chromatin.
2. **Cross-mark correlations are attenuated**, because the noise enters both variances in the
   denominator and nothing in the numerator. The attenuation factor is the *reliability*
   $\lambda_j$, and measuring it on real data gives a median of $\approx 0$: with ~11 cells per
   mark per metacell, the typical variable's entire between-metacell variance is sampling noise.
3. **Partition instability is then automatic.** At $\lambda \approx 0$ the matrix is
   $X \approx \xi$, and re-partitioning redraws $\xi$. Edges are drawn from the tail of a noise
   distribution, so they do not recur. The instability was never a clustering defect.
4. **Cross-mark orientations are not identifiable at all**, because under matched pairing the
   correct mark for a cross-channel pair is $\leftrightarrow$, never $\rightarrow$ — the shared
   metacell mean *is* a latent common cause. The measured "0 partition-stable directed cross-mark
   edges at any resolution" is what the algebra predicts.
5. **What survives is adjacency**, and it validates: 41 partition-stable cross-mark adjacencies,
   enriched for PCHi-C shared promoter hubs against a distance- and degree-matched null.

## Where the numbers are owned

Per the repo's [documentation ownership rule](../../README.md), result numbers are owned by the
`*/results/*.md` docs and caveats by [LIMITATIONS.md](../LIMITATIONS.md). Figures are reproduced
here **only** where a derivation needs them side by side with the algebra that predicts them; the
owning documents are authoritative and are linked at each point:

- [crossmark_identifiability.md](../../Evaluation/results/crossmark_identifiability.md) — the
  signal-share audit, the estimator arms, the resolution curve, the nulls, the edge list
- [partition_stability_findings.md](../../Evaluation/results/partition_stability_findings.md) —
  the original partition-instability result and the K=3 pilot
- [pairing_artifact_findings.md](../../Evaluation/results/pairing_artifact_findings.md) — the
  implementation gap and the multi-channel simulation
- [LIMITATIONS.md](../LIMITATIONS.md) — L11 (partition dependence), L14 (covariance scale
  asymmetry), L15 (most variables carry no signal)

Related: [identifiability/](../identifiability) covers FCI's guarantees in general; this folder
covers what *this* data construction does to them.
