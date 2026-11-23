# Theory: what metacell aggregation does to causal discovery

The latest [nonlinear repair and count-model program](../nonlinear_repair_2026_11_15/README.md)
contains replacement identification statements, independent proof reviews and
completed existing-count experiments. Its
[synthesis](../nonlinear_repair_2026_11_15/SYNTHESIS.md) separates conditional
nonlinear/count recovery from the unresolved biological causal target.

The [nonlinear recheck](../nonlinear_recheck_2026_11_14/ASTRA_REASSESSMENT.md)
corrects omissions in the earlier nonlinear theorem coverage and separately
reviews the literal variability premise and possible revisions.

The preceding [unpaired-assay identification program](../research_program_2026_11_14/README.md)
adds an explicit observation/target contract, primary-theorem applicability
review and a pinned Lean project. Its
[requirement ledger](../research_program_2026_11_14/REQUIREMENTS.md) preserves
the earlier review record; the notes below retain their original scope.

A derivation companion. Empirical results and population identification
arguments are separate evidence. Start with the
[2026-11-01 unpaired-integration proof](07_unpaired_integration_nonidentifiability.md)
for the current information boundary and its exceptions. Notes 01–05 retain
the earlier aggregation model and empirical history; their claims require that
model's stated conditions.

The [method-specific companion](08_modern_integration_objectives.md) applies
that proof to scGLUE, scMRDR and scConfluence objectives, including an exact
example in which optimal transport selects an incorrect residual coupling.

The [broader literature review](../integration_literature_scope_2026_11_01/REVIEW.md)
maps these limits to published methods and positive identification theorems.
In particular, a restricted unpaired model can identify a shared latent causal
graph; the broad-class counterexample is not a universal impossibility claim.
The subsequent [Sturma replication and count application](../sturma_pilot_2026_11_01/README.md)
tests the native method while keeping its published-model result separate from
its application to our existing measurements.
The [count-aware extension](../count_mdcrl_2026_11_01/METHOD.md) makes the
measurement-noise hypothesis explicit and distinguishes its finite-sample
estimator from population identification under a known Poisson kernel. Its
[results](../count_mdcrl_2026_11_01/RESULTS.md) do not establish cross-assay recovery.
The subsequent [extension method](../count_extensions_2026_11_01/METHOD.md)
derives NB/zero-mask corrections, source shrinkage and generating-function
orientation. Its [known-kernel population note](../count_extensions_2026_11_01/reviews/kernel_population_identifiability.md)
keeps within-assay law identification separate from cross-assay coupling.
The [dimension follow-up](../count_rank20_2026_11_01/README.md) tests the full
empirically selected dimensions without treating them as known biological rank.

## Read in order

| | |
|---|---|
| [01 — The aggregation algebra](01_aggregation_algebra.md) | The generative model, the covariance the search actually receives, attenuation, and why partition instability is a *corollary* rather than a separate problem |
| [02 — Identifiability under matched pairing](02_identifiability.md) | Common-cell linear equations versus missing cross-assay residual coupling; corrected graph/PAG scope and the between-metacell rank bound |
| [03 — What each algorithm actually does](03_algorithms.md) | PC / FCI / GFCI / BOSS / BFCI / LV-Heuristic in equations, the score and CI test they share, and why two "FCI" implementations disagree 5.3× |
| [04 — The simulation](04_simulation.md) | Design, the controlled contrasts, and every result matched against the prediction it was built to test |
| [05 — Synthesis](05_synthesis.md) | Why four repairs failed, what survives, and the estimator the final claim rests on |
| [06 — Latent-space aggregation](06_latent_space_aggregation.md) | Why integration aligns state but does not recreate same-cell covariance; recoverable targets, failure modes, and the next estimator contract |
| [07 — Unpaired integration and nonidentifiability](07_unpaired_integration_nonidentifiability.md) | Faithful indistinguishable causal models, algorithm-independent proof, nonlinear/count extensions, PAG distinctions, and positive exceptions |
| [08 — Modern integration objectives](08_modern_integration_objectives.md) | Why the audited objectives cannot distinguish the countermodels; method-specific input/loss audit and an exact transport counterexample |

## The story in one page

Six histone marks are six **separate assays on separate cells**. A metacell holds some cells from
each, and variable $j$ is averaged over that metacell's channel-$\mathrm{ch}(j)$ cells only. Write
the analysis matrix as a population metacell mean plus a sampling deviation, $X_{cj} = \mu_{cj} +
\xi_{cj}$. Because two variables of the same mark are averaged over the *same* cells and two
variables of different marks over *disjoint* cells,

$$\operatorname{Cov}(X_j, X_k) \;=\; \Sigma^{B}_{jk} \;+\; \mathbf{1}\!\left[\mathrm{ch}(j)=\mathrm{ch}(k)\right]\cdot \mathbb{E}_c\!\left[\frac{\Sigma^{W}_{jk}}{n_{c}}\right].$$

This decomposition assumes independent disjoint sampling within fixed state
strata. It motivates the following distinctions; it is not by itself a proof
of particular PAG endpoints or empirical instability.

1. **Same-mark and cross-mark entries are not on the same scale.** Same-mark carries a term
   cross-mark does not. This is a property of the assay design, not of chromatin.
2. **Cross-mark correlations are attenuated**, because the noise enters both variances in the
   denominator and nothing in the numerator. The attenuation factor is the *reliability*
   $\lambda_j$, and measuring it on real data gives a median of $\approx 0$: with ~11 cells per
   mark per metacell, the typical variable's entire between-metacell variance is sampling noise.
3. **Weak between-state signal can make partitions unstable.** Repartitioning
   changes sampling deviations. This supplies a possible mechanism, not a
   universal prediction about the output of a graph algorithm.
4. **Original cross-assay structure needs an identifying assumption.** Shared
   means can induce dependence among constructed aggregates. This does not
   force bidirected PAG endpoints or preclude stable wrong arrows. Note 07
   proves the precise broad-class limitation without relying on benchmark
   failures.
5. **Stable associations can remain useful.** The historical cross-mark audit
   reports recurrent adjacencies and external-support enrichment. Those are
   association and validation results, not proof of biological direct effects;
   the owning report below supplies the numbers and controls.

## Where the numbers are owned

Per the repo's [documentation ownership rule](../../../README.md), result numbers are owned by the
`*/results/*.md` project_resources/research and caveats by [LIMITATIONS.md](../LIMITATIONS.md). Figures are reproduced
here **only** where a derivation needs them side by side with the algebra that predicts them; the
owning documents are authoritative and are linked at each point:

- [crossmark_identifiability.md](../../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md) — the
  signal-share audit, the estimator arms, the resolution curve, the nulls, the edge list
- [partition_stability_findings.md](../../../pipelines/empirical/evaluation/results/partition_stability_findings.md) —
  the original partition-instability result and the K=3 pilot
- [pairing_artifact_findings.md](../../../pipelines/simulation/unpaired_multimodal/results/pairing_artifact_findings.md) — the
  implementation gap and the multi-channel simulation
- [LIMITATIONS.md](../LIMITATIONS.md) — L11 (partition dependence), L14 (covariance scale
  asymmetry), L15 (most variables carry no signal), L17 (integration validation), and L18
  (latent-space aggregation does not reconstruct the missing joint distribution)

Related: [identifiability/](../identifiability) covers FCI's guarantees in general; this folder
covers what *this* data construction does to them.
