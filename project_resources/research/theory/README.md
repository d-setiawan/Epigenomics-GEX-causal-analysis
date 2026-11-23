# Theory: what metacell aggregation does to causal discovery

**What can an unpaired assay tell us about a cell-level edge?** Even perfect
matching on biological state leaves the within-state cross-assay covariance
unobserved. Two causal models can therefore agree on every learner-visible
assay law and disagree about a direct edge. The proof states the observation
contract and shows how it applies to scGLUE, scMRDR and scConfluence. [Population
counterexample](07_unpaired_integration_nonidentifiability.md) · [Integration objectives](08_modern_integration_objectives.md).

**What can a shared-state model recover?** E/E2 identify transport between
assay log-rate subspaces under a known product-Poisson channel, full-rank
log-linear maps and a shared law of independent non-Gaussian unit-variance
sources without signed-permutation ambiguities. Recovering the count joint
also requires independent cross-assay observation noise given the shared
sources. The corollary combines established count-mixture and linear alignment
results. [Square case](../nonlinear_repair_2026_11_15/theory/COMP_COUNT_SHARED_TRANSPORT.md) · [Unequal-width case](../nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md) · [Novelty assessment](../count_transport_novelty_2026_11_15/README.md).

**Where has that worked?** In a binary-source special case with known
dimension, a count-based estimator passed the transport, covariance and
predictive-score criteria in all nine reused worlds. These worlds had one to
three sources and 12,000 unpaired fitting cells per assay. Cross-assay causal
direction and transport on empirical counts remain open. The earlier Sturma
and count-correction studies tested source matching on the original sparse
panels. [Exact-model positive control](../../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md)
· [Native Sturma results](../sturma_pilot_2026_11_01/README.md) · [Count correction](../count_mdcrl_2026_11_01/README.md) · [Count extensions](../count_extensions_2026_11_01/README.md) · [Dimension follow-up](../count_rank20_2026_11_01/README.md).

The published-method review and subsequent theory studies identify which
assumptions suffice for each target. The November 14 recheck records omissions
in the initial nonlinear review. [Scoping review](../integration_literature_scope_2026_11_01/README.md)
· [Observation and target contract](../research_program_2026_11_14/README.md) · [Nonlinear recheck](../nonlinear_recheck_2026_11_14/ASTRA_REASSESSMENT.md) · [Theorem review and study results](../nonlinear_repair_2026_11_15/README.md).

## Read in order

| Note | What it covers |
|---|---|
| [01 — The aggregation algebra](01_aggregation_algebra.md) | The model for averaged observations, correlation attenuation and how sampling noise can change an estimate across partitions |
| [02 — Identifiability under matched pairing](02_identifiability.md) | Common-cell linear equations versus missing cross-assay residual coupling; corrected graph/PAG scope and the between-metacell rank bound |
| [03 — What each algorithm actually does](03_algorithms.md) | PC / FCI / GFCI / BOSS / BFCI / LV-Heuristic in equations, the score and CI test they share, and why two "FCI" implementations disagree 5.3× |
| [04 — The simulation](04_simulation.md) | Design, the controlled contrasts, and every result matched against the prediction it was built to test |
| [05 — Synthesis](05_synthesis.md) | Why four repairs failed, what survives, and the estimator the final claim rests on |
| [06 — Latent-space aggregation](06_latent_space_aggregation.md) | Why integration aligns state but does not recreate same-cell covariance; recoverable targets, failure modes, and the next estimator contract |
| [07 — Unpaired integration and nonidentifiability](07_unpaired_integration_nonidentifiability.md) | Faithful indistinguishable causal models, algorithm-independent proof, nonlinear/count extensions, PAG distinctions, and positive exceptions |
| [08 — Modern integration objectives](08_modern_integration_objectives.md) | Why the audited objectives cannot distinguish the countermodels; method-specific input/loss audit and an exact transport counterexample |

## The story in one page

In the original PBMC design, six histone marks and RNA come from separate
assays. Consider a fixed biological stratum $c$ with $n_{c,a}$ independently
sampled cells in assay $a$. Each feature in that assay is averaged over the
same cells. Across assays the samples are independent given $c$. Write
$\Sigma^B_{jk}$ for the covariance of the two stratum means across strata and
$\Sigma^W_{c,jk}$ for within-stratum cell-level covariance. Under this model,

$$
\operatorname{Cov}(X_j,X_k)
=\Sigma^B_{jk}
+\mathbf 1_{\{\mathrm{ch}(j)=\mathrm{ch}(k)\}}
\mathbb E_c\!\left[\frac{\Sigma^W_{c,jk}}{n_{c,\mathrm{ch}(j)}}\right].
$$

Features from the same assay retain a contribution from within-stratum
covariance. For features from different assays, only the covariance of stratum
means survives. Sampling variation still affects both variances. Thus the
correlation between the aggregates equals the correlation between the
stratum means times $\sqrt{\lambda_j\lambda_k}$, where $\lambda_j$ is the
fraction of aggregate variance attributable to variation in feature $j$'s
stratum mean. This identity for fixed strata is a reference model for examining
learned partitions. [Aggregation derivation](01_aggregation_algebra.md).

The PBMC audit covered 442 features in 133 complete metacells, with a median
of about 11 cells per mark per metacell. The median estimated signal share was
−0.017. A population variance fraction cannot be negative. At this depth,
the typical feature's estimated between-metacell variation was difficult to
distinguish from sampling variation. Some adjacencies recurred across
partitions and had external promoter-contact enrichment. That evidence supports
shared-state association. Direct regulatory effects require an additional
identifying assumption. [Panel measurements and controls](../../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md).

## Where the numbers are owned

The following reports give the original data, thresholds and evaluation
procedures behind the aggregation notes. Cross-study qualifications are in
the [limitations register](../LIMITATIONS.md).

- [crossmark_identifiability.md](../../../pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md) — the
  signal-share audit, the estimator arms, the resolution curve, the nulls, the edge list
- [partition_stability_findings.md](../../../pipelines/empirical/evaluation/results/partition_stability_findings.md) —
  the original partition-instability result and the K=3 pilot
- [pairing_artifact_findings.md](../../../pipelines/simulation/unpaired_multimodal/results/pairing_artifact_findings.md) — the
  implementation gap and the multi-channel simulation
- [LIMITATIONS.md](../LIMITATIONS.md) covers partition dependence (L11), covariance differences
  across assays (L14), limited power for between-state variation (L15), integration
  validation (L17) and the missing same-cell joint (L18).

For the general assumptions behind FCI, see the [identifiability notes](../identifiability).
