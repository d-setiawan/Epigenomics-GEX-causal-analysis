# Literature check: what unpaired integration can and cannot identify

This is a bounded narrative review, checked on 2026-11-01. It supports an
information-limit argument and checks counterexamples to overly broad claims.
It is not a systematic review, a new theorem, or evidence that integration is
generally useless. Exact searches, public responses and access failures are
archived alongside this note; see [SEARCH_LOG.md](SEARCH_LOG.md).

## The claim that the literature supports

In a model class allowing different cross-assay residual couplings, observing
the separate laws of `(S, X)` and `(S, Y)` does not generally identify the joint
law of `(S, X, Y)`. Observing shared state perfectly still leaves this problem.
This is the statistical-matching setting, with state playing the role of the
common variables. D’Orazio, Di Zio and Scanu explicitly discuss the missing
joint relationship and the role of untestable identifying assumptions on page
2 of their [Istat-hosted author manuscript](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf).
Their [2006 book](https://doi.org/10.1002/0470023554) treats conditional
independence, auxiliary information and identification uncertainty separately.

The proposed family

$$
X=S+U,\qquad
Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}\,V
$$

is an illustrative specialization of that general issue, for example comparing
`rho = 0` and `rho = 1/2`. With independent standard Gaussian noises, the separate conditional laws remain fixed as
`rho` changes, although the residual dependence and the direct `X -> Y` edge
change. The manuscript should establish this algebra and the resulting
indistinguishability directly. The literature supplies context; benchmark
failures do not supply the proof.

The quantifier matters: **no procedure using only those observed laws can be
correct throughout a class containing observationally indistinguishable models
with different target structures**. This does not rule out identification in a
smaller, correctly specified model class. Priors, guidance and background
features must be held fixed across the indistinguishable models; a restriction
that rules one out changes the theorem’s scope.

## Positive results that constrain the conclusion

**Unlinked monotone regression.** Balabdaoui, Doss and Durot establish consistent
estimation of a univariate monotone regression function from unpaired covariate
and response samples under independent additive noise with known noise law,
plus the stated regularity assumptions. Thus, “unpaired data can never recover
a relationship” is false. This result fixes a response/covariate model; it is
not unrestricted multivariate causal discovery. Section 7’s unknown-noise
extensions discuss additional error/paired samples or repeated responses, so
they should not be described as solving the unrestricted two-marginal problem.
See the [canonical JMLR record](https://www.jmlr.org/papers/v22/20-689.html) and
[full text, abstract and §7](https://www.jmlr.org/papers/volume22/20-689/20-689.pdf).

**Causal direction after a joint law is available.** Peters, Mooij, Janzing and
Schölkopf’s Theorem 28 identifies the DAG for a restricted additive-noise model
with causal minimality **from the joint distribution**. Additivity or a
nonlinear neural parameterization alone is not that theorem’s condition.
Example 24 gives the familiar reversible linear-Gaussian case; Example 26
shows why conditional restrictions matter even when individual variables are
non-Gaussian. See [JMLR 15(58)](https://www.jmlr.org/papers/v15/peters14a.html)
and [full text, pp. 2023–2025](https://www.jmlr.org/papers/volume15/peters14a/peters14a.pdf).

**LiNGAM.** Shimizu and colleagues identify linear acyclic structure under
non-Gaussian independent disturbances and no hidden confounding. This is another
positive result under substantive restrictions, rather than a guarantee from
separate assay marginals. The Gaussian family above does not satisfy its
non-Gaussian disturbance assumptions. See the
[primary JMLR record](https://www.jmlr.org/papers/v7/shimizu06a.html).

**A recent transport-based challenge.** Koo, Lim and Sohn’s 2026 preprint gives
an exact Gaussian joint-recovery result for a Schrödinger bridge with an
appropriate compatibility cost and regularization parameter. Theorem 3.2
establishes existence of a recovering parameter. Its oracle cost uses the
unobserved conditional-covariance sign; the empirical alignment-cost case also
requires a residual-sign condition. Appendix A’s optimal parameter depends on
the missing conditional covariance. Section 3.2 discusses known joint
correlation or auxiliary complete triplets; §4 and Appendix B.4 use complete
triplets to select the parameter, with 50 for the weak-information variant.
This is useful positive evidence for structured completion, but it does not
identify that parameter from the two marginal laws alone. The discussion of
real-data tuning contains less precise “marginal constraints alone” wording;
the explicit theorem and tuning protocol govern this reading. See
[arXiv:2606.22770v1, §3.2 and Appendix B.4](https://arxiv.org/html/2606.22770v1).
Publication status: preprint, not treated here as established comparative
evidence.

## What the integration methods actually contribute

**GLUE/scGLUE.** The authors’ documentation defines unpaired inputs as
independent samples of a presumed common population and uses a feature-guidance
graph to orient integration. Its regulatory-inference tutorial defines feature
scores by symmetric cosine similarity and limits candidate pairs using a
skeleton. These are explicit modeling inputs and scores, not observed
cross-assay cell pairs. See the [official introduction](https://scglue.readthedocs.io/en/latest/index.html)
and [regulatory-inference definition](https://scglue.readthedocs.io/en/latest/reginf.html).
The paper is Cao and Gao (2022),
[doi:10.1038/s41587-022-01284-4](https://doi.org/10.1038/s41587-022-01284-4).
The inference for this project is that an embedding-selected coupling needs an
identification argument if it is to stand in for the true joint law; the
regulatory-score terminology alone does not provide one.

**scConfluence.** Samaran, Peyré and Cantini combine modality-specific
autoencoders with regularized inverse optimal transport using weak feature
connections; partial transport accommodates modality-specific populations.
Its reported tasks include cell alignment, clustering and feature imputation.
The Methods describe transductive evaluation and explicitly place unseen-cohort
generalization outside the original task. They also reduce the Sinkhorn
coefficient from 0.1 to 0.03 for three modalities because three terms are summed.
Those details distinguish the published experiments from our independent-cohort,
three-assay NB pilot. Neither that pilot’s failures nor its isolated successes
establish an identifiability theorem. See
[the primary article, framework and training Methods](https://www.nature.com/articles/s41467-024-51382-x).

## Citation verification

| Work | Verified bibliographic identity |
|---|---|
| D’Orazio, Di Zio, Scanu | *Statistical Matching: Theory and Practice*, Wiley, 2006; DOI `10.1002/0470023554`. Author names/year/title verified on Wiley. The Istat-hosted manuscript above has no publication year established in this review. |
| Balabdaoui, Doss, Durot | *Unlinked Monotone Regression*, JMLR **22**(172):1–60, 2021; canonical article `20-689`. No DOI supplied on the checked journal page. |
| Peters, Mooij, Janzing, Schölkopf | *Causal Discovery with Continuous Additive Noise Models*, JMLR **15**(58):2009–2053, 2014; canonical article `peters14a`. No DOI supplied on the checked journal page. |
| Shimizu, Hoyer, Hyvärinen, Kerminen | *A Linear Non-Gaussian Acyclic Model for Causal Discovery*, JMLR **7**(72):2003–2030, 2006; canonical article `shimizu06a`. No DOI supplied on the checked journal page. |
| Cao, Gao | *Multi-omics single-cell data integration and regulatory inference with graph-linked embedding*, Nature Biotechnology **40**:1458–1466, 2022; DOI `10.1038/s41587-022-01284-4`. |
| Samaran, Peyré, Cantini | *scConfluence: single-cell diagonal integration with regularized Inverse Optimal Transport on weakly connected features*, Nature Communications **15**, article 7762, 2024; DOI `10.1038/s41467-024-51382-x`. |
| Koo, Lim, Sohn | *Statistical Matching via Schrödinger Bridge beyond Conditional Independence*, arXiv:2606.22770v1, submitted 2026-06-22; DOI `10.48550/arXiv.2606.22770`. |

The skill’s citation checker was run; its results are in
[citation_verification.json](citation_verification.json). The Wiley DOI resolves,
but the checker’s Crossref metadata formatter raised a `list index out of range`
error; primary Wiley metadata supplies the verified fields. The arXiv DOI
resolves through DataCite and has no Crossref record in that check. A separate
Rässler/data-fusion candidate was excluded from claims because its primary page
and DOI could not be verified in this bounded pass. No conclusions depend on
the inaccessible pages or on unverified citations.
