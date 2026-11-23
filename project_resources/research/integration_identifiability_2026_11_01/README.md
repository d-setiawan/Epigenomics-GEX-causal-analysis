# Integration and causal nonidentifiability

_2026-11-01. User-directed theory pivot after the scConfluence/scMTR-seq pilot._

The [permanent proof](../theory/07_unpaired_integration_nonidentifiability.md)
is the main deliverable. It constructs faithful causal models with identical
observed assay-block laws but different cross-assay direct edges and effects,
then proves that no integration/readout using only those observations can
distinguish them under independent disjoint sampling.

The [modern-objective companion](../theory/08_modern_integration_objectives.md)
now applies the result to the audited scGLUE, scMRDR and scConfluence inputs
and losses. It proves equality of their objectives across the countermodels
and derives an illustrative transport problem whose unique optimum selects an
incorrect coupling. Its toy transport loss is explicitly separate from the
published and locally adapted scConfluence objectives.

This is a qualified population statement, not a conclusion from failed fits
or a claim that all restricted models are impossible. The proof also covers
nonlinear transformations and count observations, distinguishes biological
direct edges from measured PAG adjacencies, and preserves positive cases under
explicit restrictions. The [literature review](literature/FOCUSED_REVIEW.md)
places it in the established statistical-matching problem; no novelty is claimed.
Operational scope is owned by [L18](../LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution).

## Verification and independent perspectives

| Check | Record |
| --- | --- |
| Mathematical audit: faithfulness, observation law, error bound, nonlinear/count extension, PAG distinctions | [math_review.md](reviews/math_review.md) |
| Skeptical scope review: full inputs, positive restrictions, disjoint versus shuffled paired cells | [scope_review.md](reviews/scope_review.md) |
| Published results and method-attribution review of the actual draft | [DRAFT_REVIEW.md](literature/DRAFT_REVIEW.md) |
| Exact symbolic identities and two concrete model checks | [verify_algebra.py](verify_algebra.py), [receipt](symbolic_verification.json) |
| Exact literature queries, raw records and bibliography verification | [search log](literature/SEARCH_LOG.md), [citation checks](literature/citation_verification.json), [manifest](literature/review_manifest.json) |
| Modern-objective proof, global Gaussian transport optimum and cycle example | [modern_objective_math.md](reviews/modern_objective_math.md) |
| scGLUE and scConfluence objectives, local versions and paper/source differences | [modern_scglue_scconfluence.md](reviews/modern_scglue_scconfluence.md) |
| Paper-consistent scMRDR objective, input information and pinned-source verification | [modern_scmrdr.md](reviews/modern_scmrdr.md) |

The symbolic check verifies the covariance matrix, invariant observed blocks,
determinant, conditional covariances, intervention derivative, reverse-model
joint law, Poisson count covariance and shuffled-cohort mean formulas. The
extended check also verifies the transport derivatives, optimal correlation,
penalty limits and reflection identities. It does
not replace the written proof or automatically certify its assumptions.

Run from the repository root using the existing environment:

```bash
/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python \
  project_resources/research/integration_identifiability_2026_11_01/verify_algebra.py
```

No datasets were downloaded or generated, no integration fits were run, and no
frozen scientific source or result was changed in this pivot. Public literature
metadata and primary papers were retrieved for the bounded review. Existing
paired-reference results remain an empirical diagnostic, not a proof.

## Corrections to previous notes

The [matched-pairing note](../theory/02_identifiability.md) no longer claims that
every cross-assay edge becomes bidirected or that stable directed outputs are
impossible. The [project guarantee draft](../identifiability/project_identifiability_guarantees.md)
now distinguishes convergence to a state-mean distribution from convergence to
the biological joint, and adds the missing joint-completion requirement.
Current navigation and the [roadmap](../ROADMAP.md) point to this proof before
further model tuning. Historical experimental numbers are preserved.
