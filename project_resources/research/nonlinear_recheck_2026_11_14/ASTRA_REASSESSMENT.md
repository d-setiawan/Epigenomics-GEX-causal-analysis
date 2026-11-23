# Nonlinear-results recheck: Astra reassessment

2026-11-14 follow-up to: “Has there only been 2 new nonlinear results? And what
were their unresolved issues?” This record does not alter the prior sealed
program, launch a new benchmark, or claim an exhaustive literature search.

## Correction to the earlier summary

No. “Two” referred to two papers flagged late in the previous investigation:
Shrestha–Fu (2025) and Timilsina and colleagues (2026). It was not a defensible
count of nonlinear identification results. Each paper also contains several
different theorem claims. Earlier reviewed families included auxiliary nonlinear
ICA, iVAE, paired multiview identification, nonlinear nonuniqueness and manifold
alignment.

The recheck also found a genuine coverage gap. Kong (2022) and Xie (2023) had
been mentioned without adequate theorem-level dispositions; Yan (2023),
Nguyen–Fu DICA (2025), and Xu's partially observable multiview result (2024)
require distinct treatment. The [inventory](NONLINEAR_INVENTORY.md) separates
previous substantive review, incidental mention, and new primary checks.
The previous nonlinear coverage should not be described as complete.

These results do not all solve the same problem. Some concern a single observed
vector's latent representation; some require paired views or observed mask
groups; some study unpaired domains with one shared measurement map. None can
be silently relabeled as identification of the original cross-assay genomic
causal graph from our selected count matrices.

## What remains unresolved for the two flagged papers

**The population premise needs more than a cosmetic clarification.** Both
printed variability conditions require every positive-probability measurable
set that is not exactly a content cylinder to change probability across some
domains. In an atomless setting, deleting one point from the full product
space contradicts that literal requirement. An almost-everywhere reading is
natural to consider, but is not stated in those conditions.

The fresh skeptic checked a stronger obstruction for the finite-domain,
atomless-style, independent-content/style class of the 2025 model. Vector-
measure convexity gives a measurable style set H with probability one-half
under every domain's style law. The event C×H has the same probability in
every domain, yet its symmetric difference with any content cylinder B×S has
probability one-half. Thus merely replacing exact equality by equality modulo
null sets does not repair the unrestricted measurable-set premise in this
class. A restricted family of events or a different variation assumption needs
its own existence and identification proof. This strengthened argument is not
claimed for every dependent-content/style version of the 2026 model.

**The proof and estimator questions are separate.** The skeptical review flags
additional proof inferences that need justification; these are not presented
as full-model counterexamples. The 2025 practical GAN uses adversarial/l1
penalties rather than the ideal exact population/l0 constraints. The 2026
method uses finite-probe soft Jacobian penalties rather than exact global
orthogonality. An optimizer's empirical usefulness is not refuted by the
premise objection, but its results cannot automatically inherit the idealized
identification claim. The 2026 robustness bound also depends on local Jacobian
conditioning and its inherited content-identification step.

**Application to this project remains unproved even after mathematical repair.**
The shared smooth measurement coordinates, invertibility, dimension conditions
and latent independence or geometry are not supplied by different genomic
assay feature maps and stochastic count measurements. Identified content/style
blocks up to invertible transformations are not identified named genomic
variables or causal directions.

The [fresh skeptical review](SKEPTICAL_ASSUMPTION_REVIEW.md) records exact
primary statements, proof locations and the scope of both set arguments.
It did not audit every conceivable corrected theorem or establish failure of
the trained algorithms. The same literal issue appears in specific predecessor
theorems, but must not be transferred to unrelated derivative-rank,
sparsity-based or differential-geometry results without inspecting their own
assumptions.

## Decision

Correct the count and acknowledge the missing theorem-level coverage. Keep the
two papers' mathematical premise, proof, optimization and count-data applicability
issues distinct. The newly checked nonlinear alternatives deserve their own
assumption audits; no dataset fitting or new-world generation is warranted by
this follow-up question alone. No revised theorem is silently substituted for
the published statement.
