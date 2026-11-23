# Finite-mixture label alignment: an explicit boundary

Astra derivation, 2026-11-14; deterministic algebra, not a benchmark simulation.
Independent review remains required. The within-assay mixture-identification
premise is supplied by the exact primary theorem recorded in the literature
ledger; it is not proved by this note.

Suppose each assay's joint law uniquely determines its finite mixture weights
and component laws up to permutation, and all assays have the same latent
classes and selected-population class probabilities. If the probabilities
are pairwise distinct, sorting them gives a common class labeling. This
identifies matched class-conditional assay laws. With the further assumption
of mutual cross-assay conditional independence given class, the joint is

    P(A1,...,Ak) = sum_c pi_c product_a P(Aa | c).

Distinct weights identify labels, not residual dependence within a class.
Without conditional independence, arbitrary admissible within-class couplings
still give the same observed assay mixtures. If weights repeat, even class
label correspondence need not be identified.

For an exact equal-weight example, let a fair binary class C generate three
conditionally independent Bernoulli features in each assay. In assay A the
success probability is 1/5 for C=0 and 4/5 for C=1. In assay B use 3/10 and
7/10. Each class's feature blocks are linearly independent as measures, and
three jointly observed blocks satisfy the two-class within-assay rank premise.
Two cross-assay completions are possible: match low A to low B, or low A to
high B. Both have the same *complete three-feature law in each assay*, not
just the same single-feature mean. Under conditional independence across
assays given C, one A feature and one B feature have

    E[A] = E[B] = 1/2,
    E[AB] = 31/100 or 19/100,
    Cov(A,B) = 3/50 or -3/50.

The calculation is half of (1/5*3/10 + 4/5*7/10), or half of
(1/5*7/10 + 4/5*3/10), respectively. Thus within-assay latent-mixture
identification and cross-assay conditional independence do not resolve equal-
weight label permutations. The example concerns statistical dependence,
not a direct causal effect; all association can arise from the common class.
Fixed identical guidance and metadata can be appended to both models.

Actual assay-specific QC can change class probabilities, and continuous
biological variation may violate the finite-class premise. Neither observed
rank nor a visually separated mixture proves common biological class identity.
This is why the composed positive result is an explicit conditional route,
not an approved fit on the current counts.
