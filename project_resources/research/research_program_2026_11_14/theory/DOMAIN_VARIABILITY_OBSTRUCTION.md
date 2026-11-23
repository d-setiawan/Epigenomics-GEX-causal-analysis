# Literal domain-variability premise: common-null-set obstruction

Independent mathematical check, 2026-11-14. Primary source: Shrestha and Fu, [*Content-Style Learning from Unaligned Domains: Identifiability under Unknown Latent Dimensions*, §3.1–3.2, Assumption 3.2 and Theorems 3.3–3.4](https://arxiv.org/html/2411.03755), especially HTML lines 126–134 and 147–150; proof discussion Appendix C lines 516–521 and Appendix D lines 644–659. This is an analysis of the **literal written premise**, not a counterexample to an algorithm or a theorem under a corrected premise.

## Claim and proof

Let the paper's latent space be the measurable product `Z=C×S`, with nonempty C and at least two distinct points in S. Suppose there is a measurable point `z0=(c0,s0)` with zero probability under every domain distribution `P_n`. This common-null-point condition holds, for example, for finitely many distributions each absolutely continuous with respect to Lebesgue measure on a nontrivial Euclidean product. Define `A=Z\{z0}`. Then A is measurable, and `P_n(A)=1>0` for every domain n.

Yet A cannot equal `B×S` for any subset B of C. Indeed, if it did, a different style value `s1∈S`, `s1≠s0`, would give `(c0,s1)∈A`, forcing `c0∈B` and hence `(c0,s0)∈B×S=A`, a contradiction. Thus A satisfies both hypotheses of Assumption 3.2 as literally stated, but every domain gives it probability one, contrary to the demanded unequal probabilities for some pair of domains. Block independence is not needed for this contradiction.

This argument uses ordinary set equality, exactly as the printed condition says; A differs from the product Z only on a common null set. In the continuous-density setting of Theorem 3.4, singleton probability is zero provided the paper's density is with respect to a nonatomic reference such as Euclidean Lebesgue measure, and a nontrivial style space provides the second style point. Positivity of the density is not necessary for the obstruction. We do not claim this proof covers every degenerate or atomic space, or that every possible version of an informal intended assumption is impossible.

## Consequence and scope

Theorem 3.3 explicitly assumes 3.2; Theorem 3.4 explicitly assumes the conditions of 3.3 in addition to positivity and dimension conditions. On the continuous nontrivial-style class above, their **literal combined premises have no instances**. This does not logically falsify an implication from an impossible premise, show that its conclusion is false under some other sensible assumption, or establish that the proposed GAN cannot perform useful integration. The paper's appendix uses domain variability in its content-identification step; the effect on a modified premise and full proof needs a separate audit.

Replacing exact product equality by equality modulo common null sets would exclude this particular set A as a witness, but that is only a possible change of formulation. We have not proved that such a replacement is satisfiable for the required distribution family, that it suffices for the stated theorems, or that it is the authors' intended interpretation. A remedy must explicitly define the relevant measure(s), almost-everywhere equivalence, support conditions and any continuity needed to promote almost-everywhere results to pointwise conclusions. Before relying on these positive results, obtain a defensible revised statement and proof, or an author clarification. The general applicability of the model to our selected unpaired count data remains a separate question.

## The 2026 differential-independence extension

Astra and the literature worker separately inspected [*Content-Style
Identification via Differential Independence*, Assumption 3.3](https://arxiv.org/html/2605.17827),
HTML lines 133–147. It repeats the same measurable-set requirement with exact
non-cylinder equality. The argument above therefore applies without alteration
when those domain laws have a common null point and nontrivial style support.
Theorem 3.4 explicitly invokes it; Appendix B's content-identification step
references the earlier result. Added Jacobian orthogonality/full-rank assumptions
do not change the counterexample to variability. Theorem 3.6's robustness
argument is not treated as a repair. Static code penalties, including sampled
Jacobian probes, do not resolve a population-premise inconsistency.

Review provenance: the issue was first identified by the independent synthesis
reviewer, independently checked by the identification worker, and checked
against both primary statements by Astra. This is a human-readable mathematical
argument, not a theorem in the finite Lean project. No author was contacted
and no repaired theorem is asserted.
