# Primary source locators and verification excerpts

Accessed 2026-11-14 through the web reader. Short fragments below are exact quotations; mathematical expressions are copied only to identify the target, and each quotation is well below source quotation limits. Primary files could not be downloaded locally: shell HTTPS DNS resolution failed. Reviewers can inspect the linked public copies at the exact locations below.

## Timilsina, Shrestha and Fu, 2024

[Primary HTML](https://arxiv.org/html/2409.19422), Section 3, Assumption 1, Theorem 1 and Appendix B; the web reader reports lines 123–142. The theorem requires full-column-rank A^(q) and full-rank E[ccᵀ]. It first obtains Q^(q)x^(q)=Θ^(q)c. Its alternative (a) requires independence, non-Gaussianity, distinct marginal distributions under scaling and asymmetric marginals; alternative (b) replaces independence and non-Gaussianity with the full symmetric-hyperrectangle support, retaining the latter distinctness and asymmetry conditions. Exact excerpt: “either of the following is satisfied”. The source's conclusion is the common Θ under either condition. Its Assumption 1 quantifies all specified subspaces, simplices and scaled stripe probabilities; the linked text gives the formal condition.

Same paper Section 4, Assumption 2 and Theorem 2, HTML lines 154–167 and Appendix C: same mixing matrix A and same encoder Q, with full rank A and E[ccᵀ], and weaker single-stripe variability assumption. Exact fragment: “the conditions (a) and (b) in Theorem 1 are completely removed” under that additional homogeneous structure.

Same paper Section 4, Assumption 3 and Theorem 3, HTML lines 168–179 and Appendix D: exact fragment “paired samples”; at least d_C pairs are put into equality constraints, with Assumption 1, full rank A^(q) and absolutely continuous P_c. This is not an admissible unpaired fitting procedure. Source includes an associated finite-weight regularization heuristic, which is not the same as proving the theorem’s hard equality for an arbitrary β.

## Slawski and Sen, 2024

[Primary PDF](https://jmlr.org/papers/volume25/22-0058/22-0058.pdf), printed pages 14–16 (PDF pages 14–16), A1–A3 and Theorem 8; web reader lines 1264–1283 and 1374–1422. Source excerpt: “an absolutely continuous distribution” and “independent of X”. A1 is λ-strong convexity of potential ψ, A2 L-smoothness, A3 almost-sure boundedness of ||f*(X)||; theorem models unlinked iid samples and Gaussian noise and bounds mean squared prediction error with probability at least 1−5/m−C(n^−c+m^−c). Proposition 2, printed pages 6–7, is instead a noiseless *permuted cohort* assignment result. Its exact error expression and constants are at the cited theorem; Appendix B gives the proof.

## Syrota et al., 2026 preprint

[Primary HTML](https://arxiv.org/html/2611.11460), Sections 2–4, Appendix B; reader lines 103–118, 139–143, 158–164, 175–182 and 190–200. Definition 2.1 sets A=f_b⁻¹∘f_a for models of the same data manifold; Proposition 2.2 demands agreement of data-point distances. In Proposition 3.1 the alternatives are f_b⁻¹∘S∘f_a for data-space metric-measure automorphisms S; identification iff that automorphism group is trivial. Proposition 4.1 connects those isomorphisms to population GW minimizers. Proposition 4.4 uses shuffled encodings of the same observations, n≥3; independent-sample convergence uses global empirical optimizers (Appendix B.14). Source limitation excerpt: “solvers return stationary points that need not be global optima.” The web HTML hides the formal labels of the hypotheses in its display; Appendix B.2 and Section 2 should be checked in the rendered paper for their complete noise and manifold hypotheses before citing this as an exact restatement.
