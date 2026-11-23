# Fixed-known-shape negative-binomial observation bridge

Astra derivation, 2026-11-15. Independent review accepted; see [independent review](../reviews/FIXED_SHAPE_NB_INDEPENDENT_REVIEW.md). This makes the exact kernel scope of C01's shape-2 sensitivity explicit; it does not cover learned or unknown dispersion.

**Claim.** Fix known numbers r_j>0, common to all competing representations. Let Lambda be an arbitrary probability-distributed finite nonnegative rate vector in [0,infinity)^d. Given Lambda, let Y_j be mutually independent negative-binomial counts with mean Lambda_j and shape r_j (variance Lambda_j+Lambda_j^2/r_j). The population joint law of Y identifies the law of Lambda. No moment assumption on Lambda is needed.

**Proof.** Independently of Lambda, draw mutually independent G_j~Gamma(shape r_j, rate r_j), and let R_j=Lambda_j G_j. Mixing a Poisson count of rate R_j over G_j yields exactly the specified negative-binomial mass (including the degenerate Lambda_j=0 case). Thus the already independently reviewed product-Poisson injection identifies the joint law of R from the joint law of Y.

Since every G_j is strictly positive and finite almost surely, R and Lambda have the same coordinate zero pattern. Their common zero-pattern probabilities are therefore identified. Fix a pattern of positive coordinates S with positive probability. Conditioning on this pattern preserves independence of G from Lambda, because the pattern depends only on Lambda. On S, all logarithms are finite and

log R_S = log Lambda_S + log G_S.

For t in R^|S| the characteristic function of log G_S is

product over j in S of [r_j^(-i t_j) Gamma(r_j+i t_j)/Gamma(r_j)].

The gamma integral supplies this expression, and it is nonzero: the gamma function has no zeros and r_j+i t_j is not a pole since its real part is positive (see [NIST DLMF §5.2](https://dlmf.nist.gov/5.2)). Division by this known characteristic function identifies the characteristic function, hence probability law, of log Lambda_S conditional on the pattern. Exponentiation recovers its positive coordinates; remaining coordinates are zero. Combining all finitely many patterns recovers the full law of Lambda. The all-zero pattern is handled directly. This completes the injectivity argument without requiring positive rates everywhere or finite rate moments.

**Consequences and limits.** The finite distinct-atom/shared-distinct-weight theorem B therefore also holds for this fixed-known-shape product-NB kernel, with its appropriate Bayes emission probabilities. The same additional across-assay conditional-independence restriction is needed for a joint completion; no realized cell state or causal orientation follows. C01 uses r_j=2, fixed before fitting, so its kernel is in this class. This supplies a population information theorem, not consistency, stable deconvolution, optimizer convergence or model adequacy for its finite samples.

The argument fails as a justification for unknown or learned shape, because the multiplier being divided out is then unknown. It also requires independent conditional coordinate emissions and independent gamma factors with specified shapes. An endogenous library total or unknown capture factor is not automatically a known exposure. No default inVAE/scVI learned-dispersion decoder is certified by this lemma. Continuous auxiliary theorem D can be composed with this injective kernel only when the same known-shape and common measurement assumptions hold in both models; that is an explicitly restricted extension, not a proof of those implementations.

This proof is mathematical, not a generated benchmark. It is not part of the Lean algebra formalization; its characteristic-function and kernel steps require independent prose review.
