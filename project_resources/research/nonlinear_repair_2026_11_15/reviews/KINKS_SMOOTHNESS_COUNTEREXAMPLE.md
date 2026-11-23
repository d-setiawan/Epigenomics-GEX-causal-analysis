# Counterexample to the stated general kink-prior theorem

Independent primary-source check, 2026-11-15. The following refutes the literal general statement of [*Kinks vs. Smoothness: Identifiability of Real Analytic nICA for Laplace-like Sources*, arXiv:2609.21926v1, Assumption 2 and Theorem 1](https://arxiv.org/html/2609.21926v1), HTML lines 101–115 and 128–130. The source claims that every real-analytic diffeomorphism preserving an iid law satisfying Assumption 2 is a signed permutation and translation. This counterexample uses a fixed, exactly known prior satisfying that stated assumption; it does not rely on a learned or misspecified prior.

## Construction and exact check

Fix 0<a<1, and define

F(x)=x+a log(cosh x),   rho(x)=1/2 exp(−|F(x)|) F′(x),   h(x)=F⁻¹(−F(x)).

F is real analytic on R, F′(x)=1+a tanh x lies in (1−a,1+a), and F(x) tends to ±∞ at ±∞. Hence F is a real-analytic increasing diffeomorphism R→R with real-analytic inverse. Its only zero is F(0)=0, and rho is strictly positive on all R. If X has density rho, change of variables gives F(X) the standard symmetric Laplace law; thus rho is normalized and its connected support is R.

For x<0, log rho(x)=F(x)+log F′(x)−log 2; for x>0, log rho(x)=−F(x)+log F′(x)−log 2. Both expressions are real analytic on the whole real line, including across the breakpoint, because F′ is positive. The sole breakpoint is β1=0, with nonempty intervals on either side. Since F′(0)=1 and F″(0)=a, the left and right log-density derivatives at zero are a+1 and a−1. Their jump is −2, nonzero. For d=1 breakpoint (the paper's d counts breakpoints), both adjacent analytic pieces continue to a neighborhood of zero. Thus all three requirements of the stated Assumption 2 hold, including its continuation clause.

h is a globally defined real-analytic decreasing diffeomorphism and involution. Since F(h(X))=−F(X) and standard Laplace is symmetric, h(X) has exactly the same density rho. Differentiating F(h(x))=−F(x) yields h(0)=0, h′(0)=−1, and h″(0)=−2a≠0. Therefore h is not affine, and in particular is not a signed permutation plus translation in one dimension. For any iid vector (X1,…,Xk) with this same rho, the coordinatewise map H(x1,…,xk)=(h(x1),…,h(xk)) is a real-analytic diffeomorphism preserving the exact iid product law but is not a signed affine permutation. This supplies an example in any finite k, including dimensions above one.

## Location of the proof failure

[Appendix A](https://arxiv.org/html/2609.21926v1#A1), lines 362–363, says the odd-breakpoint orientation-reversing case mirrors the orientation-preserving case and concludes its analytic factor r is identically −1. In this construction h(x)=x r(x), with r analytic through 0, r(0)=−1 and r′(0)=−a≠0. Nevertheless, the proof's gamma is analytic:

log rho(h(x))−log rho(x)=log F′(h(x))−log F′(x).

Here |F(h(x))|=|F(x)|; both terms on the right are analytic globally. Thus analyticity across the kink alone does not force the claimed constant factor in the reversing case. The two different analytic pieces contribute higher-order Taylor terms under reversal; the assertion that the positive-orientation expansion carries over unchanged is insufficient. This construction simultaneously supplies an explicit violation of the theorem's conclusion, not just a suspected gap in its proof.

## Scope

The finding rules out Theorem 1 as stated for its entire allowed class. It does not show that the special **exact symmetric Laplace** prior admits this non-affine symmetry: setting a=0 makes F(x)=x and h(x)=−x, an allowed sign flip. The constructed symmetry is componentwise: it refutes the claimed signed-affine ambiguity class, not a weaker claim of identification up to arbitrary invertible componentwise transformations. It also does not show failure of any trained normalizing flow, refute other assumptions or prior families individually, or settle unrelated nonlinear identifiability results. The latent self-map counterexample is exact and deterministic; no experiments, datasets, numerical benchmark, or source-code reproduction were used.

## Restricted positive result: exact iid standard Laplace

The following self-contained proposition uses a stronger, explicit prior than Assumption 2 and avoids the invalid general-prior reversal step. It awaits a separate independent proof review.

**Proposition.** Let X have iid standard Laplace coordinates in R^k and let h:R^k→R^k be a real-analytic diffeomorphism such that h(X) has the same distribution as X. Then h is a signed coordinate permutation, with no translation.

**Proof.** Write D h for the Jacobian. Density change-of-variables and continuity give everywhere

Σ_j |h_j(x)| − Σ_i |x_i| = log |det D h(x)|.

The right side is real analytic: det D h is a nonzero real-analytic function of constant sign on connected R^k, and log of its absolute value is locally analytic. Denote the left side G.

Fix an input coordinate hyperplane P_i={x_i=0}. At every point of P_i at least one h_j vanishes: otherwise all h_j retain fixed signs locally and Σ_j|h_j| is analytic there, whereas Σ_l|x_l| has a nonzero kink across P_i; the other absolute-value terms are either smooth there or add kinks and cannot cancel the kink in Σ_l|x_l|. For the clearest local argument restrict first to points of P_i off the other coordinate hyperplanes, a dense relatively open subset.

The finite zero sets {h_j=0} cover that relatively open subset. Each restricted h_j is analytic on P_i. A finite union of proper analytic zero sets cannot cover an open subset of P_i: a nonzero analytic function's zero set has empty relative interior, and a finite union of such closed nowhere-dense sets cannot cover it. Consequently some h_j vanishes on a relatively open patch of P_i. On that patch no other h_l can vanish identically on an open subpatch: two rows of the Jacobian would then each be normal to P_i and hence linearly dependent, contradicting invertibility. Removing their zero sets leaves a nonempty relatively open patch with exactly h_j=0 and all other input coordinates and output h_l nonzero.

At a point of that patch, the nonzero transverse derivative ∂_i h_j follows from invertibility, since all tangential derivatives of h_j vanish on P_i. Thus h_j changes sign on crossing P_i. In a connected neighborhood, write the formula for G on each side using its fixed signs. The remaining input and output absolute-value terms have the same analytic expression on both sides; the jth output and ith input terms contribute respectively ±h_j and ±x_i. Since G itself is analytic, uniqueness of analytic continuation makes the two analytic side expressions agree on the whole neighborhood. Their difference is 2(σ h_j−x_i), where σ is the sign of h_j on the x_i>0 side. Therefore h_j=σ x_i on a nonempty open subset of R^k, and the real-analytic identity theorem extends it to all R^k.

Repeating for each input coordinate gives an output coordinate equal to ±x_i. Distinct input coordinates must correspond to distinct outputs; otherwise one analytic function would equal ±x_i and ±x_l identically for i≠l. Since there are k inputs and k outputs, this exhausts the output coordinates. Thus h is a signed permutation. For k=1 the hyperplane is the single point 0: its image must have the kink at the same point, and the two-side argument applies directly. ∎

**Conditional count-law composition.** For each of two candidate real-analytic diffeomorphisms f,g:R^k→R^k, let S have this fixed iid standard-Laplace law, let the positive intensity be R=exp(f(S)) componentwise, and let observed counts conditionally follow the product-Poisson law with rate R. Equality of full count laws implies equality of intensity laws by the separately reviewed product-Poisson injection; the known log map yields f(S) and g(S) equal in distribution. Then h=g⁻¹∘f is an analytic diffeomorphism preserving the exact Laplace product law, so the proposition identifies the noiseless source representation up to signed permutation within that assay. This is a population result, not a claim that noisy counts determine their individual latent rate or source realization. It assumes global square analytic diffeomorphisms and the exact standard Laplace prior, and it does not establish their suitability for the actual count data.

If separate assays are considered, the within-assay result leaves independent signed-permutation ambiguities for their source coordinates. Without additional cross-assay restrictions those permutations/signs are not aligned by unpaired marginal laws, so the proposition does not identify a same-cell cross-assay joint, an assay correspondence or a causal graph. Conditional posterior recovery under a specified model also differs from exact per-cell source recovery.
