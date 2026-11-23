# Why the NB shape must be fixed: an observation-channel countermodel

Astra derivation, 2026-11-15; independent review accepted; see [review](../reviews/UNKNOWN_SHAPE_NB_INDEPENDENT_REVIEW.md). This is a deterministic analytic countermodel, not a new benchmark world or simulated dataset.

Use NB(shape r, mean lambda), with mass Gamma(y+r)/(Gamma(r)y!) times (r/(r+lambda))^r times (lambda/(r+lambda))^y. Model A has shape1 and constant latent mean lambda=1. Thus P_A(Y=y)=2^(-y-1) for every integer y>=0.

Model B has shape2 and latent mean Lambda uniformly distributed on(0,2). Its count mass is

P_B(Y=y) = (1/2) integral_0^2 (y+1) [2/(2+lambda)]^2 [lambda/(2+lambda)]^y d lambda.

Set t=lambda/(2+lambda). Then lambda=2t/(1-t), d lambda=2/(1-t)^2 dt, and t ranges from0 to1/2. The integral becomes (y+1) integral_0^(1/2) t^y dt =2^(-y-1). The two count laws agree at every count, but one latent mean law is a point mass and the other is uniform. Both means average1, while their rate-law variances differ. All rates and shapes are finite and positive almost surely.

Therefore an unrestricted NB mixing model with unknown shape does not identify its latent mean distribution from the count law. The fixed-known-shape theorem compares models with the same r and is unaffected. This example does not refute finite-atomic-mixture identification with unknown shape (the second mixing law is continuous), nor any sufficiently restricted auxiliary or decoder model; such additional assumptions require their own theorem. It rules out using the fixed-shape observation-channel argument alone to certify arbitrary learned-dispersion models.
