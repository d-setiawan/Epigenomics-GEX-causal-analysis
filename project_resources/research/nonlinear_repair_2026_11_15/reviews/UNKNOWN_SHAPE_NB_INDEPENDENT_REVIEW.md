# Independent review: unknown-NB-shape countermodel

2026-11-15. Reviewed source: `theory/UNKNOWN_SHAPE_NB_COUNTERMODEL.md`, SHA-256 `348a8cc1d6603982df30a156705054bb4c424e11b91742b10f7b6bf7efa0855d`. Reviewer did not author the countermodel. **Accept**, with the model-class scope stated below. No simulation, fitting or data access was used.

The stated NB convention is `P(Y=y | λ,r) = Γ(y+r)/(Γ(r)y!) [r/(r+λ)]^r [λ/(r+λ)]^y`, for integer `y≥0`; it has mean λ and variance `λ+λ²/r`. At shape r=1 and λ=1, its mass is `2^(−y−1)`.

For r=2 and Λ uniform on (0,2), the normalization coefficient is `(y+1)/2`. Under `t=λ/(2+λ)`, `dλ=2(1−t)^(−2)dt` and `[2/(2+λ)]²=(1−t)²`. The transformed integrand is exactly `(y+1)t^y`, with endpoints 0 and 1/2. Its integral is `(1/2)^(y+1)` for **every** nonnegative integer y. Thus the entire count distributions coincide. The latent mean distributions differ: Model A has variance zero, Model B has variance 1/3; both means equal 1. The open endpoints are immaterial and Λ is strictly positive almost surely.

Consequently the observation map does not identify the mixing law over a class admitting both these models and unknown shape. The result does not contradict an injectivity theorem **at a specified common known shape**. It also does not resolve identification over finite atomic mixing distributions with unknown shape, because the second mixing law here is continuous, or over a more restricted class with validated auxiliary information or decoder constraints. Nothing in this example says all unknown-dispersion models or algorithms fail. No mathematical or scope error was found in the reviewed source.
