# Lean algebra core

`RepairCore.lean` was compiled 2026-11-15 with the previously pinned Lean 4.19.0 / Mathlib 4.19.0 cache, using the older project only as the build environment. The resulting `RepairCore.olean` is in this new directory. From the sealed older project's `lean/` directory:

```sh
ELAN_HOME=/tmp/epigenomics-elan /tmp/epigenomics-elan/bin/lake env lean --root=../../nonlinear_repair_2026_11_15/lean -o ../../nonlinear_repair_2026_11_15/lean/RepairCore.olean ../../nonlinear_repair_2026_11_15/lean/RepairCore.lean
```

The command exited 0 and the printed axiom dependencies for every named checked theorem are `[propext, Classical.choice, Quot.sound]`; there are no custom axioms, sorry or admit. The source imports Mathlib and does not modify the old source or caches.

`recover_two_atom` proves over real numbers that two two-atom mixtures with strictly interior weights and distinct atoms, whose first three algebraic moments match, have identical atom/weight pairs up to one common swap. `variance_identity`, `variance_positive`, `sum_recovery`, `product_recovery`, and `unordered_from_sum_product` expose the steps. The rational fixture verifies moments 3, 11, 43 for weights 1/3, 2/3 and rates 1, 4. `repeated_weight_swap` proves that swapping equal-weight atoms preserves every algebraic moment; the 1-versus-4 fixture makes the two assignments distinct. `distinct_weight_match` records a small uniqueness implication for matching a single weight, not a general finite-mixture matching theorem. `scalar_contrast` checks the one-dimensional affine, nonzero-slope step in proposal D.

The moment quantities are algebraic power sums, not encoded count distributions; interpreting Poisson factorial moments as rate moments requires the separate probability argument. The formalization does not prove the multivariate Poisson/Laplace injection, Radon–Nikodym derivatives, measure equivalence, general K-component matching, matrix contrast rank, empirical validity or numerical estimation. The result is a sufficient two-component special case, not a replacement for the full reviewed population propositions. Independent proof-to-model review remains required.
