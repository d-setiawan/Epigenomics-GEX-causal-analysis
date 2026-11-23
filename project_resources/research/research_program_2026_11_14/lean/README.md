# Finite identification in Lean

Lean 4.19.0 and Mathlib v4.19.0 (`c44e0c8ee63ca166450922a373c7409c5d26b00b`) are pinned by `lean-toolchain`, `lakefile.toml`, and `lake-manifest.json`. Reproduce in this directory with a compatible Lean/elan installation and network access:

```sh
lake update
lake exe cache get
lake build
```

The last command completed successfully on 2026-11-14; the source includes `#print axioms` for the listed theorems. `structural_pushforward` has no axioms; `different_direct_parent` and `independence_identifies` use `propext`; all other printed theorems use only Lean/Mathlib's standard `propext`, `Classical.choice`, and `Quot.sound`. There are no custom axioms or unfinished proofs. Imported Mathlib supplies rational arithmetic, finite sums, order facts, and proof tactics; the statements and proofs in this project are local.

## Proved statements

* `Model` is a normalized nonnegative rational 2×2 probability table, with fixed natural-number guidance, depth, and metadata fields. `obs` contains the two complete Bernoulli marginal laws (each determined by its success probability) and these fixed fields; `Identifiable` uses equality as its target equivalence. The proved `countBlock_depth` and `countBlock_first` embed each binary variable b as the two-feature count row (b, 1-b), whose first feature is the original detection and whose total count is exactly the fixed depth 1. This is a post-selection toy class, not an assertion that S1 fits it.
* Four uniform exogenous outcomes generate X=U, Y=V and X=U, Y=X. `structural_pushforward`, `same_observation`, `different_joint`, `different_effect`, `different_direct_parent`, and `scm_nonidentification` link these explicit equations, the induced tables, fixed metadata, functional parent criterion, and intervention-by-equation-replacement risk differences 0 and 1. `joint_not_identifiable` and `no_deterministic_recovery` prove the corresponding population-observation claims.
* `positive_scm_table`, `positive_same_observation`, `positive_different_joint`, and `positive_scm_effect` use eight uniform outcomes and X XOR W, with W true on one quarter of outcomes. The joint has diagonal 3/8 and off-diagonal 1/8; it has the same observation as the independent table and effect 1/2. The finite enumeration realizes independence of X and W. The strictly positive model is a model of the table class.
* `frechet` proves necessity and attainability of the sharp interval for every rational p, q, r, with an explicitly constructed normalized table. `independence_completion_exists` and `independence_complete_table` prove existence and uniqueness of the entire table under the stated binary unconditional independence equation. The result is statistical, not causal.
* `randomized_countermodel_error` models a forced binary decision by its output-1 probability as a function of the population observation; one of the two error probabilities is at least 1/2. It does not formalize finite-sample dataset distributions or a stochastic-kernel construction. Its conclusion is algebraic even without imposing 0 ≤ q ≤ 1.

## Scope and independent review

This formalization does not encode arbitrary local measurement kernels, count-vector distributions, randomized metadata, joint filtering, shared strata and conditional independence, same-cohort shuffling, finite-sample experiments, or general SCM semantics. The intervention and parent definitions are explicit for these finite equations, not a general causal framework. The fixed metadata construction is a counterexample with identical metadata, not a claim that all possible metadata are uninformative. Nothing proves that the actual biological selected distributions satisfy these assumptions. The unconditional binary independence condition is an added cross-assay assumption and is not testable from separate marginals alone. The [independent correspondence review](../reviews/LEAN_INDEPENDENT_REVIEW.md), including the subsequent count-block amendment, records successful fresh builds and axiom checks within this scope.
