# Final independent theory cross-review

Reviewer: `causal_reliability_pilot`  
Date: 2026-11-01  
Reviewed artifact: [REVIEW.md](../REVIEW.md)  
Reviewed SHA256: `26be6e5c9c02bc801bb2bd45e8803b3953bfb94ba14b30da63c348f8ae516291`

The reviewed synthesis has no blocking mathematical or theoretical scope error within this review's remit. It rejects a universal impossibility claim, states the negative result over an admissible model class, and preserves the positive unpaired identification result. This is a bounded review of the argument and theory interpretation, not independent verification of all 64 papers or their implementations.

The following points were checked against the existing proofs and the primary theory sources recorded in [theory_methods.json](theory_methods.json):

- **Quantifiers:** The negative claim requires two admissible models with identical laws for every permitted learner input and different values of the chosen target. A restricted model class that excludes a countermodel is not ruled out by that pair.
- **Sampling and algorithmic scope:** The identical finite-sample law and binary worst-case error bound are confined to independent, disjoint assay pools with the same supplied information. The text distinguishes this from shuffled same-cell measurements and permits abstention or an ambiguity set.
- **State and residuals:** The Gaussian pair has invariant assay laws conditional on the observed state, while its conditional residual covariance and direct effect differ. The displayed derivative and the one-half binary error bound are correct. The argument does not claim every benchmark must score 50%.
- **Objectives and downstream tests:** An objective depending only on indistinguishable inputs cannot distinguish the models. A well-calibrated test on a synthetic joint concerns that joint; optimization or calibration alone does not establish equality with the biological joint.
- **Sturma's positive result:** The observed joint law is exactly identified under its model. Signed-block representation ambiguities leave that law unchanged. The shared latent graph is identified up to relabeling, while its coefficients can retain sign ambiguity. The graph theorem requires separated shared/private blocks; the joint-law result permits the shared-to-private extension in Remark 3.3. The synthesis retains the linear measurement and disturbance restrictions and does not claim a generic count observation layer is covered.
- **Targets:** Shared representations, joint laws, latent DAGs, and original-feature direct effects are kept separate. The other positive sources are not presented as proving unrestricted causal discovery from separate assay marginals. A paired or auxiliary-variable theorem is not transferred to inferred pairings without justification.
- **Empirical evidence:** Published benchmark success and local benchmark failures are not treated as universal identification proofs. Current-data applicability is explicitly unvalidated.

One nonblocking precision addition was sent to the lead reviewer: the displayed Gaussian pair changes the direct edge and the oracle-state conditional independence. If the target is instead the two-node PAG over only `X,Y`, with `S` omitted, both models have dependent focal variables. That coarser target needs its own counterexample, as already supplied in [Note 07](../../theory/07_unpaired_integration_nonidentifiability.md). Repeating this qualification in the synthesis would prevent the displayed pair from being read as a proof for every graph equivalence target.

The requested Sturma extraction clarification has been applied to the theory ledger. Its post-correction SHA256 is `284b9fbf89eb3c37d3e83c944e058deb682b026922aaf507149eb81a3bd4868b`. No datasets, experiments, model fits, or discovery settings were changed.
