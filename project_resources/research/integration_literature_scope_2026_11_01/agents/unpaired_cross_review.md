# Cross-review of theory and bridge classifications

Reviewer: scconfluence_adapter. Date: 2026-11-01.

Scope: independent reading of the 16-record theory extraction and 17-record bridge extraction, checking target definitions, information contracts and consistency with the unpaired-method extraction. This was not a second full-text audit of every bridge paper.

I independently checked the [Sturma et al. primary text](https://arxiv.org/html/2302.00993v2), including Definition 2.1, Conditions C1–C4, Theorems 3.1 and 4.4, and Remark 3.3. Its positive classification is supported: separate assay laws identify the joint within the specified linear, distinguishable-error model. Shared latent graph recovery adds sparsity and rank conditions. I requested a wording clarification separating exact identification of the observed joint law from representation sign/permutation ambiguities. The unrestricted Gaussian counterfamily does not refute this restricted theorem.

The theory extraction preserves several other positive routes and correctly distinguishes auxiliary-variable, true multiview, temporal and directed-prior information from independent assay marginals. The prior focused checks of unlinked monotone regression, statistical matching and the restricted additive-noise result agree with their current scope statements.

The bridge extraction consistently treats observation patterns by mode: paired-training methods can process unimodal queries without becoming unpaired-only estimators; computational anchors and pseudo-modalities do not add measured joint observations. MOFA/MOFA+, MultiVI, JAMIE and scMoMaT are appropriately qualified rather than assigned one blanket class. Same-feature batch correction is also separated from direct cross-assay causal recovery.

Two cautions should survive synthesis: masking cell IDs in a co-assayed dataset is not the same sampling experiment as independently drawing disjoint cohorts; and neither a shared factor nor a well-performing regulatory score automatically identifies a directed feature-level causal graph.

Verdict: scoped approval of the classifications, with the Sturma target-wording clarification above. No broad claim that all integration methods fail is supported or endorsed. Input hashes and the primary-source check are retained beside this note.

