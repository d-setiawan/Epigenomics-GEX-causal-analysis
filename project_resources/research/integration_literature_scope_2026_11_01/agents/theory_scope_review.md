# Independent scope review: what unpaired integration can identify

Reviewer: `causal_reliability_pilot`  
Date: 2026-11-01  
Evidence ledger: [theory_methods.json](theory_methods.json)

The proposed blanket conclusion, “integration cannot recover causal structure,” is not supported. There are published identification results for genuinely unpaired observations, including recovery of a shared latent causal graph. The defensible negative statement is narrower: an integration algorithm cannot identify a target that varies across admissible biological models producing the same distribution of its permitted inputs. Establishing that premise requires specifying both the target and the admissible model class.

## The most important challenge to a universal negative claim

Sturma et al. identify a joint observation law and then a shared latent DAG from unpaired domain marginals under explicit linear measurement and source-distribution restrictions. Additional sparsity and rank conditions identify the graph. This is a positive result using model assumptions rather than additional paired measurements. See the official PDF, Theorems 3.1 and 4.4. [Sturma et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf)

This result does not contradict our [existing counterexample](../../theory/07_unpaired_integration_nonidentifiability.md): the Gaussian countermodels violate the paper’s source assumptions, so they cannot prove nonidentification inside its restricted model class. Conversely, calling a fitted representation a shared state does not establish those assumptions. A theorem for linear measurements of latent variables cannot be transferred to a Poisson count observation layer without checking what remains identifiable. Recovery of a latent graph also needs a separate argument before its nodes or arrows can be interpreted as particular peak–gene effects.

Timilsina et al. provide another unpaired positive result for identifying shared content under linear mixture and distributional restrictions. Their central shared-representation target retains an invertible linear ambiguity; identifying a common representation is therefore a different claim from identifying its causal graph. [Timilsina et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6c967a0e4168df9614a81ed4195ba78-Paper-Conference.pdf)

## Observation contracts must be compared before conclusions

| Literature route | What makes its positive claim possible | Why it is not automatically our result |
|---|---|---|
| Unlinked monotone regression | A monotone regression function and specified independent noise law | A restricted regression problem does not identify an arbitrary multivariate DAG. [Balabdaoui et al., 2021](https://www.jmlr.org/papers/v22/20-689.html) |
| Identified statistical matching | An identified parametric model; an instrumental-variable construction is one example | Shared covariates alone do not validate exclusions or identify the model. [Kim et al., 2016](https://www150.statcan.gc.ca/n1/pub/12-001-x/2016001/article/14539-eng.pdf) |
| Auxiliary-variable nonlinear ICA / iVAE | Jointly observed informative auxiliary variables and conditional source restrictions | Labels must satisfy the theorem’s variability assumptions; ordinary latent integration does not imply this. [Hyvärinen et al., 2019](https://proceedings.mlr.press/v89/hyvarinen19a.html), [Khemakhem et al., 2020](https://proceedings.mlr.press/v108/khemakhem20a.html) |
| Multiview nonlinear ICA | Views of the same latent realization and restrictions on the measurement functions | Its true joint pairs cannot be replaced by invented matches without an argument. [Gresele et al., 2020](https://proceedings.mlr.press/v115/gresele20a.html) |
| Nonlinear biomedical causal representation | Complete multimodal joint observations and structural restrictions | Theorems 4.2/4.4 compare full joint laws, not separate assay marginals. [Sun et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/2ec91b31ac4aa578d99309d4f00c81ad-Paper-Conference.pdf) |
| Propensity alignment | Observed perturbation labels constrain a common matching representation | The target is alignment; equal propensity scores need not identify a complete state. [Xi et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ff56c8dc4c61017eab9771481b7fda0d-Abstract-Conference.html) |
| Temporal OT followed by GRN inference | Joint gene panels at known timepoints, plus a dynamic inference model | This is a temporal design, with empirical benchmark evidence rather than a general static unpaired-modality theorem. [Tong et al., 2026](https://link.springer.com/chapter/10.1007/978-3-032-25311-8_21) |
| Causal multiomics hypotheses | Directed prior networks combined with biological contrasts | Selected mechanistic hypotheses are not de novo identification from disjoint marginals. [Dugourd et al., 2021](https://link.springer.com/article/10.15252/msb.20209730) |

The data distinction is mathematical, not merely terminological. Independently sampled disjoint assay pools have a different observed-data law from two shuffled lists of measurements on the same finite cells. Pairing identifiers can be hidden while dependence remains in whole-sample summaries. The unlinked-regression literature discusses this distinction explicitly; our finite-sample impossibility proof should retain its independent-pool assumption. [Balabdaoui et al., 2021, introduction](https://www.jmlr.org/papers/volume22/20-689/20-689.pdf)

## What the negative evidence actually establishes

Statistical matching illustrates why separate block distributions generally leave multiple compatible joint laws. Conditional independence, directional exclusions, known noise, or another justified restriction can reduce that set. The restriction is part of the identification argument; a numerical optimizer selecting one solution is not itself evidence that the selected completion is biological truth. [D’Orazio et al., §§1–3](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf)

Likewise, a Schrödinger-bridge theorem showing that some cost or regularization value recovers a true coupling does not show that the correct value is determined by the available marginals. The reviewed statistical-matching construction uses missing cross-block information in its recovery conditions or tuning route. [Koo et al., 2026, §3 and Appendix B.4](https://arxiv.org/html/2606.22770v1)

Locatello et al. prove an unsupervised latent-disentanglement ambiguity. It is relevant to claims that a generic representation has a uniquely meaningful coordinate system, but it is not a substitute for a proof about our original measured-variable DAG. Conversely, additive-noise causal identification assumes access to the appropriate joint distribution, so it cannot silently supply a missing cross-assay dependence structure. [Locatello et al., 2019, Theorem 1](https://proceedings.mlr.press/v97/locatello19a.html), [Peters et al., 2014, Theorem 28](https://www.jmlr.org/papers/v15/peters14a.html)

## Implications for the project

The existing proofs remain useful as limits on a broad class that allows unknown cross-assay residual relationships. They cannot rule out every identifiable model compatible with unpaired data. The current benchmarks also cannot establish a universal impossibility result: a failed fit may reflect measurement noise, sample size, optimization, model mismatch, or an invalid discovery test.

The next theoretical decision should distinguish three possible scientific claims:

1. A shared state or shared latent subspace can be recovered under stated measurement assumptions.
2. A graph among identifiable latent biological factors can be recovered under additional causal restrictions.
3. A particular observed chromatin feature directly regulates a particular observed expression feature.

A result for the first or second claim does not automatically prove the third. For each claim, the review should specify the observed data, restrictions, target equivalence class, and measurement model. When these leave several possibilities, reporting that ambiguity is more informative than presenting one algorithmic completion as identified causality.

## Review limits and traceability

The ledger contains 16 core records, three screened exclusions, primary evidence locators, publication-version distinctions, and supplementary search/access logs. OpenAlex supplied the initial central discovery; known primary sources and targeted follow-up searches supplemented it. This is a bounded scoping review, not an exhaustive systematic review. Full theorem statements and relevant methods were inspected; every published proof and implementation was not independently verified.

No datasets, simulations, integration fits, causal tests, or benchmark settings were changed for this review. Tong’s final publication metadata is verified, while detailed methods evidence comes from its available 2024 preprint. The ledger preserves that distinction. The undated Istat manuscript has no invented publication year. Related OpenAlex preprint records are kept separate from final-publication metadata.
