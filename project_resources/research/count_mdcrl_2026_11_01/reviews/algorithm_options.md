# Count-aware source estimation: bounded method review

This is a focused narrative assessment, completed on 2026-11-01, of estimators that could replace ordinary ICA on the existing count panels. It is not a systematic review. OpenAlex was searched first using the three exact queries in [algorithm_options_search_log.json](algorithm_options_search_log.json); their first pages returned 41, 100 and 100 records, before deduplication. Supplementary title searches and primary-source reads are preserved in `count_options_*.json`. No new dataset, environment installation, or scientific fit was used for this assessment.

The selected route is a factorial-cumulant estimator motivated by discrete ICA (DICA). This changes the covariance and third-order source-estimation equations to account for conditional Poisson sampling. It does not merely normalize counts before ordinary ICA. Selection of this route is an implementation decision for a bounded diagnostic, not a published guarantee for these biological panels.

| Primary method | What was verified | Relevance and limits |
|---|---|---|
| Podosinnikova, Bach and Lacoste-Julien, *Rethinking LDA: Moment Matching for Discrete ICA*, NeurIPS 2015 | Full paper, Sections 2–4: independent latent intensities, conditional Poisson observations, corrected second/third cumulants and diagonalization. | Direct foundation for a count-noise correction. The paper's topic model uses nonnegative loadings; our exposure-aware affine formulation and signed-support restriction are local extensions. It supplies no unpaired biological causal-graph guarantee. [Primary paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/9be40cee5b0eee1462c82c6964087ff9-Paper.pdf) |
| Cemgil, *Bayesian Inference for Nonnegative Matrix Factorisation Models*, 2009 | Full methods: Poisson latent allocations, conjugate priors, variational/Gibbs inference and model-order selection. | A compact count-likelihood alternative. A good factorization alone would not establish independent, distributionally distinguishable causal sources. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2688815/), [DOI](https://doi.org/10.1155/2009/785152) |
| Levitin et al., *De novo gene signature identification from single-cell RNA-seq with hierarchical Poisson factorization*, 2019 | Full methods: cell/gene capacities, Gamma weights and Poisson observations; sparse variational implementation. | Strong biological count-model precedent. Shared cell capacities induce dependence among marginal factors, and the published factor-count procedure uses cell-type/cluster information. Those choices cannot simply inherit independent-source identification or our no-label contract. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6386217/), [DOI](https://doi.org/10.15252/msb.20188557), [official code](https://github.com/simslab/scHPF) |
| Podosinnikova, Bach and Lacoste-Julien, *Beyond CCA: Moment Matching for Multi-View Models*, ICML 2016 | Full methods and Theorem 1: moment methods with identifiable multi-view mixing under stated rank and source restrictions. | Its cross-view estimator uses joint observations. Those cross-view moments are not available from our disjoint assays; its identification result cannot justify manufacturing them. [Primary paper and PDF](https://proceedings.mlr.press/v48/podosinnikova16.html) |
| Chaussard, Bonnet and Le Corff, *Independent Component Discovery in Temporal Count Data*, ICML 2026 | Primary proceedings abstract and official repository only; full theorem not audited. | Poisson-lognormal emissions and regime-adaptive temporal dynamics provide a relevant positive example. We do not have the temporal/regime information used by that framework, so its stated identification guarantee is not transferred to this pilot. [Primary record](https://proceedings.mlr.press/v306/chaussard26a.html), [official code](https://github.com/AlexandreChaussard/PLNICA) |

The DICA paper's linked `anastasia-podosinnikova/dica` repository returned 404 during access. The present estimator was implemented from the mathematical construction and checked with exact fixtures; it is not a reproduction of unavailable author code. The scHPF and PLNICA repository landing pages were accessible, but neither package was installed or used here.

## Local adaptation and its assumptions

The proposed working model is

$$
Y_{ij}\mid t_i,\lambda_i\sim\operatorname{Poisson}(t_i\lambda_{ij}),
\qquad \lambda_i=\mu+B\varepsilon_i,
$$

with conditional independence across observed coordinates, independent source coordinates, and positive rates on the full source support. Signed loadings require a compatible bounded support or another explicit positivity restriction. A large intercept does not make an unbounded signed source globally valid.

With $R_{ij}=Y_{ij}/t_i$, the covariance target is obtained from

$$
C_2=\operatorname{Cov}(R)-\operatorname{diag}\!\left(\mathbb E[Y_j/t^2]\right).
$$

Exposure-adjusted factorial third moments supply a corresponding third cumulant. Whitened tensor contractions estimate source directions, retaining all 30 permitted features. This is a within-assay estimator. It cannot determine which independent source in another assay represents the same biological variable without additional model restrictions and distinguishable source laws.

The local protocol estimates directions on one training partition and fingerprints on another. Fixed low-order moments and a finite characteristic-function grid support an uncertainty-aware compatibility screen. Agreement at these summaries does not establish equality of complete distributions. The selection rule may abstain or retain ambiguity; no shared sources are forced.

## Why this remains a diagnostic

- Feature-specific negative-binomial overdispersion is not removed by a Poisson correction. The marginal-preserving permutation screen reduces reliance on a Poisson-only rank null but does not certify biological dimension.
- A library-derived exposure can carry biological signal and is not known exogenous technical depth. The outside-panel proxy is a declared sensitivity analysis.
- Zero or weak third cumulants can prevent this particular tensor estimator from orienting sources, even where another estimator could use higher-order information.
- Posterior-mean matching from a variational count model would mix biological distributions with assay-dependent shrinkage. Common fixed priors would also risk constructing the distributional similarity used for alignment. These are reasons to separate observation modeling, source estimation, and uncertainty calibration explicitly.
- The existing simulated backgrounds are not guaranteed to satisfy an affine independent-source count model. Success or failure here tests this specified adaptation; it cannot settle unrestricted integration or direct peak–gene identification.

The primary papers support count-aware modeling and restricted source estimation. The final choice, implementation extensions, uncertainty gates, and cautions above are our synthesis and are not claims made collectively by those papers.
