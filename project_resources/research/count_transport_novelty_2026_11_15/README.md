# Novelty audit of the count-transport corollaries

Date: 2026-11-15. Focused primary-literature audit, with two independent agent reviews. This is an assessment of prior art, not external peer review or an exhaustive priority certification.

**Assessment: E and E2 remain valid under their stated assumptions, but we should not present them as a substantially new identification theorem.** Published results already supply the count-observation reduction and the unpaired linear alignment argument. The unequal-width step is standard subspace reduction. No additional identification mechanism was found in our derivation.

The exact assembled statement was not located in one paper. That does not establish novelty: a useful corollary can follow directly from existing theorems without having been printed in the same notation or biological setting.

## Statements audited

- [E: shared count-intensity transport](../nonlinear_repair_2026_11_15/theory/COMP_COUNT_SHARED_TRANSPORT.md): separate product-Poisson count laws, positive rates with a known log link, square linear mixing of common independent non-Gaussian sources, and no nonidentity signed-permutation symmetry.
- [E2: unequal-width transport](../nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md): full-column-rank rectangular mixing; identify the transport on the latent log-rate subspaces.

The target is population transport of assay signals. It is not identification of a regulatory DAG or reliable estimation from our finite count panels.

## Closest prior results

| Our proof component | Primary prior art and exact location | Assessment |
|---|---|---|
| Unique unpaired linear transport | Foster et al., **CoMP**, ICML 2022, Theorem 2 and Appendix E.1 (Theorem 6; Assumption 7). [Paper](https://proceedings.mlr.press/v162/foster22a/foster22a.pdf) | The same ICA and source-symmetry argument already identifies the relative linear map. It applies after our log-rate laws are identified. |
| Recovering latent output laws through count noise and a known link | Lopez et al., **Toward the Identifiability of Comparative Deep Generative Models**, CLeaR 2024, Theorem 2 and Appendix D.2, Proposition 2. [Paper](https://proceedings.mlr.press/v236/lopez24a/lopez24a.pdf) | The count-noise-to-noiseless reduction is already explicit. Its main representation theorem has different latent and decoder assumptions, so that whole theorem should not be claimed to subsume E. |
| Poisson observations with rates exp(f(S)) preserve noiseless identification | Chaussard, Bonnet and Le Corff, **Independent Component Discovery in Temporal Count Data**, ICML 2026, Proposition 3.1 and Appendix A.1. First preprint: 2026-01-29. [Proposition and proof](https://arxiv.org/html/2601.21696v2#S3.SS2); [publication record](https://proceedings.mlr.press/v306/chaussard26a.html) | This is a particularly close explicit statement of our entire count/log-link step. Its proposition allows more features than sources. Its temporal source-identification corollary is a separate result. |
| Unequal observation dimensions in unpaired domains | Sturma et al., **Unpaired Multi-Domain Causal Representation Learning**, NeurIPS 2023, setup equations (1)-(2), conditions C1-C2, Theorem 3.1. [Paper](https://papers.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf) | Rectangular full-rank observation maps and heterogeneous feature dimensions already appear. Their centered-source assumptions differ from E's raw linear-symmetry restriction; not every E example is literally included. E2 also follows directly from CoMP by standard subspace reduction. |

## Explicit implication for E and E2

This is our mathematical comparison of the cited results, rather than a claim that any one paper prints our complete corollary.

Apply the count/log-link reduction separately to each assay, taking trajectory length one and choosing the mixing function to be that assay's linear map. Equal count distributions imply equal log-rate distributions. The resulting population laws satisfy the square ICA alignment model already used by CoMP. Its theorem identifies the relative linear map; exponentiating returns the intensity-space map.

For E2, the log-rate covariance has rank equal to the latent dimension and range equal to the loading matrix's column space. Choosing coordinates on each identified subspace reduces the argument to the square case. Lifting the result back gives the same map regardless of basis. No new count inversion, source matching, or linear-algebra mechanism is required.

The conclusion is **low substantive theoretical novelty**, with high confidence for the individual ingredients and their routine composition. Whether an identical complete proposition has already appeared remains unresolved. That narrower bibliographic uncertainty does not support a first-discovery claim.

## Additional literature and exclusions

| Source | Relevance and boundary |
|---|---|
| Chaussard, Bonnet, Gassiat and Le Corff, **PLN-Tree**, 2024 preprint / 2025 publication, Section 4 Lemma 1 and Appendix C. [Author paper](https://www.imo.universite-paris-saclay.fr/~elisabeth.gassiat/sn-article.pdf) | Explicit multivariate Poisson mixing-law identification predates E. It does not establish unpaired assay alignment. |
| Ferrari, Letac and Tourneret, **Exponential families of mixed Poisson distributions**, 2007. [Paper](https://perso.tesa.prd.fr/jyt/publis_fichiers/Letac2007.pdf) | Classical scalar mixing-law identification appears explicitly in the introduction. Do not cite that scalar passage alone as the full multivariate proof. |
| Xi and Bloem-Reddy, **Indeterminacy in Generative Models: Characterization and Strong Identifiability**, AISTATS 2023, Assumptions 1/3, Theorem 2.2 and Proposition 3.1. [Paper](https://proceedings.mlr.press/v206/xi23a/xi23a.pdf) | Prior framework for identification-preserving observation channels and generator/prior symmetries. Does not itself prove our specific Poisson corollary. |
| Moriakov, Adler and Teuwen, **Kernel of CycleGAN as a Principal Homogeneous Space**, ICLR 2020, Propositions 2.1-2.2. [Paper](https://arxiv.org/pdf/2001.09061) | Alternate exact alignments differ by probability-preserving automorphisms. This general symmetry principle is established; the paper does not supply our count channel or ICA restrictions. |
| Podosinnikova, Bach and Lacoste-Julien, **Beyond CCA**, ICML 2016, Theorem 1 and equations (15)-(18). [Paper](https://proceedings.mlr.press/v48/podosinnikova16.pdf) | Unequal view dimensions and common component alignment, but uses observed cross-view moments and additive rates. Not an unpaired-marginal substitute. |
| Zhou and Wei, **pi-VAE**, NeurIPS 2020, Theorem 1. [Paper](https://proceedings.neurips.cc/paper/2020/file/510f2318f324cf07fce24c3a4b89c771-Paper.pdf) | Identifiable nonlinear Poisson latent models using observed auxiliary modulation. Those auxiliaries are not supplied by E. |
| Shrestha and Fu, **Towards Identifiable Unsupervised Domain Translation**, ICLR 2024. [Paper](https://arxiv.org/html/2401.09671v2) | Multiple corresponding auxiliary-conditioned distributions remove ambiguities. Different information assumptions from E. |
| Syrota et al., **Identifiable alignment of unpaired representations**, November 2026 preprint, Proposition 3.1, Lemma B.6 and E.6. [Paper](https://arxiv.org/html/2611.11460) | Shared metric-manifold geometry supplies alignment restrictions. The single-cell example aligns models reading the same genes; it is not a theorem for heterogeneous Poisson-count assays. |
| Yi et al., **Concept Modulation Models**, June 2026 preprint, Definition 3 and Theorem 1. [Paper](https://arxiv.org/html/2606.18509v1) | General identification-preserving kernel factorization and transition arguments. Its common-support and attribute-modulation setup is not a direct replacement for E's assay-specific maps. |
| Ben-Ari et al., **Unpaired Canonical Correlation Analysis**, October 2026 preprint, Section 4.1 and Assumption 1. [Paper](https://arxiv.org/html/2610.09530v1) | Correlation-maximizing proxy pairing under permutation/optimality conditions. Does not prove the same Poisson-ICA transport result. |
| **Identifying Dependent Components from Multi-domain Linear Mixtures**, October 2026 preprint, Theorems 3.2-3.3. [Paper](https://arxiv.org/abs/2610.28330) | Common observation matrix and feature space with domain-varying diagonal scalings. Different domain design; no count channel. |
| **Causal Discovery for Linear Models with Latent Confounding via Multi-domain Identifiable Representation Learning**, October 2026 preprint, equation (4). [Paper](https://arxiv.org/abs/2610.28134) | Same observed variables and invariant causal coefficients with changing noise distributions; conditions on an identifiable noise representation. Not heterogeneous count integration. |
| **Identifiable Multimodal Causal Representation Learning under Partial Latent Sharing**, May 2026 preprint, Theorem 4.2/B1. [Paper](https://arxiv.org/abs/2605.19135) | Noiseless nonlinear maps and partial sharing; this theorem excludes globally shared sources. No separate-marginal identification statement located. The referenced proof appendix was absent from the retrieved PDF, so no complete proof audit is claimed. |

Other screened leads included additive point-process ICA, discrete ICA, and Poisson matching-quantiles regression. Their observation designs or intensity models differ. Their exclusion is not a judgment that the papers are invalid or unimportant.

## Proof qualifications and citation corrections

The existing full-box Laplace-transform proof supports E's count step. Some retrieved presentations need routine analytic care: Lopez's multivariate continuation cannot be justified by a diagonal alone; Chaussard 2026's negative generating-function arguments require integrability. Restricting to a positive open box and then using analytic continuation supplies the standard argument. These details do not make the underlying count-identification result new.

Lopez 2024 is the comparative-DGM paper, not PLN-Tree. PLN-Tree is by Chaussard, Bonnet, Gassiat and Le Corff. The pi-VAE paper is by Zhou and Wei. The identifier `2401.09671` belongs to Shrestha and Fu; Lopez's preprint is `2401.15903`. This audit uses the corrected source identities.

## Recommended wording

“We derive an application-specific count-data corollary by composing established Poisson-mixture and linear-ICA alignment results, and express the induced transport on unequal-dimensional assay subspaces.”

Keep the result as a documented modeling foundation. Do not describe it as a new general nonlinear causal-identification theorem, a first unpaired alignment result, or a demonstrated recovery algorithm. A future contribution would need a substantive advance, such as a justified relaxation of the model or an estimation/stability result with evidence under the permitted data design. None is established by this audit.

## Scope and reproducibility

OpenAlex was the initial discovery index. Eight fresh queries retrieved the first 100 results each: 800 appearances, 737 unique OpenAlex work IDs. These are retrieval counts, not 737 reviewed papers. The broad searches were deliberately capped; work versions and DOI duplicates are tracked in the inventory. Targeted web discovery and primary texts from arXiv, conference proceedings, PMC and author repositories supplied the theorem comparisons. Searches were in English with no lower date bound, through 2026-11-15.

Both independent agents agreed with the low-novelty assessment. Their scopes were unpaired transport and count/ICA theory. This is internal agent review, not outside expert endorsement. The report distinguishes source statements from our implication argument and does not rely on a failure to find an exact textual duplicate.

See [search log](SEARCH_LOG.md), [retrieved metadata inventory](openalex_inventory.csv), [citation verification](citation_verification.json), and [review record](REVIEWS.md). Existing E/E2 statements, experiments and sealed historical artifacts were not changed. No new worlds, datasets or fits were generated.

This bounded audit cannot exclude every earlier statement, unindexed paper or unpublished manuscript. It provides positive evidence that the essential mathematical content already follows from prior results.
