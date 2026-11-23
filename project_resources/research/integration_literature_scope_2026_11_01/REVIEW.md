# Can integration recover cross-assay causal structure?

Literature synthesis, 2026-11-01. Read alongside the [paper matrix](PAPER_MATRIX.md), [search record](SEARCH_LOG.md), and [formal proof](../theory/07_unpaired_integration_nonidentifiability.md).

**The literature supports a general nonidentification result, but not the claim that causal recovery from unpaired data is always impossible.** Ordinary state alignment does not establish that the reconstructed cross-assay dependence is the biological one. However, a published theorem identifies a shared latent causal graph from genuinely unpaired observations under restrictive assumptions. This positive result changes the scope of the claim we should make. [Sturma et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract-Conference.html)

The project can defend this statement:

> For independently sampled, disjoint assays, integration cannot generally identify original cross-assay causal edges when the admissible models allow different residual couplings with the same complete observed assay distributions and the same supplied prior information. Successful recovery requires a restriction that identifies the target, or a narrower target. Such a restriction can be a model assumption; it need not be new data.

This statement concerns a specified class of biological models. It does not say every fitted graph is wrong, that all relationships are unrecoverable, or that integration has no biological value.

## What was reviewed

The selected evidence comprises **64 papers**: 24 unpaired/diagonal method papers, 17 paired/bridge/factor/anchor papers, 16 identification and causal-model papers, and seven primary benchmarks. Relevant methods, arguments or theorem statements were inspected for 59; five records remain explicitly limited to abstracts and indexed primary results. Three independent reviewers covered the method and theory groups; the lead reviewed benchmark targets and reconciled the synthesis.

Discovery began with seven OpenAlex queries, supplemented by PubMed-indexed records through Europe PMC, arXiv discovery and primary reference following. The indexed searches returned 981 records, reduced automatically to 866 metadata entries. These are **not 866 fully screened papers**: 37 entries link directly by DOI or title to 35 selected papers, while 29 selected papers entered through nominated-method or primary-reference follow-up. The remaining 829 entries are not fully assessed. Version differences can leave related records unmatched. Query caps, English-language access and selective extraction prevent an exhaustive claim. [Search protocol](SEARCH_PROTOCOL.md), [screening counts](SCREENING_SUMMARY.json).

## Four targets that must remain separate

| Target | What success means | What it does not establish by itself |
|---|---|---|
| State alignment | Corresponding populations occupy a useful common representation. | The correct joint distribution of individual measurements. |
| Joint-distribution recovery | The reconstructed cross-assay law equals the biological same-cell law. | A unique causal direction without causal assumptions. |
| Shared latent causal structure | A graph among identified latent factors is recovered to a stated equivalence. | That each latent node is a particular histone peak or gene. |
| Original-feature causal structure | Direct effects or an appropriate equivalence class among specified biological features are identified. | This requires its own measurement, confounding and causal argument. |

An integration paper may legitimately solve the first problem without claiming the fourth. Prediction, clustering, regulatory enrichment and causal identification therefore need different evidence. Even with a full observational joint, causal directions can remain ambiguous; restricted additive-noise results supply additional conditions rather than bypassing the need for the relevant joint. [Peters et al., 2014](https://www.jmlr.org/papers/v15/peters14a.html)

## The proof that applies across algorithms

Let $M$ denote a possible biological model, $P_O(M)$ the probability law of everything a learner receives, and $T(M)$ the desired causal target. A target is identified only if

$$
P_O(M_0)=P_O(M_1)\quad\Longrightarrow\quad T(M_0)=T(M_1)
$$

for all admissible models. To disprove general identification, it is enough to exhibit two admissible models with identical inputs and different targets. Listing failed algorithms is unnecessary.

Our [Note 07](../theory/07_unpaired_integration_nonidentifiability.md) provides such models, even when exact biological state is supplied to both assays. Let $S,U,V$ be mutually independent standard normal variables and define

$$
X=S+U,\qquad
Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}\,V.
$$

For both $\rho=0$ and $\rho=1/2$, each assay sees exactly the same distributions:

$$
X\mid S=s\sim\mathcal N(s,1),\qquad
Y\mid S=s\sim\mathcal N(s,1).
$$

But the original joint relationship differs:

$$
\operatorname{Cov}(X,Y\mid S)=\rho,
\qquad
\frac{\partial\mathbb E[Y\mid\operatorname{do}(X=x),S=s]}{\partial x}=\rho.
$$

The first model has no direct $X\to Y$ edge; the second does. Both are faithful to their respective observed-state DAGs. For independently sampled assay pools, the observed likelihood is a product of the two assay likelihoods. It contains no information distinguishing these values of $\rho$.

This pair distinguishes the direct edge and state-conditional independence. If $S$ is omitted and the target is only the two-node PAG over $\{X,Y\}$, both models have a dependence adjacency. Note 07 gives a separate construction for that coarser target; the target must be specified before choosing a counterexample.

Any algorithm, including an integration network followed by aggregation and causal discovery, is a function of those inputs and its random seed. Its output distribution is therefore identical in the two models. For a forced binary decision about this edge,

$$
\max_{j\in\{0,1\}}
\Pr_{M_j}\{\widehat T\ne T(M_j)\}\geq\frac12.
$$

This is a worst-case guarantee over this pair, not a prediction that every benchmark score equals 50%. Abstaining or returning an ambiguity set is possible. An algorithm can also succeed on one model while failing on its indistinguishable alternative.

The proof holds fixed **all supplied information**, not just one gene's marginal: full assay feature blocks, observable depth/background variables and guidance must be matched across the two models. The existing note gives those extensions and nonlinear/count constructions. If a proposed model restriction excludes one member of the pair, this particular proof no longer establishes nonidentification inside that restricted class. The statistical-matching literature frames the same need to restrict the unknown cross-block dependence. [D’Orazio, Di Zio and Scanu, §§1–3](https://www.istat.it/en/files/2013/12/Old-and-new-approaches-in-statistical-matching.pdf)

### Why the training objective cannot fix missing information

Under identical permitted input laws, a population objective built from those inputs is also identical in the two models. Reconstruction, distribution matching, geometry preservation, cycle consistency or a fixed guidance graph may choose a particular completion; they do not distinguish models that produce the same evidence. A unique numerical optimum is different from an identified biological coupling. [Note 08](../theory/08_modern_integration_objectives.md) proves this implication and gives an exact transport example.

This is our mathematical application of the input-law argument, not a negative theorem claimed by every method author. A method with a more restrictive, identifiable generative model must be evaluated within that model. A prior that actually supplies the disputed direction also changes the argument: it is informative input, not evidence learned from separate counts.

### Why conditional-independence testing after integration is not a remedy

Both countermodels have the same state-dependent assay means. Only one satisfies $X\perp Y\mid S$. If integration supplies a synthetic joint $Q$, a well-calibrated test can correctly test conditional independence **under $Q$**. Without identifying $Q=P(X,Y,S)$, that does not validate the same conclusion for the original biological joint. Nonlinear CI tests, more cells and stronger optimization do not remove this information distinction. Within-assay joint measurements remain usable; this argument concerns the missing cross-assay dependence.

## What the integration papers contribute

The [complete matrix](PAPER_MATRIX.md) records individual inputs, targets and evidence depth. The following are the main mechanisms relevant to our pipeline.

| Family | Representative primary papers | What the mechanism supplies | Implication for the causal claim |
|---|---|---|---|
| Generative embeddings with alignment | [GLUE](https://doi.org/10.1038/s41587-022-01284-4), [scMRDR](https://doi.org/10.52202/085713-5170) | Within-assay modeling with alignment; GLUE uses feature guidance, while scMRDR uses geometry constraints without a guidance graph. | Check the actual inputs and training losses. Alignment alone does not identify unrestricted cross-assay residual dependence. |
| Transport and manifold alignment | [SCOT](https://doi.org/10.1089/cmb.2021.0446), [Pamona](https://doi.org/10.1093/bioinformatics/btab594), [scConfluence](https://doi.org/10.1038/s41467-024-51382-x), [uniPort](https://doi.org/10.1038/s41467-022-35094-8) | A coupling favored by distances, weak feature links and regularization. | Cost optimality requires an additional argument before the selected plan is biological truth. |
| Topological, adversarial and cycle objectives | [UnionCom](https://doi.org/10.1093/bioinformatics/btaa443), [sciCAN](https://doi.org/10.1038/s41540-022-00245-6), [scTopoGAN](https://doi.org/10.1093/bioadv/vbad171) | Agreement of distributions, reconstructed observations or manifold geometry. | Symmetry can leave alternative alignments; generic objectives are not a causal-identification theorem. |
| Feature maps, anchors and contrastive pairs | [bindSC](https://doi.org/10.1186/s13059-022-02679-x), [scDART](https://doi.org/10.1186/s13059-022-02706-x), [BiCLUM](https://doi.org/10.1371/journal.pcbi.1013932) | Structured feature relations and inferred correspondences. | Derived anchors are functions of the available data; genuine supervision must be recorded separately. |
| Restricted trajectory or regression models | [MATCHER](https://doi.org/10.1186/s13059-017-1269-0), [UnpairReg](https://doi.org/10.1186/s13059-022-02726-7) | Additional assumptions about progression, orientation, regression or residual structure. | These are substantive restrictions and deserve their own identification analysis. |

GLUE reports useful regulatory validation against paired correlations and external regulatory evidence. Those findings should not be dismissed. They support particular predictions under its guidance and evaluation design; they do not establish arbitrary direct causal effects from disjoint assay marginals. The published scMRDR and scConfluence objectives and the versions used locally are separately audited in Note 08; published-method claims should not be inferred from our patched pilot alone. scMRDR's canonical citation is NeurIPS 2025; the inspected manuscript is its January 2026 arXiv v2. [GLUE paper](https://doi.org/10.1038/s41587-022-01284-4), [scMRDR inspected version](https://arxiv.org/html/2510.24987v2), [method audit](../theory/08_modern_integration_objectives.md).

UnpairReg is especially relevant because it estimates cross-assay coefficients rather than only an embedding. It matches covariance structure under a linear regression model, conditional error assumptions and cis-distance restrictions. Its simulations and paired evaluations are positive empirical evidence for that restricted approach. The inspected methods do not provide a general coefficient-uniqueness theorem. Covariance matching alone can admit alternative coefficient matrices; whether the full restrictions remove them requires a separate proof. [UnpairReg](https://doi.org/10.1186/s13059-022-02726-7)

Names can obscure training information. MinNet applies to unpaired targets but learns from paired references and labels. SCIM uses labels to orient alignment. SIMBA's paired regulatory demonstration must be distinguished from its unpaired integration construction. These are legitimate methods, but their evidence cannot be treated as if every learner received only independent unlabeled assay pools. [MinNet](https://doi.org/10.1186/s12859-022-05126-7), [SCIM](https://doi.org/10.1093/bioinformatics/btaa843), [SIMBA](https://doi.org/10.1038/s41592-023-01899-8)

## Positive results that prevent a universal impossibility claim

### A shared latent causal graph really can be identified from unpaired domains

**Sturma et al. is the strongest exception found.** Theorem 3.1 identifies the observed joint law from separate domain marginals. Theorem 4.4 adds shared latent DAG recovery. Conditions include linear latent equations, full-column-rank linear views, independent nonsymmetric standardized disturbances with distinguishable distributions, and, for graph recovery, two partial pure observed children per shared latent and rank faithfulness. [Official paper, Definition 2.1 and Theorems 3.1/4.4](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Paper-Conference.pdf)

The joint is exactly identified; sign and relabeling ambiguities concern its representation. The graph theorem separates shared and private latent blocks; Remark 3.3 allows shared-to-private paths for the joint-law result. No paired observations are required. Our Gaussian countermodels violate the disturbance conditions. However, this is a **latent graph**, not automatically a peak–gene DAG, and adding a Poisson/negative-binomial measurement layer needs another argument. These assumptions have not been validated for our counts.

### Other positive routes solve narrower or differently observed problems

| Result | Why recovery becomes possible | Boundary for this project |
|---|---|---|
| [Timilsina et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e6c967a0e4168df9614a81ed4195ba78-Abstract-Conference.html) | Linear shared/private mixtures with rank and distributional variability restrictions. | Shared-component identification, generally with an invertible linear ambiguity, is not itself a causal DAG theorem. |
| [Unlinked monotone regression](https://www.jmlr.org/papers/v22/20-689.html) | Monotonicity and a specified independent noise law restrict the regression relationship. | A restricted univariate regression target; count noise is not automatically the required structural noise law. |
| [Statistical matching by fractional imputation](https://www150.statcan.gc.ca/n1/pub/12-001-x/2016001/article/14539-eng.pdf) | An identified parametric model, including instrumental-variable constructions. | An exclusion restriction must be justified; numerical convergence is not a test of identification. |
| [Auxiliary-variable nonlinear ICA](https://proceedings.mlr.press/v89/hyvarinen19a.html) and [iVAE](https://proceedings.mlr.press/v108/khemakhem20a.html) | Informative observed auxiliaries and restricted modulation of independent latent sources. | Ordinary labels or estimated clusters do not automatically meet the theorem conditions; representation equivalences remain. |
| [Multiview nonlinear ICA](https://proceedings.mlr.press/v115/gresele20a.html) and [biomedical causal representation learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2ec91b31ac4aa578d99309d4f00c81ad-Abstract-Conference.html) | Joint views plus measurement and causal restrictions. | Their joint observations cannot simply be replaced by inferred pairings. |
| [Propensity-score alignment](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ff56c8dc4c61017eab9771481b7fda0d-Abstract-Conference.html) | Observed perturbation labels constrain alignment. | Different information and an alignment target; equal scores need not resolve every state. |
| [Temporal OT plus GRN inference](https://doi.org/10.1007/978-3-032-25311-8_21) and [CD-NOD](https://www.jmlr.org/papers/v21/19-232.html) | Known temporal/heterogeneous settings and jointly measured variables, with model assumptions. | Neither establishes arbitrary static cross-assay recovery from disjoint feature blocks. |

Two additional distinctions prevent misleading use of theory. Locatello et al.'s impossibility result concerns unsupervised latent disentanglement, so it is not a substitute for our measured-feature graph proof. A recent Schrödinger-bridge matching theorem characterizes a correct coupling under conditions involving unavailable cross-block dependence or its tuning; existence of a suitable cost does not show that separate marginals identify that cost. [Locatello et al., 2019](https://proceedings.mlr.press/v97/locatello19a.html), [Koo et al., 2026 preprint](https://arxiv.org/abs/2606.22770).

## Paired and bridge methods fall outside the strict-disjoint proof

WNN, BABEL, scMVP, scMM and Mowgli use paired measurements in their native or published training settings. Seurat v5 uses a paired bridge; MultiVI, totalVI and Cobolt can exploit paired observations alongside incomplete data. Applying a model to an unpaired query does not erase its paired training information. The proof above cannot rule out joint-distribution estimation in those settings. Their embedding or prediction objectives still do not automatically establish causal directions. [WNN](https://doi.org/10.1016/j.cell.2021.04.048), [BABEL](https://doi.org/10.1073/pnas.2023070118), [Seurat v5](https://doi.org/10.1038/s41587-023-01767-y), [MultiVI](https://doi.org/10.1038/s41592-023-01909-9)

MOFA/MOFA+ use identified sample indices and can handle missing views. That differs from knowing only completely disconnected assay blocks. JAMIE can receive genuine correspondences or infer them from geometry; the former adds information, the latter does not. scMoMaT can use genuinely shared cell measurements, which supply joint information. Shared feature identities and annotation-derived pseudo-modalities alone do not supply measured cell pairs. [MOFA+](https://doi.org/10.1186/s13059-020-02015-1), [JAMIE](https://doi.org/10.1038/s42256-023-00663-z), [scMoMaT](https://doi.org/10.1038/s41467-023-36066-2)

There is also a sampling distinction: shuffling two assay lists measured on the same finite cells is not independent sampling of different cells. Whole-sample summaries can retain dependence after identifiers are hidden. MultiVI's artificial-unpairing experiments additionally retain paired observations. Such designs are valuable benchmarks, but they do not instantiate the strict product-law experiment used in our impossibility proof. [MultiVI](https://doi.org/10.1038/s41592-023-01909-9), [unlinked-regression discussion](https://www.jmlr.org/papers/volume22/20-689/20-689.pdf).

## What benchmark success tells us

The seven benchmark records examine batch removal, biological conservation, neighborhood or label accuracy, matching, imputation, feature selection and computational performance. Those endpoints answer useful questions. A good score does not certify equality of the full biological joint, preservation of every cross-assay conditional independence, or recovery of direct effects. The inspected endpoints were not a general certification of original-feature causal-graph recovery. [scIB](https://doi.org/10.1038/s41592-021-01336-8), [Xiao et al.](https://doi.org/10.1093/bib/bbae095), [Hu et al.](https://doi.org/10.1038/s41592-024-02429-w), [Fu et al.](https://doi.org/10.1038/s41592-025-02737-9)

The multitask Registered Report separates seven tasks rather than treating integration quality as one quantity. SCMBench assesses domain-specific and foundation methods using integration and biological-consistency endpoints; using a GRN-based simulation generator does not mean the reported score measures exact GRN recovery. [Liu et al., 2025](https://doi.org/10.1038/s41592-025-02856-3), [SCMBench, 2026](https://doi.org/10.1038/s41467-026-72570-x)

The component benchmark is especially relevant to our histone setting because it includes paired RNA–histone references. Its separate RNA–ATAC motivating example shows how annotation errors affect pseudobulk peak–gene correlations; neighborhood preservation can coexist with weak gene-level agreement. This motivates checking downstream targets separately. It is empirical evidence of distortion, not a population impossibility proof or interventional ground truth. [Naqing, Yuan and Duren, 2026](https://doi.org/10.1186/s13059-026-04071-5)

## What this means for our existing data and algorithms

For scGLUE, scMRDR and scConfluence as currently used, we have not established assumptions that identify original-feature cross-assay direct structure. Good alignment, guidance agreement or stable fitted arrows cannot close that gap. Conversely, the literature does not justify declaring all future methods on the same assays impossible.

The useful theoretical next step is an **assumption audit using the variables already permitted**: specify whether the target is a shared latent graph or a peak–gene graph; write the count measurement model; and ask whether a positive theorem remains valid for it. Sturma's model and UnpairReg's regression restrictions are concrete places to start, with very different guarantees. Fitting either successfully would not itself verify the assumptions. No additional modality, paired training set, temporal label or perturbation is assumed available by this recommendation.

For a proposed new model class, there are two honest routes: prove that all compatible models agree on the target, or characterize the remaining set of compatible targets. The existing broad-class counterexample is an essential negative control for that argument. It cannot be used to reject a restricted class without first checking membership.

## Evidence limits and review records

This is a bounded scoping review, not an exhaustive systematic review or a performance meta-analysis. We did not reproduce the selected studies, rederive every published proof, or audit every implementation. The five limited records are scJoint, the MMD-MA scaling study, Seurat v5, Hu's benchmark and Fu's benchmark. Funding and conflicts were not extracted systematically; missing entries mean unassessed, not absent. Preprint/final-publication differences and source-access failures are retained in the ledgers.

The mathematical claim rests on the stated observation contract and model class. Its applicability to a biological dataset requires checking those premises. The positive identification assumptions have not been validated for our sparse count data. No new worlds, biological datasets or integration fits were created for this review.

- [Paper matrix](PAPER_MATRIX.md), [full extraction](PAPER_EVIDENCE.json), [tab-separated extraction](PAPER_EVIDENCE.tsv).
- [Search log](SEARCH_LOG.md), [discovery catalog](DISCOVERY_CATALOG.tsv), [screening accounting](SCREENING_SUMMARY.json), [citation verification](citation_verification.json).
- Independent component reviews: [unpaired methods](agents/unpaired_review.md), [paired/bridge methods](agents/bridge_methods.md), [identification theory](agents/theory_scope_review.md), and [initial cross-review](agents/unpaired_cross_review.md).
- Final synthesis reviews: [unpaired-method reviewer](agents/final_unpaired_cross_review.md), [bridge-method reviewer](agents/final_bridge_cross_review.md), [theory reviewer](agents/final_theory_cross_review.md). The lead incorporated their five wording/target clarifications; [reconciliation](REVIEW_RECONCILIATION.md) records the final artifact hashes.
- Existing proofs and exact symbolic checks: [research record](../integration_identifiability_2026_11_01/README.md).
