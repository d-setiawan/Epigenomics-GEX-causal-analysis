# Jianle scMRDR Paper Notes and Comparison to Our Implementation

## Scope

This note summarizes what the Jianle paper in [IntegrationJianLe.pdf](../../docs/papers/IntegrationJianLe.pdf) is doing conceptually and mathematically, and then compares that method to the current repo-native Jianle track under `/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/`.

The paper presents a method called single-cell Multi-omics Regularized Disentangled Representations, or scMRDR. The core problem is unpaired single-cell multi-omics integration: given separate collections of cells measured in different modalities, learn a shared latent space that aligns modalities while preserving biological structure.

## What Problem the Paper Is Solving

The paper focuses on a hard regime:

1. The data are unpaired, so the same biological cell is not observed across modalities.
2. The modalities may have different feature sets and very different coverage.
3. The datasets can be large.
4. The method should extend beyond two modalities.

The paper argues that many earlier approaches either:

1. Need paired data.
2. Need prior feature or sample correspondences.
3. Rely on a global coupling or transport matrix, which scales poorly.

The proposed solution is to avoid explicit pair matching and instead learn a disentangled latent representation in which:

1. One part is shared across modalities.
2. One part is modality-specific.

The shared part is then regularized to be:

1. Aligned across modalities.
2. Structure-preserving within each modality.

## Core Representation in the Paper

For a cell from modality $m$, the latent representation is split into:

1. A shared latent variable $z_u$.
2. A modality-specific latent variable $z_s^{(m)}$.

The full latent vector is:

$$
z = \left(z_u, z_s^{(m)}\right).
$$

The modeling assumption is that modality-invariant biological state should live mainly in $z_u$, while modality-specific effects should live in $z_s^{(m)}$.

The paper factorizes the latent prior as:

$$
p(z \mid m) = p(z_u) p(z_s \mid m).
$$

With covariates $c$ such as batch, the full generative factorization is:

$$
p(x, z, m, c) = p(x \mid z, c)\, p(z_u)\, p(z_s \mid m)\, p(m)\, p(c).
$$

This means the paper is explicitly not treating all nuisance effects as part of the modality label alone. Batch and related effects are meant to be handled as covariates.

## Generative Model in the Paper

The paper uses a $\beta$-VAE style framework. The encoder maps observed data $x$ to approximate posteriors over $z_u$ and $z_s^{(m)}$, and the decoder reconstructs $x$ from the latent code and covariates.

The approximate posteriors are diagonal Gaussians:

$$
q(z_u \mid x), \qquad q(z_s \mid x, m).
$$

The priors are:

$$
p(z_u) = \mathcal{N}(0, I),
$$

and

$$
p(z_s \mid m) = \mathcal{N}(\mu_m, \sigma_m^2 I).
$$

So the paper allows each modality to have its own prior mean and variance for the modality-specific latent space.

## Reconstruction Distribution

The paper assumes count-like, sparse, overdispersed single-cell measurements, and uses a zero-inflated negative binomial, or ZINB, decoder.

For feature $g$ in cell $n$:

$$
x_{ng} \sim \operatorname{ZINB}(l_n \rho_{ng}, r_g, \pi_{ng}),
$$

where:

1. $l_n$ is the library size for cell $n$.
2. $\rho_{ng}$ is the mean proportion for feature $g$ in cell $n$.
3. $r_g$ is the feature-specific dispersion.
4. $\pi_{ng}$ is the dropout probability.

The decoder predicts these parameters from the latent representation and covariates.

The paper also notes that Gaussian decoding could be used for normalized non-count data, but the main method is framed around ZINB.

## Base $\beta$-VAE Objective

The paper starts from the VAE evidence lower bound and adds a $\beta$ multiplier to the KL term:

$$
L_{\beta\text{-VAE}} = L_{\text{recon}} + \beta L_{\text{KL}}.
$$

More explicitly:

$$
L_{\beta\text{-VAE}}
=
- \mathbb{E}_{z \sim q(z \mid x, m)} \log p(x \mid z, c)
+ \beta \left[
\operatorname{KL}(q(z_u \mid x)\,\|\,p(z_u))
+ \operatorname{KL}(q(z_s \mid x, m)\,\|\,p(z_s \mid m))
\right].
$$

The role of $\beta > 1$ is to encourage stronger disentanglement between shared and modality-specific factors.

Conceptually, the paper wants the full latent space $z$ to reconstruct well, but also wants $z_u$ and $z_s$ to become separated enough that the shared space can be used for integration.

## Adversarial Alignment Term

Even if the latent space is disentangled, the paper argues that $z_u$ from different modalities may still not be well aligned. To fix this, it introduces a discriminator $D$ that tries to predict which modality a shared latent code came from.

The discriminator minimizes:

$$
L_{\text{discriminator}}
=
- \sum_m \log D_m(z_u^{(m)}).
$$

The encoder is trained adversarially so that $z_u$ becomes harder to classify by modality. Equivalently, the encoder minimizes an alignment objective that pushes the shared latent distributions toward modality invariance.

In practical terms:

1. The discriminator learns modality separation in $z_u$.
2. The encoder learns to remove modality-identifying signal from $z_u$.

This is what turns the shared latent space into an integration space rather than merely a disentangled one.

## Isometric Structure-Preservation Term

The paper also argues that if we push too hard on alignment, we may wash out biological structure. To avoid that, it introduces an isometric preservation loss.

The idea is:

1. The full latent representation $z = (z_u, z_s)$ should already preserve within-modality structure reasonably well because it is optimized for reconstruction.
2. The shared space $z_u$ should preserve that same geometry as much as possible.

For each modality, the paper compares pairwise Euclidean distances in the shared latent space to pairwise distances in the full latent space:

$$
L_{\text{preserve}}
=
\sum_m \sum_{i,j \in X^{(m)}}
\left(
\lVert \mu_{z_u}(x_i) - \mu_{z_u}(x_j) \rVert_2
- \lVert \mu_z(x_i) - \mu_z(x_j) \rVert_2
\right)^2.
$$

This encourages the geometry of the shared latent space to remain faithful to the geometry of the full latent representation.

Conceptually, this is the paper's main bio-conservation mechanism.

## Final Paper Objective

Putting the terms together, the paper optimizes:

$$
L_{\text{total}}
=
L_{\text{recon}}
+ \beta L_{\text{KL}}
+ \lambda L_{\text{alignment}}
+ \gamma L_{\text{preserve}}.
$$

Training alternates between:

1. Updating the discriminator.
2. Updating the VAE with the full objective above.

So the training loop is adversarial, but still grounded in a generative reconstruction objective.

## Masked Reconstruction for Missing Features

This is one of the most important details in the paper.

Different modalities may not measure the same feature set. The paper explicitly warns against two bad options:

1. Restricting to only the features shared across all modalities, which throws away too much information.
2. Filling unmeasured features with zeros, which distorts the distribution.

Instead, the paper defines a binary availability mask $b_{ng}$ so that only observed features contribute to the reconstruction loss:

$$
L_{\text{recon}}
=
- \frac{1}{N}
\sum_{n=1}^N
\left(
\frac{G}{\sum_{g=1}^G b_{ng}}
\sum_{g=1}^G
b_{ng}
\log P_{\text{ZINB}}(x_n; l_n \rho_{ng}, r_g, \pi_{ng})
\right).
$$

There are two conceptual points here:

1. Gradients should not flow through features that were never measured for that modality.
2. The per-cell loss is rescaled by the proportion of observed features so losses remain comparable across modalities with different feature coverage.

This masked-loss design is central to how the paper supports integration of more than two omics, especially when one modality has far fewer features than the others.

## Data Representation Assumptions in the Paper

The paper is written around a feature-aligned cross-modality representation, ideally at the gene level.

The examples in the appendix are especially informative:

1. For scATAC-seq in the two-omics setting, peak counts are converted into gene activity scores using EpiScanpy.
2. For two-modality integration, they select highly variable genes measured in both RNA and ATAC-derived gene activity and do not need masked loss.
3. For tri-modal integration, they use highly variable genes from RNA and ATAC plus all nonzero protein features, and masked loss becomes necessary.

So although the paper is flexible, its concrete experimental setup is strongly feature-aligned across modalities. It is not working with unrelated mark-specific bin identifiers as the core feature universe.

## Why the Paper Scales

The paper claims scalability for two reasons:

1. It avoids building a global cell-to-cell coupling matrix across modalities.
2. It uses one generative model that can naturally absorb multiple modalities without redesigning a pairwise alignment step for every modality pair.

In other words, the method scales by making integration a latent-variable learning problem rather than a global correspondence problem.

## What Our Repo Implementation Is Doing

The repo contains a separate Jianle track described as a practical, scMRDR-inspired implementation in [Integration/README.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/Integration/README.md) and [Integration/scripts/jianle/README.md](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/README.md).

The workflow is:

1. Build a local integration workspace and manifest.
2. Optionally build a shared gene feature universe from histone peak matrices.
3. Preprocess one RNA modality and all histone-mark chromatin modalities.
4. Train a joint model.
5. Validate the learned shared embeddings.

The main code paths are:

1. `Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/preprocess_joint_jianle.py`
2. `Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/preprocess_pilot_jianle.py`
3. `Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/train_joint_jianle.py`
4. `Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/validate_joint_jianle.py`
5. `Epigenomics-GEX-causal-analysis/Integration/scripts/jianle/build_joint_gene_feature_universe.py`

### Inputs used by our implementation

The workspace builder constructs a shared manifest of chromatin inputs from the `Data/` directory. The manifest currently points to per-mark bin matrices such as:

1. `H3K27ac`
2. `H3K27me3`
3. `H3K4me1`
4. `H3K4me2`
5. `H3K4me3`
6. `H3K9me3`

The default joint Jianle path still uses bin matrices rather than gene-level activity matrices.

This means the current implementation is integrating:

1. RNA counts.
2. Multiple mark-specific chromatin bin-count modalities.

That is a different data design from the paper's main gene-aligned examples.

However, the repo now also includes a bootstrap script that can aggregate histone peak matrices onto the RNA gene universe. That narrows one major gap with the paper, but it is still a preprocessing side path rather than the default joint Jianle training path.

### Preprocessing in our implementation

RNA preprocessing:

1. Filters cells and genes.
2. Stores raw counts in `layers["counts"]`.
3. Selects highly variable genes.
4. Normalizes, log-transforms, and scales the matrix.
5. Computes PCA for inspection.

Chromatin preprocessing:

1. Reads each mark-specific bin matrix.
2. Stores raw counts in `layers["counts"]`.
3. Selects top chromatin features by total counts.
4. Computes TF-IDF and LSI.
5. Writes one `h5ad` per mark.

However, the actual training code later uses the `counts` layer rather than the precomputed PCA or LSI coordinates.

### Model structure in our implementation

The implemented model still follows the same high-level latent decomposition:

$$
z = \left(z_u, z_s^{(m)}\right).
$$

It also keeps modality-specific Gaussian priors for $z_s^{(m)}$ and a standard Gaussian prior for $z_u$.

The training objective also has the same broad form:

$$
L_{\text{total}}
=
L_{\text{recon}}
+ \beta L_{\text{KL}}
+ \lambda L_{\text{alignment}}
+ \gamma L_{\text{preserve}}.
$$

Concretely, the implementation contains:

1. A shared/specific encoder.
2. A ZINB decoder.
3. A modality discriminator on $z_u$.
4. An isometric preservation loss comparing distances in $z_u$ and $z = (z_u, z_s)$.

So the main idea of the Jianle paper is genuinely present.

## Detailed Differences Between the Paper and Our Implementation

This section is the most important for deciding how faithfully we are following the paper.

### 1. Shared feature universe vs per-modality feature spaces

This used to be the single biggest difference, and it is still the biggest difference in the default pipeline.

In the paper, the masked reconstruction loss is defined over a common feature universe, with a modality-specific availability mask:

$$
b_{ng} \in \{0,1\}.
$$

That setup only makes sense if all modalities are represented against one aligned feature index, even if some entries are unmeasured.

In the default joint Jianle implementation, each modality keeps its own feature space:

1. RNA has RNA genes.
2. Each chromatin mark has its own selected bin set.

Those chromatin features are even namespaced by mark, which makes the feature spaces intentionally separate. So our model is not reconstructing a single common feature universe with masks. It is reconstructing each modality in its own native feature space using separate output heads.

This changes the meaning of reconstruction substantially:

1. The paper reconstructs a single aligned feature space with missingness handled by masking.
2. Our code reconstructs different modality-specific feature spaces.

This is why the repo's own README says the full paper-faithful masked reconstruction over a single explicit cross-modality feature universe is not implemented yet.

There is now an important update: we have implemented a shared-gene preprocessing path that consumes the outputs of `build_joint_gene_feature_universe.py`. So the repo can now run an end-to-end Jianle workflow in which RNA and histone modalities share one gene index. However, this is still an explicit shared-gene mode rather than the default native bin-based path, so the paper-closer representation exists but is not yet the only operational path.

### 2. Masked reconstruction is not functionally active in the paper's sense

The training code does contain a masked ZINB loss function, and after the shared-gene update it can now receive real modality-specific availability masks through `var["feature_available"]`.

So operationally:

$$
b_{ng} \in \{0,1\}.
$$

In shared-gene mode, this now does reflect structural feature availability inside one aligned gene universe.

In effect:

1. The paper uses masking to handle missing cross-modality features.
2. Our shared-gene mode now does this in a real way.
3. The older native-feature pipeline still effectively uses all-ones masks because its modalities are already separated by design.

There used to be a second mismatch in the reconstruction scaling. The paper rescales the reconstruction term by the fraction of available features using a factor like

$$
\frac{G}{\sum_g b_{ng}},
$$

and the current shared-gene training path now mirrors that idea explicitly rather than merely averaging over observed features. So the important remaining gap is no longer the presence of masks themselves, but rather the exact biological meaning of feature availability and the broader training/evaluation protocol around the model.

### 3. Paper uses gene-aligned cross-modality inputs; our default pipeline still uses RNA plus mark-specific bins

In the paper's experiments, the multimodal comparison is largely gene-level:

1. RNA is gene-level.
2. ATAC is converted to gene activity scores.
3. Protein is added with masking where needed.

Our default implementation uses chromatin bin matrices as the integration input. That choice is reasonable for this repo because the project README explicitly recommends bin matrices for integration, but it is still a conceptual departure from the paper.

The consequences are important:

1. The paper's modalities are linked by biological feature semantics at the gene level.
2. Our chromatin modalities are linked mostly through shared cell-state geometry, not through a shared feature definition.
3. Multiple histone marks are each treated as separate modalities.

So we are applying the scMRDR idea to a harder and somewhat different data design than the paper's benchmark setup.

The new shared-gene mode narrows this gap substantially, but it is still not identical to the paper's setup:

1. The paper converts ATAC to gene activity scores.
2. Our implementation aggregates histone peaks to gene-level histone scores.
3. That is biologically sensible, but it is not exactly the same object as ATAC gene activity.
4. Shared-gene mode is now supported end-to-end, but the default repo path is still the native bin-based representation unless that mode is selected explicitly.

### 4. The paper models covariates $c$ explicitly; our implementation now conditions the decoder on them, with a metadata caveat

The paper writes the decoder as conditioned on $z$ and covariates $c$, where $c$ can include batch.

Our preprocessing step joins per-cell metadata from the clean-cell tables, and the current training path now one-hot encodes requested covariates and concatenates them into the decoder input. In other words, the model is no longer fully covariate-free.

So batch handling differs:

1. In the paper, batch can be modeled directly via $c$.
2. In our code, the decoder can now condition on covariates such as `batch_id` and `donor_id`.
3. Those covariates are not yet fed into the encoder, so our implementation is still a decoder-conditioned variant rather than the most general possible conditioning scheme.

There is also a practical caveat in the current workspace:

1. chromatin modalities have clear `donor_id` metadata from HTO demultiplexing,
2. RNA does not currently have matched donor labels in the workspace,
3. so RNA is conditioned on a missing-donor category unless richer RNA metadata are added later.

So this change brings the implementation substantially closer to the paper, but the available covariates are still limited by the current dataset packaging.

### 5. Unified encoder-decoder in the paper vs shared backbone plus modality adapters/heads in our code

The paper emphasizes a unified encoder-decoder architecture treating different omics as samples from one flexible generative framework.

Our code is partly unified and partly modality-specific:

1. It has a shared trunk.
2. It has modality embeddings.
3. It has modality-specific input linear layers.
4. It has modality-specific output heads and per-modality dispersions.

This is a reasonable engineering choice, but it is not the cleanest interpretation of a single unified encoder-decoder over one common feature universe.

So the implementation is best viewed as:

1. A shared latent-model family.
2. With modality-specific adapters around a shared core.

rather than a strict paper-faithful unified feature-space generator.

### 6. The paper's examples are mostly RNA plus ATAC or RNA plus ATAC plus protein; our code handles RNA plus multiple histone-mark modalities

The paper demonstrates:

1. Two-omics integration.
2. Large-scale two-omics integration.
3. Triple-omics integration.

Our implementation is architected for:

1. One RNA modality.
2. Multiple histone-mark chromatin modalities, each treated as separate modalities in the discriminator and model.

This is a useful extension for the project, but it is not the same biological setup as the paper. The adversarial alignment problem is also harder because the discriminator now separates RNA from several chromatin marks rather than just two or three modalities defined in the paper.

### 7. Preprocessing artifacts like PCA and LSI are computed but not used as model input

The preprocessing scripts compute:

1. RNA PCA.
2. Chromatin LSI.

But the training code later reads the `counts` layer and internally performs its own normalization and log transform for the encoder input.

So PCA and LSI are currently useful mainly as preprocess diagnostics or saved artifacts, not as the actual representation the model consumes.

This is not wrong, but it means the practical workflow differs from what a reader might infer from the preprocess outputs.

There is now a second preprocessing path that produces shared gene-level histone matrices from peaks. That path is closer to the paper conceptually than the default bin-based path, and it is now wired into the main Jianle preprocess and train scripts when a shared-gene manifest is provided.

### 8. Training protocol differs from the paper appendix

The paper appendix reports details such as:

1. Batch size $128$.
2. Epochs $200$.
3. Shared latent dimension $20$.
4. Specific latent dimension $20$.
5. Adam optimizer.
6. Learning rate starting at $10^{-3}$.
7. Cosine annealing scheduler.
8. A validation split with early stopping.

Our defaults overlap partly:

1. Latent dimensions default to $20$ and $20$.
2. Adam and learning rate $10^{-3}$ are used.
3. Epoch count defaults to $200$.

But our code does not currently implement:

1. A validation split.
2. Early stopping.
3. Cosine annealing.

So our optimization protocol is simpler than the one described in the paper.

### 9. Loss weights are not matched to the paper's reported experimental settings

The paper's appendix reports different $\beta$, $\gamma$, and $\lambda$ settings depending on the experiment. For example, some datasets use stronger values than the defaults in our code.

Our default training arguments are:

$$
\beta = 2.0, \qquad \lambda = 1.0, \qquad \gamma = 1.0.
$$

These are sensible starting points, but they are not a direct reproduction of the paper's benchmark-specific settings.

So even when the loss terms match conceptually, the operating regime is not paper-matched by default.

### 10. Evaluation is much lighter than the paper's benchmark suite

The paper evaluates with a broad set of integration and biology-preservation metrics, including:

1. isolated label scores,
2. NMI,
3. ARI,
4. silhouette scores,
5. LISI variants,
6. PCR,
7. kBET,
8. graph connectivity.

Our validation now computes and exports a richer scGLUE-style sanity-check bundle, including:

1. UMAP,
2. Leiden clustering,
3. modality silhouette,
4. same-modality neighbor fraction,
5. cluster-by-modality tables,
6. cluster-by-modality fractions,
7. cluster-by-modality heatmaps,
8. modality facet UMAPs,
9. RNA annotation overlays when the RNA label tarball is available,
10. optional RNA-to-non-RNA harmonized label transfer in the learned joint space.

That is a much better integration-evaluation layer than the earlier minimal validator, and it makes the Jianle track easier to compare directly against the scGLUE track inside this repo.

However, it is still not paper-level benchmarking. We still do not reproduce the full evaluation suite reported in the paper, such as NMI, ARI, LISI-family metrics, kBET, PCR, graph connectivity, and related benchmark protocol details. So we still cannot claim parity with the paper's reported evaluation quality from the current validation script alone.

### 11. The structure-preservation loss is implemented as a minibatch approximation

The paper writes the preserve loss over within-modality pairs:

$$
\sum_{i,j \in X^{(m)}} (\cdots)^2,
$$

which reads like a modality-wide pairwise distance comparison.

Our implementation computes the pairwise distance matrices only for the current sampled minibatch from each modality. That is a practical approximation and is computationally sensible, but it is still a difference:

1. The paper's equation is expressed over all within-modality pairs.
2. Our code uses only the current minibatch pairs.

So our preserve loss is an unbiased practical approximation only insofar as the minibatches are representative enough; it is not the exact full-modality pairwise objective.

### 12. Our shared-gene builder now supports union mode, but its masking semantics are still a design choice

The shared-gene builder is now more flexible than the earlier RNA-anchored bootstrap. It can build:

1. an RNA-only universe,
2. a `union_linked` universe containing RNA genes plus genes linked in at least one selected histone mark,
3. or a broader GTF-derived universe.

That means the repo can now represent a larger union of features rather than only an RNA-anchored intersection-like set.

However, there is still an implementation choice hidden inside the feature-availability definition:

1. for RNA, a gene is available if it exists in the RNA feature matrix,
2. for a histone mark, a gene is available if at least one peak was linked to that gene under the chosen windowing rule.

That is reasonable for our project, but it is still a modeling choice rather than something uniquely prescribed by the paper. In other words, we now have a real union-plus-mask implementation, but the exact meaning of "available feature" for histone-derived gene scores remains assay- and preprocessing-dependent.

This is still conceptually different from the paper's most general setup, where one can keep a larger feature universe and then use:

$$
b_{ng}
$$

to decide which features contribute to reconstruction in each modality.

So the gap is now much smaller, but not fully closed.

## What Is Faithful in Our Implementation

Even with the differences above, several key parts are clearly faithful to the Jianle paper's main methodological idea:

1. The latent split into $z_u$ and $z_s^{(m)}$.
2. The use of a $\beta$-VAE objective.
3. A modality-specific prior for $z_s^{(m)}$.
4. A ZINB reconstruction model.
5. Adversarial alignment on the shared latent space.
6. Isometric preservation from the full latent space to the shared latent space.
7. Joint training across more than two modalities.

So this is not a superficial naming reuse. It is a real scMRDR-style implementation.

## Bottom Line

The current Jianle track should be described as:

1. A practical scMRDR-inspired implementation.
2. Adapted to this repo's RNA plus multi-histone bin-matrix setup.
3. Faithful to the main latent-factorization and regularization ideas.
4. Now able to run in a shared-gene mode with real modality-specific feature masks.
5. Now uses paper-closer masked reconstruction scaling and decoder conditioning on covariates.
6. Still not fully faithful to the paper's benchmark protocol and full covariate availability across modalities.

The most important remaining gaps relative to the paper are:

$$
\text{paper benchmark protocol and covariate coverage} \neq \text{current repo implementation}.
$$

If the goal is to capture the spirit of Jianle's method for this project, the current code is already doing that in a meaningful way. If the goal is strict reproduction of the paper, the main missing pieces are:

1. Paper-style training protocol and evaluation metrics.
2. A decision about whether the shared-gene path should replace the older native bin-based path as the main workflow.
3. Better covariate coverage for RNA, especially donor labels if they exist outside the current workspace.
4. Potential encoder-side conditioning if we want to go beyond decoder-only conditioning.

## Suggested Next Steps

The most useful next steps are now fairly clear.

### 1. Validate and, if it looks good, promote shared-gene mode to the default Jianle path

That main representation step is now implemented. The next question is whether we want to keep two parallel Jianle paths indefinitely:

1. the older native bin-based path,
2. and the newer shared-gene path.

If the shared-gene path behaves well empirically, it would make sense to promote it to the default because it is much closer to the paper conceptually.

### 2. Validate the new masks and covariates empirically

Now that every modality can be represented on one gene index, we can decide whether we want:

1. a strict shared intersection of genes, where masking may be unnecessary, or
2. a larger union of genes, where masking becomes important.

The real masks are now present in shared-gene mode, and the reconstruction term now uses the paper-style observed-feature rescaling. So the next job is empirical rather than purely architectural:

$$
b_{ng} = 0
$$

for structurally unavailable features, not just for padded tensor positions.

And on the covariate side, we should verify that decoder conditioning on `batch_id` and `donor_id` improves reconstruction or integration quality rather than simply giving the decoder an easy shortcut.

### 3. Decide what gene-level histone representation we want to treat as the default

We now have enough infrastructure to support different biologically sensible definitions, but we should choose one explicitly.

The main options are:

1. promoter-only aggregation, which stays closer to sparse count-like inputs,
2. promoter plus distal linked peaks, which is biologically richer,
3. mark-specific rules, such as narrower promoter windows for H3K4me3 and broader windows for H3K27me3 or H3K9me3.

This choice matters because it changes whether the downstream decoder is still well modeled as ZINB-like count data or whether a continuous decoder would become more appropriate.

### 4. Improve covariate coverage and optionally extend conditioning

The decoder is now explicitly conditioned on covariates, which closes one important methodological gap with the paper. The next question is how far we want to take that conditioning.

The main open options are:

1. improve the RNA-side metadata so donor conditioning is not missing there,
2. add more technical covariates if they are available,
3. optionally feed covariates into the encoder as well if we want a stronger version of

$$
p(x \mid z, c).
$$

### 5. Bring training and evaluation closer to the paper

After the representation issue is fixed, the next layer is protocol parity:

1. add a validation split,
2. add early stopping,
3. add cosine annealing,
4. benchmark with stronger paper-style integration metrics beyond the current scGLUE-style repo validator.

At that point we could make a much stronger claim about similarity to the paper rather than similarity in spirit only.

### Recommended order

If the goal is maximum progress per unit effort, the best order is:

1. compare the shared-gene path against the native bin-based path on the same datasets,
2. if it is stable, make shared-gene mode the default Jianle path,
3. then compare covariate-conditioned and covariate-free runs quantitatively,
4. then add benchmark-style evaluation metrics and, if useful, extend conditioning beyond the decoder.

That order keeps the largest conceptual fix in place first and focuses the next effort on whether the new masking and covariate logic actually improves the model in practice.
