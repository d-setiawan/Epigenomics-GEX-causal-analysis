# Objective audit: scGLUE and scConfluence under unpaired sampling

Checked 2026-11-01. This audit connects the [unpaired-data proof](../../theory/07_unpaired_integration_nonidentifiability.md) to the actual integration objectives. It does not use the pilot's benchmark outcomes as evidence for impossibility. No model was fitted or changed for this audit.

**Conclusion:** under independent, disjoint assay sampling, the two countermodels have the same complete input law and the same guidance. Consequently, scGLUE and scConfluence have the same population objective for every parameter setting in both models. Their training and downstream discovery outputs have the same distribution as well. Either method can select an informative alignment under substantive assumptions; neither can guarantee the correct cross-assay causal structure throughout a class containing these two models.

## Inputs that must be held fixed

Let \(D_a\) contain the complete observed cell vectors for assay \(a\), including any permitted background features, depth and metadata. Let \(G\) denote fixed feature annotations/guidance. Sampling is independent between assays. All preprocessing, feature selection, costs and balancing weights are computed from these inputs, with the same rules in both worlds. Randomness in training is independent of the unknown world.

For independent standard normal \(S,U,V\), compare \(\rho=0\) and \(\rho=1/2\) in

$$
X=S+U,\qquad Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}\,V.
$$

Both have \(X\mid S=s\sim\mathcal N(s,1)\) and \(Y\mid S=s\sim\mathcal N(s,1)\). Thus even oracle state supplied separately with each assay does not distinguish them. Yet \(\operatorname{Cov}(X,Y\mid S)=\rho\), and the direct \(X\to Y\) coefficient is \(\rho\). The main proof explains the extension to identical full assay-block laws using fixed local measurement/background kernels. Equality of only two selected scalar marginals would not be sufficient for this audit: **every input available to integration must have the same joint observed-data law**.

## scGLUE: what each loss observes

The public method combines modality-specific variational reconstruction, a feature-guidance graph and adversarial modality alignment. The current official source documents the same loss decomposition and separately exposes paired and supervised modes. See the [GLUE paper](https://doi.org/10.1038/s41587-022-01284-4) and [official source documentation, `SCGLUETrainer.compute_losses`](https://scglue.readthedocs.io/en/latest/_modules/scglue/models/scglue.html#SCGLUETrainer.compute_losses).

The executable local audit is of **installed scGLUE 0.2.3**, not the current documentation's 0.4.1. Its [ordinary unpaired trainer](/home/dgsetiawan/anaconda3/envs/scrna_r/lib/python3.11/site-packages/scglue/models/scglue.py:603) implements:

| Term | Information used by the audited code |
|---|---|
| Data reconstruction | Encoded cell features and reconstructed counts from that same assay; feature embeddings learned from the guidance graph. |
| Cell KL | Each assay's encoded distribution and the latent prior. |
| Graph reconstruction and graph KL | Guidance vertices, signed/weighted edges, negative sampling and the feature prior. |
| Discriminator | Concatenated encoded cells, their assay labels and balancing weights; optional batch covariates. |
| Generator alignment | The negative of the discriminator loss, encouraging assay overlap in the embedding. |

With implementation normalizations absorbed into the terms, the unpaired generator objective has the schematic form

$$
\mathcal J_{\mathrm{GLUE}}(\theta,\psi)
=\lambda_D\sum_a w_a R_a(\theta;P_a,G)
+\lambda_G M R_G(\theta;G)
-\lambda_A C(\theta,\psi;P_1,\ldots,P_M,G).
$$

Here \(R_a\) includes reconstruction and cell KL, \(R_G\) includes graph reconstruction and KL, and the discriminator minimizes \(C\) while the generator opposes it. \(P_a\) is the complete observed assay law. The actual code averages graph-positive/negative losses separately and normalizes the KL terms; none of those operations observes a missing biological pairing. This is an information-dependence formula, not a substitute for those exact normalizations.

Our [adapter](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/pipelines/simulation/integration_recoverability/scglue_adapter.py:275) configures NB counts with `use_batch=None`, `use_cell_type=None`, `use_uid=None` and an annotation-derived graph. Its numerical `lam_sup` setting adds no labeled information when the classifier is absent. Native pretraining, learned balancing weights and fine-tuning remain functions of the same available inputs. The package's paired trainer is a different information setting and is not the local run audited here.

## scConfluence: what its coupling observes

The paper uses stochastic autoencoders without the VAE prior-KL term, a partial transport plan from converted common features, inverse-OT alignment, and an unbalanced Sinkhorn penalty on latent distributions. Its equations 7–9 use squared latent distance and sum over modality pairs. These statements concern the published objective. See [Samaran, Peyré and Cantini (2024), Methods](https://www.nature.com/articles/s41467-024-51382-x).

The local source is pinned to [upstream commit `14d81e2ca679e161ec5c6806103a66c14fcdcd9a`](https://github.com/cantinilab/scconfluence/tree/14d81e2ca679e161ec5c6806103a66c14fcdcd9a). Its [autoencoder loss](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/pipelines/empirical/integration/methods/scconfluence/upstream/scconfluence/unimodal.py:280) returns reconstruction only, despite using a stochastic `VariationalEncoder` internally. The [pair losses](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/pipelines/empirical/integration/methods/scconfluence/upstream/scconfluence/model.py:321) do the following:

1. Compute a partial OT plan \(\pi_{ab}\) from uniform minibatch weights and the supplied common-feature cost matrix, transporting fixed mass \(m\).
2. Penalize \(\sum_{ij}\pi_{ab,ij}\,d(z_{ai},z_{bj})\).
3. Penalize latent-distribution discrepancy with `SamplesLoss(loss="sinkhorn", p=2, ...)`.

At the population level the corresponding information-dependence form is

$$
\mathcal J_{\mathrm{Conf}}(\theta)
=\sum_a\lambda_a R_a(\theta;P_a)
+\sum_{a<b}\left[
\lambda_I\int d(E_a(o),E_b(o'))\,\mathrm d\pi_{ab}(o,o')
+\lambda_S\,\mathcal S(E_{a\#}P_a,E_{b\#}P_b)
\right].
$$

The partial plan \(\pi_{ab}\) has mass \(m\); the integral does not renormalize it to unit mass. For stochastic encoders, the integrand also averages encoder randomness and \(E_{a\#}P_a\) means the resulting encoded law. The selected \(\pi_{ab}\) depends on \(P_a,P_b,G\) and the declared transport rule. It is **not an independently measured biological joint law**. Nonunique transport plans do not change the argument: a fixed or randomized tie rule receives the same inputs in both worlds. Minibatch expected objectives can differ from this idealized population-OT functional, but their equality between countermodels follows directly from equality of the full finite-data law.

Our [cost builder](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/pipelines/empirical/integration/methods/scconfluence/pilot_adapter.py:169) maps permitted feature annotations to common gene summaries, applies training-only normalization/scaling, then computes cross-assay correlation distances. A distance between two observed cells is computable without knowing whether their residuals belong together biologically. The adapter uses full retained normalized-log inputs, NB count reconstruction and frozen independent-cohort encoding.

### Published method versus the local pilot

The pinned upstream IOT implementation uses **unsquared** Euclidean distance; the paper specifies squared distance. The local patch retains the upstream distance. The pinned pair loop also overwrites accumulated terms and self-adds the final eligible pair. Our [reviewed patch](/home/dgsetiawan/MachineLearning/CLeaR/epigenomics/Epigenomics-GEX-causal-analysis/pipelines/empirical/integration/methods/scconfluence/loss_patch.py:10) instead sums every eligible pair once and skips empty-assay losses; native pair functions remain unchanged.

The paper's three-modality Sinkhorn coefficient is 0.03 and its original evaluation encodes the supplied training/validation dataset. Its high-dimensional experiments use PCA inputs and modality-specific reconstruction losses. Our count-input pilot uses 0.1, NB for all three assays and independent-cohort evaluation. It is an adaptation, not a reproduction of the published experiments. The [paper's Methods](https://www.nature.com/articles/s41467-024-51382-x) and local source establish these distinctions. The information argument applies separately to the published, native and patched objectives; it does not depend on which performs better.

## Why equality covers the entire modern objective

Write \(\mathcal O_\rho=(P_{1,\rho},\ldots,P_{M,\rho},G)\). For the constructed full-block models,

$$
\mathcal O_0=\mathcal O_{1/2}.
$$

Every term audited above is a functional of \(\mathcal O_\rho\) and declared parameters. Therefore, **for every admissible parameter setting**, not just at an optimum,

$$
\mathcal J^{(0)}_{\mathrm{GLUE}}(\theta,\psi)
=\mathcal J^{(1/2)}_{\mathrm{GLUE}}(\theta,\psi),\qquad
\mathcal J^{(0)}_{\mathrm{Conf}}(\theta)
=\mathcal J^{(1/2)}_{\mathrm{Conf}}(\theta).
$$

This remains true for data-dependent weights or preprocessing because their input laws coincide. Where minimizers, saddle points or gradients exist, the corresponding sets or values coincide too. No uniqueness, convergence or model-capacity assumption is needed. At finite sample sizes, the random objective functions and training outputs have the same law. Adaptive tuning and downstream conditional-independence tests using only these outputs cannot distinguish the worlds either.

This does not say the methods necessarily produce independent assay embeddings, or that OT always chooses the conditional-product coupling. They can produce dependent, biologically useful completions. The limitation is that the selected completion cannot be validated as the unique true residual coupling from these inputs. A unique optimizer is insufficient: it can reflect a unique modeling preference even when multiple biological models fit all observed data equally well.

Actual paired observations, distinguishing guidance, or a scientifically justified structural restriction excluding one countermodel changes the assumptions. This audit establishes failure of a uniform causal-recovery guarantee over the stated class, not failure in every biological problem. The proof also does not rely on PCA, imperfect integration, sparse counts or the local pilot's settings.

## Audit provenance

The [machine-readable receipt](modern_scglue_scconfluence_sources.json) records exact local source hashes and accessed primary URLs. Current online scGLUE source was checked only for the same decomposition; it is not substituted for the version used by the existing pilot. The GLUE PMC page returned a browser check, so detailed objective claims are grounded in official source documentation and the installed code. All scConfluence paper/source differences above are reported explicitly.
