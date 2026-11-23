# scMRDR: source audit and limits of causal recovery

_Independent bounded review, 2026-11-01. No training, new data, or benchmark changes._

The paper-consistent scMRDR implementation is covered by the strict-disjoint
nonidentifiability argument in [the main proof](../../theory/07_unpaired_integration_nonidentifiability.md).
Its active objectives can use rich within-assay distributions and feature
correspondences, but they do not observe the missing same-cell residual
coupling. This rules out a general causal-recovery guarantee over the model
class in that proof. It does not rule out successful state alignment or
recovery under additional, valid identifying restrictions.

## Exact method and evidence boundary

**scMRDR means single-cell Multi-omics Regularized Disentangled
Representations.** The primary publication is Sun et al., *scMRDR: A Scalable
and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration*,
NeurIPS 2025; the inspected revision is arXiv 2510.24987v2, dated 17 January
2026. Its stated targets are integration, biological conservation and batch
correction. The paper describes a disentangled VAE, adversarial alignment,
isometric regularization and masked reconstruction. It explicitly discusses
nonunique disentangled representations; it does not give a theorem identifying
the original cross-assay causal graph. [Primary paper, Sections 2–4](https://arxiv.org/html/2510.24987v2)

The remaining implementation findings below come from the local source and
the authors' pinned code, independently compared during this review. The
audited package is
`pipelines/empirical/integration/methods/scmrdr/paper_faithful/`, pinned to
commit `7280e64ae19f7391c6f5adf00d9d353f79e96603`. It is distinct from
`train_joint_jianle.py`, the older local approximation. Findings about one
must not be attributed to the other. No similarly named scMDR model is used.
[Package audit](../../../../pipelines/empirical/integration/methods/scmrdr/paper_faithful/README.md)

## Inputs and active computations

Line references below use the local pinned files, whose hashes appear at the
end of this review.

| Component | Verified behavior | Source location |
| --- | --- | --- |
| Features | Every assay supplies raw counts on one gene-aligned feature axis and an assay-specific availability mask. Zero observed counts remain observations; unmeasured features are masked. | `adapter.py` 211–250; `upstream/loss.py` 27–53 |
| Shared encoder | One dense network used across assays, with Linear, BatchNorm, LeakyReLU and Dropout layers; Gaussian mean and log-variance heads. Count inputs receive `log1p`. | `upstream/model.py` 49–108, 357–360, 393–405 |
| Specific encoder | A second unified dense network receives the counts plus modality one-hot. This is not a separate encoder for every assay. | `upstream/model.py` 357–360, 403–405 |
| Priors | Shared standard Gaussian; specific Gaussian with modality-dependent scalar mean and log-variance broadcast across dimensions. | `upstream/model.py` 371–374, 398–405; `upstream/loss.py` 74–109 |
| Decoder | A unified dense decoder receives concatenated shared and specific latents and optional covariates. It produces softmax proportions, dispersion and zero-inflation probabilities. | `upstream/model.py` 164–231, 415–418 |
| Reconstruction | Masked negative ZINB log likelihood, with mean equal to decoded proportion times that row's observed count sum. Loss sums features, scales by total/available features, and averages cells. | `upstream/model.py` 425–432, 531–550; `upstream/loss.py` 27–53 |
| KL | Sum of specific-to-specific-prior and shared-to-standard-prior KL terms; latent dimensions summed, then cells averaged. | `upstream/model.py` 439–440; `upstream/loss.py` 74–109 |
| Adversarial alignment | Discriminator minimizes modality classification cross-entropy; the VAE minimizes its negative, encouraging modality-confused shared embeddings. | `upstream/model.py` 451–460, 467–487 |
| Geometry | Within each modality, squared discrepancies between pairwise distances of full posterior means and shared posterior means. Both representations receive gradients. | `upstream/model.py` 442; `upstream/loss.py` 116–143 |
| Export | Frozen deterministic shared/specific posterior means for each independently encoded row. | `adapter.py` 390–417 |

There is **no graph encoder, graph neural network, regulatory-graph
reconstruction, cycle-consistency term, or true cross-assay paired-cell
reconstruction term** in this active architecture. `kneighbors_graph` is
imported but unused in the loss module; several alternative geometry/MMD
expressions are commented out. They are not active objectives. Geometry here
means distances among cells, not conditional independences or causal edges
among measured features. [Authors' model source](https://github.com/sjl-sjtu/scMRDR/blob/7280e64ae19f7391c6f5adf00d9d353f79e96603/src/scMRDR/model.py),
[authors' loss source](https://github.com/sjl-sjtu/scMRDR/blob/7280e64ae19f7391c6f5adf00d9d353f79e96603/src/scMRDR/loss.py)

For a minibatch of size $B$, let $u_i$ and $v_i$ be shared and specific
posterior means, and let $m_i$ denote modality. The implemented VAE objective
can be written as

$$
\mathcal L_{\rm VAE}
=\frac1B\sum_i R_i
+\frac{\beta}{B}\sum_i K_i
-\frac{\lambda}{B}\sum_i
\operatorname{CE}\!\left(D(z_{u,i}),m_i\right)
+\frac{\gamma}{B}\sum_{i,j}\mathbf1\{m_i=m_j\}
\left(\|[u_i,v_i]-[u_j,v_j]\|_2-\|u_i-u_j\|_2\right)^2.
$$

$R_i$ is the scaled masked reconstruction loss and $K_i$ the two KL terms.
The discriminator minimizes the corresponding positive CE. Notice that the
distance term pairs cells only **within** an assay. Its denominator is $B$,
not the number of within-assay pairs. The decoder reconstructs each observed
row using that row's own latent variables; it has no observed target from
another assay on the same cell.

The local adapter fixes adversarial gradient isolation, aggregates all
validation batches and restores the best checkpoint. It explicitly selects
feature-wise dispersion, whereas the native decoder default is
feature-by-modality. These details matter for implementation fidelity, but
none introduces observations of the missing cross-assay joint law.
[Adapter implementation](../../../../pipelines/empirical/integration/methods/scmrdr/paper_faithful/adapter.py)

## What information is present or absent

Gene alignment and a region-to-gene transformation are **feature
correspondences**. Thus an informal statement that this application uses
“no prior information” would be too strong. Gene identity, promoter
annotations and availability masks are supplied. Histone promoter counts
are a local application of the architecture, not the paper's original
RNA/ATAC/protein experiment. Such annotation does not establish that a
promoter mark directly regulates a gene, fix its effect size, or specify the
joint distribution of cell-specific residuals.

The native model exposes optional covariate and cell-type inputs. In the
reviewed adapter, `celltype_num=0`, `encoder_covariates=False`, and the
cell-type tensor has zero columns (`adapter.py` 130–137, 243–250). Optional
legitimately available covariates can condition the decoder; this benchmark
uses no additional covariates. There is no input for true cell pairing,
perturbation targets, temporal order, or a causal orientation graph.

A batch label is not automatically an instrument, intervention, or valid
multiple-environment identification condition. If a later application uses
such information with identifying assumptions, its information contract and
theorem must be changed explicitly.

## Applying the strict-disjoint proof to this objective

Let $W_m$ contain the **entire** observed row in assay $m$: its counts,
permitted covariates, and background features. Let $A$ denote fixed feature
annotations/masks. Under independently sampled disjoint cells, the learner
receives

$$
\mathcal D\sim\bigotimes_m P_m^{\otimes n_m},\qquad A\text{ fixed}.
$$

For any fixed parameters, reconstruction and KL are functions of observed
rows. The adversary sees the mixture of modality-labeled row distributions.
The geometry term sees within-assay row pairs. Every one of these
distributions is determined by $\{P_m\}$ and the sampling scheme. BatchNorm
and minibatch composition do not escape this argument: the **entire
minibatch law** remains the same when all $P_m$ are unchanged.

Consequently, if two biological models agree on every $P_m$ and on $A$, the
distribution of the empirical objective, gradients, optimization path,
selected checkpoint and exported representations is identical, when all
algorithmic randomness follows the same policy. The population objectives
are identical as well. The subsequent matching and causal-discovery steps
are further functions of the same observations and randomness.

The main proof supplies two such biological models, even with the exact
aligned state $S$ given:

$$
X=S+U,\qquad
Y=(1-\rho)S+\rho X+\sqrt{1-\rho^2}\,V,
\quad S,U,V\overset{\rm iid}{\sim}\mathcal N(0,1).
$$

Both assay conditionals are $\mathcal N(s,1)$ for every $\rho$, while
$\operatorname{Cov}(X,Y\mid S)=\rho$ and the direct effect of $X$ on $Y$ is
$\rho$. Choose $\rho=0$ and $\rho=1/2$. Whole assay blocks and count
observations can be added with the same assay-local measurement kernels,
preserving the equal input laws. No scMRDR loss term distinguishes these
worlds. Therefore a binary edge decision derived from scMRDR cannot be
correct uniformly over them; its worst-case error is at least $1/2$ under
this two-world comparison. This is a mathematical input-law argument, not
an inference from poor benchmark results.

It is unnecessary to prove that both biological worlds equal the exact
finite neural decoder family: an algorithm cannot distinguish equal input
laws even when its fitted model is misspecified. Conversely, a theorem
restricted to a genuinely narrower, correctly specified biological family
requires checking whether both countermodels belong to that family. The
broad-class result alone does not resolve that separate question.

## Assumptions that could change the conclusion

The model's Gaussian priors, latent factorization, shared weights and
geometry penalty restrict or favor representations. They do not by
themselves establish biological causal identifiability. In particular,
independence of a row's shared and specific latent priors is not evidence
that unobserved same-cell residuals from different assays are independent.

One could define a joint generative extension using a common latent state
and independent assay-private variables. Conditional independence would
then enter **as an assumption of that extension**. The separate
reconstruction objectives cannot validate it against a different residual
copula with the same observed assay laws. Selecting a unique optimum by
regularization also does not make the selected biological explanation
identifiable from observations.

Restricted positive cases remain legitimate: an exact aligned sufficient
state together with a known conditional copula; deterministic assay views
of that state; or a known structural exclusion, invariant conditional-mean
equation and full-rank state variation. Unlinked monotone-regression
identification requires its own assumptions about direction, function
class and structural noise. The reviewed scMRDR objective does not impose
or verify these causal conditions. A ZINB measurement likelihood is not a
known biological structural-noise distribution.

Even perfect preservation of a latent distance geometry would not imply
preservation of conditional independence among named biological variables:
arbitrary changes of coordinates can mix variables. Similarly, the graph
drawn for a VAE's latent generative process is not an identified regulatory
DAG among histone and RNA variables.

Finally, the theorem assumes genuinely disjoint independent samples. The
scMTR pilot's independently shuffled **paired holdout** arrays contain
measurements of the same cells, so the theorem is not automatically a
statement about that full evaluation experiment. Paired evaluation labels
must also remain outside fitting and tuning. This review makes no claim
that every real dataset, every target, or every restricted model class is
unrecoverable.

## Provenance and verification

The exact pinned primary `model.py` and `loss.py` were fetched through the
GitHub connector. Their text hashes reproduce the original hashes in
`upstream_provenance.json` after removing the one terminal newline added by
the temporary text capture. AST comparisons after excluding import
statements were equal to the local copies. No retrieved code was executed.
Only source inspection and text/AST/hash comparisons were performed.

| Inspected local file | SHA-256 |
| --- | --- |
| `paper_faithful/upstream_provenance.json` | `69297f8e7e9b22a3365cebd3f7913ab3248c9d84a1b7636effb1c852b4202e58` |
| `paper_faithful/README.md` | `32526015222a655e811acdd70f1bd82cc782d93bc83b4b9a50e876f4ca520eee` |
| `paper_faithful/adapter.py` | `0b2857139120573013d7e6bd30398523d2a6680bca8b7f162c2453a9c828874a` |
| `paper_faithful/upstream/model.py` | `99a2b10b6bcce8f20422392ecc64b889ca30fd154577be22490664a817aba52b` |
| `paper_faithful/upstream/loss.py` | `5fbd90962ccb08b7bbfc2d2adfbc7fd47e008f77bde7ec4112d592bd0085252c` |

Primary model SHA-256:
`221ff4fbddb63cc08613f9086ce52e230d5175f5c473c92be8f73df210aab078`.
Primary loss SHA-256:
`6da9df9e2602aabf38d8e43c23583ce3b42f850b1389fd51be1d71cb11b93600`.

The paper check was limited to the named primary publication, particularly
Sections 2 and 3 and its stated evaluation targets. This is not a systematic
literature search, a convergence audit, or a numerical re-run of scMRDR.

## Review of the actual companion draft

Reviewed [Note 08](../../theory/08_modern_integration_objectives.md), SHA-256
`df524dfc350f59124b52fb4546d4f5890dc6494dc56388ace5e91dac850deed2`.
**Scoped approval:** its scMRDR subsection accurately identifies the active
masked-ZINB/KL/adversarial/isometry losses, gene-aligned inputs and masks, and
absence of graph encoders and cycle consistency. The identical-input-law
argument is correctly applied to these losses. It does not require multiple
optima, successful fitting, or empirical failure. The stated restrictions to
independent disjoint sampling and a class containing the countermodels, and
its distinct biological-edge/conditional-independence/PAG targets, prevent
the main overclaims identified in the earlier scope review.

One wording correction was sent to the author: the sentence about the missing
residual relationship should refer to the **same cell's** unobserved modality,
not “another cell's unobserved modality.” This is a terminology correction,
not a defect in the proof. No other blocking scMRDR or theorem-applicability
issue was found. The generic cycle illustration is appropriately separated
from scMRDR's active loss. The transport calculation is explicitly a simplified
example, not a numerical claim about the fitted scConfluence model.
