# GLUE Paper Notes (Beginner-First, Theory-Deep)

Source: `papers/GLUE.pdf`  
Paper: **Cao, Z.-J. & Gao, G. (2022), Nature Biotechnology**

---

## 1) Start Here: What GLUE is solving

You have multiple single-cell assays:

1. RNA has gene features.
2. ATAC has peak/bin features.
3. Other modalities (methylation, protein) have their own features.

These feature spaces are different, and cells are often unpaired across assays.

GLUE solves this by learning:

1. A shared cell representation.
2. A shared feature representation.
3. A biologically constrained connection between modalities through a prior regulatory graph.

---

## 2) Minimal Probability + Deep Learning Crash Course

This section is intentionally slow.

### 2.1 Random variable and distribution

A random variable $x$ is a quantity that can take different values.  
A probability distribution $p(x)$ tells you how likely each value is.

### 2.2 Latent variable

A latent variable $u$ is hidden, not directly observed.  
In GLUE, $u$ represents hidden cell state.

### 2.3 Prior and posterior

1. Prior: what you believe before seeing data, e.g. $p(u)$.
2. Posterior: what you believe after seeing data, e.g. $p(u\mid x)$.

### 2.4 Why posterior is hard

In deep latent models, exact $p(u\mid x)$ is usually intractable.

### 2.5 Variational posterior (your key question)

A **variational posterior** is an approximating distribution:

$$
q(u\mid x;\phi)\approx p(u\mid x).
$$

You pick a tractable family $q$ (often Gaussian with neural-network outputs), then optimize parameters $\phi$ so $q$ gets close to the true posterior.

So in plain language:

1. True posterior is too hard.
2. Build a learnable approximation $q$.
3. Train it to be close.

### 2.6 ELBO (your second key question)

ELBO means **Evidence Lower BOund**.

You want to maximize data log-likelihood:

$$
\log p(x).
$$

But direct optimization is hard with latent variables.  
Using $q(u\mid x)$, we get:

$$
\log p(x)=\underbrace{\mathcal{L}_{\text{ELBO}}(x)}_{\text{tractable lower bound}}
+
\mathrm{KL}\!\big(q(u\mid x)\,\|\,p(u\mid x)\big).
$$

Since KL is nonnegative, ELBO is a lower bound:

$$
\mathcal{L}_{\text{ELBO}}(x)\le \log p(x).
$$

Equivalent useful form:

$$
\mathcal{L}_{\text{ELBO}}(x)
=
\mathbb{E}_{q(u\mid x)}[\log p(x\mid u)]
-\mathrm{KL}\!\big(q(u\mid x)\,\|\,p(u)\big).
$$

Interpretation:

1. Reconstruction term: make decoder explain data well.
2. KL term: keep latent encoding regularized to prior.

This is why VAEs do not just memorize.

---

## 3) Per-Modality Latent Model in GLUE

Suppose $K$ omics layers.  
Layer $k$ has features $V_k$, data $x_k$, and sample size $N_k$.

GLUE models each layer with a modality-specific VAE:

$$
p(x_k;\theta_k)=\int p(x_k\mid u;\theta_k)\,p(u)\,du.
$$

Encoder (approximate posterior):

$$
q(u\mid x_k;\phi_k).
$$

Per-layer ELBO:

$$
\mathcal{L}_{X_k}
=
\mathbb{E}_{x_k\sim p_{\text{data}}}
\left[
\mathbb{E}_{u\sim q(u\mid x_k)}\log p(x_k\mid u)
-\mathrm{KL}\!\big(q(u\mid x_k)\,\|\,p(u)\big)
\right].
$$

Important issue: if layers are trained independently, their latent spaces can have inconsistent orientation.

---

## 4) Prior Regulatory Graph (What it means)

The guidance graph $G=(V,E)$ is a **prior biological structure**:

1. Nodes $V$: all features across modalities.
2. Edge sign $s_{ij}\in\{+1,-1\}$: positive or negative regulation.
3. Edge weight $w_{ij}\in(0,1]$: confidence.

This graph injects biological prior knowledge into integration.

In simple terms:

1. Without graph, alignment may be purely geometric.
2. With graph, alignment is guided by known biology.

---

## 5) Graph-Linked Joint Model

GLUE augments per-modality VAEs with feature latent variables $v_i\in\mathbb{R}^m$:

$$
p(x_k,G;\theta_k,\theta_G)
=
\int p(x_k\mid u,V;\theta_k)\,p(G\mid V;\theta_G)\,p(u)\,p(V)\,du\,dV.
$$

Typical priors:

$$
p(u)=\mathcal{N}(0,I_m),\quad
p(v_i)=\mathcal{N}(0,I_m),\quad
p(V)=\prod_{i\in V}p(v_i).
$$

Why this matters:

1. Cells and features are learned together.
2. Feature embeddings transfer information across modalities.

---

## 6) Graph Decoder Intuition and Formula

Graph likelihood encourages feature pairs linked in $G$ to have compatible embedding geometry.

Core quantity:

$$
v_i^\top v_j.
$$

Signed, weighted, negative-sampling objective:

$$
\log p(G\mid V)
=
\mathbb{E}_{(i,j)\sim p(i,j;w_{ij})}
\left[
\log \sigma\!\left(s_{ij}\,v_i^\top v_j\right)
+
\mathbb{E}_{j'\sim p_{\text{ns}}(\cdot\mid i)}
\log\!\left(1-\sigma\!\left(s_{ij}\,v_i^\top v_{j'}\right)\right)
\right].
$$

Interpretation:

1. Positive edge: push embeddings together.
2. Negative edge: push embeddings apart.
3. Weight: stronger prior has stronger effect.

---

## 7) Data Decoder in GLUE

For count modalities (RNA/ATAC), GLUE uses negative binomial likelihood:

$$
p(x_k\mid u,V;\theta_k)=\prod_{i\in V_k}\mathrm{NB}(x_{ki};\mu_i,\theta_i).
$$

Mean parameter depends on inner product between cell embedding and feature embeddings:

$$
\mu_i \propto \mathrm{Softmax}_i\!\left(\alpha\odot V_k^\top u+\beta\right)\times(\text{library size}).
$$

For methylation-like data, GLUE swaps in other distributions (for example ZILN in the paper).

So GLUE is modular in decoder family.

---

## 8) Variational Factorization and Combined ELBO

GLUE uses:

$$
q(u,V\mid x_k,G;\phi_k,\phi_G)
=
q(u\mid x_k;\phi_k)\,q(V\mid G;\phi_G),
$$

where:

1. $q(V\mid G)$ is graph encoder (GCN-based).
2. $q(u\mid x_k)$ is modality encoder.

Objective before adversarial part:

$$
K\,\mathcal{L}_G(\theta_G,\phi_G)+\sum_{k=1}^{K}\mathcal{L}_{X_k}(\theta_k,\phi_k,\phi_G),
$$

with:

$$
\mathcal{L}_G
=
\mathbb{E}_{V\sim q(V\mid G)}\log p(G\mid V)
-\mathrm{KL}\!\big(q(V\mid G)\,\|\,p(V)\big).
$$

---

## 9) Adversarial Alignment (slow explanation)

This is a two-player game.

### 9.1 Discriminator role

Discriminator $D(u;\psi)$ receives cell embedding $u$ and predicts modality label.

Loss:

$$
\mathcal{L}_D
=
-\frac{1}{K}
\sum_{k=1}^{K}
\mathbb{E}_{x_k}
\mathbb{E}_{u\sim q(u\mid x_k)}
\log D_k(u;\psi).
$$

### 9.2 Encoder role

Encoders are trained in the opposite direction to make modality hard to predict.

Intuition:

1. If modality is easy to predict, embeddings are not aligned.
2. If modality is hard to predict, embeddings are modality-invariant.

### 9.3 Minimax form

$$
\min_{\psi}\ \lambda_D\mathcal{L}_D,
$$
$$
\max_{\theta,\phi}\ \lambda_D\mathcal{L}_D+\lambda_G K\mathcal{L}_G+\sum_{k=1}^{K}\mathcal{L}_{X_k}.
$$

This is what “adversarial alignment” means in GLUE.

---

## 10) Weighted Adversarial Alignment

When modality compositions differ a lot, naive adversarial training can overfit common populations.

GLUE adds weights $w^{(n)}$:

$$
\mathcal{L}_D^{\text{weighted}}
=
-\frac{1}{K}
\sum_{k=1}^{K}
\frac{1}{W_k}
\sum_{n=1}^{N_k}
w^{(n)}
\mathbb{E}_{u\sim q(u\mid x_k^{(n)})}
\log D_k(u;\psi),
\quad
W_k=\sum_{n=1}^{N_k}w^{(n)}.
$$

The paper uses a two-stage approach:

1. coarse pretraining with noise,
2. cluster-based balancing for fine tuning.

---

## 11) Batch Covariate Extension

GLUE can include batch label $b$ directly in decoder:

$$
p(x_k\mid u,V,b;\theta_k).
$$

In practice, decoder parameters become batch-specific (for example $\alpha_b,\beta_b,\theta_b$).

---

## 12) Regulatory Inference in GLUE

After training, compute feature-feature similarity:

$$
s_{ij}=\cos(v_i,v_j)=\frac{v_i^\top v_j}{\|v_i\|\,\|v_j\|}.
$$

Then:

1. define null via shuffled embeddings,
2. compute empirical $P$-values,
3. control FDR (Benjamini-Hochberg).

This yields statistically filtered regulatory hypotheses.

---

## 13) Benchmark Metrics (paper context)

The paper evaluates:

1. biology conservation,
2. omics mixing,
3. alignment error (FOSCTTM),
4. robustness to corrupted guidance graphs,
5. scalability.

FOSCTTM form:

$$
\mathrm{FOSCTTM}
=
\frac{1}{2N}\sum_{i=1}^{N}\left(\frac{n_i^{(1)}}{N}+\frac{n_i^{(2)}}{N}\right),
$$

where $n_i^{(1)}$, $n_i^{(2)}$ count imposters closer than the true matched cell.

Lower is better.

---

## 14) Mapping to your histone-modification integration

For your current pipeline outputs:

1. Use bin matrices for robust shared feature integration.
2. Use peak matrices for interpretability and regulatory follow-up.
3. Keep QC rules consistent across marks.
4. Build guidance graph progressively:
   1. start with distance-based peak-gene links,
   2. add stronger evidence later (Hi-C/eQTL/TF-ChIP).

---

## 15) Quick glossary

1. Latent variable: hidden representation (cell state).
2. Posterior: updated belief after observing data.
3. Variational posterior: tractable approximation to true posterior.
4. ELBO: lower bound to log-likelihood used for training latent models.
5. Adversarial alignment: discriminator-vs-encoder game to remove modality identity.
6. Guidance graph: prior regulatory feature network constraining integration.

