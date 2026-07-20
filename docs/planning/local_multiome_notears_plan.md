# Local NOTEARS Plan for Paired scRNA-seq + scATAC-seq

## Goal

Build a **local causal discovery algorithm** for one target gene at a time using paired single-cell multiome data. The output should be a small directed graph containing:

- local peaks near the target gene,
- the target gene expression node,
- observed confounder nodes,
- optional peak-to-peak edges,
- candidate peak-to-gene causal edges.

The project is not trying to recover a whole-genome DAG. It is trying to recover a **local causal hypothesis graph** around one gene.

## Why This Is a Good Project

- It is small enough to implement from scratch.
- It is still a real causal discovery problem rather than plain prediction.
- It respects the sparse and non-Gaussian nature of single-cell RNA and ATAC data.
- It lets us encode biological assumptions directly into the graph search space.

## Local Graph Definition

For one target gene $Y$, define the local variable set

$$
X = (C_1, \dots, C_q, A_1, \dots, A_p, Y),
$$

where:

- $C_1, \dots, C_q$ are observed confounders,
- $A_1, \dots, A_p$ are candidate local peaks near the gene,
- $Y$ is the target gene expression.

Possible examples of confounders:

- batch,
- donor,
- RNA library size,
- ATAC fragment count,
- cell cycle score.

## Causal Assumptions

We assume:

1. The local system can be approximated by a DAG.
2. Observed confounders are **root nodes**.
3. Confounders may affect peaks and the target gene.
4. Peaks may affect other peaks.
5. Peaks may affect the target gene.
6. The target gene does **not** directly affect local chromatin accessibility in this local model.

This yields the following allowed edge types:

$$
C_m \to A_j, \qquad C_m \to Y, \qquad A_j \to A_k, \qquad A_j \to Y.
$$

Forbidden edge types:

$$
A_j \to C_m, \qquad Y \to C_m, \qquad Y \to A_j.
$$

This is the main way causal structure is injected into the model: not just through acyclicity, but through an **edge mask** reflecting biology.

## How Observed Confounding Works

A confounder $C$ is a common cause of two variables of interest.

For a peak-gene relation:

$$
C \to A_j, \qquad C \to Y.
$$

Then $A_j$ and $Y$ can be associated even if there is no direct edge $A_j \to Y$.

For a peak-peak relation:

$$
C \to A_j, \qquad C \to A_k.
$$

Then two peaks can appear dependent because of shared upstream variation rather than a direct causal relationship.

In single-cell multiome data, realistic confounders include:

- cell cycle,
- technical batch,
- donor,
- sequencing depth,
- global cell state.

Including them as explicit graph nodes is better than only regressing them out beforehand, because the graph can explicitly explain dependencies through fork structures such as $A_j \leftarrow C \to Y$.

## Statistical Model

Use a **mixed likelihood** rather than a Gaussian loss.

### Peak Nodes

Treat each peak $A_j$ as binary accessibility:

$$
A_{ij} \sim \mathrm{Bernoulli}(\pi_{ij}),
$$

with

$$
\mathrm{logit}(\pi_{ij}) = \alpha_j + \sum_{u \in \mathrm{Pa}(A_j)} W_{u,j} X_{iu}.
$$

This is a logistic node model.

### Gene Node

Treat the target gene expression $Y_i$ with a negative binomial model:

$$
Y_i \sim \mathrm{NB}(\mu_i, \theta),
$$

with

$$
\log \mu_i = \beta_0 + \sum_{u \in \mathrm{Pa}(Y)} W_{u,Y} X_{iu}.
$$

This is better aligned with sparse, overdispersed single-cell RNA counts.

## NOTEARS-Style Objective

Let $W$ be the weighted adjacency matrix over all local nodes. Learn $W$ by solving

$$
\min_{W} \; \mathcal{L}(W; X) + \lambda \|W\|_1
$$

subject to

$$
h(W) = 0,
$$

where:

- $\mathcal{L}(W; X)$ is the total negative log-likelihood over all node models,
- $\lambda \|W\|_1$ is a sparsity penalty,
- $h(W)$ is the acyclicity constraint.

Use the standard NOTEARS acyclicity function:

$$
h(W) = \mathrm{tr}\left(e^{W \circ W}\right) - d,
$$

where $d$ is the number of nodes and $\circ$ is the elementwise product.

## Edge Mask

Let $M$ be a binary matrix of allowed edges. Optimize only over

$$
W = M \circ B,
$$

where $B$ is the unconstrained parameter matrix.

This means:

- acyclicity is enforced only over allowed edges,
- forbidden edges are always zero,
- observed confounders stay root nodes automatically.

## Suggested Local Pipeline

For each target gene:

1. Select peaks in a cis window, for example $\pm 100 \text{kb}$.
2. Filter to a manageable number of peaks, for example the top $10$ to $25$ by variance or marginal association with the gene.
3. Build the local variable matrix with confounders, peaks, and the target gene.
4. Binarize peaks.
5. Keep gene counts as counts for the negative binomial node model.
6. Construct the allowed-edge mask.
7. Optimize the masked mixed-likelihood NOTEARS objective.
8. Threshold small edge weights to zero.
9. Report the final local DAG and bootstrap edge stability.

## Optimization Strategy

Implement the optimization in PyTorch.

Use the augmented Lagrangian version of NOTEARS:

$$
\min_W \; \mathcal{L}(W;X) + \lambda \|W\|_1 + \alpha h(W) + \frac{\rho}{2} h(W)^2
$$

with iterative updates of $\alpha$ and $\rho$ until $h(W)$ is close to zero.

This is much easier to implement than a constrained solver from scratch.

## Minimal Version

A good first version of the project should include:

- observed confounders only,
- binary peaks,
- one negative binomial target gene,
- peak-to-peak edges,
- peak-to-gene edges,
- no latent variables.

This already gives a real and defensible local causal model.

## Optional Extension: Latent Nuisance Factors

If there is time, extend the model to

$$
X = (C_1, \dots, C_q, L_1, \dots, L_k, A_1, \dots, A_p, Y),
$$

where $L_1, \dots, L_k$ are latent nuisance factors.

This would require alternating optimization:

1. initialize latent factors,
2. optimize $W$ given latent factors,
3. optimize latent factors given the graph,
4. repeat.

This is more ambitious and should be treated as an extension, not the core deliverable.

## What the Project Can Honestly Claim

The project does **not** identify ground-truth biology from observational data alone.

It **does** produce:

- local causal hypothesis graphs,
- explicit adjustment for observed confounding,
- sparse candidate peak-to-peak and peak-to-gene edges,
- a framework that is better aligned with single-cell distributions than Gaussian DAG learning.

## Recommended Deliverables

- Implementation of masked local NOTEARS with Bernoulli and negative binomial node models.
- A visualization of one local graph per selected gene.
- A ranked list of stable $A_j \to Y$ edges.
- An ablation comparing:
  - no confounder nodes,
  - confounder nodes included,
  - peak-to-peak edges allowed versus forbidden.

## Implementation Milestones

### Milestone 1

- Read paired multiome matrix.
- Select one target gene and nearby peaks.
- Build confounder matrix.

### Milestone 2

- Implement masked adjacency matrix.
- Implement Bernoulli loss for peaks.
- Implement negative binomial loss for gene node.

### Milestone 3

- Implement NOTEARS acyclicity penalty.
- Optimize with augmented Lagrangian.

### Milestone 4

- Threshold edges.
- Visualize the learned local DAG.
- Run bootstrap stability analysis.

## Core Pitch for the Report

The project learns a **local biologically constrained DAG** from paired single-cell RNA and ATAC data. It combines:

- explicit confounder nodes,
- peak-to-peak and peak-to-gene edges,
- mixed single-cell likelihoods,
- and NOTEARS-style acyclicity enforcement.

This makes it a causal discovery project rather than a standard predictive model, while keeping the problem small enough to implement from scratch.
