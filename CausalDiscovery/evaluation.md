# Evaluation

This file collects practical ways to evaluate the current causal-discovery pipeline.

## Goal

We do not want to evaluate the graph learner in isolation. We want to evaluate the full pipeline:

$$
\text{pairing} \rightarrow \text{feature construction} \rightarrow \text{graph learning} \rightarrow \text{biological interpretation}
$$

The easiest way to get confused is to collapse "feature construction" and "graph stability" into the same step. They are related, but they answer different questions.

## Four evaluation layers

### 1. Pairing quality

Before causal discovery, check whether the pseudo-paired rows are believable.

Useful checks:

- coarse-label agreement across matched modalities
- fine-label agreement across matched modalities
- distribution of GLUE match distances
- sensitivity to keeping only the closest matches
- donor consistency, if donor labels are available

Two useful summaries are:

$$
\text{label agreement}=
\frac{\#\{\text{matched pairs with same label}\}}{\#\{\text{all matched pairs}\}}
$$

and graph stability after restricting to the top $q\%$ best matches by distance.

### 2. Feature construction quality

This layer is about the variables that go into the graph learner.

In this project, feature construction means:

- choosing which regions belong to a locus
- deciding whether to use broad or expanded region definitions
- deciding whether region scores come from retained `scGLUE` bins or raw clean bins
- aggregating bins into promoter or enhancer region scores
- deciding which region-mark variables are kept or dropped

So feature construction happens before the graph-learning step. The graph learner only sees the final matrix:

$$
\text{rows} \times \text{constructed variables}
$$

Useful checks:

- number of unique values per node
- zero fraction per node
- whether promoter marks behave like promoter variables
- whether enhancer marks behave like enhancer variables
- broad-locus versus expanded-locus consistency
- whether one region definition disconnects expression while another does not

This is important because a graph can fail for feature-construction reasons even when pairing is fine.

Example:

- if a curated enhancer is split too finely, some region-mark nodes may become almost constant
- then PC may return a sparse or empty graph
- that is a feature-construction problem, not necessarily a graph-learning problem

### 3. Graph stability

This layer starts after feature construction is fixed.

Here the question is:

$$
\text{given the same locus matrix, does the learned graph stay similar under reasonable analysis choices?}
$$

Compare graphs across:

- depth
- prior versus no prior
- transform versus no transform
- CI test choice
- pairing-distance filtering
- bootstrap resampling

Useful summaries:

$$
J(E_a,E_b)=\frac{|E_a\cap E_b|}{|E_a\cup E_b|}
$$

for edge-set Jaccard overlap, and

$$
\text{orientation stability}=
\frac{\#\{\text{runs with same edge direction}\}}{\#\{\text{runs where the edge exists}\}}
$$

for direction consistency.

So the distinction is:

- feature construction asks whether we built sensible variables
- graph stability asks whether the algorithm gives similar answers once those variables are fixed

### 4. Biological validation

This is the final layer, not the only layer.

Questions to ask:

- do expected promoter or enhancer regions attach to expression at all?
- do edge directions make sense under background knowledge?
- do expected edges survive across multiple settings?
- do positive-control loci outperform weaker or exploratory loci?

Current practical positive controls:

- `CD14`
- `CSF1R`

## Null tests

Null tests are important because they tell us whether the pipeline is creating structure even when it should not.

Recommended nulls:

- permuted pairing null:
  shuffle RNA anchors within the same cell type before locus scoring
- region-shift null:
  move curated regions away from the true locus
- random gene-region null:
  pair one gene's expression with another locus's regions

Under a good pipeline, we expect:

$$
\text{edge count under null} < \text{edge count under real data}
$$

and biologically coherent expression-linked edges should weaken or disappear.

## Practical scorecard

A locus is more convincing when it performs reasonably well on all four layers:

1. pairing quality
2. feature quality
3. graph stability
4. biological plausibility

That means a good graph should be:

- built from believable pseudo-pairs
- based on usable locus variables
- stable across reasonable settings
- more aligned with known biology than null controls are

## Current interpretation of stability

One useful result already observed in this project is that the current `PC + KCI` graphs with tiered background knowledge were stable across:

$$
\text{max depth} = 1,2,3,4,5
$$

for the active broad and expanded `CD14` and `CSF1R` raw-bin loci.

That suggests the present graph patterns are not being driven mainly by very high-order conditioning sets.
