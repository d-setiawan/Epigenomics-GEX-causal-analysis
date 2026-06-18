# Literature Review: Evaluating Causal Graph Discovery

This note summarizes how causal-discovery and biological-network papers evaluate inferred graphs, and maps those practices to the current CUT&Tag/RNA graph workflow.

## Main Takeaways

Most papers use one of four evaluation regimes:

1. **Known graph or simulated graph recovery**: compare inferred edges to a ground-truth DAG, CPDAG, PAG, or synthetic regulatory network.
2. **Ranked biological edge recovery**: compare weighted/ranked edges to a gold standard using AUROC, AUPRC, early precision, or edge precision ratios.
3. **Stability and robustness**: compare inferred graphs across bootstrap samples, perturbations, hyperparameters, data transformations, or methods.
4. **External biological validation**: test whether high-priority edges overlap perturbation evidence, curated databases, motif/chromatin evidence, or cell-type-specific regulatory annotations.

For this project, we do not have a full gold-standard causal graph for `CSF1R`, `CD14`, or `IL1B`. The right first-pass evaluation is therefore:

$$
\text{graph stability}+\text{external biological support}+\text{null controls}
$$

rather than only graph-recovery metrics such as SHD.

## Pure Causal Discovery Evaluation

### Structural Recovery Metrics

General causal-discovery benchmarks usually evaluate an estimated graph against a known or simulated graph with:

- adjacency precision and recall
- arrowhead/orientation precision and recall
- structural Hamming distance, or SHD
- F1 or Matthews correlation for edge existence
- sometimes structural intervention distance, or SID

The mixed-data causal-discovery benchmark by Andrews et al. explicitly reports adjacency precision/recall, arrowhead precision/recall, and SHD for methods such as PC variants and mixed graphical model pipelines [Andrews et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6510516/). The `causal-learn` evaluation utilities expose the same style of metrics, including SHD and adjacency precision/recall, when a reference graph exists [causal-learn evaluations](https://causal-learn.readthedocs.io/en/latest/utilities_index/Evaluations.html).

For this project, SHD is only appropriate if we create a reference graph, for example through simulation, curated local test graphs, or high-confidence perturbation edges. Without a reference graph, Jaccard overlap and edge frequency across runs are more honest.

### Intervention-Aware Metrics

Peters and Buhlmann introduced SID to evaluate whether two DAGs imply similar intervention statements, not just whether their edge lists match [Peters and Buhlmann, 2015](https://is.mpg.de/en/publications/petbuh2015). This is important because two graphs can have similar SHD but different causal intervention implications.

SID would become useful later if we define a fully directed local DAG and want to ask whether graph differences change implied peak-to-expression interventions. It is less directly applicable to current `FCI` PAGs with circle endpoints and to graphs without a known target DAG.

### Continuous Optimization DAG Methods

`NOTEARS` evaluates graph recovery on synthetic and benchmark settings using metrics such as SHD, false discovery rate, true positive rate, and false positive rate [Zheng et al., 2018](https://arxiv.org/abs/1803.01422). `DAGMA` similarly evaluates against known graphs and reports improved structural Hamming distances in its experiments [Bello et al., 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/36e2967f87c3362e37cf988781a887ad-Abstract-Conference.html).

For this project, `DAGMA` can be compared to `PC`/`FCI` through shared adjacency recovery, stable expression-adjacent edges, and biological support. Weight threshold sweeps should be evaluated with edge-frequency curves or top-$k$ edge stability, because DAGMA returns weighted directed graphs.

### PC, FCI, and KCI

The PC algorithm is a constraint-based method that uses conditional-independence testing and returns a partially directed equivalence-class graph under causal sufficiency assumptions. The FCI algorithm extends constraint-based discovery to settings with possible latent confounders and returns PAG endpoint marks; `causal-learn` documents `FCI` with Fisher Z, chi-square/G-square, KCI, and related tests [causal-learn FCI documentation](https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/FCI.html). KCI is useful when relationships are nonlinear or non-Gaussian, but is computationally expensive; the original KCI test was proposed by Zhang, Peters, Janzing, and Scholkopf [Zhang et al., 2012](https://arxiv.org/abs/1202.3775).

For this project, PC and FCI should not be evaluated as if every non-directed endpoint is a failure. Instead:

- evaluate adjacency existence separately from orientation
- treat `FCI` circle endpoints as uncertainty, not as wrong directions
- report whether peak-expression adjacencies survive when moving from PC to FCI

## Biological Network Evaluation

### DREAM-Style Gene Regulatory Network Benchmarks

The DREAM network inference challenges established a common GRN evaluation pattern: methods submit ranked edge lists and are evaluated by AUROC/AUPR against known or semi-known regulatory networks. The DREAM5 consortium paper emphasizes aggregate performance across networks and evidence types [Marbach et al., 2012](https://pubmed.ncbi.nlm.nih.gov/22796662/). NetBenchmark describes this as a de facto protocol built around ROC/PR curves and AUPR/AUROC [Bellot et al., 2015](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0728-4).

This is useful if we later convert graph outputs to ranked peak-gene hypotheses. For unweighted `PC`/`FCI`, ranks could come from stability frequency, expression-adjacent frequency, or recurrence across methods. For `DAGMA`, ranks can combine absolute weight with stability across thresholds.

### Single-Cell GRN Benchmarks

Pratapa et al. benchmarked single-cell GRN methods using AUPRC, AUROC, early precision, stability across simulations/dropouts/algorithms, motif analysis, and scalability [Pratapa et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7098173/). This is directly relevant: they separate ranking accuracy from stability and scalability rather than relying on one score.

For this project, the analogous plan is:

- AUPRC/AUROC only when an external peak-gene truth set is available
- early precision for top stable expression-adjacent peaks
- stability across depth, alpha, CI test, structural filters, and method
- scalability/runtime notes for KCI versus Fisher Z

### Single-Cell Causal Discovery

CausalCell evaluates causal discovery for single-cell settings under high dimensionality, missingness, and incomplete-variable challenges, and specifically discusses `PC + KCI` as a strong practical choice across varied situations [CausalCell](https://pmc.ncbi.nlm.nih.gov/articles/PMC10229139/). Their framing supports using kernel CI tests only where the candidate graph is small enough, while documenting runtime and sample requirements.

For this project, this supports the current staged plan:

$$
\text{structural filters and Fisher Z baselines first}
\rightarrow
\text{KCI on smaller candidate sets}
$$

### Perturbation-Based and Curated Biological Ground Truth

Recent biological causal-discovery papers increasingly evaluate against perturbation-derived or curated causal graphs. PC-NOTEARS uses Perturb-seq, eQTL-derived causal graphs, and curated causal resources, evaluating graph structure with SHD/F1 and regulator prediction with AUPRC/AUROC [Wang et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11373380/). CausalBench focuses on network inference from real single-cell perturbation data and highlights scalability and interventional evidence as central challenges [CausalBench](https://arxiv.org/abs/2210.17283).

CICT evaluates inferred GRNs using AUPR, partial AUPR, ratio partial AUPR, edge precision ratios, sensitivity analyses, and biological enrichment of predicted regulator targets [CICT](https://pmc.ncbi.nlm.nih.gov/articles/PMC10612495/).

For this project, the closest analogue is:

- `ePerturbDB` overlap as stronger perturbation-style support
- `ENCODE SCREEN` and cCRE annotations as regulatory plausibility support
- curated locus configs as local prior knowledge for evaluation, not graph construction
- enrichment among graph-adjacent peaks relative to all selected nearby peaks

## Recommended Metrics For This Project

### Required First-Pass Metrics

These are implemented in `evaluate_graph_stability.py`:

- `n_edges`
- `n_directed_edges`
- `n_partial_edges`
- `adjacency_jaccard`
- `exact_jaccard`
- `orientation_agreement_given_adjacency`
- `adjacency_frequency`
- `dominant_orientation_frequency`
- `orientation_entropy`
- expression-adjacent peak counts
- curated-overlap expression-adjacent peak counts
- hypergeometric enrichment of curated regions among expression-adjacent peaks

### Recommended Reported Edge Classes

For each gene and dataset, report:

- stable adjacencies: $\mathrm{freq}(e)\ge 0.5$
- core adjacencies: $\mathrm{freq}(e)\ge 0.8$
- stable expression-adjacent peaks
- stable peak-to-expression directions
- peaks supported by curated/literature regions
- peaks supported by external perturbation or cCRE evidence when available

### Suggested Nulls

The literature consistently shows that graph quality cannot be judged only by attractive-looking edges. We should add nulls:

- permuted RNA-anchor pairing within monocytes
- shifted genomic regions
- random gene-expression to wrong-locus peak matrix
- shuffled expression column with fixed peak matrix
- matched node-count random peak subsets

The desired behavior is:

$$
\text{stable expression-adjacent curated support}_{\mathrm{real}}
>
\text{stable expression-adjacent curated support}_{\mathrm{null}}
$$

### Suggested Later Extensions

Once external support tables are populated:

- precision at top $k$ stable edges
- enrichment odds ratios for `ePerturbDB` and `ENCODE SCREEN`
- rank-based AUPRC/AUROC using stability or DAGMA weight as the score
- cross-method consensus score:

$$
s(e)=\frac{1}{M}\sum_{m=1}^M \mathbf{1}\{e\in G_m\}
$$

- split or cross-fit screening evaluation if elastic-net screening is used before graph discovery

## Practical Interpretation Rule

For the current monocyte CUT&Tag/RNA graphs, I would treat an edge as high priority only if it satisfies most of:

1. stable adjacency across reasonable graph settings
2. stable or biologically interpretable orientation
3. expression-adjacent under the expression-sink background
4. peak overlaps curated or external regulatory evidence
5. signal is stronger than null-control graphs
6. recurrence across `PC` and `FCI`, or agreement between constraint-based methods and `DAGMA`

This keeps the evaluation aligned with the literature while respecting the fact that these are uncurated nearby-peak graphs, not fully known causal DAGs.

