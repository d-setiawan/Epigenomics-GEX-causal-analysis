# Graph Evaluation Pipeline

This pipeline evaluates saved `PC`, `FCI`, and `DAGMA` graph outputs without rerunning graph discovery.

It is designed for the current output layout:

```text
CausalDiscovery/outputs/methods/<method>/<dataset>/<gene>/<result-run>/
CausalDiscovery/outputs/methods/dagma/<dataset>/<family-config>/<gene>/<result-run>/
```

Each graph run should contain:

- `run_summary.json`
- `<method>_edges.tsv`
- optional `node_support.tsv`
- nearby-peak metadata reachable from the matrix path in `run_summary.json`

## Command

Run a focused evaluation:

```bash
python3 CausalDiscovery/scripts/causal_cli.py \
  --python-bin /home/dgsetiawan/anaconda3/bin/python \
  evaluate \
  --methods pc,fci \
  --datasets joint_v2_monocyte_cuttag_peak_genes_300kb \
  --genes cd14 \
  --out-dir CausalDiscovery/outputs/evaluation/graph_stability/cd14_pc_fci_300kb
```

Run across all saved graph runs:

```bash
python3 CausalDiscovery/scripts/causal_cli.py \
  --python-bin /home/dgsetiawan/anaconda3/bin/python \
  evaluate \
  --out-dir CausalDiscovery/outputs/evaluation/graph_stability/all_runs
```

By default, stability is computed within:

```text
dataset,gene,method,family_config
```

To compare `PC` and `FCI` directly within the same gene and dataset, use:

```bash
python3 CausalDiscovery/scripts/causal_cli.py \
  --python-bin /home/dgsetiawan/anaconda3/bin/python \
  evaluate \
  --group-cols dataset,gene \
  --methods pc,fci \
  --out-dir CausalDiscovery/outputs/evaluation/graph_stability/pc_fci_by_gene
```

## Outputs

The evaluator writes:

- `graph_runs.tsv`: one row per graph run with graph-size, parameter, and biological-support summaries.
- `normalized_edges.tsv`: one row per graph edge after normalizing the edge schema across `PC`, `FCI`, and `DAGMA`.
- `edge_stability.tsv`: edge-level frequency and orientation consistency within each comparison group.
- `pairwise_graph_similarity.tsv`: pairwise graph overlap metrics.
- `group_summary.tsv`: compact group-level metrics.
- `evaluation_report.md`: a short rendered summary.

## Metrics

### Adjacency Stability

An adjacency is the unordered pair of nodes connected by an edge, ignoring orientation.

For two graph runs $a$ and $b$:

$$
J(E_a,E_b)=\frac{|E_a\cap E_b|}{|E_a\cup E_b|}
$$

This is reported as `adjacency_jaccard`.

For each edge within a group:

$$
\mathrm{freq}(e)=\frac{\#\{\text{runs containing adjacency }e\}}{\#\{\text{runs in group}\}}
$$

This is reported as `adjacency_frequency`.

### Orientation Stability

Orientation is evaluated separately because `PC` outputs CPDAG-like edge states and `FCI` can output PAG-style partial marks.

For each adjacency, the evaluator counts normalized endpoint signatures. Examples:

- `TAIL|ARROW`: directed edge
- `TAIL|TAIL`: undirected edge
- `CIRCLE|CIRCLE`: ambiguous/partial `FCI` edge

The main orientation fields are:

- `dominant_exact_edge_key`
- `dominant_orientation_frequency`
- `orientation_entropy`
- `dominant_directed_key`

This lets us separate:

$$
\text{edge exists stably}
$$

from:

$$
\text{edge direction is stably identified}
$$

### Biological-Support Metrics

For nearby-peak datasets, the evaluator loads each gene's `*_nearby_peak_metadata.tsv` and computes:

- number of selected peak nodes
- number of curated-overlap peak nodes
- number of peak nodes adjacent to expression
- number and fraction of expression-adjacent peaks that overlap curated regions
- hypergeometric enrichment p-value for curated overlap among expression-adjacent peaks

The enrichment test asks whether expression-adjacent peaks are enriched for curated/literature regions relative to all selected peak nodes:

$$
P(X\ge x),\quad X\sim\mathrm{Hypergeometric}(N,K,n)
$$

where $N$ is the number of peak nodes, $K$ is the number of curated-overlap peaks, $n$ is the number of expression-adjacent peaks, and $x$ is the observed number of curated expression-adjacent peaks.

This is validation evidence, not proof of causality.

## Recommended Evaluation Pattern

Use the pipeline in layers:

1. Compare structural filters such as `nz001`, `nz005`, and `nz010`.
2. Within each dataset and gene, compare graph settings: alpha, depth, CI test, transform, and background mode.
3. Compare `PC` and `FCI` with `--group-cols dataset,gene`.
4. Inspect stable expression-adjacent peaks first.
5. Treat stable peak-expression edges with matching target-gene external support
   as high-priority hypotheses. For peak-to-expression edges, interval-only
   ENCODE/JASPAR support is node context, not edge validation.
6. Report adjacency stability and orientation stability separately.

For the current project, the most useful first-pass edge classes are:

- high `adjacency_frequency`
- high `dominant_orientation_frequency`
- expression-adjacent
- peak-to-expression under `tiered_distal_promoter_expr`
- matching target-gene support in `edge_support.tsv` for peak-to-expression
  edges
- overlapping curated/literature regions or ENCODE/JASPAR interval support for
  peak/peak-context prioritization
- recurring across `PC` and `FCI`, not only within one method

When using the external support layer, inspect these files in each graph
directory:

- `node_support.tsv`: summarized node support.
- `node_support_matches.tsv`: one row per external evidence match.
- `edge_support.tsv`: edge validation, source families, and highlight class.
- `support_report.md`: source counts and interpretation notes.
