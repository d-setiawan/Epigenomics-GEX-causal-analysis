# Evaluation

↑ [Central README](../README.md) · [MISSION](../docs/MISSION.md) (why) · [ROADMAP](../docs/ROADMAP.md) (plan) — for evaluation *framing / methodology* see [results/evaluation_overview.md](results/evaluation_overview.md).

Validation of the graphs produced by [`CausalDiscovery/`](../CausalDiscovery/README.md): do the
discovered edges hold up against orthogonal evidence?

## Layout

```
Evaluation/
├── scripts/   validators and external-evidence builders
│   ├── peak_peak_validator.py     element<->element axes w/ distance-matched null
│   ├── coaccess_contrast.py       Cicero-style co-accessibility contrast (the novelty comparison)
│   ├── build_chip_occupancy.py    ChIP-Atlas TF occupancy (shared-TF axis)
│   ├── scan_peak_motifs.py        JASPAR motif scan (shared-TF axis, presence)
│   ├── build_{re2g,eqtl,pchic}_gene_link_support.py   the 3 peak->gene support axes
│   ├── liftover.py                UCSC chain staging used by the PCHi-C builder
│   ├── null_calibration.py        false-positive calibration of the discovery step
│   └── simulate_aggregation_identifiability.py   controlled sim: does aggregation preserve identifiability?
├── outputs/   evidence tables + run artifacts
│   ├── peak_peak_pairs.tsv, peak_tf_chip.tsv, peak_tf_motifs.tsv, *_summary.json  (tracked)
│   ├── external/cache/            download cache (gitignored)
│   └── bootstrap_stability/, graph_stability/, null_calibration/  (run outputs, gitignored)
└── results/   the write-ups — what the numbers mean
```

The curated evidence tables in `outputs/` are tracked deliberately: they are what the claims in
`results/` rest on. The per-run search outputs and the download cache are not.

## Key results

- [results/peak_peak_validation_results.md](results/peak_peak_validation_results.md) — the four element↔element axes and the co-accessibility contrast.
- [results/expanded_panel_confounder_results.md](results/expanded_panel_confounder_results.md) — the confounder-conditioned 15-gene panel.
- **[results/same_mark_finding.md](results/same_mark_finding.md) — ⚠️ the one significant element↔element signature, and it is probably technical. Blocks the four-negative endpoint.**
- **[results/simulation_identifiability_results.md](results/simulation_identifiability_results.md) — the pivotal sim: cell-level identifiable, aggregation collapses it (the mechanism).**
- **[results/partition_bootstrap_results.md](results/partition_bootstrap_results.md) — ⚠️ DEFINITIVE partition test (K=10): no edge in all 10 partitions; the stable-edge list; same-mark enrichment is partition-fragile (4/10). Read before citing any edge.**
- **[results/pairing_artifact_findings.md](results/pairing_artifact_findings.md) — is the cross-assay pairing manufacturing our bidirected edges? multi-channel sim: same-channel structure survives the construction, cross-channel does not (-44 to -57%); and only FCI produces bidirected edges (0 from every score-based search).**
- [results/partition_stability_findings.md](results/partition_stability_findings.md) — how the partition-instability was found (3-seed controls + pilot); background for the K=10 result.
- [results/run_audit_report.md](results/run_audit_report.md) — saved-run audit and rigor tiers.
- [results/evaluation_overview.md](results/evaluation_overview.md) — framing, validation layers, honesty rules.

## The two evaluation questions

Graph variables are **uncurated** nearby CUT&Tag peaks; curated promoter/enhancer regions are never
graph inputs, only evaluation annotations. That separation is what keeps validation independent of
discovery. Two distinct questions, with different machinery:

**1. peak → gene — "is this element a real regulator *of this gene*?"**
Triangulated across three orthogonal, cell-type + target-gene aware axes; an edge counts as validated
when the support is target-gene-aware (≥2 axes agreeing is the gold standard).

| axis | source | builder |
| --- | --- | --- |
| prediction | ENCODE-rE2G | `build_re2g_gene_link_support.py` |
| physical | BLUEPRINT PCHi-C | `build_pchic_gene_link_support.py` |
| genetics | eQTL Catalogue SuSiE | `build_eqtl_gene_link_support.py` |

**2. peak ↔ peak — "what explains the latent common cause behind a bidirected edge?"**
This is the project's novel object and has **no accepted community null**, so we built one: a
distance-matched permutation test (`peak_peak_validator.py`) against which each candidate driver is
tested. Current scope: 15-gene panel, confounder-conditioned metacell PAGs.

Status of both lives in [results/](results) — see **Key results** above.

## Evidence Sources

External evidence can include:

- `ePerturbDB`: stronger perturbation-based support when available.
- `ENCODE Portal`: PBMC or monocyte peak-file support from released GRCh38 ENCODE BED files.
- `ENCODE SCREEN`: broader regulatory annotation support, especially cCREs and enhancer-like elements.
- `JASPAR`: online motif support for peak nodes using JASPAR matrices plus UCSC hg38 sequence retrieval.
- Curated project locus configs under `CausalDiscovery/configs/loci/`.
- Nearby peak metadata emitted by dataset generation.

See [external_sources.md](results/external_sources.md) for the supported online source
formats, default ENCODE SCREEN URLs, and plotting semantics.

## Evidence table format

Column templates for external evidence tables live in `outputs/templates/`. The key template is:

```text
Evaluation/outputs/templates/external_support_regions_template.tsv
```

Expected columns:

- `chrom`
- `start`
- `end`
- `gene`
- `support_label`
- `external_id`
- `biosample`
- `notes`

Only `chrom`, `start`, and `end` are required for interval overlap, but the other columns make reports easier to interpret.

## Nearby-Peak Overlap Checks

For the current datasets, start by reading each gene's nearby peak metadata:

```text
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/<gene>/nearby_peaks/<GENE>_nearby_peak_metadata.tsv
CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_rawcounts/<gene>/nearby_peaks/<GENE>_nearby_peak_metadata.tsv
```

Useful columns include:

- `gene`
- `mark`
- `signed_distance_to_tss`
- `overlaps_curated_region`
- `overlapping_curated_region_names`

These columns let us evaluate graph edges without injecting curated biology into the graph-learning matrix.

## External Support Workflow

For the downloaded ePerturbDB files, place the database TSV, enhancer-gene BED,
and enhancer-TF overlap BED in `Data/raw/external/eperturbdb/`, then annotate a
saved graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --support-preset local_eperturbdb
```

To rebuild support and render the graph in one plotting command:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --online-support \
  --support-preset local_eperturbdb
```

You can also prepare locus-focused template TSVs from `ePerturbDB` and/or
`ENCODE SCREEN`, then annotate a saved graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --locus-config CausalDiscovery/configs/loci/cd14_regions678.tsv \
  --eperturbdb-tsv Evaluation/outputs/external/cd14_eperturbdb.tsv \
  --encode-screen-tsv Evaluation/outputs/external/cd14_encode_screen.tsv
```

This writes support tables into the graph directory:

- `node_support.tsv`
- `node_support_matches.tsv`
- `edge_support.tsv`
- `support_report.md`

The local ePerturbDB adapter writes three source labels when matching records
are present:

- `eperturbdb`: perturbation database rows with `Gene (literature)` target-gene support.
- `eperturbdb_enhancer_gene`: enhancer interval/nearest-gene annotation.
- `eperturbdb_tf_overlap`: enhancer interval/TF overlap context.

You can also fetch ENCODE SCREEN directly:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --fetch-encode-screen
```

For PBMC- or monocyte-context ENCODE peak evidence, query ENCODE Portal files:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --fetch-encode-pbmc \
  --encode-biosample-term "CD14-positive monocyte" \
  --encode-assay-term DNase-seq \
  --max-encode-files 10
```

For motif-level regulatory support, query JASPAR and UCSC sequences:

```bash
python3 CausalDiscovery/scripts/causal_cli.py support \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --fetch-jaspar \
  --jaspar-matrix-id MA0139.1 \
  --jaspar-min-score 0.75
```

Re-plot the graph:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --layout local
```

Or build external support and plot in one step:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/pc/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/pc_kci_alpha_0_05_depth_2_bg_tiered_distal_promoter_expr \
  --online-support \
  --fetch-encode-pbmc \
  --encode-biosample-term "CD14-positive monocyte" \
  --encode-assay-term DNase-seq \
  --layout local
```

New graph runs can also build support before rendering their final plot:

```bash
python3 CausalDiscovery/scripts/causal_cli.py graph \
  --matrix-tsv CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb/cd14/nearby_peaks/CD14_nearby_peak_matrix.tsv \
  --method fci \
  --online-support \
  --fetch-encode-pbmc \
  --encode-biosample-term "CD14-positive monocyte" \
  --encode-assay-term DNase-seq \
  --plot-layout local
```

Add `--fetch-jaspar --jaspar-matrix-id <ID>` to the same `graph` or `plot`
commands when motif-level peak support should be included.

For the current monocyte/PBMC analysis, the compact preset is:

```bash
python3 CausalDiscovery/scripts/causal_cli.py plot \
  --graph-dir CausalDiscovery/outputs/methods/fci/joint_v2_monocyte_cuttag_peak_genes_300kb/cd14/fci_kci_alpha_0_05_depth_5_bg_tiered_distal_promoter_expr \
  --online-support \
  --support-preset monocyte_regulatory \
  --layout local
```

This expands to ENCODE SCREEN, ENCODE Portal `CD14-positive monocyte`
`DNase-seq`, and JASPAR CTCF (`MA0139.1`) support.

## Matching Logic

Support annotation uses:

- Expression nodes: exact gene-symbol matching.
- Coordinate peak nodes such as `chr5:140612011-140612339__H3K27ac`: genomic interval overlap.
- Named curated regions: coordinates resolved from the locus config, then interval overlap.
- Peak-expression edges: interval evidence on the peak endpoint validates the
  edge only when the external target gene matches the expression endpoint. This
  keeps ePerturbDB-style target-gene evidence edge-specific and prevents
  ENCODE/JASPAR interval-only context from validating peak-to-gene edges.

## How To Interpret Support

`ePerturbDB` is the stronger causal-support layer. It is most useful when graph-adjacent peaks overlap experimentally perturbed enhancers.

`ENCODE Portal` peak files provide cell-context support when the chosen biosample term matches the analysis, for example `peripheral blood mononuclear cell` or `CD14-positive monocyte`.

`ENCODE SCREEN` is a broader regulatory-plausibility layer. It is useful for checking whether graph-adjacent peaks overlap cCREs, enhancer-like signatures, promoter-like signatures, or cCRE-target gene links.

`JASPAR` is motif-level regulatory plausibility. It helps prioritize peak nodes
with TF motif support, especially for peak-peak or TF-centered follow-up, but it
is not perturbation evidence by itself.

The strongest first-pass evidence is:

$$
\text{graph-adjacent peak}
\cap
\text{curated/literature region}
\cap
\text{external regulatory evidence}
$$

This is still validation, not proof of causality. It tells us whether the uncurated graph learner is rediscovering plausible biology before we move to stronger perturbation-based claims.

## Graph Stability Pipeline

Saved `PC`, `FCI`, and `DAGMA` graph runs can now be compared directly with:

```bash
python3 CausalDiscovery/scripts/causal_cli.py \
  --python-bin /home/dgsetiawan/anaconda3/bin/python \
  evaluate \
  --methods pc,fci \
  --datasets joint_v2_monocyte_cuttag_peak_genes_300kb \
  --genes cd14 \
  --out-dir Evaluation/outputs/graph_stability/cd14_pc_fci_300kb
```

The evaluator writes:

- `graph_runs.tsv`
- `normalized_edges.tsv`
- `edge_stability.tsv`
- `pairwise_graph_similarity.tsv`
- `group_summary.tsv`
- `evaluation_report.md`

See [graph_evaluation_pipeline.md](results/graph_evaluation_pipeline.md) for usage and metric definitions. See [literature_review_graph_evaluation.md](results/literature_review_graph_evaluation.md) for the citation-backed rationale.
