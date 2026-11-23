# Epigenomics-GEX Causal Analysis

Which relationships between histone marks and RNA can we recover when each assay measures different cells? We study this at individual gene loci in human blood cells, with simulations where the same-cell biology is known. A shared state can explain why two assays change together. A direct peak–gene edge makes a stronger claim about what happens within a cell. [Project mission](project_resources/research/MISSION.md)

## Start here

- For the scientific story, read the [research recap through 22 November 2026](project_resources/reports/2026-11-22-unpaired-assay-recap.md).
- For the evidence and open questions, see the [roadmap](project_resources/research/ROADMAP.md), [decision rationale](project_resources/research/DECISIONS.md) and [limitations](project_resources/research/LIMITATIONS.md).
- To run an analysis from a checkout, start with the [transfer guide](TRANSFER.md) and [verification record](reproduce/VERIFICATION.md).

## Research status

| Question | What the records support |
| --- | --- |
| Can unpaired raw counts recover relationships carried by a common latent state? | **Yes in the exact-model experiment.** The calibration-selected source search passed the covariance, transport and prediction criteria in **9/9 reused worlds** with one to three sources and the correct source count supplied. The unbounded fit in the new coordinates, initialized from the old starts, passed **5/9**. Success varied across individual starts. [Experiment and per-world scores](pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md) |
| What makes that possible? | With a known product-Poisson count model, full-column-rank linear log-rate maps and a common law of independent non-Gaussian sources distinguishable under signed permutations, the assay laws identify transport on the supported log-rate subspaces. A common latent draw and independent measurement noise across assays given that draw also determine a model-implied count joint. An earlier test on background features from two cached simulations did not establish this recovery. [Corollary](project_resources/research/nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md) · [Earlier count experiment](pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_DEMONSTRATION.md) |
| Does alignment identify direct cross-assay effects? | Even perfect state alignment leaves the within-state coupling open in the general unpaired problem. The scGLUE, scMRDR-style and scConfluence studies did not establish reliable recovery of causal edges. The bounded integration validation is complete. The broader frozen integration suite recorded **118/178** scheduled model fits. [Population counterexample](project_resources/research/theory/07_unpaired_integration_nonidentifiability.md) · [Validation](pipelines/simulation/integration_recoverability/INTEGRATION_VALIDATION.md) · [Execution ledger](pipelines/simulation/integration_recoverability/IMPLEMENTATION_STATUS.md) |
| What survives in the biological analysis? | Partition-stable cross-mark adjacencies in the monocyte panel were enriched for shared promoter contacts. Cross-mark directions varied across partitions. The separate scMTR-seq S1 pilot reserved paired observations for evaluation and exposed sparse-count calibration problems. [Monocyte evidence](pipelines/simulation/aggregation_identifiability/results/crossmark_identifiability.md) · [S1 pilot](project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md) |

The November 22 clean-checkout replay reproduced the full *earlier matched comparison* and exercised one fit from the later source search. The complete 9/9 result and its independent numerical audit come from the original November 15 search. [Transfer verification](reproduce/VERIFICATION.md)

## Repository structure

| Area | What to find there |
| --- | --- |
| [Empirical pipeline](pipelines/empirical/README.md) | Data provenance, integration methods, local graph analysis and external biological evidence |
| [Simulation pipelines](pipelines/simulation/README.md) | Aggregation controls, the parked CA-BOSS benchmark and integration-recoverability studies |
| [Project resources](project_resources/README.md) | Mission, theory, study reviews, reports and archive |
| [Environments](envs/README.md) | CPU replay requirements, Linux empirical environment and study-specific overlays |
| [Reproduction](TRANSFER.md) | Input restoration, replay commands and checks completed on a clean export |

## Pipeline at a glance

The historical monocyte workflow preprocesses CUT&Tag, integrates the cells with RNA, constructs metacells, screens nearby peaks, fits local graphs and compares selected edges with external evidence. Simulations let us check separately which results come from the underlying biology, the counts, the matching rule and the graph learner. [Pipeline map](pipelines/README.md)

## Existing real-data setup

The PBMC analysis uses six histone marks from GSE195725 and RNA from a separate 10x experiment. A shared donor identifier has not been verified. The chromatin hashtags identify technical groups. The separate scMTR-seq S1 study uses RNA, H3K27ac and H3K4me3, with disjoint cells during fitting. [PBMC provenance](pipelines/empirical/data/results/data_sources.md) · [S1 cohort](pipelines/empirical/data/scmtr_seq/README.md)

## Quick start

From the repository root, the exact-model CPU tests require Python 3.11 and no biological downloads.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r envs/requirements-replay-cpu.txt
python -m pytest -q pipelines/simulation/integration_recoverability/tests/test_count_transport*.py
```

For a full simulation replay, use a new run ID and choose the stage in the transfer guide. For a biological run, restore and preprocess the inputs first. The per-gene driver requires the prepared peak–gene matrices. [Replay and restore commands](TRANSFER.md) · [Empirical driver](pipelines/empirical/analysis/PIPELINE.md)

## Documentation ownership (read before editing any doc)

The [mission](project_resources/research/MISSION.md) defines the questions and evidence standard. The [roadmap](project_resources/research/ROADMAP.md) owns project status, [decisions](project_resources/research/DECISIONS.md) records the rationale and the [limitations ledger](project_resources/research/LIMITATIONS.md) qualifies each claim. Dataset provenance belongs to the data stage. Detailed scores, denominators, controls and audit receipts belong to the study that produced them. Link to those records when summarizing a result. Dated plans, source snapshots and hash-verified study records preserve the version used for that study. To inspect a guide as it appeared in the November 22 handoff before this documentation refresh, use `git show 8338034:path/to/README.md`.

## Where did this number come from?

For a saved empirical run, `python project_resources/research/provenance.py --gene CD14` follows recorded and inferred links through the original manifests. For a simulation result, start at the study's comparison table, evaluation seal and review. Checkout contents and files generated during a run are described in the transfer guide. [Research tools](project_resources/research/README.md#tooling) · [Count-search records](pipelines/simulation/integration_recoverability/runs/count_transport_search_v1/README.md)

## Documentation map

The [research index](project_resources/research/README.md) connects the theory, literature, simulation designs and independent reviews. The [pipeline index](pipelines/README.md) links the operating instructions for each stage. The [reports index](project_resources/reports/README.md) links shareable summaries and dated presentations.
