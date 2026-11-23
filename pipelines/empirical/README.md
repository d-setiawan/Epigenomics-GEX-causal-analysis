# Empirical pipeline

The PBMC analysis asks whether monocyte neighborhoods in a shared embedding produce reproducible relationships between RNA and six histone marks. The CUT&Tag-pro and 10x RNA studies have separate cells and no verified shared donor. The protein panel can check chromatin-cell neighborhoods. Hashtags identify technical groups. Neither supplies a same-cell RNA–histone reference. [Input provenance](data/results/data_sources.md) · [Biological alignment checks](integration/results/biological_alignment_2026-09-16.md)

The separate scMTR-seq H9 differentiation pilot permits a stronger measurement check. The learner sees disjoint RNA, H3K27ac and H3K4me3 cells. The evaluator holds the original pairing for 595 other cells. Their sparse panel yielded no positive cross-assay adjacency with the specified readout, so causal sensitivity could not be estimated from this cohort. For known causal truth, see the simulation benchmark. [scMTR-seq cohort](data/scmtr_seq/README.md) · [Pilot results](../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md) · [Integration recoverability](../simulation/integration_recoverability/README.md)

| Stage | What it produces |
| --- | --- |
| [Data](data/README.md) | Source provenance, cell identifiers and assay-specific count matrices. |
| [Integration](integration/README.md) | Shared coordinates, candidate neighbors and alignment diagnostics. |
| [Analysis](analysis/README.md) | Locus graphs under different ways of pairing or aggregating cells. |
| [Evaluation](evaluation/README.md) | External evidence for individual loci and reproducibility across partitions. |

A fresh Git export reproduced the 24 compared PBMC preprocessing files and all six scMTR learner H5ADs. Fitted integration models and large run directories require a separate training run or an existing cache. [Restore commands](../../TRANSFER.md) · [Transfer verification](../../reproduce/VERIFICATION.md)
