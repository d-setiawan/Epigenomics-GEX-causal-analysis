# Empirical evaluation

The PBMC analysis generates candidate peak–gene and peak–peak edges from aggregated monocyte measurements. This stage asks two questions about those edges.

- **Does an annotation support the proposed relationship?** Peak–gene checks match both the interval and the target gene against ENCODE-rE2G predictions, promoter capture Hi-C contacts and eQTL credible sets. For peak–peak edges, the dedicated analysis compares independent evidence against distance-matched interval pairs. These assays provide evidence about the selected loci. A graph arrow still depends on the discovery assumptions. [External sources](results/external_sources.md) · [Peak–peak results](results/peak_peak_validation_results.md)
- **Does the same edge recur when the cells are regrouped?** Over 15 genes, the 133- and 240-metacell partitions each yielded 11 externally supported peak–gene edges. No supported edge appeared in both sets. A separate test held partition size fixed and changed the seed. [Partition stability findings](results/partition_stability_findings.md) · [Partition bootstrap](results/partition_bootstrap_results.md)

The [analysis driver](../analysis/PIPELINE.md) produces graph inputs. The [evaluation overview](results/evaluation_overview.md) describes the evidence builders and reports. For the paired scMTR-seq dependence reference and the synthetic tests with known causal truth, use the [pilot record](../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md) and [simulation benchmark](../../simulation/integration_recoverability/README.md).
