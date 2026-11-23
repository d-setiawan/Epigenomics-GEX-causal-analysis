# Independent evaluation readout

All 225 frozen arms completed and were evaluated without changing a model setting. The evaluator verified the full fit seal before scoring. A separate agent then reconstructed the readouts without importing the evaluator or learner: 1,350 within-assay views, 900 real pair/level/width views, 2,250 state-correlation matrices, 2,700 literal bootstrap comparisons, and 180,000 conditional model-versus-zero identities passed, with maximum numerical difference \(1.44\times10^{-15}\). No zero-scale abstention, dropped feature pair, or scoring anomaly occurred.

The [machine-readable contrasts](frozen_readout_contrasts.json) contain every variant, matching width comparison, real unit-exposure covariance table, fitted depth coefficient, and both-seed/control state summary. Covariance errors were invariant to source seed within \(8.33\times10^{-17}\). Therefore each comparison uses nine unique dataset-by-assay units, averaging the two numerically equivalent seeds. Those nine units are still not independent biological replications.

## Cross-assay result and the cap qualification

No arm produced a unique moment match or a CF-confirmed match. All real cross-assay predictions therefore use the empty-candidate zero default. Primary improvement over zero is exactly zero in every pair/level/view. Bootstrap improvement intervals are zero because the two predictions coincide, not because the biological cross-assay covariance was shown to vanish.

Every original 300-feature unit-exposure assay fit exceeded the preset cap of six: 108 of 108 assay fits, with uncapped selections ranging from seven to nineteen. The protocol consequently disallowed their source matching. For real Poisson counts the uncapped selections were RNA 11, H3K27ac 12, and H3K4me3 12. Thus the expanded unit-exposure arms are **cap-limited abstentions**; they do not establish failure at an unrestricted source dimension. The separately frozen cap-20 follow-up preserves this original block and directly examines that restriction.

Real offset and fitted-elasticity Poisson variants selected zero sources at either width. Their fitted elasticity coefficients were RNA 1.01727, H3K27ac 0.98826, and H3K4me3 1.00881. This depth sensitivity is a property of the observed proxy and model. It does not identify technical capture efficiency separately from biological state.

## Within-assay improvements on fixed targets

Expanding the Poisson unit-exposure model from 30 to 300 features reduced the original-30 full covariance error in eight of nine dataset-by-assay units and increased it in one. The real original-30 errors illustrate the mixed assay response:

| Assay | 30-feature model | 300-feature model, same 30 targets | Diagonal training baseline |
|---|---:|---:|---:|
| RNA | 0.14166 | 0.13412 | 0.14473 |
| H3K27ac | 0.11288 | 0.12061 | 0.10537 |
| H3K4me3 | 0.10794 | 0.10786 | 0.10526 |

These fixed-scale errors show improvement for RNA and a very small full-matrix change for H3K4me3, while H3K27ac worsens. Both histone models remain worse than the diagonal baseline in this comparison. Whole-300 averages describe different features and must not substitute for the original-30 comparison.

The source-noise sensitivities also had mixed effects. For example, the real unit-exposure 30-feature NB-shape-two model reduced RNA full covariance error from 0.14166 to 0.11143, while increasing H3K27ac from 0.11288 to 0.11406 and H3K4me3 from 0.10794 to 0.11036. These are descriptive contrasts among prespecified variants, not a selected model or evidence that NB shape two is the true measurement process. The full machine-readable table includes off-diagonal errors and both baselines so that restored observation variance cannot be mistaken for latent structure recovery.

Third-moment, shrinkage, and PGF orientations can have equal covariance predictions because orthogonal source rotations leave \(BB^T\) unchanged. Their distinct orientation, stability, and matching diagnostics remain in the fit artifacts. Noisy projected-score correlations with simulated state remain descriptive within-assay associations; none yielded accepted cross-assay sources in this block.

The [data/evaluation receipt](data_and_evaluation_audit.json) and [independent replay](extension_metric_audit.json) support the numerical readout. They do not validate the conditional noise model, latent-source assumptions, biological identity of matched laws, or direct causal interpretation. The existing source QC, promoter overlap, endogenous depth, one real sublibrary, and repeated development use remain material limits.
