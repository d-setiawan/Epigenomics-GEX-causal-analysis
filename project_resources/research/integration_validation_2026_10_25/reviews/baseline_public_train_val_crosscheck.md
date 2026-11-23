# Public baseline state-transport crosscheck

_Independent check · 2026-10-25 · post-baseline diagnostics; original metrics and selection rule retained_

The [saved baseline report](../../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261025_v1/reports/linear_chain_0_informative_val_seed0_scglue-scmrdr_baseline_5e025d63.json) reproduces from the permitted public inputs. The disagreement between scMRDR transport and its RNA-trained standardized Ridge is real, but does not establish a generally incorrect shared-state integration.

The independent calculation loaded only public baseline `train`/`val` embedding `cell_ids` and `shared`, plus `cell_ids`, `assigned_assay`, and `state` from the corresponding development truth archives. It checked all 12 embedding hashes against the public export seals, both truth hashes, all six common-pool ID hashes, assay membership, unique IDs, and train/validation disjointness. It reproduced the exact saved embedding-only assignment IDs, every state-cost field, and all 18 declared source/target probe scores. The calculation did not import the production evaluator or metric module. Its numerical record, input hashes, environment versions, and reviewer-script hash are in [baseline_public_train_val_crosscheck.json](baseline_public_train_val_crosscheck.json).

The observed training cohort contains 4,000 cells per assay, while the legacy exports contain 1,936 RNA and 2,000 cells in each mark. The common training pool has 1,936 per assay. Validation contains 500 observed cells per assay; RNA QC leaves 487 exported cells, giving 487 per assay after intersection and outcome-independent equalization. The original statement in the theoretical review that these cohort sizes were combined assay totals was corrected.

The common training state center is `[0.0027658891300102162, 0.0036931498334113874]`; population standard deviations are `[1.0071856125953662, 1.0145039876265425]`. Both match the report within `1e-14`. This same scale is used for both methods and every pair. Probe scoring uses raw state, with per-coordinate target-validation variance in the R² denominator; it does not fit target normalizations. The three pairwise squared-state costs are:

| Assay pair | Squared-cost oracle | Exact random expectation | scGLUE matched | scMRDR matched |
| --- | ---: | ---: | ---: | ---: |
| RNA–H3K27ac | 0.061828 | 4.060545 | 0.498899 | 0.174759 |
| RNA–H3K4me3 | 0.069946 | 3.937864 | 0.468411 | 0.154153 |
| H3K27ac–H3K4me3 | 0.078561 | 3.882940 | 0.618787 | 0.132447 |

Independent Euclidean assignment and squared-state oracle calculations agree with the saved costs and gaps within `1e-12`. The exact random expectation was additionally computed using state moments rather than the cost-matrix average. Worst normalized gaps reproduce as `0.14200130451844178` for scGLUE and `0.0282418403234005` for scMRDR. These are finite-pool state-transport results, not biological pairing or causal-graph scores.

The additional source-trained probes below are **post-baseline secondary diagnostics**. All use the same common training/validation identities, no target refitting, Ridge `alpha=1`, or fixed 15-neighbor uniform averaging in raw Euclidean latent geometry. None changes a declared metric, candidate, threshold, or selection criterion.

| Method and direction | Declared standardized Ridge R² | Secondary raw Ridge R² | Secondary raw kNN15 R² |
| --- | ---: | ---: | ---: |
| scGLUE RNA→H3K27ac | 0.857974 | 0.868748 | 0.916625 |
| scGLUE RNA→H3K4me3 | 0.935583 | 0.940584 | 0.928445 |
| scMRDR RNA→H3K27ac | −0.910162 | 0.876413 | 0.969661 |
| scMRDR RNA→H3K4me3 | −0.975870 | 0.838399 | 0.981341 |

This supports a readout-conditioning and extrapolation explanation. In scMRDR's RNA training pool, 17 of 20 coordinates have standard deviation below `0.02`; most are near `0.002–0.005`. The standardized Ridge's effective decoder has raw-coordinate operator norm `165.49`, compared with `1.34` for raw Ridge. Every mark validation row lies outside the RNA training coordinate range in at least one coordinate. Coordinate 8 (zero based), whose source standard deviation is `0.2263`, shifts by about nine source standard deviations; some very small-variance coordinates shift by about three to four. Thus the issue is not confined to the 17 nearly collapsed dimensions.

For a standardized linear decoder with coefficient vector `w`, its raw coefficient is `beta_j=w_j/sigma_j`. Ridge penalizes `sum_j w_j² = sum_j beta_j² sigma_j²`, so low-source-variance directions can receive large raw slopes cheaply. StandardScaler is invertible; the interaction with Ridge regularization and changed target support is the limitation. The standardized RNA probe's squared mean bias accounts for about 80% and 83% of total target squared error in H3K27ac and H3K4me3, respectively. Small Euclidean mismatch need not imply a small error for this high-slope probe. Raw kNN's strong held-out transfer demonstrates useful local state correspondence in this particular world, without proving a unique mechanism for the Ridge failure or validating all latent dimensions.

Two post-baseline controls used exactly the same validation pools. A sorted-ID identity coupling, equivalent to constant embeddings with diagonal tie resolution, gives normalized pairwise gaps `0.9573`, `1.0075`, and `1.0101`. A fixed random triple coupling with seed `20261025` gives `1.0370`, `1.0475`, and `0.9970`. These checks show that the low actual costs do not arise simply from sorted cell-ID order. They are descriptive fixed controls, not confidence intervals or additional independent worlds.

Keep the original standardized Ridge failures visible and label the secondary probes exploratory. The justified conclusion is strong state-transport performance for scMRDR on this one informative development world and common retained population, accompanied by a fragile declared linear transfer readout. No result here identifies cross-assay residual coupling or establishes correct integration of real cells.

No checkpoint, count matrix, assessment/discovery cohort, or CUDA operation was accessed or invoked. No integration model was fitted and no production file changed. Only lightweight state readouts explicitly permitted for this crosscheck were fitted; the live alternative integration was left untouched.
