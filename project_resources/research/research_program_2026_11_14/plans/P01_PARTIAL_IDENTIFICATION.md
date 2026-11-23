# P01: marginally identified detection-joint intervals

Astra decision, 2026-11-14: approved before implementation or new evaluator access.

## Target and theorem

For two explicitly named count features define X=1[count>0], Y=1[count>0].
Given population detection probabilities p and r, the sharp set of possible
same-cell joint detection probabilities is

    max(0,p+r-1) <= P(X=1,Y=1) <= min(p,r).

Every point in the interval gives a nonnegative normalized 2x2 probability
table with these marginals. Covariance subtracts p*r from both endpoints;
the independent completion p*r is always included. These are pairwise joint
targets, not source identities, activity effects or genomic causal edges.
Simultaneous endpoint choices for many pairs need not define one joint law.

This result requires no chosen biological coupling and no paired training.
It is compatible with observed count detection in a specified selected
population. It cannot remove inherited selection, assay population differences,
or count noise. The mathematical core is to be independently checked and
formalized in Lean if feasible.

The actual biological assay pools have different local QC selection events.
Consequently, the computed sets are couplings of the supplied assay-specific
empirical marginal distributions. They are not certified identified sets for
the all-assay-QC biological holdout population. Transferring that interpretation
requires equality/transport assumptions that are not established. Preserve this
distinction in every result summary and evaluator label.

## Why this is a distinct next step

Earlier source fits tried to select compatible cross-assay sources. This step
reports the entire pairwise identified set supported by marginal detections,
including uncertainty that an integration objective hides by choosing one
completion. It is not another source-matching variant and does not rerun the
published Sturma simulation or acquire data.

## Frozen implementation and evaluation contract

- Inputs: existing `count_extensions_2026_11_01/data_v1/public/{dataset}/train`
  for `scmtr_seq_s1`, `linear_chain_0`, `constrained_collider_0`.
- Assays: RNA, H3K27ac, H3K4me3; first 30 frozen columns per assay, exactly
  the historical common panel. Enumerate all three unordered assay pairs
  and all 30x30 feature pairs, with no feature/pair filtering by outcomes.
- Transform: count>0, without depth normalization or thresholds learned from
  evaluation. Record original feature names and all input hashes.
- Output: p, r, lower/upper joint bounds, independent completion, lower/upper
  covariance bounds, interval widths for all 8,100 dataset-feature pairs.
- Numerical acceptance: finite p,r in [0,1], ordered bounds, valid endpoint
  tables, independence inside every interval, exact zero-width recognition.
- Deterministic negative control: independently reverse or permute assay rows;
  marginal bounds must be unchanged. This is an algebra/invariance check, not
  a new benchmark world or biological false-discovery estimate.
- Unit checks: fair Bernoulli [0,1/2] coupling interval; deterministic endpoint
  marginals; nontrivial asymmetric marginals with endpoint attainability;
  malformed counts or feature mismatch must fail.
- Seal protocol, source, inputs and complete predictions before evaluator reads.
- Separate evaluator: existing biological `paired_reference.npz` only, using
  its documented assay-major 300-column layout and original-30 anchors. Report
  paired empirical joint detections inside/outside the training-marginal
  plug-in bounds, discrepancy magnitudes, and widths. Do not tune anything
  using this readout. Independently reconstruct bounds from evaluator marginals
  as a labeled tautological consistency check, never an admissible learner.
- Simulation existing state references do not contain the target binary joint;
  do not score detection joints against latent states or unrelated graph truth.
  Report marginal-set results for those cached worlds without invented truth.
- Summarize all pairs and datasets, not only narrow/nonzero intervals. No
  graph adjacency/direction score follows from this target.

## Interpretation and rejection criteria

The exact theorem concerns population marginals. Plug-in intervals concern the
empirical training-marginal law, not confidence intervals for the population
identified set. Historical panel selection used these data, so this plan makes
no nominal post-selection coverage claim. Evaluation outside a plug-in interval
can arise from different sample marginals and is not a theorem refutation.
Biological batch/selection differences remain alternative explanations.

A failed endpoint identity, row-invariance check, input provenance check or
independent arithmetic replay blocks acceptance. Numerical correctness and
wide/zero-spanning sets support faithful partial-identification reporting;
they do not establish a biological absence of edges. The next step after this
bounded evaluation is synthesis with the exact theorem and assumption audit,
not threshold widening or forced coupling selection.
