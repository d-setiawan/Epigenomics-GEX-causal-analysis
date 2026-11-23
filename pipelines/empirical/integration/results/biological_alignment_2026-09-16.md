# Biological alignment baseline — 2026-09-16

The execution phase is authorized for the **existing RNA and six histone assays**.
This report records completed diagnostics and preparation, not a newly trained
or biologically validated integration. CA-BOSS remains paused; causal aggregation
has separate ownership. No new datasets/worlds or protein-assisted discovery
inputs were introduced.

## Outcome

The saved scGLUE embedding conserves substantial protein-associated structure
within assays, but some cross-assay neighborhoods reverse that structure.
The saved scMRDR comparator does not resolve the problem. An independent reviewer
reproduced a discordant and a concordant comparison directly from raw ADTs and
Euclidean distances without importing the evaluator.

These are quantitative checks in the full latent space, **not UMAP judgments**.
They support investigating cross-assay correspondence; they do not identify a
unique cause or establish correct biological labels. Interpretation boundaries
are owned by [LIMITATIONS L17](../../../../project_resources/research/LIMITATIONS.md#l17-biological-integration-is-not-established-by-modality-mixing).

## Evaluation contract

Implementation: [evaluate_biological_alignment.py](../scripts/evaluate_biological_alignment.py).
Each completed run has a hashed `protocol.json`, `cell_partitions.tsv` and
`metrics.json` under `pipelines/empirical/integration/comparisons/biology_alignment_2026-09-16/`.

- Protein measurements were excluded from the historical integration training.
  The new evaluation reserves approximately 25% of assay-qualified cell identities
  using a fixed SHA256 split. Only development proteins are loaded for analysis
  or used as neighbor sources. The reservation starts now, after earlier
  exploratory analyses; it is not a claim of never-before-seen data.
- Each ordered source→target comparison predicts target protein scores from 25
  source neighbors in the full embedding. A fixed, outcome-independent hash
  selects up to 1,000 development target cells per assay. Within-assay searches
  explicitly exclude the query cell. Source→target denotes prediction, not causation.
- Fixed scores use log1p ADTs: T panel (CD3-1, CD7, CD2), myeloid panel (CD14,
  CD33, CD11b-1, CD64), B panel (CD19, CD20, CD79b, CD22), their T−myeloid and
  B−T contrasts, isotypes and depth controls. Individual-protein correlations
  are also retained. These are descriptive panels, not validated classifiers.
- Raw Spearman correlations are primary descriptive effects. Sensitivity
  analyses regress scores on log ADT depth, log upstream fragment depth, their
  squared terms and within-assay hashtag indicators. Hashtags are not verified
  donors. Adjustment can remove real biology as well as nuisance effects.
- Sixty-four whole-row source-protein shuffles preserve hashtag × ADT-depth
  quartile × fragment-depth quartile strata. Raw and adjusted null ranges are
  descriptive references, **not confidence intervals, p-values or FDR control**.
- Reports include source-neighbor concentration, effective source-cell counts,
  distances, target hashtag/depth strata, zero rates and all 30 cross-assay
  directions. Twelve within-assay records compare joint embeddings with the
  common historical scGLUE LSI representation, not a method-specific scMRDR LSI.
- Reciprocal embedding intersections ensure the two saved methods use identical
  source/query identities. Full-cohort scGLUE results are reported separately.
  These embeddings were fit transductively; this is not unseen-cell generalization.

The matched cohort contains 38,668 chromatin cells: 29,022 development and 9,646
reserved. Both methods have the same membership hash,
`01a97cc35b4265e9ebd643d0b5af63645387a1b95e37808cbd19d8b55cb382ad`,
and identical source/query hashes in every direction.

Full scGLUE covers 44,433 chromatin cells: 33,322 development and 11,111 reserved.
H3K9me3 has 7,164 cells in full scGLUE but only 1,659 in the matched comparison.
Thus the intersection is not a representative substitute for the full cohort.

## Measured results

Selected **matched-cohort** T−myeloid correlations:

| Prediction direction | scGLUE raw | scGLUE adjusted | scMRDR raw | scMRDR adjusted |
| --- | ---: | ---: | ---: | ---: |
| H3K27me3 → H3K4me1 | −0.608 | −0.564 | −0.056 | −0.086 |
| H3K4me2 → H3K4me3 | 0.671 | 0.558 | 0.191 | 0.085 |
| H3K9me3 → H3K4me1 | −0.301 | −0.309 | 0.022 | −0.010 |

For the first scGLUE comparison, the adjusted conditional-shuffle range is
−0.198 to 0.116; for the second it is 0.009 to 0.229. Independent reconstruction
reproduced both observed raw/adjusted correlations to numerical precision.
Raw conditional nulls can be strongly nonzero: preserving depth and hashtags
also preserves some biological structure. A positive raw correlation alone
is therefore not sufficient evidence of valid correspondence.

In **full-cohort scGLUE**, within-assay T−myeloid conservation is:

| Assay | Joint embedding | Historical local LSI |
| --- | ---: | ---: |
| H3K27ac | 0.735 | 0.759 |
| H3K27me3 | 0.726 | 0.756 |
| H3K4me1 | 0.658 | 0.666 |
| H3K4me2 | 0.663 | 0.669 |
| H3K4me3 | 0.699 | 0.700 |
| H3K9me3 | 0.406 | 0.440 |

Yet H3K27me3→H3K4me1 is −0.618 raw/−0.574 adjusted, and
H3K9me3→H3K4me1 is −0.396/−0.382. The discordance therefore persists outside the
restricted comparator cohort. This combination of within-assay conservation
and cross-assay disagreement motivates alignment diagnostics; it does not prove
that feature truncation, guidance signs or a specific cell-type permutation caused it.

All three artifacts contain 30 cross-assay and 12 within-assay records, pass
input-stat rechecks and bind the evaluator source hash. An independent reviewer
checked cohort/split counts, matched identity hashes and null movement. No
assessment proteins were released. No overall accuracy score or acceptance
claim is assigned. RNA alignment still needs its own evidence.

## Isolated inputs and runtime readiness

[prepare_biology_integration_inputs.py](../scripts/prepare_biology_integration_inputs.py)
successfully created `compact_preprocess_v1/` with seven H5ADs and a corrected
absolute-path manifest. It preserves counts, PCA/LSI and identities exactly by
chunked hashes and verifies sources were unchanged. The historical `joint_v2`
preprocessing symlink resolves to `joint_v1`; neither was edited.

The historical RNA `X` is a dense 9,462 × 25,215 float64 array (~1,820 MiB before
copies). The isolated files omit those values and make `X` a local hard link to
the copied raw counts. **These are training-specific files: `X` now means raw
counts, not scaled expression.** Training must explicitly use `layers/counts`
and unchanged `X_pca`/`X_lsi`. This storage change is not new preprocessing or
feature selection. The files occupy approximately 132 MiB on disk; compressed
file size is not a training RAM estimate.

The helper does not claim a whole-file hash of the unread source dense `X`.
It records protected-array hashes, file identity/size/mtime/ctime, repaired
manifest paths, and its source hash. Ten fixture tests cover sparse/dense source
formats, AnnData 0.9.2 read-back, no reading of dense `X`, source-change detection
and overwrite/path/identity guards. Together with 13 evaluator tests, **23 tests pass**.
An independent read-only check of the actual seven outputs compared all 227
non-`X` datasets (86,886,716 values) against their sources in bounded chunks,
recomputed protected hashes on both sides, verified the local CSR hard links
and corrected paths, and confirmed unchanged source file stats. No source `X`
values were read; verification peak RSS was approximately 77 MiB. Full real-data
AnnData eager loading was deferred; compatibility was tested on small fixtures.

CUDA works outside the restricted execution sandbox: a real tensor calculation
succeeded with PyTorch 2.10.0/CUDA 13.0 on the RTX 3070 Ti Laptop GPU (8 GiB VRAM;
approximately 7 GiB free at the check). The sandbox's blocked NVML/device view
must not be interpreted as absent hardware. WSL-visible RAM was 11.68 GiB with
other workloads consuming much of it; this is not total host capacity or a
permanent resource limit. Full training has not yet been launched. Check current
RAM and GPU availability immediately before launch; do not terminate user jobs.

The independent readiness audit estimates ~0.695 GiB for compact input numeric
arrays (~0.624 GiB after graph subsetting), plus ~0.765 GiB graph overhead from
a small systematic sample. The graph estimate is approximate and likely inflated
by the sample's vertex/edge ratio. Installed scGLUE also copies the graph and
allocates negative-sampling arrays and framework state. Consequently the observed
~1.7 GiB available RAM is not a safe full-training budget. **Use at least 6 GiB
currently available RAM as a conservative launch gate**, then measure actual
peak usage; this is not a measured requirement or a claim that the host lacks RAM.

## Controlled rerun sequence

1. Establish a current-environment scGLUE control with fixed original counts,
   representations, graph, cells, seed and recorded `joint_v2` hyperparameters.
   Use a fresh output directory and verify actual CUDA execution. Compare it
   with the saved baseline to separate runtime changes from scientific changes.
2. Isolate guidance strength (control `lam_graph=0.005`, weaker arm `0.001`)
   while keeping the other inputs fixed. Add a degree/sign-aware shuffled-guidance
   negative control. Agreement with guidance is an internal diagnostic, not
   independent validation. Freeze graph construction and randomization before
   reading that arm's results.
3. Compare abundance selection with depth-aware variability at the same feature
   budget, then larger budgets and mark-appropriate coarse bins/domains. Start
   from the existing sparse full-bin data; do not treat six million possible
   bins as a dense matrix requirement. Separate the encoder feature budget from
   the independent guidance/reconstruction cap. Pin LSI implementation and seed;
   silent backend/HVG fallbacks must not define experimental arms.
4. Evaluate all six assays, RNA anchors, unmatched populations, technical effects,
   within-assay conservation, cross-assay proteins and seed stability. Keep ADTs
   out of model fitting. Report coverage loss rather than improving scores by
   dropping difficult cells. Trace feature/count retention, but do not equate
   retained abundance with recovered biology.
5. Select on development diagnostics only, record a fixed configuration and
   comparison criteria, then explicitly release reserved assessment outcomes.
   Keep positive/negative controls and per-assay results; do not promote a model
   merely because a pooled score or UMAP improves. Held-out-feature/RNA checks,
   repeated seeds and downstream prior-sensitivity tests remain to be implemented.

No revised integration has been promoted. Mark-specific coverage/background QC,
external sample identity reconciliation and RNA anchoring remain open. Live
status is in the [roadmap](../../../../project_resources/research/ROADMAP.md#biology-execution-2026-09-16).

## Reproduction

Run from the repository root in `scrna_r`. Completed output names below are
immutable: choose a new suffix to repeat. The evaluator refuses existing output
directories and requires an explicit assessment-release flag.

```bash
python pipelines/empirical/integration/scripts/evaluate_biological_alignment.py \
  --embedding pipelines/empirical/integration/runs/scglue_joint_v2/train/all_cells_glue_embeddings.tsv \
  --prefix GLUE_ \
  --match-embedding pipelines/empirical/integration/runs/scmrdr_scMRDR_gaussian/train/all_cells_jianle_embeddings.tsv \
  --out-dir pipelines/empirical/integration/comparisons/biology_alignment_2026-09-16/glue_development_v1

python pipelines/empirical/integration/scripts/evaluate_biological_alignment.py \
  --embedding pipelines/empirical/integration/runs/scmrdr_scMRDR_gaussian/train/all_cells_jianle_embeddings.tsv \
  --prefix JIANLE_ \
  --match-embedding pipelines/empirical/integration/runs/scglue_joint_v2/train/all_cells_glue_embeddings.tsv \
  --out-dir pipelines/empirical/integration/comparisons/biology_alignment_2026-09-16/jianle_development_v1

python pipelines/empirical/integration/scripts/evaluate_biological_alignment.py \
  --embedding pipelines/empirical/integration/runs/scglue_joint_v2/train/all_cells_glue_embeddings.tsv \
  --prefix GLUE_ \
  --match-embedding pipelines/empirical/integration/runs/scglue_joint_v2/train/all_cells_glue_embeddings.tsv \
  --out-dir pipelines/empirical/integration/comparisons/biology_alignment_2026-09-16/glue_full_development_v1

python pipelines/empirical/integration/scripts/prepare_biology_integration_inputs.py \
  --source-dir pipelines/empirical/integration/runs/scglue_joint_v2/preprocess \
  --out-dir pipelines/empirical/integration/comparisons/biology_alignment_2026-09-16/compact_preprocess_v1

python -m unittest discover -s pipelines/empirical/integration/tests -p 'test_*.py' -v
```
