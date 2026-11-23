# Existing-data biological investigation — 2026-09-11

_Follow-up to the [initial review](biological_foundation_review_2026-09-10.md).
Existing inputs only; CA-BOSS and aggregation development were not resumed._

---

## Main conclusion

The leading concern is now **cross-assay biological correspondence**, not local
ADT count corruption, matrix reordering, or failure to reproduce the saved label
transfer. Those implementation checks pass. H3K27me3's protein/label disagreement
persists under several descriptive checks, while its within-assay neighborhoods
retain protein-associated structure. Separately, abundance-based feature selection
discards a particularly large fraction of H3K9me3 bin counts.

These findings narrow the investigation; they do not establish a unique cause or
a validated replacement integration. Interpretation limits are maintained in
[L17](../../../../project_resources/research/LIMITATIONS.md#l17-biological-integration-is-not-established-by-modality-mixing).
Live status and next actions are in [ROADMAP](../../../../project_resources/research/ROADMAP.md#biology-investigation-2026-09-11).
No production labels, inputs, filtering rules, embeddings or models were changed.

## Audit scope and reproducibility

Four read-only diagnostics inspect the existing six-mark scCUT&Tag inputs and
`scglue/joint/joint_v2` outputs. Formats are feature-by-cell ADT/HTO text tables,
cell-by-bin sparse MatrixMarket counts, HDF5-backed annotated integration matrices,
embedding/label TSVs, and the original 10x RNA count/annotation archives. Every
comparison aligns explicit assay/barcode/feature identities rather than assuming
row order. Input paths, sizes, timestamps, source hashes, and selected input hashes
are recorded in the linked artifacts; large matrix files are not all content-hashed.

| Investigation | Script | Saved artifact |
| --- | --- | --- |
| Raw-to-integration identity | [investigate_adt_lineage.py](../scripts/investigate_adt_lineage.py) | [adt_lineage.json](../outputs/biology_investigation_2026-09-11/adt_lineage.json) |
| Label replay and protein contrasts | [investigate_label_transfer.py](../scripts/investigate_label_transfer.py) | [label_transfer_replay.json](../outputs/biology_investigation_2026-09-11/label_transfer_replay.json) |
| Chromatin count retention | [investigate_chromatin_retention.py](../scripts/investigate_chromatin_retention.py) | [chromatin_retention.json](../outputs/biology_investigation_2026-09-11/chromatin_retention.json) |
| RNA annotations and QC | [investigate_rna_annotations.py](../scripts/investigate_rna_annotations.py) | [rna_annotations.json](../outputs/biology_investigation_2026-09-11/rna_annotations.json) |

Commands are in the [Data README](../README.md#existing-data-investigation-2026-09-11).
Reports refuse overwrite; use a new output filename to repeat. The run used CPU-only
bounded/sparse reads on a machine with about 8.2 GB available RAM. No new datasets,
worlds or trained integrations were generated. Diagnostic neighbor searches index
saved embeddings; they do not train new representations.

## Local identity and count lineage pass

| Comparison | Scope | Mismatches |
| --- | --- | ---: |
| Raw → clean ADT | 7,687,082 values; 44,434 cells | 0 |
| Raw → clean HTO | 222,170 values | 0 |
| Source bin counts → integration counts | Six 15,000-bin matrices | 0 |
| Demux → clean → integration metadata | Shared metadata columns | 0 |
| HDF5 → embedding → label identities | 53,895 cells, including RNA | 0 |

The six ADT tables retain the identical ordered panel of 173 unique features.
ADT/HTO totals agree with demultiplexing metadata. Embedding TSV values match
the HDF5 embeddings within serialization precision (maximum absolute discrepancy
2.37 × 10⁻⁷).

**ADT columns and clean-metadata rows are not in the same order.** Explicit barcode
joins validate correctly; a positional join would be unsafe. This audit found no
such mismatch in the evaluated pipeline outputs.

The one clean H3K27ac cell absent from integration became zero-count after the
top-30,000 feature selection, as recorded in preprocessing metadata. It was not
an initially empty source-bin cell. These local checks do not establish the
original experiment's antibody specificity or donor identity.

## Saved label transfer reproduces, but biological disagreements persist

An independent Euclidean-neighbor replay at the saved settings, `k=25` and
minimum winning-vote confidence 0.5, reproduces **all 44,433 chromatin coarse
labels and confidence values exactly**. The replay uses saved RNA annotations;
the separate RNA audit verifies their source mapping. Fine-label transfer was not
replayed. Changing only the number of neighbors to 10 or 50 changes 1,787 (4.02%)
or 1,245 (2.80%) coarse labels, respectively; this is sensitivity, not improved accuracy.

To check whether the earlier single-marker patterns extend across proteins, fix
the following panels for this diagnostic:

- T-associated panel: CD3-1, CD7, CD2.
- Myeloid-associated panel: CD14, CD33, CD11b-1, CD64.
- B-associated panel: CD19, CD20, CD79b, CD22.

For each cell, take the mean `log1p(count)` in each panel and subtract panel means.
No threshold is used to assign a biological label. These panels were fixed for this
follow-up, not preregistered or validated classifiers. All group summaries and
isotype features are retained in the artifact, including discordant sensitivity results.

| Assay | T-minus-myeloid median in monocyte-labeled cells | Same contrast in T-labeled cells |
| --- | ---: | ---: |
| H3K27ac | −1.003 | 1.190 |
| H3K27me3 | 1.230 | −0.566 |
| H3K4me1 | −0.896 | 1.733 |
| H3K4me2 | −1.062 | 1.607 |
| H3K4me3 | −1.047 | 1.506 |
| H3K9me3 | 0.836 | 0.747 |

H3K27me3's monocyte-labeled group has a positive contrast in **all five HTO
groups** (medians 1.092–1.400), among confidence ≥0.9 cells (1.249), and in the
assay-wide upper ADT-count quartile (1.638). Its B-labeled group instead has a
negative T-minus-myeloid contrast (−0.888). H3K4me1's B- and T-labeled groups
have similar positive T-minus-myeloid contrasts (1.754 and 1.733); the B-labeled
group's B-minus-T contrast is −1.991. These extend the initial protein discrepancies.

The depth-subset result is **not uniform**: H3K9me3 monocyte-labeled cells in the
upper ADT-count quartile have a negative contrast (−0.589), unlike the full group's
positive median. H3K27ac's T-labeled group also changes sign in that subset.
Upper-quartile selection changes population composition, is not depth matching,
and includes threshold ties. Zero median isotype counts do not rule out antibody
background. Do not interpret persistence across hashtags as donor replication.

### Within-assay neighborhoods retain protein-associated structure

For each cell, average the T-minus-myeloid contrast of its 25 nearest *same-assay*
neighbors, explicitly excluding itself, then compute Spearman correlation with
its own contrast. Neither protein values nor labels enter the neighbor search.

| Assay | Saved LSI | Saved GLUE |
| --- | ---: | ---: |
| H3K27ac | 0.756 | 0.719 |
| H3K27me3 | 0.757 | 0.728 |
| H3K4me1 | 0.673 | 0.672 |
| H3K4me2 | 0.693 | 0.680 |
| H3K4me3 | 0.707 | 0.697 |
| H3K9me3 | 0.417 | 0.440 |

H3K27me3 therefore has protein-associated within-assay neighborhood structure
despite its cross-assay label disagreement. This is consistent with a population
correspondence problem; it does not prove a permutation of cell types. The statistic
is descriptive, not depth-adjusted, and lacks uncertainty estimates. LSI and GLUE
have different geometries; this is not a definitive method ranking or causal test.

## Feature selection truncates broad-mark counts disproportionately

The clean source has 6,304,308 genomic 500-bp bins. Integration selects 30,000
abundance-ranked bins for LSI; training reconstructs 15,000 graph-capped bins.

| Assay | Median detected bins/cell: source → 30k → 15k | Overall source counts retained: 30k / 15k |
| --- | --- | ---: |
| H3K27ac | 380 → 94 → 66 | 24.46% / 16.65% |
| H3K27me3 | 626 → 125 → 81 | 21.34% / 13.64% |
| H3K4me1 | 341 → 98 → 64 | 29.15% / 19.09% |
| H3K4me2 | 457 → 257 → 170 | 56.98% / 37.97% |
| H3K4me3 | 518 → 363 → 250 | 70.00% / 48.37% |
| H3K9me3 | 309 → 29 → 18 | 10.59% / 6.52% |

Percentages divide total retained matrix counts by all-clean-cell source counts;
for H3K27ac, this includes the one omitted cell in the denominator. Matched-cell
retention distributions are provided separately in the artifact. Counts are **bin
matrix units**, not unique fragments, reads, FRiP, or measured biological information.

All stage matrices are exact source subsets. All six training LSI arrays are
unchanged from their 30,000-bin inputs: the encoder representation is not computed
solely from the 15,000 reconstructed features. Source matrices have no empty cells;
the later cap introduces one empty H3K27me3 and three empty H3K9me3 reconstruction rows.

The largest H3K9me3 count reduction occurs at the initial 30,000-bin selection,
with further reduction at the graph cap. Saved guidance also covers fewer H3K9me3
genes/bins: 5,786 genes and 9,847/15,000 bins, versus 14,893 genes and 14,738/15,000
bins for H3K4me3. This motivates a mark-aware feature/guidance comparison; it does
not establish that discarded counts contain useful biology or caused misalignment.

**Peak calling is a separate branch.** Fixed-bin construction consumes fragments,
chromosome sizes and clean barcodes—not called peaks. MACS records confirm broad
calling and paired-end mode were off, with no control input. Those settings remain
appropriate subjects for a peak-definition review, but the 51 H3K9me3 narrow peaks
cannot directly explain count truncation in the separate fixed-bin branch.

## RNA quality and annotation provenance are now localized

Exact joins cover all 9,506 raw RNA cells/annotations and 9,462 retained cells.
Saved labels match the harmonization table with zero discrepancies. The 44 omitted
cells exactly equal those with fewer than 200 detected genes; all 44 exceed 20%
mitochondrial counts. **96 retained cells still exceed 20%**, with maximum 87.38%.
This resolves the previous raw-versus-retained denominator question; no exclusion
rule has been applied or validated by this investigation.

Major RNA label groups have coherent expression summaries: median raw counts are
CD3D/CD3E = 6/7 in T cells, MS4A1/CD79A = 16/11 in B cells, and
LST1/LYZ/S100A8/CD14 = 16/306/94/18 in monocytes. These checks reuse the expression
that generated annotations and therefore are not independent accuracy estimates.

All 12 “microglial” labels originate in both the source annotation CSV and its
per-cell match JSON. None of those cells detects TMEM119, P2RY12, TREM2, C1QA/B/C,
LST1 or CD14. GNLY is detected in 9/12, and NKG7 and CX3CR1 in 7/12. Source matches
include competing T/leukocyte assignments, sometimes almost tied (134 versus 132
supporting reference matches, not posterior probabilities). This supports review
of source-reference ambiguity, not an automatic alternative cell type.

Only two source JSON/CSV fine-label differences were found: JSON “native cell”
becomes CSV “cell,” then harmonized “unknown.” This is a label naming difference,
not evidence of barcode corruption. The rare annotation issue does not establish
the cause of widespread chromatin label disagreements.

## Review checks and next decision

Independent agents reviewed lineage, RNA provenance, count retention and the root
label-transfer diagnostic. A second implementation reproduced H3K27me3/H3K9me3
group medians and neighborhood correlations exactly; direct distance calculations
reproduced label/confidence values for 129 queries per mark. Independent source-MTX
and HDF5 arithmetic reproduced H3K9me3 retention totals. Five small diagnostic-contract
tests cover vote semantics, invalid representations and overwrite protection.

The data-audit and critical-review workflows kept **observed processing loss**,
**biological interpretation**, and **proposed fixes** separate. The next useful
experiment is a controlled, existing-data comparison of mark-aware feature/guidance
choices, evaluated against held-out protein agreement and within-assay conservation
alongside mixing. Peak-definition changes need their own comparison. No fix, new
integration run, relabeling, or dataset replacement was implemented in this turn.
