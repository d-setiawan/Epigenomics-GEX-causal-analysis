# Biological foundation review — 2026-09-10

↑ [Data README](../README.md) · [Integration README](../../integration/README.md) ·
[Two-goal status](../../../../project_resources/research/ROADMAP.md#two-goals-2026-09-10)

> **Follow-up 2026-09-11:** the [existing-data investigation](biological_investigation_2026-09-11.md)
> tests local identity/count lineage, replays label transfer, measures stage-specific
> count retention, and resolves the raw-versus-retained RNA QC question. This dated
> initial review is preserved; use the follow-up for the newer measurements.

## Decision supported by this review

**Investigate the existing biological alignment and preprocessing before replacing
the data or interpreting the real-data graphs.** The most actionable finding is
disagreement between several RNA-transferred chromatin labels and the protein
measurements already available for those cells. Broad-mark preprocessing is a
second priority. This review does not establish that the original experiment is
poor quality, that a replacement dataset is better, or that a particular alternative
integration will fix the problem.

CA-BOSS is paused, not completed. The separate aggregation agent owns the
[aggregation restart](../../../../project_resources/research/aggregation_restart_plan.md). This document owns
the initial biology-review measurements and candidate comparisons; live actions
belong to [ROADMAP](../../../../project_resources/research/ROADMAP.md#two-goals-2026-09-10), source identity to
[data_sources.md](data_sources.md#provenance-correction-2026-09-10), and interpretation
limits to [L1](../../../../project_resources/research/LIMITATIONS.md#l1-donor-is-not-an-available-variable),
[L2](../../../../project_resources/research/LIMITATIONS.md#l2-pseudo-pairs-are-not-within-individual), and
[L17](../../../../project_resources/research/LIMITATIONS.md#l17-biological-integration-is-not-established-by-modality-mixing).

## Inputs and reproducibility

Inspected the six existing scCUT&Tag-pro marks, their clean-cell metadata and ADTs,
the source 10x RNA count HDF5, the `scglue/joint/joint_v2` count layers and transferred
labels, and the preprocessing/guidance/training/validation scripts. No raw inputs,
cell labels, model weights or causal results were changed; no datasets were
downloaded, simulated or substituted, and no model was trained.

Read-only diagnostics and their saved outputs:

| Diagnostic | Script | Artifact |
| --- | --- | --- |
| Existing count/metadata inventory | [audit_biological_inputs.py](../scripts/audit_biological_inputs.py) | [input_audit.json](../outputs/biology_review_2026-09-10/input_audit.json) |
| ADT markers versus transferred labels | [audit_adt_label_agreement.py](../scripts/audit_adt_label_agreement.py) | [adt_label_audit.json](../outputs/biology_review_2026-09-10/adt_label_audit.json) |

Run commands are in the [Data README](../README.md#biological-review-2026-09-10).
Outputs use exclusive creation. The count audit records its source hash and selected
input paths/statistics, not checksums or structural validation for every matrix.
The ADT audit records source and input hashes, runtime versions and exact transformations.
These machine outputs are local, gitignored artifacts; this report and the scripts
preserve the human-readable evidence and regeneration route.

Independent reviewers checked source provenance, integration assumptions and
alternative methods. A calculation reviewer independently reproduced every raw
RNA count/gene/mitochondrial quantile and confirmed the 10x CSC orientation and
chromatin CSR encodings. The ADT diagnostic separately checked unique joins and
count validity. This is an initial descriptive audit, not complete biological certification.

## Existing chromatin: measured QC

There are **44,434 clean chromatin cells** and **44,433 retained integration cells**;
the difference is one H3K27ac cell. Each integration assay retains 15,000 bins.

| Assay | Demultiplexing input cells | Clean cells | Median fragment rows/cell | Median detected integration bins/cell | Called peaks | Median peak width, bp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H3K27ac | 8,386 | 7,099 | 234 | 66 | 12,044 | 419 |
| H3K27me3 | 8,232 | 7,172 | 390 | 81 | 2,130 | 291 |
| H3K4me1 | 6,393 | 5,683 | 209 | 64 | 2,574 | 279 |
| H3K4me2 | 9,575 | 8,441 | 278 | 170 | 21,475 | 683 |
| H3K4me3 | 10,386 | 8,875 | 317 | 250 | 18,533 | 829 |
| H3K9me3 | 8,304 | 7,164 | 190 | 18 | 51 | 251 |

No duplicate clean-cell barcodes or nonfinite, negative or noninteger stored counts
were found in the scanned integration layers. The graph-capped count matrices have
one zero-count H3K27me3 row and three H3K9me3 rows. Their earlier, larger-feature LSI
representations may still contain information; investigate consistency between the
representation and reconstruction inputs.

The fragment metric counts fragment-file rows per barcode, as implemented by
[`02_count_fragments.py`](../scripts/02_count_fragments.py); it does not sum the
fifth-column multiplicity. Clean selection uses HTO singlets and a fragment threshold
of 100. The audited metadata have ADT/HTO totals, tag classes/margins and fragment
counts, but no recorded FRiP, TSS-enrichment or blacklist-overlap columns.

The peak caller uses the same `macs3 callpeak --nomodel --shift -100 --extsize 200`
settings without `--broad` for every mark:
[`run_cuttag_stage2_call_peaks.sh`](../scripts/run_cuttag_stage2_call_peaks.sh).
The H3K9me3 peak count and narrow intervals therefore motivate a **mark-aware
processing audit**, not an immediate verdict on the raw experiment. Integration
uses fixed bins, so peak calling alone is not established as the cause of the sparse
integration representation. Quantify coverage loss separately at each stage; see L17.

## Existing RNA: raw QC and retained annotations

The source HDF5 contains **9,506 cells × 38,606 features**, all Gene Expression.
Counts are finite, nonnegative and integer, with no duplicate barcodes or zero-count
cells. Median UMIs are **10,532.5**, median detected genes **3,159**, and median
mitochondrial fraction **2.71%**. **140 raw cells exceed 20% mitochondrial counts**;
the maximum is 95.94%. These are descriptive flags, not applied exclusion rules.

The integration label file instead contains **9,462 retained RNA cells**, including
**12 annotated “microglial cell” cells**, two unknowns, and 77 plasma cells. A
microglial annotation in a PBMC preparation warrants marker/provenance review;
it is not evidence that brain-resident cells were sampled. The 140-cell mitochondrial
flag has not been restricted to the retained integration subset. The preprocessing
script has minimum-gene/cell filters, but the inspected path does not perform an
explicit mitochondrial or RNA-doublet screen.

## Existing ADTs: an orthogonal biological consistency check

All **44,433 chromatin label rows** joined to exactly one clean ADT cell using an
assay-specific join, removing only the terminal `.1` from ADT barcodes. Original
and normalized barcodes were unique within assay. One H3K27ac ADT cell has no
retained integration label. The six matrices each contain 173 unique ADT features;
their scanned counts are finite, nonnegative integers. All chromatin labels have
`label_source=rna_knn_transfer`; **1,263 are explicitly unknown** and remain in the audit.

For each cell, calculate `log1p(count) − mean(log1p(count))` across all 173 ADTs,
including controls, then summarize markers by the existing coarse transferred
label. This explicitly defined centered-log transform is not claimed to reproduce
Seurat CLR. Negative values mean below that cell's panel log-average, not absent protein.

| Assay | CD14: monocyte / T median | CD3-1: monocyte / T median | CD19: B median | CD20: B median |
| --- | ---: | ---: | ---: | ---: |
| H3K27ac | 0.899 / 0.096 | 0.243 / 1.419 | −0.455 | −0.392 |
| H3K27me3 | −0.292 / 0.468 | 1.392 / 0.230 | −0.490 | −0.482 |
| H3K4me1 | 0.676 / −0.341 | 0.294 / 2.055 | −0.462 | −0.393 |
| H3K4me2 | 0.902 / −0.314 | 0.268 / 1.931 | −0.449 | −0.331 |
| H3K4me3 | 0.935 / −0.300 | 0.241 / 1.838 | −0.399 | −0.325 |
| H3K9me3 | −0.216 / −0.195 | 0.833 / 0.797 | −0.359 | −0.331 |

Several patterns require investigation:

- H3K27me3's **2,321 monocyte-labeled cells** have raw median CD3-1 count 5 and
  CD14 count 0. Its **696 B-labeled cells** instead have raw median CD14 count 3.
  Among 1,933 monocyte-labeled cells with transfer confidence ≥0.9, centered CD3-1
  and CD14 medians remain 1.411 and −0.300.
- H3K9me3's **1,575 monocyte-labeled cells** have very similar CD14/CD3-1 medians
  to its T-labeled cells. B-labeled cells have higher median centered CD14 (0.694).
- H3K4me1's **723 B-labeled cells** have centered CD3-1 median 2.039; its
  **144 dendritic-labeled cells** have CD19/CD20 medians 1.519/1.702, with raw
  medians 7/9.5. This suggests a possible population correspondence problem.
- Activating marks often show plausible monocyte–T separation, but other labels
  still need checking: H3K4me3's B-labeled cells have raw median CD3-1 7 and
  CD19/CD20 0/0. CD16 is too sparse at these group medians to resolve the discrepancies.

These are descriptive group summaries, **not label accuracy or new annotations**.
The alternative explanations and validation boundaries are in L17. Validate
feature/barcode provenance and antibody background before blaming the integration
or relabeling cells. High neighbor-vote confidence does not resolve this check.

## Integration implementation: what was actually tested

| Finding | Local evidence | Review implication |
| --- | --- | --- |
| RNA labels are transferred by kNN in the same GLUE embedding | [`validate_joint_scglue.py`](../../integration/methods/scglue/validate_joint_scglue.py) | Use the existing held-out ADTs and within-assay structure for an additional biological check |
| Saved `compute_consistency` is false; `consistency_tsv` is null | `joint_v2/train/train_summary.json` | The README example's requested consistency check was not performed for this saved run |
| Chromatin “highly variable” selection ranks total feature abundance | [`preprocess_pilot_scglue.py`](../../integration/methods/scglue/preprocess_pilot_scglue.py) | Review whether this retains cell-state information across marks, particularly broad/sparse marks |
| One promoter window and mark-sign convention used throughout | [`build_guidance_graph_joint_scglue.py`](../../integration/methods/scglue/build_guidance_graph_joint_scglue.py), [`build_guidance_graph_pilot_scglue.py`](../../integration/methods/scglue/build_guidance_graph_pilot_scglue.py) | Review mark-specific feature/gene coverage and shuffled-guidance sensitivity |
| Locus peaks are quantified from overlapping fixed bins | [`export_locus_matrix_scglue_matches.py`](../../analysis/scripts/commands/export_locus_matrix_scglue_matches.py) | Inspect interval/coverage semantics separately from integration quality |

The saved guidance uses 1 kb promoters with a 50 kb extension and a 15,000-feature
cap per chromatin mark. Positive signs are assigned to H3K27ac/H3K4me1/2/3 and
negative signs to H3K27me3/H3K9me3. Those choices require biological sensitivity
checks rather than being treated as regulatory observations (L17).

A limited check of existing CD14/CSF1R/IL1B k10 peak-to-bin maps examined 43/19/24
intervals and found no identical-bin aliases or shared-bin peak pairs. Therefore
bin duplication has **not** been shown to explain those existing results.

## Data candidates: suitability, not a quality ranking

The bounded search did **not verify a superior drop-in replacement** providing
the same six PBMC histone marks and suitably matched RNA. These are candidates
for later assessment, not a finding that none exists.

| Candidate | Why inspect it | Compatibility / next check |
| --- | --- | --- |
| Complete the existing [GSE195725](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE195725) replicate inventory | Local H3K27ac and H3K4me1 inputs include `rep1`, but no `rep2` triplets were found | Same study and assay set; reconcile exact samples and technical versus biological replication. More cells/replicates do not prove better per-cell quality; [H3K4me1 replicate 2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5849209) is an accession lead |
| [GSE164378 CITE-seq reference](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) | RNA and protein reference used in the original scCUT&Tag-pro analysis; permits a protein bridge | Includes baseline and post-vaccination time points: select documented baseline samples for resting PBMC comparisons. Not paired histone–RNA or donor-matched to our histone cells; added RNA+ADT would be a separate input arm |
| [GSE157910 H3K27me3 PBMC sample](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5034342) | Possible mark-specific technical QC comparator | One mark only, not six-mark replacement; conflicting sample metadata need resolution first |

The original scCUT&Tag-pro study used proteins and chromatin jointly and aligned
to a CITE-seq reference through shared proteins; its reference-imputed histone
profiles are not measured histone–RNA pairs. This makes the **existing ADTs** a
particularly relevant validation resource for our histone–RNA-only baseline.
[Original study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9378363/)
The candidate reference's vaccination context is documented by
[Hao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238499/); baseline means
pre-boost in that experiment, not necessarily vaccine-naïve.

## Integration candidates: test biological conservation, not mixing alone

1. **Keep scGLUE as the reviewed baseline.** It already addresses unpaired
   modalities using a guidance graph; shared-space integration is not new by
   itself. First assess mark-aware inputs/guidance and independent protein
   concordance under the unchanged input contract.
   [scGLUE paper](https://www.nature.com/articles/s41587-022-01284-4)
2. **scConfluence is a relevant alternative for a controlled comparison.** It
   uses modality-specific autoencoders and unbalanced optimal transport with weak
   feature connections, addressing unequal or missing cell populations. Its
   published multi-omic examples do not establish success for our six histone
   assays. Relevance to this project's population mismatch is an inference,
   not a benchmark result.
   [Samaran, Peyré and Cantini, 2024](https://www.nature.com/articles/s41467-024-51382-x)
3. **Pamona provides a partial-alignment sensitivity comparator.** Partial
   Gromov–Wasserstein alignment is relevant when populations do not completely
   overlap; geometry alone is not a guarantee of correct biological matching.
   [Pamona paper](https://academic.oup.com/bioinformatics/article/38/1/211/6353029)

A protein-assisted integration modeled on the source study is a separate,
biologically motivated option, but uses information beyond histone–RNA inputs
and consumes the current orthogonal validation channel. Do not silently promote
it as success under the existing discovery contract. No new data arm or integration
method has been executed here. The first next action is to establish why the
existing protein/label correspondence is inconsistent.
