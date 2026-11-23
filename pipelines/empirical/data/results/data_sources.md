# Data sources — where every dataset came from

↑ [Data README](../README.md) · [Central README](../../../../README.md)

**Owner doc:** this is the single home for *data origin* — the study, accession, or portal each
dataset came from. `project_resources/research/provenance.py` traces a result back through the pipeline to a file on disk;
**this file is what turns that file path into a citation.**

Rule: nothing enters `pipelines/empirical/data/raw/` without a row here. If the origin is unknown, say so explicitly
rather than leaving it blank.

---

## Provenance correction (2026-09-10)

The July source review correctly identified the likely chromatin study and the
RNA library, but **overstated donor identity**. Its descriptions of “five hashed
donors,” a proven unrelated RNA donor, and “no shared individuals by construction”
are superseded by this correction.

The scCUT&Tag-pro methods describe a PBMC sample divided into five tubes,
hashtagged and pooled. Thus the local `donor_id` column, assigned from the dominant
HTO, records a **hashtag group, not a verified biological donor**. The five HTO
classes are populated locally, but no donor-to-HTO key has been established.
[Published methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9378363/)

The RNA portal describes a single healthy PBMC donor; its source study is separate
from the chromatin study. There is **no verified donor matching or paired-cell
correspondence** across these inputs. Separate datasets alone do not establish
that donor identities differ. The portal was checked on September 10; this is
not a recovered original download date.
[10x dataset](https://www.10xgenomics.com/datasets/10k_Human_PBMC_5p_v3_Ultima)

Chromatin study attribution is supported by the matching panel, assays and file
structure below, but exact local-file → GSM → replicate mappings and download
checksums were not established in the September review. The H3K4me1 cell-count discrepancy remains
unresolved; “different replicate or post-QC” was a hypothesis, not an explanation.
The RNA HDF5 library ID provides a more direct source identifier. Its stored
chemistry is authoritative for this file; inconsistent generic chemistry text on
the current vendor page should be checked against the original library/web summary.

For the current download mapping and checksums, see the [Git transfer guide](../../../../TRANSFER.md).

Accession reconciliation leads verified during the September review (sample descriptions,
**not** checksum-equivalent local-file matches):

| Study sample | GEO record |
| --- | --- |
| H3K4me1 replicate 1 histone | [GSM5849206](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5849206) |
| H3K4me1 replicate 1 ADT | [GSM5849207](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5849207) |
| H3K4me1 replicate 1 HTO | [GSM5849208](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5849208) |
| H3K27ac replicate 1 HTO | [GSM5849223](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5849223) |

The consequences for donor adjustment and pseudo-pair interpretation are maintained
in [L1](../../../../project_resources/research/LIMITATIONS.md#l1-donor-is-not-an-available-variable) and
[L2](../../../../project_resources/research/LIMITATIONS.md#l2-pseudo-pairs-are-not-within-individual).
Measured input/ADT diagnostics are in the
[biological review](biological_foundation_review_2026-09-10.md).

## Primary chromatin data — study attribution recorded 2026-07-20

**Zhang, B., Srivastava, A., Mimitou, E. et al. "Characterizing cellular heterogeneity in chromatin
state with scCUT&Tag-pro." *Nature Biotechnology* **40**, 1220–1230 (2022).**
doi:[10.1038/s41587-022-01250-0](https://doi.org/10.1038/s41587-022-01250-0) ·
GEO: **[GSE195725](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE195725)** · healthy human PBMC

scCUT&Tag-pro is a multimodal assay measuring protein–DNA interactions (CUT&Tag) **and** surface-protein
abundance (ADT) in the same single cell — which is why every mark ships as a histone/ADT/HTO triplet.

| Local path | Contents |
| --- | --- |
| `raw/<MARK>/*_histone_fragments.tsv` | CUT&Tag fragments; 6 marks: H3K27ac, H3K27me3, H3K4me1, H3K4me2, H3K4me3, H3K9me3 |
| `raw/<MARK>/*_ADT_counts.tsv` | 173-antibody TotalSeq-A surface-protein panel (incl. isotype controls) |
| `raw/<MARK>/*_HTO_counts.tsv` | 5 hashtags (HTO5/7/9/10/13); technical sample groups, not verified donors (corrected 2026-09-10) |

### How it was confirmed

Matched against the publication and the GEO record on every checkable dimension:

| Evidence | Our files | Published | |
| --- | --- | --- | --- |
| ADT panel size | **173** antibodies | "optimized panel of **173** antibodies from the Biolegend TotalSeq-A catalog" | exact |
| Histone marks | the six above | the same six (per GSE195725 overall design) | exact |
| File structure | `histone` / `ADT` / `HTO` per mark | 24 GSM samples labelled `histone`, `ADT`, `HTO` | exact |
| Cell counts | 6,393–10,386; **H3K27me3 = 8,232** | "sample sizes ranging from **8,232** to 15,609 cells" | exact at the lower bound |
| Sample type | PBMC | healthy human PBMC | exact |
| Replicates | `_rep1` suffix on H3K27ac, H3K4me1 | replicates present in the series | consistent |

*Unresolved, updated 2026-09-10:* our H3K4me1 has 6,393 cells, below the published
8,232 minimum. The July suggestion of a different replicate or post-QC count has
not been verified; reconcile the exact sample and counting stage before formal citation.

## Primary RNA data — ✅ CONFIRMED (2026-07-21)

**10x Genomics — *10k Human PBMCs, 5' v3, Ultima* (single-cell gene expression demonstration
dataset).** Portal: <https://www.10xgenomics.com/datasets/10k_Human_PBMC_5p_v3_Ultima> · healthy-donor
PBMC · Gene Expression only (**not** multiome).

**Corrected 2026-09-10:** this is a separate dataset from the chromatin data
(GSE195725), with no verified cross-study donor matching. See the dated correction above.

| Local path | Contents |
| --- | --- |
| `raw/GexData/GEX_feature_cell_matrix_HDF5.h5` | scRNA-seq, 9,506 cells × 38,606 genes, GRCh38 |
| `raw/GexData/GEX_sample_barcodes.csv` | 9,506 barcodes |
| `raw/GexData/cell_type_annotation.tar.gz` | cell-type labels (10x Cloud Analysis format) |

### How it was confirmed — exact, from file metadata

The h5's own attributes name the dataset directly (not a probabilistic match):

| h5 attribute | value |
| --- | --- |
| `library_ids` | **`10k_Human_PBMC_5p_v3_Ultima`** — the exact dataset slug in the URL |
| `chemistry_description` | `Single Cell 5' R2-only v3` — matches "5' v3" |
| `software_version` | `cellranger-9.0.1` — matches the portal's "Cell Ranger v9.0.1" |
| genome / feature types | GRCh38 / Gene Expression only |
| cells | 9,506 (a "10k" dataset after CellRanger cell-calling) |

The file was produced directly from that 10x dataset by Cell Ranger 9.0.1.

**Citation note:** this is a 10x Genomics *demonstration* dataset, not a peer-reviewed publication.
Cite it as the vendor dataset (name + URL + "10x Genomics, Cell Ranger 9.0.1, accessed <date>"), and
record the download date. There is no DOI. The `cell_type_annotation.tar.gz` is in 10x Cloud Analysis
format and was likely produced by 10x's annotation pipeline on this dataset; confirm before citing the
labels' provenance separately.

> **July donor conclusion superseded, 2026-09-10.** Identifying the library and
> study does not establish donor identity. Use the corrected provenance section
> above, not the earlier “five donors / no shared individuals” conclusion.

<details><summary>Ruled out: scNanoSeq-CUT&Tag as the chromatin source</summary>

`project_resources/research/papers/Nano-seq_CutTag.pdf` (Li et al., *Nat Methods* 21:2044–2057, 2024) was an earlier
candidate for the **chromatin** data. It is a nanopore long-read CUT&Tag method paper with no ADT/HTO
multimodal design, and does not match the 173-antibody panel or six-mark PBMC structure. Not the source.
</details>

---

## Reference genome & annotation

| Resource | Version | Source |
| --- | --- | --- |
| Genome build | **GRCh38 / hg38** | — |
| Chrom sizes | `raw/chromsizes/hg38.chrom.sizes` | UCSC |
| Gene annotation | GENCODE **v49** primary assembly GTF | GENCODE (used by the scGLUE guidance graph) |
| liftOver chains | hg19→hg38 | UCSC, auto-staged by `pipelines/empirical/evaluation/scripts/liftover.py` |

---

## Separate scMTR-seq pilot (2026-11-01)

The user authorized the [scMTR-seq S1 pilot](../scmtr_seq/README.md) as a separate
integration-reliability dataset. It does not replace the monocyte inputs above.
The source is [GSE277778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277778),
specifically human sublibrary S1 DNA
[GSM8529544](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8529544) and RNA
[GSM8529552](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8529552).
The [study article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12327464/) and pinned
author scripts establish the sublibrary, antibody and RT-tag conventions.
Acquisition URLs, source hashes and author-source commit are recorded in the
cohort directory.

Only RNA, H3K27ac and H3K4me3 processed measurements are acquired for fitting.
Same-cell identifiers and RT/day tags are kept in evaluator artifacts;
training uses disjoint canonical cells with opaque assay-specific IDs. Existing
GENCODE v49 GRCh38 annotations supply fixed promoter windows. Details of
canonical-ID construction, assigned-assay QC, held-out panel masking and
source-selection limits are in the [cohort record](../scmtr_seq/README.md) and
[L19](../../../../project_resources/research/LIMITATIONS.md#l19-scmtr-seq-pilot-agreement-is-not-causal-ground-truth).

---

## External validation databases

All are public and rebuildable — each row's builder script regenerates the local table from the
upstream source, so these need no manual archiving.

| Local path | Resource | Upstream source | Rebuild with |
| --- | --- | --- | --- |
| `raw/external/re2g/` | **ENCODE-rE2G** monocyte enhancer→gene links (prediction axis) | ENCODE portal; Gschwind et al. 2023, *Nature* | `pipelines/empirical/evaluation/scripts/build_re2g_gene_link_support.py` |
| `raw/external/pchic/` | **BLUEPRINT PCHi-C** promoter-capture Hi-C, `Mon` ≥5 (physical axis) | Javierre et al. 2016, *Cell*; `https://osf.io/u8tzp/` (`PCHiC_peak_matrix_cutoff5.tsv`, hg19 → lifted) | `pipelines/empirical/evaluation/scripts/build_pchic_gene_link_support.py` |
| `raw/external/eqtl/`, `eqtl_tcell/` | **eQTL Catalogue** SuSiE credible sets (genetics axis); monocyte QTD000021 (BLUEPRINT), QTD000081-83 (Fairfax), QTD000069 (CEDAR) | `ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/`; Kerimov et al. | `pipelines/empirical/evaluation/scripts/build_eqtl_gene_link_support.py` |
| `raw/external/chipatlas/` | **ChIP-Atlas** hg38 — 515 monocyte-relevant TF ChIP experiments, 73 TFs (occupancy axis) | `https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/` | manifest tracked; BEDs fetched by `pipelines/empirical/evaluation/scripts/build_chip_occupancy.py` |
| *(no local dir — fetched live)* | **JASPAR** CORE vertebrate PWMs + UCSC sequence (motif axis) | `https://jaspar.elixir.no/api/v1/matrix/` | `pipelines/empirical/evaluation/scripts/scan_peak_motifs.py` |
| *(cache only)* | **ENCODE SCREEN** cCREs, ENCODE portal peaks | ENCODE portal | `causal_cli.py support --fetch-encode-screen` |
| `raw/external/eperturbdb/` | **ePerturbDB** perturbation support | manual download | — **dropped**: 0 hits for our targets; see [ROADMAP](../../../../project_resources/research/ROADMAP.md) Q1 |

Cell-type alignment is tracked per source — only cell-type + target-gene aware sources count toward
edge validation. See [external_sources.md](../../evaluation/results/external_sources.md).

> **Citation hygiene:** DOIs above are transcribed from project notes and are flagged in
> [MISSION.md](../../../../project_resources/research/MISSION.md) as *verify before formal citation*.
