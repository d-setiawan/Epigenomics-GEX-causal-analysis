# Data sources — where every dataset came from

↑ [Data README](../README.md) · [Central README](../../README.md)

**Owner doc:** this is the single home for *data origin* — the study, accession, or portal each
dataset came from. `docs/provenance.py` traces a result back through the pipeline to a file on disk;
**this file is what turns that file path into a citation.**

Rule: nothing enters `Data/raw/` without a row here. If the origin is unknown, say so explicitly
rather than leaving it blank.

---

## Primary chromatin data — ✅ CONFIRMED (2026-07-20)

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
| `raw/<MARK>/*_HTO_counts.tsv` | 5 hashtags (HTO5/7/9/10/13) multiplexing donors |

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

*One loose end:* our H3K4me1 has 6,393 cells, below the published 8,232 minimum — most likely a
different replicate or a post-QC count. Worth a sentence in the methods, not a reason to doubt the match.

## Primary RNA data — ⚠️ ORIGIN STILL UNRECORDED

**This is a separate dataset and still needs a citation.** It is **not** part of GSE195725.

| Local path | Contents | Origin |
| --- | --- | --- |
| `raw/GexData/GEX_feature_cell_matrix_HDF5.h5` | scRNA-seq, **9,506 cells**, GRCh38, 10x CellRanger HDF5 | **UNKNOWN — to fill in** |
| `raw/GexData/GEX_sample_barcodes.csv` | 9,506 barcodes | same |
| `raw/GexData/cell_type_annotation.tar.gz` | cell-type labels (10x Cloud Analysis output format) | same |

Record: **study/paper (+DOI) · accession · portal URL · download date · sample/library IDs.**

> **This closes the loop on the donor question.** The chromatin data is 5 hashed donors from
> GSE195725; the RNA is a *different* 9,506-cell PBMC experiment. So the two modalities genuinely
> share no individuals — which is exactly what
> [LIMITATIONS L1](../../docs/LIMITATIONS.md#l1-donor-is-not-an-available-variable) states and what
> makes [L2](../../docs/LIMITATIONS.md#l2-pseudo-pairs-are-not-within-individual) (pseudo-pairs are
> cross-individual) a structural fact rather than a suspicion.

<details><summary>Ruled out: scNanoSeq-CUT&Tag</summary>

`docs/papers/Nano-seq_CutTag.pdf` (Li et al., *Nat Methods* 21:2044–2057, 2024) was an earlier
candidate. It is a **nanopore long-read** CUT&Tag method paper with no ADT/HTO multimodal design, and
does not match the 173-antibody panel or the six-mark PBMC structure. **Not the source.**
</details>

---

## Reference genome & annotation

| Resource | Version | Source |
| --- | --- | --- |
| Genome build | **GRCh38 / hg38** | — |
| Chrom sizes | `raw/chromsizes/hg38.chrom.sizes` | UCSC |
| Gene annotation | GENCODE **v49** primary assembly GTF | GENCODE (used by the scGLUE guidance graph) |
| liftOver chains | hg19→hg38 | UCSC, auto-staged by `Evaluation/scripts/liftover.py` |

---

## External validation databases

All are public and rebuildable — each row's builder script regenerates the local table from the
upstream source, so these need no manual archiving.

| Local path | Resource | Upstream source | Rebuild with |
| --- | --- | --- | --- |
| `raw/external/re2g/` | **ENCODE-rE2G** monocyte enhancer→gene links (prediction axis) | ENCODE portal; Gschwind et al. 2023, *Nature* | `Evaluation/scripts/build_re2g_gene_link_support.py` |
| `raw/external/pchic/` | **BLUEPRINT PCHi-C** promoter-capture Hi-C, `Mon` ≥5 (physical axis) | Javierre et al. 2016, *Cell*; `https://osf.io/u8tzp/` (`PCHiC_peak_matrix_cutoff5.tsv`, hg19 → lifted) | `Evaluation/scripts/build_pchic_gene_link_support.py` |
| `raw/external/eqtl/`, `eqtl_tcell/` | **eQTL Catalogue** SuSiE credible sets (genetics axis); monocyte QTD000021 (BLUEPRINT), QTD000081-83 (Fairfax), QTD000069 (CEDAR) | `ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/`; Kerimov et al. | `Evaluation/scripts/build_eqtl_gene_link_support.py` |
| `raw/external/chipatlas/` | **ChIP-Atlas** hg38 — 515 monocyte-relevant TF ChIP experiments, 73 TFs (occupancy axis) | `https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/` | manifest tracked; BEDs fetched by `Evaluation/scripts/build_chip_occupancy.py` |
| *(no local dir — fetched live)* | **JASPAR** CORE vertebrate PWMs + UCSC sequence (motif axis) | `https://jaspar.elixir.no/api/v1/matrix/` | `Evaluation/scripts/scan_peak_motifs.py` |
| *(cache only)* | **ENCODE SCREEN** cCREs, ENCODE portal peaks | ENCODE portal | `causal_cli.py support --fetch-encode-screen` |
| `raw/external/eperturbdb/` | **ePerturbDB** perturbation support | manual download | — **dropped**: 0 hits for our targets; see [ROADMAP](../../docs/ROADMAP.md) Q1 |

Cell-type alignment is tracked per source — only cell-type + target-gene aware sources count toward
edge validation. See [external_sources.md](../../Evaluation/results/external_sources.md).

> **Citation hygiene:** DOIs above are transcribed from project notes and are flagged in
> [MISSION.md](../../docs/MISSION.md) as *verify before formal citation*.
