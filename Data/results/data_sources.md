# Data sources — where every dataset came from

↑ [Data README](../README.md) · [Central README](../../README.md)

**Owner doc:** this is the single home for *data origin* — the study, accession, or portal each
dataset came from. `docs/provenance.py` traces a result back through the pipeline to a file on disk;
**this file is what turns that file path into a citation.**

Rule: nothing enters `Data/raw/` without a row here. If the origin is unknown, say so explicitly
rather than leaving it blank.

---

## Primary data — ⚠️ ORIGIN NOT RECORDED

> **Action needed (only the project owner can supply this).** Nothing in the repo records where the
> primary single-cell CUT&Tag and GEX data came from — no accession, DOI, portal URL, or citation
> appears in any script, config, README, or manifest. It cannot be recovered from the files
> themselves. Until this is filled in, **the primary dataset is uncitable**, which blocks a methods
> section and any data-availability statement.

| Dataset | What it is | Origin |
| --- | --- | --- |
| `raw/<MARK>/*_histone_fragments.tsv` | single-cell CUT&Tag fragments, 6 histone marks (H3K27ac, H3K27me3, H3K4me1, H3K4me2, H3K4me3, H3K9me3) | **UNKNOWN — to fill in** |
| `raw/<MARK>/*_ADT_counts.tsv`, `*_HTO_counts.tsv` | antibody-derived tags + hashtag oligos (5 donors: HTO5/7/9/10/13) | **UNKNOWN — same source presumably** |
| `raw/GexData/GEX_feature_cell_matrix_HDF5.h5` | scRNA-seq expression matrix (PBMC/blood) | **UNKNOWN — to fill in** |
| `raw/GexData/cell_type_annotation.tar.gz` | cell-type labels | **UNKNOWN — to fill in** |

For each, record: **study/paper (+DOI) · accession (GEO/SRA/EGA/ArrayExpress) · portal URL ·
download date · whether it is public or restricted · the specific sample/library IDs used.**

**Candidate, NOT confirmed.** `docs/papers/Nano-seq_CutTag.pdf` is
*scNanoSeq-CUT&Tag: a single-cell long-read CUT&Tag sequencing method…* (Li, Guo, Wu et al., **Nature
Methods** 21:2044–2057, 2024, doi:10.1038/s41592-024-02453-w). It is a **methods** paper for a
nanopore long-read CUT&Tag protocol. It may be the protocol behind this data, the source of the data,
or simply background reading — **do not cite it as the data source without confirming.** Note also
that our data is HTO-multiplexed with ADT counts across 6 marks, which is not obviously the same
design.

Relevant context already established: the modalities do **not** share donors
([LIMITATIONS L1](../../docs/LIMITATIONS.md#l1-donor-is-not-an-available-variable)) — consistent with
RNA and CUT&Tag having been generated separately, possibly from different studies. Recording both
origins separately therefore matters.

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
