# Data

↑ [Central README](../README.md) · [MISSION](../docs/MISSION.md) (why) · [ROADMAP](../docs/ROADMAP.md) (plan)

**The single source of truth for every input used by `Integration/`, `CausalDiscovery/`, and
`Evaluation/`.** Nothing downstream should read sequencing data from anywhere else.

> **Where did each dataset come from?** → **[results/data_sources.md](results/data_sources.md)** —
> study, accession, and portal for every input. ⚠️ The **primary CUT&Tag/GEX origin is currently
> unrecorded** and needs filling in; it cannot be recovered from the files.

## Layout

```
Data/
├── raw/        untouched inputs — nothing here is produced by our code
│   ├── H3K27ac/ H3K27me3/ H3K4me1/ H3K4me2/ H3K4me3/ H3K9me3/
│   │              per-mark CUT&Tag fragments + ADT/HTO count tables
│   ├── GexData/  scRNA GEX matrix (h5), sample barcodes, cell-type annotation
│   ├── chromsizes/  hg38.chrom.sizes
│   └── external/    downloaded evidence DBs (see below)
├── scripts/    CUT&Tag preprocessing: raw fragments -> peaks + matrices
├── outputs/    per-mark preprocessing products, one folder per histone mark
└── results/    interpretation + QC write-ups
```

`raw/` vs `outputs/` is the important split: **`raw/` is what we were given, `outputs/` is what our
code made.** Everything in `outputs/` is reproducible by re-running `scripts/`.

## What's tracked in git

Most of `Data/` is gitignored — it is large and regenerable. Tracked: the ADT/HTO count tables,
`chromsizes`, and the ChIP-Atlas experiment manifest. Ignored: fragment files, `GexData/`, all of
`outputs/`, and every downloaded external DB.

## Preprocessing

Six marks, one pass each: `H3K27ac`, `H3K27me3`, `H3K4me1`, `H3K4me2`, `H3K4me3`, `H3K9me3`.
The staged pipeline and the per-mark output files are documented in
[results/preprocessing_notes.md](results/preprocessing_notes.md).

Env: **`scrna_r`** (snapatac2 for the fixed-bin / peak matrices) — see [envs/](../envs/README.md).

## External evidence databases (`raw/external/`)

Downloaded reference data used by `Evaluation/`. Each regenerates from a builder script:

| Folder | Contents | Rebuild with |
| --- | --- | --- |
| `re2g/` | ENCODE-rE2G monocyte gene links (prediction axis) | `Evaluation/scripts/build_re2g_gene_link_support.py` |
| `eqtl/`, `eqtl_tcell/` | eQTL Catalogue credible sets (genetics axis) | `Evaluation/scripts/build_eqtl_gene_link_support.py` |
| `pchic/` | BLUEPRINT PCHi-C gene links (physical axis) | `Evaluation/scripts/build_pchic_gene_link_support.py` |
| `chipatlas/` | monocyte TF ChIP experiment manifest (occupancy axis) | tracked; BEDs fetched by `Evaluation/scripts/build_chip_occupancy.py` |
| `eperturbdb/` | perturbation-based support | manual download |
| `chains/` | UCSC liftOver chains | auto-staged by `Evaluation/scripts/liftover.py` |

## Consumers

- [`Integration/`](../Integration/README.md) reads `raw/GexData/`, `raw/chromsizes/`, and
  `outputs/<MARK>/` through a symlinked workspace built by
  `Integration/scripts/setup_integration_workspace.py`.
- [`Evaluation/`](../Evaluation/README.md) reads `raw/external/`.
