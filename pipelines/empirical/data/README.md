# Data

↑ [Central README](../../../README.md) · [MISSION](../../../project_resources/research/MISSION.md) (why) · [ROADMAP](../../../project_resources/research/ROADMAP.md) (plan)

**The single source of truth for every input used by Integration, Analysis, and
Evaluation.** Nothing downstream should read sequencing data from anywhere else.

> **Where did each dataset come from?** → **[results/data_sources.md](results/data_sources.md)** —
> study/dataset, accession, and portal for every input. Historical monocyte inputs:
> chromatin = scCUT&Tag-pro (GSE195725, Zhang et al. 2022); RNA = 10x *10k Human PBMC 5' v3 Ultima*.
> They are **separate datasets with no verified donor matching**. The September 10
> provenance correction distinguishes hashtags from donors; see
> [L1](../../../project_resources/research/LIMITATIONS.md#l1-donor-is-not-an-available-variable).

## scMTR-seq pilot (2026-11-01)

The [scMTR-seq S1 pilot](scmtr_seq/README.md), authorized on 2026-11-01, is a
separate paired-reference dataset for evaluating unpaired integration. Its
learner inputs contain only RNA, H3K27ac and H3K4me3, with disjoint training
cells. Pairing and differentiation-day metadata are isolated for evaluation.
The historical monocyte inputs below remain unchanged.

## Biological review (2026-09-10)

This track owns data validity and integration suitability, separately from the
[causal aggregation track](../../../project_resources/research/aggregation_restart_plan.md). Start with the
[biological foundation review](results/biological_foundation_review_2026-09-10.md)
for measured QC, ADT checks, source/method candidates, and links to interpretation limits.
Status and next actions are in [ROADMAP](../../../project_resources/research/ROADMAP.md#two-goals-2026-09-10).

Read-only diagnostics (run in `scrna_r`, from the repository root):

```bash
python pipelines/empirical/data/scripts/audit_biological_inputs.py \
  --out pipelines/empirical/data/outputs/biology_review_2026-09-10/input_audit.json
python pipelines/empirical/data/scripts/audit_adt_label_agreement.py \
  --output pipelines/empirical/data/outputs/biology_review_2026-09-10/adt_label_audit.json
```

Both scripts refuse to overwrite an existing report: use a new output filename
for a repeat. They do not filter, relabel, modify source data, or train models.

## Existing-data investigation (2026-09-11)

The [follow-up investigation](results/biological_investigation_2026-09-11.md)
traces raw-to-integration counts and identities, replays saved label transfer,
checks protein-panel consistency, measures chromatin count retention, and audits
the original RNA annotations. Results and interpretation are kept in that report;
live actions are in [ROADMAP](../../../project_resources/research/ROADMAP.md#biology-investigation-2026-09-11).

From the repository root in `scrna_r`:

```bash
python pipelines/empirical/data/scripts/investigate_adt_lineage.py \
  --output pipelines/empirical/data/outputs/biology_investigation_2026-09-11/adt_lineage.json
python pipelines/empirical/data/scripts/investigate_label_transfer.py \
  --out pipelines/empirical/data/outputs/biology_investigation_2026-09-11/label_transfer_replay.json
python pipelines/empirical/data/scripts/investigate_chromatin_retention.py \
  --out pipelines/empirical/data/outputs/biology_investigation_2026-09-11/chromatin_retention.json
python pipelines/empirical/data/scripts/investigate_rna_annotations.py \
  --out pipelines/empirical/data/outputs/biology_investigation_2026-09-11/rna_annotations.json
python -m unittest discover -s pipelines/empirical/data/tests -p 'test_investigate_label_transfer.py' -v
```

These diagnostics also refuse overwrite. They preserve the source inputs and
production labels; the saved-embedding neighbor searches are not integration training.

**September 16 execution update:** the
[integration alignment baseline](../integration/results/biological_alignment_2026-09-16.md)
adds matched-method and full-cohort protein checks, conditional nulls and a fixed
development/assessment split. Existing ADTs remain evaluation-only, not discovery
inputs. Isolated training files preserve source counts and representations;
their `X` is raw counts and must not be used as scaled-expression data. No raw
data, cell exclusions, production labels or historical models were changed.

## Layout

```
pipelines/empirical/data/
├── raw/        untouched inputs — nothing here is produced by our code
│   ├── H3K27ac/ H3K27me3/ H3K4me1/ H3K4me2/ H3K4me3/ H3K9me3/
│   │              per-mark CUT&Tag fragments + ADT/HTO count tables
│   ├── GexData/  scRNA GEX matrix (h5), sample barcodes, cell-type annotation
│   ├── chromsizes/  hg38.chrom.sizes
│   └── external/    downloaded evidence DBs (see below)
├── scripts/    CUT&Tag preprocessing: raw fragments -> peaks + matrices
├── outputs/    per-mark preprocessing products, one folder per histone mark
├── scmtr_seq/  separate S1 pilot, source receipts and learner/evaluator cohorts
└── results/    interpretation + QC write-ups
```

`raw/` vs `outputs/` is the important split: **`raw/` is what we were given, `outputs/` is what our
code made.** Everything in `outputs/` is reproducible by re-running `scripts/`.

## What's tracked in git

Most of `pipelines/empirical/data/` is gitignored — it is large and regenerable. Tracked: the ADT/HTO count tables,
`chromsizes`, and the ChIP-Atlas experiment manifest. Ignored: fragment files, `GexData/`, all of
`outputs/`, and every downloaded external DB.

## Preprocessing

Six marks, one pass each: `H3K27ac`, `H3K27me3`, `H3K4me1`, `H3K4me2`, `H3K4me3`, `H3K9me3`.
The staged pipeline and the per-mark output files are documented in
[results/preprocessing_notes.md](results/preprocessing_notes.md).

Env: **`scrna_r`** (snapatac2 for the fixed-bin / peak matrices) — see [envs/](../../../envs/README.md).

## External evidence databases (`raw/external/`)

Downloaded reference data used by `pipelines/empirical/evaluation/`. Each regenerates from a builder script:

| Folder | Contents | Rebuild with |
| --- | --- | --- |
| `re2g/` | ENCODE-rE2G monocyte gene links (prediction axis) | `pipelines/empirical/evaluation/scripts/build_re2g_gene_link_support.py` |
| `eqtl/`, `eqtl_tcell/` | eQTL Catalogue credible sets (genetics axis) | `pipelines/empirical/evaluation/scripts/build_eqtl_gene_link_support.py` |
| `pchic/` | BLUEPRINT PCHi-C gene links (physical axis) | `pipelines/empirical/evaluation/scripts/build_pchic_gene_link_support.py` |
| `chipatlas/` | monocyte TF ChIP experiment manifest (occupancy axis) | tracked; BEDs fetched by `pipelines/empirical/evaluation/scripts/build_chip_occupancy.py` |
| `eperturbdb/` | perturbation-based support | manual download |
| `chains/` | UCSC liftOver chains | auto-staged by `pipelines/empirical/evaluation/scripts/liftover.py` |

## Consumers

- [`Integration/`](../integration/README.md) reads `raw/GexData/`, `raw/chromsizes/`, and
  `outputs/<MARK>/` through a symlinked workspace built by
  `pipelines/empirical/integration/scripts/setup_integration_workspace.py`.
- [`Evaluation/`](../evaluation/README.md) reads `raw/external/`.
