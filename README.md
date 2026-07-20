# Epigenomics-GEX Causal Analysis

> **New here? Read [docs/MISSION.md](docs/MISSION.md) first** — the scientific intent and *why this
> is novel* (recovering the local causal **architecture** among a gene's regulatory elements, not
> just enhancer→gene links). This README is the operational hub;
> [docs/ROADMAP.md](docs/ROADMAP.md) is the live plan, decisions, and status.

Single-cell gene expression (scRNA) + single-cell CUT&Tag histone marks → **local causal graphs**
for monocyte gene regulation. For one target gene we learn a causal graph over its nearby chromatin
peaks, observed confounders, and its expression, then test the edges against orthogonal biological
evidence.

## Repository structure

The repo is four pipeline sections plus a docs folder. **Every section has the same three
subfolders**, so you always know where to look:

| subfolder | holds |
| --- | --- |
| `scripts/` | code you run |
| `outputs/` | raw machine output (gitignored — regenerable) |
| `results/` | human-readable interpretation and write-ups |

```
Data/             raw inputs + CUT&Tag preprocessing        -> peak/bin matrices per histone mark
Integration/      scGLUE / scMRDR cross-modality alignment  -> joint latent space, pseudo-pairs
CausalDiscovery/  pairing, nearby-peak matrices, discovery  -> PAGs (FCI/GFCI, PC, DAGMA)
Evaluation/       validation of the discovered edges        -> external support, peak<->peak axes
docs/             MISSION, ROADMAP, theory, reports, papers
envs/             pinned conda envs per stage
```

Data flows left to right: **`Data/` → `Integration/` → `CausalDiscovery/` → `Evaluation/`.**
`Data/` is the single source of truth for inputs to all three downstream sections.

| Section | What it does | README |
| --- | --- | --- |
| [`Data/`](Data) | raw CUT&Tag + GEX + external DBs; preprocessing → peaks & matrices | [Data/README.md](Data/README.md) |
| [`Integration/`](Integration) | RNA ↔ CUT&Tag shared latent space (scGLUE `joint_v2` active) | [Integration/README.md](Integration/README.md) |
| [`CausalDiscovery/`](CausalDiscovery) | metacell pairing, nearby-peak matrices, graph discovery | [CausalDiscovery/README.md](CausalDiscovery/README.md) |
| [`Evaluation/`](Evaluation) | edge validation: external support + element↔element axes | [Evaluation/README.md](Evaluation/README.md) |

## Pipeline at a glance

`CUT&Tag preprocessing → scGLUE joint integration → metacell pairing → nearby-peak matrices → causal discovery (FCI/GFCI · PC · DAGMA) → edge validation`

## Current setup

See [docs/ROADMAP.md](docs/ROADMAP.md) for live status.

- Genome **GRCh38**; cell type **monocyte** (CD14+/classical); identifiability target **local PAG/MAG** (no causal sufficiency assumed).
- Integration: **scGLUE `joint_v2`** (preferred over the scMRDR comparison for modality mixing).
- Aggregation: **non-overlapping metacells** (replaced kNN-averaging — see the aggregation bake-off in `Evaluation/results/`).
- Discovery: **FCI/GFCI → PAG** with bootstrap **stability selection** as the headline output; PC/DAGMA kept as comparisons.
- Validation: **3 orthogonal, cell-type + target-gene aware axes** — prediction (ENCODE-rE2G), genetics (eQTL Catalogue), physical (BLUEPRINT PCHi-C) — plus the element↔element axes in `Evaluation/`.

## Quick start

End-to-end per-gene run (discovery + stability + 3-axis validation):

```bash
python CausalDiscovery/scripts/run_gene_pipeline.py --gene CD14 \
  --dataset-root CausalDiscovery/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_metacell \
  --methods gfci,fci --stability-method fci
```

Individual stages run through the single CLI — `causal_cli.py match | dataset | graph | support`.
See [CausalDiscovery/README.md](CausalDiscovery/README.md) for the CLI and
[CausalDiscovery/PIPELINE.md](CausalDiscovery/PIPELINE.md) for the full driver.

Rebuild the integration input workspace (regenerates all symlinks into `Data/`):

```bash
python Integration/scripts/setup_integration_workspace.py
```

## Documentation map

**Start here**
- [docs/MISSION.md](docs/MISSION.md) — scientific intent & novelty (the *why*). **Read first.**
- [docs/ROADMAP.md](docs/ROADMAP.md) — living master plan: status, method decisions, phased actions, changelog. §0.5 is the publication-push hub.

**How each stage works**
- [Data/README.md](Data/README.md) — inputs and CUT&Tag preprocessing → peaks + matrices.
- [Integration/README.md](Integration/README.md) — scGLUE (active) and scMRDR (comparison) tracks.
- [CausalDiscovery/README.md](CausalDiscovery/README.md) — pairing, nearby-peak matrices, the discovery CLI, screening.
- [CausalDiscovery/PIPELINE.md](CausalDiscovery/PIPELINE.md) — end-to-end per-gene driver.
- [Evaluation/README.md](Evaluation/README.md) — validation axes and the evidence sources.
- [envs/README.md](envs/README.md) — pinned conda envs and which env runs which stage.

**Research notes**
- [docs/identifiability/](docs/identifiability) — identifiability derivations and FCI guarantees.
- [docs/planning/](docs/planning) — earlier design plans (scGLUE→causal, local multiome NOTEARS).
- [docs/reports/](docs/reports) — LaTeX reports (Spring 2026 final report, integration summary).
- [docs/papers/](docs/papers) — reference PDFs (gitignored).
- [CausalDiscovery/results/](CausalDiscovery/results) — modeling-design notes, competitive landscape, literature reviews.
- [Evaluation/results/](Evaluation/results) — validation write-ups and the saved-run audit.
