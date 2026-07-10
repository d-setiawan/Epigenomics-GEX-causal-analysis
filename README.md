# Epigenomics-GEX Causal Analysis

> **New here? Read [MISSION.md](MISSION.md) first** — the scientific intent and *why this is novel*
> (recovering the local causal **architecture** among a gene's regulatory elements, not just
> enhancer→gene links). This README is the operational hub; [ROADMAP.md](ROADMAP.md) is the live
> plan, decisions, and status.

Single-cell gene expression (scRNA) + single-cell CUT&Tag histone marks → **local causal graphs**
for monocyte gene regulation. For one target gene we learn a causal graph over its nearby chromatin
peaks, observed confounders, and its expression, then validate edges against orthogonal biological
evidence. What makes this more than enhancer→gene link prediction is spelled out in
[MISSION.md](MISSION.md).

## Pipeline at a glance

`CUT&Tag preprocessing → scGLUE joint integration → metacell pairing → nearby-peak matrices → causal discovery (FCI/GFCI · PC · DAGMA) → 3-axis edge validation`

| Stage | Area | Docs |
| --- | --- | --- |
| raw CUT&Tag → peaks + matrices | [`sc_scripts/`](sc_scripts) | [sc_scripts/README.md](sc_scripts/README.md) |
| cross-modality integration (scGLUE `joint_v2`) | [`integration/`](integration) | [integration/README.md](integration/README.md) |
| pairing · nearby-peak matrices · discovery · validation | [`CausalDiscovery/`](CausalDiscovery) | [CausalDiscovery/README.md](CausalDiscovery/README.md) · [PIPELINE.md](CausalDiscovery/PIPELINE.md) |
| pinned conda envs per stage | [`envs/`](envs) | [envs/README.md](envs/README.md) |

## Current setup (see [ROADMAP.md](ROADMAP.md) for live status)

- Genome **GRCh38**; cell type **monocyte** (CD14+/classical); identifiability target **local PAG/MAG** (no causal sufficiency assumed).
- Integration: **scGLUE `joint_v2`** (preferred over the scMRDR comparison for modality mixing).
- Aggregation: **non-overlapping metacells** (replaced kNN-averaging — see the aggregation bake-off in `evaluation/`).
- Discovery: **FCI/GFCI → PAG** with bootstrap **stability selection** as the headline output; PC/DAGMA kept as comparisons.
- Validation: **3 orthogonal, cell-type + target-gene aware axes** — prediction (ENCODE-rE2G), genetics (eQTL Catalogue), physical (BLUEPRINT PCHi-C).

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

## Documentation map

**Start here**
- [MISSION.md](MISSION.md) — scientific intent & novelty (the *why*). **Read first.**
- [ROADMAP.md](ROADMAP.md) — living master plan: status, the method decisions, phased actions, changelog.

**How each stage works**
- [sc_scripts/README.md](sc_scripts/README.md) — CUT&Tag preprocessing → peaks + matrices.
- [integration/README.md](integration/README.md) — scGLUE (active) and scMRDR (comparison) tracks.
- [CausalDiscovery/README.md](CausalDiscovery/README.md) — pairing, nearby-peak matrices, the discovery CLI, screening.
- [CausalDiscovery/PIPELINE.md](CausalDiscovery/PIPELINE.md) — end-to-end per-gene driver.
- [envs/README.md](envs/README.md) — pinned conda envs and which env runs which stage.

**Validation & evaluation**
- [CausalDiscovery/evaluation.md](CausalDiscovery/evaluation.md) — evaluation framing (the validation layers and honesty rules).
- [CausalDiscovery/evaluation/](CausalDiscovery/evaluation) — external-source docs, the saved-run audit, the aggregation bake-off, and results tables.

**Research notes** (`notes/` and `CausalDiscovery/notes/`)
- Identifiability derivations, the original local-causal-discovery design, the modeling-design
  exploration (element / latent-factor / confounder), and the aggregation & target-gene literature reviews.
