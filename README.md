# Epigenomics-GEX Causal Analysis

> **New here, or picking this back up after time away? Read these two:**
> **[docs/MISSION.md](docs/MISSION.md)** — the scientific intent and *why this is novel* (recovering the
> local causal **architecture** among a gene's regulatory elements, not just enhancer→gene links), then
> **[docs/DECISIONS.md](docs/DECISIONS.md)** — *why the pipeline looks the way it does*: the diagnostic
> that forced each choice, the alternatives rejected, and what each one cost.
>
> This README is the operational hub; [docs/ROADMAP.md](docs/ROADMAP.md) is the live plan and status.

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

Fixed choices only. Live status, rationale and open work: **[docs/ROADMAP.md](docs/ROADMAP.md)**.

| | |
| --- | --- |
| Genome / cell type | GRCh38 · monocyte (CD14+/classical) |
| Identifiability target | local PAG/MAG — no causal sufficiency assumed |
| Integration | scGLUE `joint_v2` |
| Aggregation | non-overlapping metacells (n=133) |
| Discovery | FCI/GFCI → PAG, bootstrap stability selection as headline |
| Validation | 3 axes peak→gene (rE2G · eQTL · PCHi-C) + element↔element framework |
| Observed confounders | library depth only — donor is **not available** ([L1](docs/LIMITATIONS.md#l1-donor-is-not-an-available-variable)) |

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

## Documentation ownership (read before editing any doc)

Docs drift when the same fact is written in two places and only one gets updated. **Every fact has
exactly one owner; everything else links to it.** When you update something, update the owner —
never copy the content outward.

| Fact | Owner — the only place it is written | Everyone else |
| --- | --- | --- |
| Why / novelty / competitive positioning | [docs/MISSION.md](docs/MISSION.md) | link |
| Status · open work · changelog · one-line decisions | [docs/ROADMAP.md](docs/ROADMAP.md) | link |
| **Decision *rationale*** (why, options, cost) | [docs/DECISIONS.md](docs/DECISIONS.md) | link |
| **Caveats & limitations** | [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | link to `#lN` |
| **Result numbers** | the relevant `*/results/*.md` | link — never restate |
| Repo structure · quick start | this README | link |
| How to run one section | that section's README | link |
| Driver stages + axis-aware rule | [CausalDiscovery/PIPELINE.md](CausalDiscovery/PIPELINE.md) | link |

Rules of thumb: a **results doc** carries numbers and gets *superseded*, never silently edited — add
a dated correction instead. The **ROADMAP** carries verdicts and links, not tables. A **caveat**
written outside LIMITATIONS.md is a bug.

Drift is checked automatically by the pre-commit hook (enable once per clone:
`git config core.hooksPath .githooks`), or manually:

```bash
python docs/check_docs.py
```

## Where did this number come from?

Any result can be traced back to the raw data it derives from:

```bash
python docs/provenance.py --gene CD14        # --list to see what's traceable
```

It walks the manifests each run wrote — pipeline → dataset → pairing → metacells → scGLUE run →
per-mark matrices → raw CUT&Tag/GEX — and labels every hop `[recorded]` or `[CONVENTION]` so an
inferred link is never mistaken for a documented one. See [docs/README.md](docs/README.md#tooling).

## Documentation map

**Start here**
- [docs/MISSION.md](docs/MISSION.md) — scientific intent & novelty (the *why*). **Read first.**
- [docs/DECISIONS.md](docs/DECISIONS.md) — **why the pipeline looks like this**: each decision, the diagnostic that forced it, the options rejected, and the cost.
- [docs/ROADMAP.md](docs/ROADMAP.md) — living master plan: status, method decisions, phased actions, changelog. §0.5 is the publication-push hub.

**How each stage works**
- [Data/README.md](Data/README.md) — inputs and CUT&Tag preprocessing → peaks + matrices.
- [Integration/README.md](Integration/README.md) — scGLUE (active) and scMRDR (comparison) tracks.
- [CausalDiscovery/README.md](CausalDiscovery/README.md) — pairing, nearby-peak matrices, the discovery CLI, screening.
- [CausalDiscovery/PIPELINE.md](CausalDiscovery/PIPELINE.md) — end-to-end per-gene driver.
- [Evaluation/README.md](Evaluation/README.md) — validation axes and the evidence sources.
- [envs/README.md](envs/README.md) — pinned conda envs and which env runs which stage.

**Research notes**
- **[docs/README.md](docs/README.md)** — index of all project-level docs: theory (identifiability), earlier design plans, LaTeX reports, reference papers.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — every known caveat (L1–L10), the single source of truth.
- [CausalDiscovery/results/](CausalDiscovery/results) — modeling-design notes, competitive landscape, literature reviews.
- [Evaluation/results/](Evaluation/results) — validation write-ups and the saved-run audit.
