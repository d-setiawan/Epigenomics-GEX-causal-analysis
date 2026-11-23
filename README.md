# Epigenomics-GEX Causal Analysis

This project studies local regulatory structure using existing single-cell RNA
and CUT&Tag histone measurements. Empirical pipelines handle preprocessing,
integration, causal discovery and biological evaluation. Separate simulations
study what structure survives integration and aggregation of unpaired cells.

## Start here

- [Mission](project_resources/research/MISSION.md),
  [decisions](project_resources/research/DECISIONS.md), and
  [limitations](project_resources/research/LIMITATIONS.md) explain the scientific scope.
- [Pipeline map](pipelines/README.md) connects the runnable workflows.
- [Git transfer and reproduction guide](TRANSFER.md) covers environment setup,
  verified input downloads and simulation replay.

## Research status

The latest [exact-model count-source search](pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md)
meets the shared-state transport and cross-assay count recovery criteria in all nine existing
one-, two- and three-source worlds using raw counts, multiple starts and
unpaired calibration selection. Parameter boxes and cumulative optimization
budgets are absent. Individual starts remain unreliable; the report preserves
all failures and independent audits. This is a known-dimension, matched-model
simulation result, not general cross-assay causal-direction recovery.

The [nonlinear repair and count-model program](project_resources/research/nonlinear_repair_2026_11_15/README.md)
contains the expanded literature review, independently checked identification
results and completed experiments on existing count inputs. Its
[synthesis](project_resources/research/nonlinear_repair_2026_11_15/SYNTHESIS.md) and
[requirement audit](project_resources/research/nonlinear_repair_2026_11_15/reviews/FINAL_REQUIREMENT_AUDIT.md)
distinguish conditional population recovery from the empirical results.
Reliable cross-assay causal recovery remains unestablished.

The [nonlinear literature recheck](project_resources/research/nonlinear_recheck_2026_11_14/ASTRA_REASSESSMENT.md)
corrects gaps in the earlier theorem coverage and distinguishes the papers
with assumption issues from other nonlinear identification approaches.

The preceding [unpaired-assay identification program](project_resources/research/research_program_2026_11_14/README.md)
connects the observation contract, primary theorem review, Lean proofs and
existing-input experiments. Its [requirement ledger](project_resources/research/research_program_2026_11_14/REQUIREMENTS.md)
preserves the earlier review record and evidence; the recheck above corrects
its nonlinear coverage. Earlier frozen studies remain intact.

The preceding work is the [count-source extension comparison](project_resources/research/count_extensions_2026_11_01/README.md)
and its [selected-dimension follow-up](project_resources/research/count_rank20_2026_11_01/README.md).
It tests noise models, depth treatment, source estimation and larger nested
feature panels using the existing observations. Independent agents check the
algebra, fit decisions and heldout readouts. Within-assay covariance gains are
reported separately from cross-assay source matching and causal recovery.
Neither completed block accepted a cross-assay source pair under the fixed
rules; raising the computational dimension ceiling did not resolve this.

The preceding [count-aware source pilot](project_resources/research/count_mdcrl_2026_11_01/README.md)
introduced explicit Poisson correction with a matched-dimension comparison.
It improved a narrower within-assay covariance task but did not recover
supported cross-assay sources or a causal graph on the existing panels.

The preceding [Sturma unpaired causal-representation pilot](project_resources/research/sturma_pilot_2026_11_01/README.md) contains
a reduced native replication followed by a frozen application to existing counts,
with independent source, implementation and result reviews. It separates recovery
under the published linear model from the outcome on our sparse count panels.

The [integration literature and identification review](project_resources/research/integration_literature_scope_2026_11_01/REVIEW.md)
distinguishes general nonidentification from published positive results for
restricted unpaired latent models. The [population proof](project_resources/research/theory/07_unpaired_integration_nonidentifiability.md)
establishes why general cross-assay direct-structure recovery can fail even with
perfect state alignment. Its [research record](project_resources/research/integration_identifiability_2026_11_01/README.md)
links independent reviews and symbolic checks. Those theory studies are separate
from the subsequent Sturma fitting experiment.

The [method-specific companion](project_resources/research/theory/08_modern_integration_objectives.md)
connects this proof to the objectives of scGLUE, scMRDR and scConfluence.

The [scConfluence/scMTR-seq pilot](project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)
completed a count-input integration comparison on cached simulations and a
separate real-data holdout, with independent reviews. It exposed integration
seed instability and a sparse-count causal-test calibration problem. It does
not replace the historical monocyte setup or establish reliable causal recovery.

The bounded [integration-validation study](pipelines/simulation/integration_recoverability/INTEGRATION_VALIDATION.md)
is complete; the broader frozen simulation benchmark remains unfinished.
Use its [execution record](pipelines/simulation/integration_recoverability/IMPLEMENTATION_STATUS.md)
for current results and remaining work. State alignment has not established
reliable cross-assay causal recovery.

The empirical biology track retains its
[reviewed alignment baseline](pipelines/empirical/integration/results/biological_alignment_2026-09-16.md).
The [CA-BOSS track](pipelines/simulation/unpaired_multimodal/README.md) is parked;
its code, cached worlds, results and negative controls remain available.
[ROADMAP](project_resources/research/ROADMAP.md) records the project history and
scientific priorities. Result numbers belong to the linked result reports.

## Repository structure

The repository is organized around two workflow families and shared project resources:

```text
pipelines/empirical/    data → integration → analysis → evaluation
pipelines/simulation/   aggregation, unpaired multimodal, and integration-recoverability studies
project_resources/      research, reports, shared outputs, and archive
envs/                   pinned environments used by both workflow families
```

Start with the [pipeline map](pipelines/README.md). The empirical pipeline’s
[`data/`](pipelines/empirical/data/README.md) directory remains the source of
truth for biological inputs. Each integration run has a separate directory
under [`integration/runs/`](pipelines/empirical/integration/runs/) so new method
and parameter experiments can be compared without overwriting one another.
The former combined Evaluation area is divided between empirical validation and
the simulation tracks that produced the formal worlds and runs.

## Pipeline at a glance

`CUT&Tag preprocessing → scGLUE joint integration → metacell pairing → nearby-peak matrices → causal discovery (FCI/GFCI · PC · DAGMA) → edge validation`

## Existing real-data setup

Historical monocyte configuration. The separate scMTR-seq pilot is linked above.
Live status, rationale and open work: **[ROADMAP](project_resources/research/ROADMAP.md)**.

| | |
| --- | --- |
| Genome / cell type | GRCh38 · monocyte (CD14+/classical) |
| Desired graph target | local PAG/MAG; its biological joint is not certified by integration |
| Integration | scGLUE `joint_v2`; biological review tracked in [ROADMAP](project_resources/research/ROADMAP.md#two-goals-2026-09-10) |
| Aggregation | non-overlapping metacells (n=133) |
| Discovery | FCI/GFCI → PAG, bootstrap stability selection as headline |
| Validation | 3 axes peak→gene (rE2G · eQTL · PCHi-C) + element↔element framework |
| Observed confounders | library depth only; donor identity is not verified ([L1](project_resources/research/LIMITATIONS.md#l1-donor-is-not-an-available-variable)) |

## Quick start

End-to-end per-gene run (discovery + stability + 3-axis validation):

```bash
python pipelines/empirical/analysis/scripts/run_gene_pipeline.py --gene CD14 \
  --dataset-root pipelines/empirical/analysis/outputs/datasets/joint_v2/monocyte_cuttag_peak_genes_300kb_metacell \
  --methods gfci,fci --stability-method fci
```

Individual stages run through the single CLI — `causal_cli.py match | dataset | graph | support`.
See [Analysis/README.md](pipelines/empirical/analysis/README.md) for the CLI and
[Analysis/PIPELINE.md](pipelines/empirical/analysis/PIPELINE.md) for the full driver.

Rebuild the integration input workspace (regenerates all symlinks into `pipelines/empirical/data/`):

```bash
python pipelines/empirical/integration/scripts/setup_integration_workspace.py
```

## Documentation ownership (read before editing any doc)

Docs drift when the same fact is written in two places and only one gets updated. **Every fact has
exactly one owner; everything else links to it.** When you update something, update the owner —
never copy the content outward.

| Fact | Owner — the only place it is written | Everyone else |
| --- | --- | --- |
| Why / novelty / competitive positioning | [project_resources/research/MISSION.md](project_resources/research/MISSION.md) | link |
| Status · open work · changelog · one-line decisions | [project_resources/research/ROADMAP.md](project_resources/research/ROADMAP.md) | link |
| **Decision *rationale*** (why, options, cost) | [project_resources/research/DECISIONS.md](project_resources/research/DECISIONS.md) | link |
| **Caveats & limitations** | [project_resources/research/LIMITATIONS.md](project_resources/research/LIMITATIONS.md) | link to `#lN` |
| **Result numbers** | the relevant `*/results/*.md` | link — never restate |
| **Data origin and sample identity** | [Data/results/data_sources.md](pipelines/empirical/data/results/data_sources.md) | link |
| Repo structure · quick start | this README | link |
| How to run one section | that section's README | link |
| Driver stages + axis-aware rule | [Analysis/PIPELINE.md](pipelines/empirical/analysis/PIPELINE.md) | link |

Rules of thumb: a **results doc** carries numbers and gets *superseded*, never silently edited — add
a dated correction instead. The **ROADMAP** carries verdicts and links, not tables. A **caveat**
written outside LIMITATIONS.md is a bug.

Drift is checked automatically by the pre-commit hook (enable once per clone:
`git config core.hooksPath .githooks`), or manually:

```bash
python project_resources/research/check_docs.py
```

## Where did this number come from?

Any result can be traced back to the raw data it derives from:

```bash
python project_resources/research/provenance.py --gene CD14        # --list to see what's traceable
```

It walks the manifests each run wrote — pipeline → dataset → pairing → metacells → scGLUE run →
per-mark matrices → raw CUT&Tag/GEX — and labels every hop `[recorded]` or `[CONVENTION]` so an
inferred link is never mistaken for a documented one. See [project_resources/research/README.md](project_resources/research/README.md#tooling).

## Documentation map

**Start here**

- [project_resources/research/MISSION.md](project_resources/research/MISSION.md) — scientific intent & novelty (the *why*). **Read first.**
- [project_resources/research/DECISIONS.md](project_resources/research/DECISIONS.md) — **why the pipeline looks like this**: each decision, the diagnostic that forced it, the options rejected, and the cost.
- [project_resources/research/ROADMAP.md](project_resources/research/ROADMAP.md) — living master plan: status, method decisions, phased actions, changelog. §0.5 is the publication-push hub.

**How each stage works**

- [Data/README.md](pipelines/empirical/data/README.md) — inputs and CUT&Tag preprocessing → peaks + matrices.
- [Integration/README.md](pipelines/empirical/integration/README.md) — scGLUE baseline under review and scMRDR comparison.
- [Analysis/README.md](pipelines/empirical/analysis/README.md) — pairing, nearby-peak matrices, the discovery CLI, screening.
- [Analysis/PIPELINE.md](pipelines/empirical/analysis/PIPELINE.md) — end-to-end per-gene driver.
- [Evaluation/README.md](pipelines/empirical/evaluation/README.md) — validation axes and the evidence sources.
- [envs/README.md](envs/README.md) — pinned conda envs and which env runs which stage.

**Research notes**

- [Biological foundation review](pipelines/empirical/data/results/biological_foundation_review_2026-09-10.md) — existing-data QC, ADT consistency, and data/integration candidates.
- [Latent-space aggregation](project_resources/research/theory/06_latent_space_aggregation.md) — why integrated state
  geometry does not itself reconstruct the missing same-cell joint distribution, and what the next
  aggregation benchmark must identify.
- **[project_resources/research/README.md](project_resources/research/README.md)** — index of project-level research notes, theory, earlier design plans, reports, and reference papers.
- [project_resources/research/LIMITATIONS.md](project_resources/research/LIMITATIONS.md) — every known caveat, the single source of truth.
- [Analysis/results/](pipelines/empirical/analysis/results) — modeling-design notes, competitive landscape, literature reviews.
- [Evaluation/results/](pipelines/empirical/evaluation/results) — validation write-ups and the saved-run audit.
