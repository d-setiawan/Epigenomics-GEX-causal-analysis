# Research and project documentation

This index connects the empirical results, the identifiability arguments and the experiments that tested them. For a narrative across the studies, start with the [November 22 research recap](../reports/2026-11-22-unpaired-assay-recap.md). For commands and available inputs, start with the [transfer guide](../../TRANSFER.md).

## Start here

| Question | Guide |
| --- | --- |
| What are we trying to recover? | [Mission](MISSION.md) |
| What has been completed and what remains open? | [Roadmap](ROADMAP.md) |
| Why did we change methods or change the target? | [Decisions](DECISIONS.md) |
| Which assumptions limit each claim? | [Limitations and observation boundaries](LIMITATIONS.md) |
| Which files own the detailed results? | [Documentation ownership](../../README.md#documentation-ownership-read-before-editing-any-doc) |

## From unpaired integration to count transport

| Study | How to read the result |
| --- | --- |
| Integration evidence | The [nonidentification study](integration_identifiability_2026_11_01/README.md) and [literature audit](integration_literature_scope_2026_11_01/README.md) distinguish state alignment from same-cell structure. The [scConfluence/scMTR-seq pilot](scconfluence_scmtr_pilot_2026_11_01/README.md) used held-out pairs for evaluation. |
| Published linear source model | The [Sturma replication](sturma_pilot_2026_11_01/README.md), [count-aware pilot](count_mdcrl_2026_11_01/README.md), [extension comparison](count_extensions_2026_11_01/README.md) and [dimension follow-up](count_rank20_2026_11_01/README.md) distinguish within-assay count improvements from evidence for source matching across assays. |
| November 14 identification program | The [study record](research_program_2026_11_14/README.md) and [requirement ledger](research_program_2026_11_14/REQUIREMENTS.md) document the observation model and experiments. The [Lean record](research_program_2026_11_14/lean/README.md) checks finite observationally equivalent models, direct-effect counterexamples and conditions that restrict the joint. The [nonlinear literature recheck](nonlinear_recheck_2026_11_14/ASTRA_REASSESSMENT.md) revises the original coverage assessment. |
| November 15 count-model program | The [study record](nonlinear_repair_2026_11_15/README.md), [synthesis](nonlinear_repair_2026_11_15/SYNTHESIS.md) and [requirement audit](nonlinear_repair_2026_11_15/reviews/FINAL_REQUIREMENT_AUDIT.md) document the analytic transport corollaries and the initial count experiments. The [Lean companion](nonlinear_repair_2026_11_15/lean/README.md) checks the two-atom moment algebra, equal-weight ambiguity and a scalar contrast. The full population transport argument is in the [analytic proof](nonlinear_repair_2026_11_15/theory/COMP_TALL_MATRIX_COROLLARY.md). |
| Finite-sample demonstration | The [initial existing-world test](../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_DEMONSTRATION.md), [exact-model comparison](../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_MATCHED.md) and [count-informed source search](../../pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md) follow the theory studies. The last search passed the specified criteria on nine reused worlds of known dimension. The [novelty audit](count_transport_novelty_2026_11_15/README.md) traces the population result to prior work. |

The two November theory programs and their Lean guides belong to hash-verified historical study trees. Read their statements about pending experiments as of the study date. The finite-sample reports above document the subsequent estimation work. For the scope of the independent checkout replay, see the [November 22 verification record](../../reproduce/VERIFICATION.md).

For individual theorem checks in the November 14 record, the [primary-paper locators](research_program_2026_11_14/literature/PRIMARY_LOCATORS.md) link the literature to its assumptions. The [mixture alignment argument](research_program_2026_11_14/theory/MIXTURE_ALIGNMENT_BOUNDARY.md), its [independent review](research_program_2026_11_14/reviews/MIXTURE_ALIGNMENT_INDEPENDENT_REVIEW.md) and the [adversarial synthesis review](research_program_2026_11_14/reviews/SYNTHESIS_ADVERSARIAL_REVIEW.md) explain specific limits. The November 15 [initial inventory](nonlinear_repair_2026_11_15/INITIAL_INVENTORY.md) and [independent literature-coverage review](nonlinear_repair_2026_11_15/reviews/LITERATURE_COVERAGE_INDEPENDENT_REVIEW.md) document the follow-up.

<details>
<summary>Original count-study protocols and audited source snapshots</summary>

The November count studies preserve the [C01 shared-state protocol](nonlinear_repair_2026_11_15/plans/C01_COUNT_STATE_VALIDATION.md), [C02 observed-count DCRL protocol](nonlinear_repair_2026_11_15/plans/C02_DCRL_VALIDATION.md) and [C03 results](nonlinear_repair_2026_11_15/experiments/C03/RESULTS.md). For the design history, compare the [DCRL candidate](nonlinear_repair_2026_11_15/plans/DCRL_CANDIDATE_DESIGN.md), [first C02 plan](nonlinear_repair_2026_11_15/experiments/C02/prefit_revisions/v1/plan_snapshot.md) and [review of that first plan](nonlinear_repair_2026_11_15/reviews/c02_independent_prefit/REVIEW_V1.md). The [Zhang–Kileel proposal](nonlinear_repair_2026_11_15/plans/ZHANG_KILEEL_CANDIDATE.md) records the initial tensor-moment design.

The literature audits saved selected files from the authors’ repositories. The upstream READMEs describe those full repositories and can reference files omitted from the local snapshots. The November 14 sources include [differential independence](research_program_2026_11_14/literature/sources/csdi_code/README.md) and [content–style learning with unknown latent dimensions](research_program_2026_11_14/literature/sources/shrestha_code/README.md). The November 15 sources include [CoMP](nonlinear_repair_2026_11_15/sources/comp_root/README.md), [DCRL](nonlinear_repair_2026_11_15/sources/dcrl_root/README.md), [PT-SEM](nonlinear_repair_2026_11_15/sources/ptsem_root/README.md) and [mixture moment estimation](nonlinear_repair_2026_11_15/sources/zk_meitd_root/README.md).

</details>

## Progress updates — [updates/](../reports/updates)

Point-in-time summaries written for someone outside the day-to-day. Each is a **snapshot**: never
edit an old update to reflect new results — write a new one.

- [2026-07-20 — mentor update](../reports/updates/2026-07-20_mentor_update.md): evaluation layer complete;
  3-axis peak→gene triangulation, the element↔element framework, four negative axes + the
  co-accessibility contrast, and what remains before scaling.

## Consults — [consults/](consults)

Self-contained briefs written to share with an outside expert for advice. Abstracted (no biology),
portable (no repo-relative links in the body).

- [aggregation_brief.md](consults/aggregation_brief.md) — how we construct samples (disjoint metacell
  aggregation in a shared latent space) and the partition-dependence problem, framed against
  Tillman (2009) *Structure learning with independent non-identically distributed data* (ICML).

## Theory — [identifiability/](identifiability)

What FCI's guarantees are, in general.

- [project_identifiability_guarantees.md](identifiability/project_identifiability_guarantees.md) — the full guarantee analysis for this project's design.
- [identifiability_guarantees_intro.md](identifiability/identifiability_guarantees_intro.md) — introduction to what is and isn't identifiable here.
- [fci_identifiability_summary.md](identifiability/fci_identifiability_summary.md) — what FCI/PAG output does and does not license.

## Theory — [theory/](theory)

What *this data construction* does to those guarantees. Derivations with the algebra worked through,
each connected to the measurement that tested it. Start at [theory/README.md](theory/README.md).

- [01_aggregation_algebra.md](theory/01_aggregation_algebra.md) — the covariance the search actually
  receives, attenuation and sampling variation across partitions.
- [02_identifiability.md](theory/02_identifiability.md) — matched-pair assumptions, missing
  cross-assay residual dependence, and the rank bound on between-metacell covariance.
- [03_algorithms.md](theory/03_algorithms.md) — PC / FCI / GFCI / BOSS / BFCI / LV-Heuristic in
  equations, and why two implementations of FCI disagree 5.3×.
- [04_simulation.md](theory/04_simulation.md) — the controlled contrasts, and seven predictions
  against their outcomes.
- [05_synthesis.md](theory/05_synthesis.md) — four repairs, why each failed, and the estimator the
  surviving claim rests on.
- [06_latent_space_aggregation.md](theory/06_latent_space_aggregation.md) — why a shared latent
  space can align biological state without reconstructing missing same-cell covariance; the
  recoverable targets, failure modes and aggregation-aware estimator contract.
- [07_unpaired_integration_nonidentifiability.md](theory/07_unpaired_integration_nonidentifiability.md)
  and [08_modern_integration_objectives.md](theory/08_modern_integration_objectives.md) —
  the general counterexample and its application to audited integration objectives.
- [Integration literature review](integration_literature_scope_2026_11_01/README.md) —
  method inputs and targets, benchmarks, and positive identification results that
  prevent a universal impossibility claim.

## Simulation design

The [original integration design](../../pipelines/simulation/integration_recoverability/design.md) and its [extended simulation plan](integration_recoverability_extended_plan.md) specify the early benchmarks. The [aggregation restart plan](aggregation_restart_plan.md) explains the two research tracks. The [simulation index](../../pipelines/simulation/README.md) gives the current entry points.

The [completed validation review](integration_validation_2026_10_25/reviews/objective_completion_final_audit_2026_10_26.md)
links the final evidence. Earlier scoped reviews are retained for provenance:

- [Partial matrix review](integration_validation_2026_10_25/reviews/final_matrix_integrity_interpretation_draft_2026_10_26.md), superseded by the final matrix audit.
- [Interpretation review](integration_validation_2026_10_25/reviews/integration_validation_interpretation_review_2026_10_26.md).
- [Execution-wrapper review](integration_validation_2026_10_25/reviews/matrix_wrapper_readonly_review.md).
- [scMRDR collider-state cross-check](integration_validation_2026_10_25/reviews/scmrdr_all_constrained_collider_0_public_state_crosscheck.md).
- [scMRDR linear-chain state cross-check](integration_validation_2026_10_25/reviews/scmrdr_all_linear_chain_0_public_state_crosscheck.md).

## Earlier design plans — [planning/](planning)

These plans document the original biological targets and workflow before the observation limits and count-transport tests were established.

- [local_multiome_notears_plan.md](planning/local_multiome_notears_plan.md) — the original local-causal-discovery design (`C→A`, `C→Y`, `A→A`, `A→Y`). See the [current mission](MISSION.md).
- [scGLUE_to_causal_discovery_plan.md](planning/scGLUE_to_causal_discovery_plan.md) — the integration→discovery hand-off plan.

## Reports & references

- [Reports index](../reports/README.md) links the November research recap, the historical PDFs and the July mentor update.
- Local reference PDFs live under `papers/` when available. The [literature inventory](nonlinear_recheck_2026_11_14/NONLINEAR_INVENTORY.md) and [prior-art audit](count_transport_novelty_2026_11_15/README.md) are available in the checkout.

## Tooling

- **[check_docs.py](check_docs.py)** — doc-drift checker: link/anchor integrity, stale paths,
  known-false claims, orphaned documents. Runs automatically via the pre-commit hook (below); run it
  manually with `python project_resources/research/check_docs.py`. When a fact is corrected, add it to the `FACTS`
  registry so the claim cannot be reintroduced.
- **[provenance.py](provenance.py)** — backtrack any result to the raw data behind it. Walks the
  recorded manifests from a pipeline result up through dataset → pairing → metacells → scGLUE run →
  per-mark matrices → raw CUT&Tag/GEX, labelling every hop `[recorded]` (read from a manifest the
  run wrote) or `[CONVENTION]` (inferred from layout — verify before citing).

  ```bash
  python project_resources/research/provenance.py --list              # what can be traced
  python project_resources/research/provenance.py --gene CD14         # human-readable lineage
  python project_resources/research/provenance.py --gene CD14 --json  # machine-readable
  ```

  Missing files are flagged inline, so it doubles as a reproducibility check on a saved run.

### Enabling the pre-commit hook

The hook lives in version control at [`.githooks/pre-commit`](../../.githooks/pre-commit). Each clone
enables it once:

```bash
git config core.hooksPath .githooks
```

It runs `check_docs.py` when Markdown is staged. The checker also scans historical source snapshots and can report unavailable upstream assets, machine-local source-line citations and unlinked audit files there. [Transfer verification](../../reproduce/VERIFICATION.md)
