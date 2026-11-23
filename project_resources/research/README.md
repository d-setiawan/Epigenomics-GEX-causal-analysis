# Research and project documentation

↑ [Central README](../../README.md) · ownership rules: [README.md → Documentation ownership](../../README.md#documentation-ownership-read-before-editing-any-doc)

Project-wide research and decision documents live here. Workflow-specific methods and results remain with their owning pipeline stage.

The latest [nonlinear repair and count-model program](nonlinear_repair_2026_11_15/README.md)
links its [scientific synthesis](nonlinear_repair_2026_11_15/SYNTHESIS.md), completed
count experiments and [independent requirement audit](nonlinear_repair_2026_11_15/reviews/FINAL_REQUIREMENT_AUDIT.md).

The [nonlinear literature recheck](nonlinear_recheck_2026_11_14/ASTRA_REASSESSMENT.md)
corrects the earlier nonlinear coverage and revisits the flagged assumptions.
The preceding [unpaired-assay identification program](research_program_2026_11_14/README.md)
collects the observation and target contract, theorem applicability decisions,
formal proofs and existing-input experiments. Its
[requirement ledger](research_program_2026_11_14/REQUIREMENTS.md) preserves the
earlier review record and links the permanent evidence.

## Start here

| Doc | Owns |
| --- | --- |
| **[MISSION.md](MISSION.md)** | why this project is novel — the claim, the competitive positioning. **Read first.** |
| **[ROADMAP.md](ROADMAP.md)** | live status, decisions, open work, changelog. §0.5 = publication push. |
| **[DECISIONS.md](DECISIONS.md)** | *why* the pipeline is built this way — the reasoning behind each choice. Start here when returning to the project. |
| **[LIMITATIONS.md](LIMITATIONS.md)** | every caveat, L1–L19. The *only* place caveats are written. |

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
  receives; attenuation; why partition instability is a corollary rather than a separate problem.
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

The active integration-recoverability study is now organized with the simulation workflows: [design](../../pipelines/simulation/integration_recoverability/design.md). Project status remains in [ROADMAP](ROADMAP.md#integration-recoverability-design-2026-09-23).

The [completed validation review](integration_validation_2026_10_25/reviews/objective_completion_final_audit_2026_10_26.md)
links the final evidence. Earlier scoped reviews are retained for provenance:

- [Partial matrix review](integration_validation_2026_10_25/reviews/final_matrix_integrity_interpretation_draft_2026_10_26.md), superseded by the final matrix audit.
- [Interpretation review](integration_validation_2026_10_25/reviews/integration_validation_interpretation_review_2026_10_26.md).
- [Execution-wrapper review](integration_validation_2026_10_25/reviews/matrix_wrapper_readonly_review.md).
- [scMRDR collider-state cross-check](integration_validation_2026_10_25/reviews/scmrdr_all_constrained_collider_0_public_state_crosscheck.md).
- [scMRDR linear-chain state cross-check](integration_validation_2026_10_25/reviews/scmrdr_all_linear_chain_0_public_state_crosscheck.md).

## Earlier design plans — [planning/](planning)

Superseded in places, kept because they define the target design the project converged back onto.

- [local_multiome_notears_plan.md](planning/local_multiome_notears_plan.md) — the original local-causal-discovery design (`C→A`, `C→Y`, `A→A`, `A→Y`); MISSION §1 still cites it.
- [scGLUE_to_causal_discovery_plan.md](planning/scGLUE_to_causal_discovery_plan.md) — the integration→discovery hand-off plan.

## Reports & references

- [reports/](../reports) — LaTeX write-ups (Spring 2026 final report, joint_v2 integration summary, scATAC/scRNA summary).
- [papers/](papers) — reference PDFs (gitignored, not redistributed).

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

It runs `check_docs.py` only when markdown is staged, and blocks the commit on drift
(`git commit --no-verify` bypasses it).
