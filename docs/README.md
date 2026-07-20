# docs/ — project-level documentation

↑ [Central README](../README.md) · ownership rules: [README.md → Documentation ownership](../README.md#documentation-ownership-read-before-editing-any-doc)

Cross-cutting docs that belong to no single pipeline section. Section-specific docs live in
`<Section>/results/`.

## Start here

| Doc | Owns |
| --- | --- |
| **[MISSION.md](MISSION.md)** | why this project is novel — the claim, the competitive positioning. **Read first.** |
| **[ROADMAP.md](ROADMAP.md)** | live status, decisions, open work, changelog. §0.5 = publication push. |
| **[LIMITATIONS.md](LIMITATIONS.md)** | every caveat, L1–L10. The *only* place caveats are written. |

## Theory — [identifiability/](identifiability)

- [project_identifiability_guarantees.md](identifiability/project_identifiability_guarantees.md) — the full guarantee analysis for this project's design.
- [identifiability_guarantees_intro.md](identifiability/identifiability_guarantees_intro.md) — introduction to what is and isn't identifiable here.
- [fci_identifiability_summary.md](identifiability/fci_identifiability_summary.md) — what FCI/PAG output does and does not license.

## Earlier design plans — [planning/](planning)

Superseded in places, kept because they define the target design the project converged back onto.

- [local_multiome_notears_plan.md](planning/local_multiome_notears_plan.md) — the original local-causal-discovery design (`C→A`, `C→Y`, `A→A`, `A→Y`); MISSION §1 still cites it.
- [scGLUE_to_causal_discovery_plan.md](planning/scGLUE_to_causal_discovery_plan.md) — the integration→discovery hand-off plan.

## Reports & references

- [reports/](reports) — LaTeX write-ups (Spring 2026 final report, joint_v2 integration summary, scATAC/scRNA summary).
- [papers/](papers) — reference PDFs (gitignored, not redistributed).

## Tooling

- [check_docs.py](check_docs.py) — doc-drift checker: link/anchor integrity, stale paths, known-false
  claims, orphaned docs. Run `python docs/check_docs.py` before committing doc changes.
