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

- **[check_docs.py](check_docs.py)** — doc-drift checker: link/anchor integrity, stale paths,
  known-false claims, orphaned docs. Runs automatically via the pre-commit hook (below); run it
  manually with `python docs/check_docs.py`. When a fact is corrected, add it to the `FACTS`
  registry so the claim cannot be reintroduced.
- **[provenance.py](provenance.py)** — backtrack any result to the raw data behind it. Walks the
  recorded manifests from a pipeline result up through dataset → pairing → metacells → scGLUE run →
  per-mark matrices → raw CUT&Tag/GEX, labelling every hop `[recorded]` (read from a manifest the
  run wrote) or `[CONVENTION]` (inferred from layout — verify before citing).

  ```bash
  python docs/provenance.py --list              # what can be traced
  python docs/provenance.py --gene CD14         # human-readable lineage
  python docs/provenance.py --gene CD14 --json  # machine-readable
  ```

  Missing files are flagged inline, so it doubles as a reproducibility check on a saved run.

### Enabling the pre-commit hook

The hook lives in version control at [`.githooks/pre-commit`](../.githooks/pre-commit). Each clone
enables it once:

```bash
git config core.hooksPath .githooks
```

It runs `check_docs.py` only when markdown is staged, and blocks the commit on drift
(`git commit --no-verify` bypasses it).
