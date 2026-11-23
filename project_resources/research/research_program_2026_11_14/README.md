# Unpaired-assay causal identification research program

Started 2026-11-14. Status: scientific program completed and reviewed;
[mechanical validation and artifact hashes](reviews/final_validation.json)
are recorded separately. Start with the [synthesis](SYNTHESIS.md)
and [complete requirement audit](FINAL_REQUIREMENT_AUDIT.md). Open scientific
assumptions and literature gaps remain explicit; no genomic causal graph is
claimed recovered.

This program asks which observational, latent and causal targets are identified
by the project's existing unpaired assays, under which explicit assumptions,
and which identified targets can be estimated with the available counts.
It extends, rather than replaces, the frozen November 1 studies.

Scientific decisions and final review belong to Astra. Literature, proofs,
implementation, Lean and independent replication use GPT-6.1 Terra workers.
Each substantive proof or implementation requires review by someone other
than its author. No new benchmark worlds or biological datasets are authorized.

## Record layout

- [Requirement and evidence ledger](REQUIREMENTS.md): completion criteria and evidence.
- `theory/`: exact observation/target contract, model classes and proof arguments.
- `literature/`: reproducible searches, primary theorem ledger, applicability and gaps.
- `plans/`: Astra decisions and frozen evaluation acceptance criteria before implementation.
- `experiments/`: justified existing-input computations, immutable protocols and results.
- `lean/`: pinned Lean/Mathlib project, complete proofs, builds and axiom inventory.
- `reviews/`: independent correspondence, implementation and result audits.
- [Synthesis](SYNTHESIS.md): final target-by-model conclusions after review.

Existing published-model replication and count-source studies were inspected
and reused where adequate. Repetition requires an identified scientific gap.
Reused worlds remain development evidence even when a new configuration is frozen.

## Initial scientific decision

No integration score will be interpreted as a proof of coupling or causal
identification. Distinct targets are: full cross-assay joint law; shared-state
associations; latent sources; shared latent causal graph; original genomic
feature graph; and specified directions/effects/equivalence classes. A result
about one target transfers to another only through an explicit argument.

Population identification, objective identification, estimator consistency,
finite-sample recovery and biological credibility receive separate verdicts.
Incompatible information requirements are recorded, not supplied artificially.
