# Final unpaired-method cross-review

Reviewer: scconfluence_adapter. Date: 2026-11-01.

Reviewed document: ../REVIEW.md, SHA256 `26be6e5c9c02bc801bb2bd45e8803b3953bfb94ba14b30da63c348f8ae516291`.

Verdict: substantive approval of method characterization and theoretical scope, with two minor clarity edits requested below. This is a source-attribution and scope review, not an independent reproof of every theorem or a reproduction of the benchmarks.

## Findings

The synthesis states the relevant nonidentification claim conditionally on the admissible model class and the full supplied information. It explicitly permits recovery from restrictive assumptions without requiring new measurements. It does not infer a universal impossibility result from failed integration experiments.

The strongest positive case is accurately retained: Sturma et al. identify the observed joint law under their assumptions and add shared latent graph recovery under further conditions. Exact observed-law identification is separated from representation ambiguities. The text distinguishes this latent target from named original-feature causal edges and does not assume an extra count-observation layer inherits the theorem.

The treatment of UnpairReg is fair. Its direct coefficient target, restrictive model and positive evaluations remain visible; covariance-factorization ambiguity is presented as a question for a separate identification proof. MATCHER and the other restricted positive routes are not dismissed as mere visualization.

The treatment of GLUE retains its regulatory validation. SCIM's orientation labels, MinNet's paired references, and SIMBA's paired regulatory example versus unpaired integration are distinguished. Published scConfluence claims are separated from the local patched pilot. The artificial-unpairing versus independent-cohort sampling distinction is also preserved.

## Requested wording clarifications

1. Rename the family “Guidance-linked generative embeddings” to “Generative embeddings with alignment,” or equivalent. scMRDR has gene-aligned input features and geometry regularization, but no feature-guidance graph in the inspected formulation. The current row's explanatory text is qualified; its heading could still imply the wrong architecture.
2. State explicitly that scMRDR's canonical publication is NeurIPS 2025 while the inspected author text is arXiv v2 dated 17 January 2026. The synthesis correctly uses the proceedings DOI and Note 08 does cover scMRDR, but a direct version sentence would remove ambiguity.

Neither point changes the central mathematical conclusion. The structured record already contains the publication/version distinction.

## Bibliographic date reconciliation

The two year discrepancies are different publication events, not conflicting papers:

| Method | Online publication | Journal issue | Preferred year in extraction |
|---|---|---|---|
| Pamona | 16 August 2021 | January 2022, Bioinformatics 38(1) | 2022 |
| SIMBA | 29 May 2023 | June 2024, Nature Methods 21(6) | 2024 |

Both now consistently use the publisher's issue citation year, with explicit online-date fields. The SIMBA year changed from the original extraction's online year. Primary verification: [Pamona publisher page](https://academic.oup.com/bioinformatics/article/38/1/211/6353029), [SIMBA publisher page](https://www.nature.com/articles/s41592-023-01899-8). Exact checks are retained in unpaired_publication_date_verification.json.

The source and search ledgers preserve access limitations, so the two abstract-limited method records do not support full-method absence claims. No new papers, training, simulations or biological datasets were added in this final review.

