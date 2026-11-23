# Literature coverage and stopping evidence

Status 2026-11-15: the adaptive ranked-query screen is complete under the narrow rule below; the focused per-branch citation crosswalk is recorded, with explicit caps and source gaps. This is a reproducible scoping review, not exhaustive retrieval or a full proof audit of every source.

## What was actually screened

The raw 23 OpenAlex pages contain 2,253 result appearances and2,065 distinct OpenAlex work IDs. The latter is not a deduplicated-publication count. All(query,rank,ID) appearances have nonempty screening decisions; see [accounting receipt](reviews/literature_record_integrity.json). DOI/title-year candidate groups and conflicting metadata are explicitly reconciled in [duplicate audit](openalex/WORK_ID_DUPLICATE_RECONCILIATION.csv). Different-title DOI collisions and uncertain related versions are not silently merged. A MultiCOAP title/COAP abstract mismatch was found; primary source text controls its theorem disposition.

| Retrieval block | Returned appearances | Recorded screening depth |
|---|---:|---|
| Five first pages |500|241 retained records individually abstract/metadata-dispositioned; other appearances have title exclusion or duplicate/related-record decisions. Twelve retained abstracts were initially unavailable, ten recovered through PMC/primary routes and two remain metadata-only. |
| Five second pages |453|384 title-only exclusions,38 abstract/primary dispositions,31 duplicate IDs. |
| Four third pages |400|323 title-only exclusions,36 abstract/primary dispositions,41 duplicate IDs. |
| Four fourth pages |400|318 title-only exclusions,43 abstract/primary dispositions,39 duplicate IDs. |
| Four fifth pages |400|303 title-only exclusions,58 abstract/primary dispositions,39 duplicate IDs. |
| Query 05 sixth page |100|62 title-only exclusions,30 abstract/metadata dispositions,8 duplicate IDs. |

These are appearance-level decisions and selected abstract/primary inspections, not a claim that all 2,253 abstracts or full texts were read. Raw requests/responses and query/sort/page parameters are retained in `openalex/root*_retrieval_manifest.json` and individual page files; decisions are in `MANUAL_ABSTRACT_DISPOSITIONS.csv` and `PAGE2_TRIAGE.csv` through `PAGE6_TRIAGE.csv`. All 241 initial retained IDs and all 38 retained citation IDs reconcile exactly with their disposition ledgers. The original 20 flagged primary followups have theorem/observation dispositions. Primary inspection remains distinct from independent proof review, source-code inspection and executed reproduction.

## Operational relevance and stopping rule

A resetting discovery is a distinct **established current-input-compatible shared-recovery population class or faithful estimator of an established shared-recovery class**. A new within-assay observed-feature DAG method, a count factor package without a shared-recovery argument, or a theorem requiring unavailable observations is still recorded and assessed; it does not automatically reset this particular search criterion. This boundary is a scientific judgment about this project, not a finding that adjacent methods are invalid or useless.

| Query | Retrieved depth | Stopping evidence |
|---|---|---|
|01|Pages 1–2,153 indexed results|Index exhausted at retrieval time. |
|02–04|Through page 5|Pages 4 and5 yielded no additional established class/estimator under the stated shared-recovery boundary. RCI-NB, dNBFA and related real count methods remain separately recorded as adjacent targets or requiring other inputs. |
|05|Through page 6|The static discrete-auxiliary Lachapelle result on page 4 counted as a positive conditional population lead. Pages 5–6 produced no further established class/estimator under the same boundary. Liu–Lee–Gu's bottom result is explicitly inherited from the reviewed DDE result; its distinct higher-order model requires known anchors and proposes MCEM, whose implementation was not located. |

Earlier continuation was productive: page 2 found CoMP, inVAE, cICA and domain-mixture/copula candidates; page 3 found the executable Zhang–Kileel moment estimator. Those findings triggered additional primary/source audits, independent CoMP correction and the approved C03 experiment. Search stopped after declining yield under the prespecified refined relevance boundary, not because a fixed page quota was reached. This is not theorem-family saturation, full citation-index saturation, or a guarantee against a later-ranked relevant paper.

## Citation routes and primary evidence

Current raw focused receipts cover pi-VAE, DDE, López and Ferrari; reusable historical Sturma/Timilsina citation work and further focused routes are mapped in the [ten-branch crosswalk](BRANCH_CITATION_CROSSWALK.md). The 38 retained current citation records are reconciled individually:17 of18 formerly unresolved plausible records had abstracts inspected, one historical source lacked an indexed abstract; some remaining background records are explicitly title-only. Ferrari's index returned 10 of11 backward references and six forward items. López's index returned only two references and no forward records, demonstrating index incompleteness. Primary backward references, author pages and publisher/arXiv searches supplement the index. The crosswalk now records two discovery routes and key backward/forward paths for all ten branches, including a direct sparse-ICA/DICA route. Six additional seed batches contain 463 title-screened appearances, with 78 available full abstracts among 99 selected titles. All 21 missing-indexed-abstract appearances have [individual source dispositions](openalex/focused_citations/MISSING_ABSTRACT_DISPOSITION.md). Some full primary proofs remain inaccessible; Allman and GLUE forward pages are capped at 100. These are bounded relevance judgments, not exhaustive citation saturation. A direct DICA forward citation surfaced the 2026 analytic kink-source theorem; explicit source review challenged its general signed-affine conclusion and separately investigates the exact Laplace subclass. This late finding is retained rather than obscured by the stopping rule.

## Access and interpretive limits

The two initially retained metadata-only sources (algebraic-geometry thesis and predictive-recursion survey), a microbiome thesis and a historical structural-identification citation remain limited at source level; exact-title/alternate searches are logged. Kano–Takai's former publisher403 gap was resolved by an author-associated rendered full text with matching publisher header/DOI; extraction/formula fidelity remains limited and no full proof audit is claimed. Huang–Gu's HTML404 was resolved with its primary PDF. Direct publisher restrictions are not evidence that the paper lacks an accessible source.

Count-native methods such as DCRL and MEITD were taken to approved existing-input applications; other true results need common measurement mechanisms, auxiliary/intervention information, known anchors, linked temporal observations, actual paired views, or unverified structural/source-distribution assumptions. CoMP's differing-feature semantic exclusion was corrected after independent review: the population square theorem needs equal dimension, not identical meanings, and the new E2 corollary relaxes even that width restriction. Its released Gaussian conditional VAE still does not implement the new count-theorem class. The [current disposition](THEOREM_ALGORITHM_DISPOSITION.md) records these distinctions. No fit, local optimization success, good prediction or estimated-Q pass establishes population assumptions or a biological causal graph.
