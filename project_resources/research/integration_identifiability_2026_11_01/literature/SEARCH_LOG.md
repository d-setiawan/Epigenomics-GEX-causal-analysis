# Search and access log

Review type: bounded narrative methods/identifiability review. Date: 2026-11-01
UTC. Scope: statistical matching with shared covariates; unlinked positive
identification results; causal direction under restricted structural models;
the stated inputs/objectives of GLUE and scConfluence. English primary papers,
author manuscripts and official method documentation were eligible. This was
not a comprehensive evidence search or a PRISMA review. Empirical effect-size
synthesis, funding/conflict assessment and a citation-tree expansion were
outside scope.

## Discovery sequence

1. Read `/home/dgsetiawan/.agents/skills/literature-review/SKILL.md`.
2. Checked only whether `OPENALEX_API_KEY` existed: it was absent. No credential
   from the conversation was used, echoed, saved or placed in a URL.
3. Attempted `GET https://api.openalex.org/works` with exact parameters:
   `search=statistical matching identification conditional independence`,
   `per_page=100`, `select=id,doi,title,publication_year,primary_location`,
   `sort=relevance_score:desc`. Page 1; no cursor or pagination.
   Sandbox DNS resolution failed; see `openalex_01_search.json`.
4. The identical URL was also inaccessible to the web tool; see
   `openalex_web_attempt.json`. Primary web discovery proceeded while the access
   limitation was documented.
5. One host-network retry of the identical unauthenticated request succeeded
   with HTTP 200: 100 records returned, 30,644 broad matches. Exact timestamp and
   parameters are in `openalex_host_search.json`; the unmodified public JSON is
   in `openalex_host_raw.json`. This corrected the initial diagnosis: the API
   was available, while the first execution context lacked DNS access.
6. The first 25 ranked OpenAlex titles were inspected. The 2026 bridge preprint
   and the older data-fusion identification paper were followed up. Other
   matching/imputation methods were not exhaustively screened. The latter
   candidate’s primary landing page and DOI verification failed, so no claims
   rely on it. The former received one focused full-text pass of its recovery
   theorem, cost assumptions and tuning protocol.

## Supplementary primary discovery queries

The exact web requests and returned responses are preserved in the named JSON
artifacts. Main queries were:

- `statistical matching identification conditional independence D Orazio Scanu missing joint distribution`
- `Unlinked Monotone Regression Balabdaoui Doss Durot 2021 JMLR`
- `Causal Discovery with Continuous Additive Noise Models Peters Mooij Janzing Schölkopf 2014 JMLR`
- `scGLUE Multi-omics single-cell data integration regulatory inference graph-linked embedding scConfluence weakly connected features`
- `scConfluence single-cell diagonal integration regularized Inverse Optimal Transport weakly connected features 51382 Samaran 2024`
- `A Linear Non-Gaussian Acyclic Model for Causal Discovery 2006 JMLR Shimizu`
- `Old and new approaches in statistical matching D Orazio Di Zio Scanu 2010 DOI`

The year in the final discovery query was a search hypothesis, not a verified
publication year, and is not assigned to the Istat manuscript. Two later
site-specific GLUE searches returned mostly irrelevant items; those were not
used. Official documentation supplied the needed precise score definition.

## Selection and primary checks

The focused note uses eight primary works: the Istat manuscript, its authors’
verified 2006 book, three JMLR methods papers, the GLUE and scConfluence papers,
and one explicitly labeled 2026 preprint. Official GLUE documentation supplies
additional implementation-level definitions. Deduplication uses DOI when
available, otherwise canonical JMLR article identity/title. No count of all
internet hits screened or coverage completeness is claimed.

Primary checks included the Istat manuscript p. 2; JMLR20-689 abstract/§7;
Peters Theorem 28 and Examples 24/26; LiNGAM’s stated assumptions;
scConfluence framework/training Methods; GLUE’s official guidance and
regulatory-score definitions; and the bridge preprint’s Theorem 3.2,
Appendix A, §4 “Choice of lambda,” and Appendix B.4.

The GLUE Nature page redirected to an inaccessible identity-provider URL, and
PMC returned a browser check on direct opens. Its DOI/title/authors/year were
verified through the skill checker and primary indexed article record; precise
claims in the note additionally use accessible official author documentation.
No private, assessment or empirical datasets were accessed for this review.

## Verification artifacts

- `primary_discovery_searches.json`: initial exact web query batch/responses.
- `primary_pages_and_followup_searches.json`: canonical journal pages and
  follow-up searches.
- `verified_primary_fulltext.json`: full-text/page access attempts and results.
- `scope_and_assumptions_checks.json`,
  `positive_results_and_nonidentification_primary.json`,
  `final_assumption_checks.json`,
  `integration_objective_and_bridge_checks.json`: focused source checks.
- `citation_verification.json`: results of the supplied skill’s
  `scripts/verify_citations.py`, with DOI handle checks, available Crossref
  metadata and journal URL checks. Five DOIs and three canonical JMLR URLs
  were checked. Metadata omissions/errors are reported, not filled by guessing.

No model fitting, simulation generation, automated literature polling or
algorithm modification was performed.
