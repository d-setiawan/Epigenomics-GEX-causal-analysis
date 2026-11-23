# Unpaired-family search and source log

Date: 2026-11-01. Reviewer: scconfluence_adapter.

## Discovery sequence

1. The parent used OpenAlex first and completed seven query result files, each capped at the first 100 records. This component began supplementary discovery only after the parent confirmed the first raw OpenAlex result was available. The capped result sets cannot establish exhaustive coverage.
2. Representative named methods and references in the previously verified scConfluence/GLUE papers supplied initial seeds.
3. Four recorded supplementary web calls used 14 exact queries. Their query strings, dates and returned results are retained in [unpaired_supplementary_searches.json](unpaired_supplementary_searches.json). No query output was treated as sufficient evidence when methods text was accessible.
4. Central OpenAlex DOI matching finds 15 of these 24 papers within the seven capped result sets; [unpaired_discovery_provenance.json](unpaired_discovery_provenance.json) preserves file, rank and identifier. The remaining nine came through nominated seeds, title search or citation following. These numbers describe retrieval provenance, not screening sensitivity.

## Primary verification and access

Primary publisher/PMC pages, official method documentation and original full texts were inspected. For retained Europe PMC sources, exact DOI search parameters, raw metadata responses, fullTextXML URLs, HTTP status, retrieval times and hashes are in [unpaired_sources](unpaired_sources/). Public retrieval used no credentials.

Europe PMC fullTextXML returned HTTP 500 for SCOT, SCOTv2, the MMD-MA scaling paper and scJoint. Browser access supplied the first two papers' relevant methods. Browser access to the latter two remained blocked, so they are explicitly abstract-limited. No statement about the absence of a theorem is inferred solely from their abstracts.

The scConfluence and scMRDR full methods were reused from the preceding focused audit and verified primary browser text. Those accesses and original reference following are retained in [unpaired_primary_reuse.json](unpaired_primary_reuse.json), [unpaired_primary_following.json](unpaired_primary_following.json) and [unpaired_primary_checks.json](unpaired_primary_checks.json). These files preserve tool results, including blocked routes. The separate version-qualified modern-method audit remains under [the earlier reviews](../../integration_identifiability_2026_11_01/reviews/).

## Extraction and checks

The extraction has 24 unique DOI records and all required fields. Full-method evidence means the objective, input contract and relevant claims were inspected; it does not mean every supplement was reread or every proof independently proved. The parent performs one consolidated citation-verifier run for all review components, avoiding redundant DOI traffic.

No new experiments, simulation worlds, model training, paired datasets or credentials were introduced. This component does not derive a universal impossibility claim from performance benchmarks.


## Final date reconciliation

The final synthesis review added three narrow primary-source date checks for Pamona and SIMBA. Exact requests and returned primary headers are retained in unpaired_publication_date_verification.json. Both records now prefer journal issue years and retain their earlier online publication dates; this adds no method to the selection.
