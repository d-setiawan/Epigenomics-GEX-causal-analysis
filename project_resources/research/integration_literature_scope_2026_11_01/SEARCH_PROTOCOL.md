# Protocol: integration methods and causal identifiability

_Frozen scope before new discovery, 2026-11-01._

## Question and review type

This is a broad scoping review of single-cell multi-omics integration and the
identification theory relevant to recovering original-feature cross-assay
causal structure. It is not a claim to enumerate every integration paper, a
systematic meta-analysis of performance, or an effort to prove a predetermined
universal negative.

The question is whether a method's actual training information and structural
assumptions identify the missing same-cell cross-assay joint law and the causal
target, or merely select a completion. Both positive results and limitations
are eligible. The current information-limit proofs are in
[Note 07](../theory/07_unpaired_integration_nonidentifiability.md) and
[Note 08](../theory/08_modern_integration_objectives.md).

## Scope and eligibility

- Search cutoff: publications and accessible preprints dated through
  2026-11-01, with older foundational identification work included.
- Language: English sources available to the reviewers; disclose this limit.
- Core: unpaired/diagonal single-cell integration using representations,
  manifold alignment, transport, adversarial methods, or feature guidance.
- Boundary comparisons: paired, partly paired, bridge, factor and anchor methods;
  horizontal integration is labeled separately.
- Theory: statistical matching, unlinked/shuffled regression, nonlinear ICA,
  identifiable representations, causal discovery and explicit causal multimodal
  models where the observation design is relevant.
- Benchmarks: primary evaluations used to determine what published success
  criteria measure, not to infer an impossibility theorem from failures.
- Exclude from method-level claims: secondary summaries without primary support,
  unrelated clinical data fusion, duplicate versions, and unverified method names.

## Discovery and screening

1. Start with OpenAlex: seven thematic queries, relevance ordering, first 100
   records per query, publication-date cutoff, selected metadata and abstracts.
   Save exact parameters, total hits, returned counts and raw responses.
2. Supplement with PubMed and arXiv searches and primary publisher/conference
   pages. Record access failures and any fallback routes.
3. Merge discovery records by DOI, then normalized title/year. Preprint/journal
   version relationships require explicit reconciliation when titles differ.
4. Screen the resulting candidate titles for relevant scope and send named
   families to the assigned reviewers. Records not fully assessed must remain
   labeled as such; retrieval is not equivalent to inclusion or full-text review.
5. Use a bounded citation pass from primary benchmark references to fill missing
   method families, recording each addition. No claim of search saturation is
   made solely because the query caps were reached.

## Extraction and assessment

Each included paper records bibliographic identity and version, method family,
actual training inputs, use of paired cells/bridges/labels/guidance, intended
target, main objective or theorem, evidence location and depth, applicable
assumptions, and the classification of the strict-disjoint proof. Keep author
claims distinct from our mathematical inference. Distinguish biological direct
edges, observed-variable PAGs, latent coordinates, associations and imputation.

The theory assessment asks whether the paper proves identification, under what
observation contract and model class, and whether our two countermodels both
remain admissible. No performance ranking or clinical risk-of-bias score is
appropriate for this question. Record funding/conflicts where available; no
unreported absence is inferred.

Three independent reviewers cover unpaired integration, bridge/paired methods,
and skeptical identification theory. The lead reconciles the evidence matrix,
checks primary benchmark targets and citations, and obtains cross-review of
the final synthesis. No new scientific datasets, simulations or model fits are
part of this review. Only public bibliographic records and papers are retrieved.

## Planned deliverables

A sourced synthesis with a method/evidence matrix, a deduplicated discovery
catalog and explicit screening statuses, exact search logs/raw records, verified
citations, and independent scope reviews. State what the general theorem covers
and retain positive exceptions. Literature coverage is finite even though an
information-theoretic theorem can quantify over every algorithm with the same
input contract.
