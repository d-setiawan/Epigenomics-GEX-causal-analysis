# Search and selection record

Search date and publication cutoff: **2026-11-01**. The [protocol](SEARCH_PROTOCOL.md) was written before discovery. This is a capped scoping review, not exhaustive retrieval or a PRISMA systematic review. No biological data were downloaded.

## Central indexed searches

OpenAlex was the first discovery source. Each query requested the first 100 results in relevance order, with `filter=to_publication_date:2026-11-01`, `per_page=100`, `page=1`, `sort=relevance_score:desc`. Selected fields were `id,title,doi,publication_year,publication_date,type,primary_location,abstract_inverted_index,cited_by_count`. Public unauthenticated requests were sufficient.

| Query | Total indexed hits at retrieval | Returned | Raw response |
|---|---:|---:|---|
| unpaired single-cell multi-omics integration | 8,855 | 100 | [Query 1](raw/openalex_01.json) |
| single cell multimodal integration | 215,439 | 100 | [Query 2](raw/openalex_02.json) |
| single cell integration optimal transport | 441,744 | 100 | [Query 3](raw/openalex_03.json) |
| statistical matching identifiability | 96,481 | 100 | [Query 4](raw/openalex_04.json) |
| causal discovery unpaired multimodal | 1,539 | 100 | [Query 5](raw/openalex_05.json) |
| identifiable multi view nonlinear independent component analysis | 24,804 | 100 | [Query 6](raw/openalex_06.json) |
| benchmark single-cell multi-omics integration | 25,402 | 100 | [Query 7](raw/openalex_07.json) |

[Exact OpenAlex parameters, URLs, timestamps and status](raw/openalex_search_log.json). Sandbox DNS failed on an initial probe; host read-only retrieval succeeded.

The direct NCBI PubMed ESearch endpoint returned HTTP 500 for all three requests. The [failure log](raw/pubmed_search_log.json) preserves exact terms and date filters. No successful PubMed API results are claimed. Equivalent title/abstract searches were then run through Europe PMC, restricted to `SRC:MED` (PubMed-indexed records), with `FIRST_PDATE:[1900-01-01 TO 2026-11-01]`, `pageSize=100`, `resultType=core`, `format=json`, initial `cursorMark=*` and default relevance order.

| Europe PMC query (date and MED filters also applied) | Total hits | Returned | Raw response |
|---|---:|---:|---|
| `(TITLE_ABS:"single-cell" OR TITLE_ABS:"single cell") AND (TITLE_ABS:"multi-omics" OR TITLE_ABS:multimodal) AND TITLE_ABS:integration` | 1,347 | 100 | [Query 1](raw/europepmc_01.json) |
| `(TITLE_ABS:unpaired OR TITLE_ABS:diagonal) AND TITLE_ABS:integration AND (TITLE_ABS:cell OR TITLE_ABS:omics)` | 81 | 81 | [Query 2](raw/europepmc_02.json) |
| `(TITLE_ABS:causal OR TITLE_ABS:identifiability) AND (TITLE_ABS:"data fusion" OR TITLE_ABS:multimodal OR TITLE_ABS:"multi-omics")` | 2,511 | 100 | [Query 3](raw/europepmc_03.json) |

[Exact Europe PMC parameters and timestamps](raw/europepmc_search_log.json). The 81-hit query was fully returned; other large result sets were capped. Broad query hit counts include irrelevant literature and are not counts of eligible papers.

## Supplementary discovery and primary checks

arXiv discovery used these web queries, with primary arXiv/conference follow-up:

```text
site:arxiv.org "unpaired" "causal" "integration"
site:arxiv.org "single-cell" "multi-omics" "integration" identifiability
site:arxiv.org "single-cell" "integration" "benchmark"
```

The [arXiv response](raw/arxiv_discovery_response.json) is retained. These overlapping supplementary hits are not added to the indexed retrieval total. Primary benchmarks and their method references informed a bounded family-coverage pass. Named methods already used by this project were retained even when the capped searches missed them. No saturation claim is made.

Additional exact queries, primary access attempts and reference-following evidence are recorded in:

- [Unpaired search narrative](agents/unpaired_search_log.md), [queries](agents/unpaired_supplementary_searches.json), [selection provenance](agents/unpaired_discovery_provenance.json), and [source manifest](agents/unpaired_manifest.json).
- [Bridge searches](agents/bridge_search_records.json) and [addendum](agents/bridge_search_records_addendum.json).
- [Theory access log, discovery origins and three explicit exclusions](agents/theory_methods.json).
- [Root benchmark searches](raw/root_benchmark_search.json), [initial primary reads](raw/root_primary_benchmarks.json), [follow-up](raw/root_primary_followup.json), [additional sources](raw/root_benchmark_more.json), [final search](raw/root_benchmark_final_search.json), and [abstract-limited checks](raw/root_benchmark_abstracts.json).

## Accounting and deduplication

The indexed searches returned **700 OpenAlex + 281 Europe PMC records = 981 records**. Automatic DOI, then normalized title/year deduplication produced **866 metadata entries**. Publication/preprint/repository versions can remain; this is not a claim of 866 unique intellectual works.

The selected extraction contains **64 papers** (24 unpaired-method papers, 17 paired/bridge/factor papers, 16 theory/boundary papers and seven benchmarks). Matching selected papers back by exact DOI or normalized title links 37 catalog entries to 35 papers; 29 selected papers were added by named-method or primary-reference follow-up under this matching rule. The remaining **829 entries are not fully assessed**, not declared excluded. Version aliases and related OpenAlex records are retained in agent ledgers.

The selected set has 59 methods/argument/theorem inspections and five limited records. The scoping process did not apply two-reviewer screening to every retrieved abstract; independent reviewers extracted different groups and cross-reviewed conclusions. No invented full-text screening or exclusion counts are reported. [Machine-readable accounting](SCREENING_SUMMARY.json), [catalog](DISCOVERY_CATALOG.tsv), [selected records](PAPER_EVIDENCE.json).

Canonical versions were reconciled for scMRDR (NeurIPS 2025 with later inspected arXiv v2), Sturma (NeurIPS 2023), Timilsina and Xi (NeurIPS 2024), Sun (ICLR 2025), Tong (2026 chapter, inspected 2024 preprint), and Liu (2025 Registered Report rather than counting its 2024 preprint again). Online and issue-year differences are noted where relevant. MMD-MA original and scaling papers are distinct studies.

## Citation and evidence checks

The literature-review skill's `verify_citations.py` checked DOI handles and Crossref metadata, or primary URLs where no DOI was assigned. The report records its SHA256 and reuses the bridge reviewer's verified responses rather than requesting them twice. Missing registry metadata is reconciled against primary metadata; DOI existence alone is not taken as evidence for a scientific claim. [Verification report](citation_verification.json), [bibliographic reconciliation](CITATION_CHECKS.md).

The structured extraction preserves the inspected source/version and precise evidence locator. Funding/conflicts were not systematically extracted. Publication status and access limitations are stated; unreported evidence is not treated as negative evidence. No performance meta-analysis, new integration runs, simulations or causal tests were undertaken.
