# Search log

Date: 2026-11-15. Review type: focused novelty/prior-art audit. English-language searches; no lower date bound. Include theory bearing directly on count-channel injection, unpaired linear alignment, or rectangular observation maps. Adjacent papers are retained only to distinguish their information assumptions. This is not a systematic review with exhaustive title/full-text screening.

## OpenAlex discovery

Endpoint: `https://api.openalex.org/works`. Default relevance sort, `per_page=100`, no cursor, first page only. Selected fields: id, doi, title, publication_year, publication_date, primary_location, referenced_works, cited_by_count, abstract_inverted_index. No credentials were stored. The first sandbox request failed DNS; the same public service was then reachable through the approved network route.

| Query | Returned | Indexed total | Raw response |
|---|---:|---:|---|
| Poisson independent component analysis identifiability | 100 | 20977 | [JSON](raw/oa_01.json) |
| unpaired count data identifiability transport | 100 | 2975 | [JSON](raw/oa_02.json) |
| log linear Poisson identifiability | 100 | 17000 | [JSON](raw/oa_03.json) |
| unpaired multi domain causal representation learning | 100 | 3063 | [JSON](raw/oa_04.json) |
| unpaired observations Poisson | 100 | 5658 | [JSON](raw/oa_05.json) |
| Contrastive Mixture Posteriors counterfactual | 100 | 3490 | [JSON](raw/oa_06.json) |
| nonlinear independent component analysis Poisson | 100 | 139581 | [JSON](raw/oa_07.json) |
| unpaired distribution alignment identifiability symmetry | 100 | 493 | [JSON](raw/oa_08.json) |

Retrieval totals: 800 appearances; 737 unique work IDs; 737 canonical keys using DOI first and normalized title/year only when DOI is absent. This limited deduplication does not merge every preprint/publication version. 14 retrieved work IDs match the selected primary-comparison titles; other full texts were also found through targeted web search. The inventory explicitly marks unreviewed retrievals. These totals are not counts of papers read.

## Root web queries

Default web-search relevance, no domain/recency filter. Exact strings:

1. `"Poisson" "unpaired" "identifiability"`
2. `"log-linear" "independent component analysis" Poisson`
3. `"multi-domain" "count" "identifiability" Sturma`
4. `"counterfactual" "Poisson" "CoMP"`
5. `identifiability unpaired Poisson latent variable model`
6. `"Poisson" "independent component" identifiability log`
7. `"multi-domain" "Poisson" "Sturma"`
8. `"unpaired" "log-linear" latent`
9. `"Unpaired Canonical Correlation Analysis" identifiability`
10. `"Domain Transfer Becomes Identifiable via a Single Alignment"`
11. `"Concept Modulation Models" identifiability`
12. `"Independent Component Discovery in Temporal Count Data" Proposition 3.1`
13. `"Toward the Identifiability of Comparative Deep Generative Models" PMLR`
14. `"Tree-based variational inference for Poisson log-normal models" Chaussard Gassiat 2024`
15. `"Poisson" "unpaired" "ICA" identifiability count transport`
16. `"count" "Unpaired Multi-Domain Causal Representation Learning"`

## Independent unpaired-transport reviewer

Exact recorded queries (2026-11-15; default relevance, no lower date bound):

1. `unpaired domain translation identifiability Xi Bloem Reddy indeterminacy generative models`
2. `Shrestha Fu identifiability unsupervised domain translation distribution preserving`
3. `Sturma unpaired multi domain causal representation learning identifiability`
4. `"Towards Identifiable Unsupervised Domain Translation" pdf`
5. `"Kernel of CycleGAN" principal homogeneous space`
6. `"unpaired" "Poisson" identifiability transport`
7. `"unpaired" "Poisson" "identifiability" ICA`
8. `"unpaired" "log-linear" identifiability`
9. `"Poisson" "Contrastive Mixture of Posteriors"`

## Independent count/ICA reviewer

Exact recorded queries (2026-11-15; default relevance, no lower date bound):

1. `Poisson log linear independent component analysis identifiability`
2. `multivariate mixed Poisson identifiability Laplace transform`
3. `Poisson nonlinear ICA identifiability unpaired`
4. `"Independent Component Discovery in Temporal Count Data"`
5. `"Poisson" "log-linear" "independent component"`
6. `"Poisson" "independent component analysis" "identifiability"`
7. `"multivariate mixed Poisson" "identifiability"`
8. `"Independent component models for replicated point processes"`
9. `"discrete independent component analysis" identifiability multi view`
10. `"Tree-based variational inference for Poisson log-normal models"`
11. `"Beyond CCA: Moment Matching for Multi-View Models"`
12. `"Rethinking LDA" "Discrete ICA"`
13. `"Poisson" "unpaired" "ICA" transport identifiability`
14. `"Identifiability, separability, and uniqueness of linear ICA models" pdf`
15. `"independent component analysis" "more mixtures" "sources" Hyvarinen Oja 2000`
16. `"Poisson" "ICA" "logarithm" identifiability`
17. `"Poisson log-normal" "Lemma 1" identifiability Chaussard`
18. `site.proceedings.neurips.cc pi-VAE 2020 identifiable latent models neural activity`
19. `site.web.stanford.edu pritchard Lopez24 identifiable latent count Poisson`

## Source follow-up and limits

Primary URLs and exact theorem locations are in [README.md](README.md). The count reviewer also screened Gervini2016, Podosinnikova2015/2016, Lim-Kim2024, Luo2021, and Zocher2005; distinct additive-rate, paired-view, observed-covariate or process-law settings prevented treating them as exact substitutes. The transport reviewer checked the November2026 Syrota paper plus the three additional 2026 preprints listed in the report. Root checked count/log-link propositions, CMM kernel assumptions and UCCA pairing assumptions.

Unavailable direct ICLR/OpenReview routes were replaced by primary arXiv texts. Some PDF screenshot requests failed; HTML and extracted text were available. The partial-sharing preprint referred to an appendix absent from its retrieved PDF; only its setup/theorem was classified. OpenAlex pages are capped and index completeness is unverified. No forward-citation saturation or exhaustive priority search is claimed.

Citation links were checked with the literature-review skill verifier; failed HEAD/DOI metadata requests are recorded as access/metadata limits, not as nonexistent publications. Primary theorem claims are based on the cited full texts, not on the HTTP status alone.
