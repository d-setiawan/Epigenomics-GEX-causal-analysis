# Sturma unpaired causal representation pilot

Completed 2026-11-01. The native method recovered shared dimensions and approximated latent coefficients in some simulations of its published continuous linear model. It did not recover shared sources on the small existing count panels tested here. These are separate findings; neither establishes reliable original peak–gene causal recovery.

## Read the results

- [Published-model replication](PUBLISHED_REPLICATION.md): 60 native fits, original saved-statistic comparison, conditional graph errors and an unconditional coefficient criterion.
- [Existing-count application](APPLICATION_RESULTS.md): frozen real/cached-world fits, paired moment evaluation and the exploratory pairwise follow-up.
- [Independent theory and implementation review](reviews/assumption_evaluation_review.md): actual algorithm, theorem assumptions and native-code limitations.
- [Data contract](reviews/data_contract_review.md): existing sources, frozen feature selection, train/evaluator boundary and source audits.
- [Validation and artifact manifest](validation.json): checked links, tests, source identities and reviewed result artifacts.

## What was executed

The [authors' implementation](https://github.com/NilsSturma/multi-domain-crl) is pinned at `578b5900590164b2ed1750fb6a7f98ae7153e220` in [upstream](upstream/README.md). Its core files were independently checked against the [official NeurIPS supplement](https://proceedings.neurips.cc/paper_files/paper/2023/file/6c8985579293e0209bdaa4f21bb1d237-Supplemental-Conference.zip). No upstream algorithm was changed.

The method fits ICA within each domain, matches estimated independent source distributions across domains, constructs a joint mixing matrix, and estimates shared latent graph coefficients when shared sources are found. The matched ICA sources are disturbances, not automatically the causally related latent-state coordinates. The count application supplies observed feature columns directly to this pipeline; internal ICA whitening remains necessary.

The published-model arm uses the authors' generator, learner and scoring functions with two or three domains and three shared latent factors. The existing-data arm uses RNA, H3K27ac and H3K4me3 from the scMTR S1 cohort and two cached informative simulation worlds. It receives direct raw or explicitly normalized count columns, not scGLUE embeddings, PCA coordinates, pairing labels, state or causal truth. Evaluation uses previously available heldout references.

This is a bounded replication and application, not a production integration backend. No new biological datasets, modalities or project benchmark worlds were introduced. The only new synthetic draws are the small published-model replication control. All work ran on CPU in the existing environment; no packages were installed or upgraded. Actual versions are recorded in the replication protocol and data manifest.

## Protocols and execution

| Stage | Frozen specification | Runner | Results |
|---|---|---|---|
| Published-model control | [Protocol](published_replication_protocol.json) | [replicate_published_model.py](replicate_published_model.py) | [Summary](published_replication_summary.json), [manifest](published_replication_manifest.json) |
| Existing-data extraction | [Protocol](data_extraction_protocol.json) | [prepare_cached_data_pilot.py](prepare_cached_data_pilot.py) | [Data manifest](data_pilot_v1/manifest.json) |
| Three-assay application | [Protocol](application_protocol.json) | [apply_native.py](apply_native.py) | [Summary](application_v1/summary.json), [provenance](application_v1/provenance.json) |
| Exploratory pairwise checks | [Protocol](pairwise_protocol.json) | [pairwise_followup.py](pairwise_followup.py) | [Summary](pairwise_v1/summary.json), [provenance](pairwise_v1/provenance.json) |

The application runners refuse an existing output directory. Retain current results and create a separately versioned protocol/output path before changing any scientific setting. The replication runner can summarize its sealed records without rerunning fits. See its result report for commands.

The application adds a training-fixed heldout encoder and moment scoring, with focused algebra tests. An omitted transpose was caught during preflight and corrected; the earlier attempt and the frozen protocol's formula erratum are preserved in the [engineering correction receipt](implementation_checks/encoder_correction.json). Final results use the corrected encoder and independently verified native-source parity. Data, method parameters and outcome rules were not tuned in response to that correction.

## Independent checks

- [Published-result replay](reviews/native_replication_audit.json): every native score, denominator and group summary.
- [Original-source extraction replay](reviews/data_extraction_audit.json) and [consumer contract audit](reviews/data_consumer_audit.json).
- [Application code review](reviews/application_code_review.md), [readout replay](reviews/application_readout_audit.json), and [interpretation review](reviews/application_data_and_readout_review.md).
- [Pairwise follow-up review](reviews/pairwise_followup_audit.json).

The review checks validate the executed calculations and information boundaries. They do not establish the identifying assumptions for sparse counts, a causal graph for real data, or general failure of the published theory. The central project claim boundary remains [L18](../LIMITATIONS.md#l18-latent-space-aggregation-does-not-reconstruct-the-missing-joint-distribution).
