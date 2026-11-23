# Unpaired and diagonal integration: scoped extraction

Reviewed 2026-11-01. This component contains 24 primary method papers: 22 inspected at the methods level and two limited to abstract/indexed evidence (scJoint and the MMD-MA scaling paper). The original MMD-MA methods paper was inspected separately. The selection combines central OpenAlex discovery, nominated representative methods, primary citation following and targeted title searches. It is a scoping sample, not an exhaustive review, an implementation audit or a reproduction of reported experiments.

The structured extraction is [unpaired_methods.json](unpaired_methods.json). Each record separates training information, outputs, validation information, restrictions and the scope of any theorem. “No general causal-identification theorem found” means none was located in the inspected relevant methods; it is not a claim about every appendix, implementation or later variant.

## Main finding

Many methods optimize reconstruction, geometry conservation, distribution alignment, graph links or a combination. An optimum or computationally unique transport plan is not, by itself, a proof that the plan is the biological joint distribution. The separate indistinguishable-model argument applies when all full observed assay laws and all supplied prior information are held fixed. It does not imply that integration lacks useful predictive or biological performance.

The records preserve several important positive and boundary cases:

- **UnpairReg** directly models cross-assay coefficients from unpaired covariances under a linear model, conditional error independence and cis-distance restrictions. It is a meaningful restricted alternative, not merely a visualization method; the inspected methods do not establish unrestricted coefficient uniqueness. [Original paper](https://doi.org/10.1186/s13059-022-02726-7)
- **MATCHER** targets a one-dimensional biological progression and requires a biological choice of trajectory orientation. Its cross-assay relationship results therefore address a restricted state model. [Original paper](https://doi.org/10.1186/s13059-017-1269-0)
- **GLUE** reports substantive regulatory validation. This must remain visible alongside the distinction between guidance-supported predictions and identification of arbitrary direct causal effects. [Original paper](https://doi.org/10.1038/s41587-022-01284-4)
- **SCIM** uses labels for orientation; **MinNet** learns from paired reference cells; **SIMBA** separates unpaired integration from its paired multimodal regulatory demonstration. These information contracts cannot be silently merged into a strict unpaired baseline. [SCIM](https://doi.org/10.1093/bioinformatics/btaa843), [MinNet](https://doi.org/10.1186/s12859-022-05126-7), [SIMBA](https://doi.org/10.1038/s41592-023-01899-8)

Geometry symmetries, orientation choices and partial population overlap are acknowledged in several papers. Priors and structural restrictions can resolve particular ambiguities; their adequacy for the intended biology requires separate justification. The companion theory extraction contains actual positive identification theorems and should govern claims about what is possible in restricted model classes.

## Version and evidence cautions

The scMRDR canonical record is the NeurIPS 2025 proceedings DOI; the inspected text is the later arXiv v2. The scConfluence extraction describes the published method, not the locally patched pilot. Local pilot results do not invalidate the published method. Source access failures are explicit in [unpaired_search_log.md](unpaired_search_log.md); abstract-only records support narrower conclusions.

The parent review performs consolidated citation verification. Local primary-text metadata and retrieval receipts provide an independent source trail. The source manifest binds the extraction, search records and retained public full texts.

