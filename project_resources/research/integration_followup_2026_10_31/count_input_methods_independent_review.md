# Bounded independent review — 2026-11-01 UTC

**Pass as an exploratory comparison and conditional research recommendation. No new-input or dataset-modality blocker was found. The acknowledged scConfluence loss-accumulation defect remains an implementation prerequisite for its future three-assay pilot.** This review did not inspect active fit outputs, execute model code, install packages, search new literature, or change production documents/sources.

One wording correction is warranted in [COUNT_INPUT_FOLLOWUP.md](../../../pipelines/simulation/integration_recoverability/COUNT_INPUT_FOLLOWUP.md): lines 27–31 describe both arms as having identical “architecture,” while explicitly checking different input-layer widths. The latent/hidden-layer settings and training procedure can be held fixed; the input-dependent encoder layers necessarily differ. Say that directly. The note already correctly identifies the experiment as an input-pipeline comparison, including normalization, rather than an isolated PCA-rank intervention. This clarification does not warrant interrupting the running fits.

The [alternative-method recommendation](alternative_methods.md), especially lines 26–40, stays within the declared fitting-input boundary: existing assay counts and permitted gene/region annotations define reconstruction inputs and weak alignment costs. It does not import ADTs, paired-cell bridges, atlas labels or additional datasets. Derived gene scores and costs are processing of allowed inputs, not new measurements. This permission does not justify treating annotation links as known regulation; the text correctly calls them proxies and reserves hidden state/pairing for evaluation. A pilot still needs the prospective protocol revision described there.

The scientific limits are appropriately stated. Count likelihoods and avoiding PCA do not establish correspondence; a learned bottleneck and nuisance effects remain. Relaxed/partial transport can represent unmatched populations but cannot identify the correct shared mass or resolve measure-preserving state symmetries by itself. The proposal explicitly requires transported-mass/unmatched-population reporting and separates native partial-coupling results from forced full-pool matching. Thus it does not gain an apparently favorable full-pool transport gap merely by excluding difficult cells. Decoder transfer, geometry preservation and hubness remain complementary diagnostics, not certificates of residual covariance, conditional independence or causal recovery.

The source finding is accurate in the saved pinned capture: scConfluence commit `14d81e2ca679e161ec5c6806103a66c14fcdcd9a`, `model.py` lines 408/417 overwrite the two accumulator variables; lines 426/427 self-add them; lines 428/429 return only the final eligible pair's doubled losses. The recommendation correctly blocks an unmodified three-assay call and does not present unrelated pairwise fits as one shared coordinate system. No training-based conclusion about a corrected implementation follows from this static check.

Dataset triad qualifications pass against the [verified shortlist](dataset_shortlist.md): scMTR-seq observes RNA and both requested marks in the same cells; Paired-Tag observes RNA with one mark per nucleus, not the complete triad; the unpaired brain option retains the P21/P25 mismatch. These are optional external evidence with changed biological context, expressly excluded from the four current fits. Revealed pairs would permit tests of specified **observed** cross-assay dependence; they would not make noisy counts latent biological truth or supply a causal graph.

Reviewed byte identities:

| Artifact | SHA256 |
|---|---|
| COUNT_INPUT_FOLLOWUP.md | `e16821a0fd8154b11899e58fcd518cef342382caf7ba96e82f21cf863a2c03b5` |
| alternative_methods.md | `af0d635c34fb79543915f76665911d687681e29f8d551338561da81be9534aea` |
| dataset_shortlist.md | `9714a08689239f662539e88457cc256ab490c6f002c8bc1330f7de0f37c715e8` |
| Pinned model.py text inside saved source capture | `78cca16d1c9941d693642bd3c1d0144664d746e1cd5d10d884dae74c6e8cf2ac` |

The code hash covers the captured text, not an independently checked Git blob or installed package. This is a bounded interpretation/input-contract review, not a runtime or completed-outcome audit.
