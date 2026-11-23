# Independent Lean correspondence review

2026-11-15. Reviewer did not author `lean/RepairCore.lean`. **Accept as a kernel-checked algebraic two-atom special case**, with the count-law and cross-assay scope limits below. This review does not evaluate the C01 implementation or its outcomes.

## Independent build and axiom audit

Using the pinned older project only as the Lean/Mathlib environment, I ran from `project_resources/research/research_program_2026_11_14/lean/`:

```sh
ELAN_HOME=/tmp/epigenomics-elan /tmp/epigenomics-elan/bin/lake env lean --root=../../nonlinear_repair_2026_11_15/lean -o /tmp/RepairCore_independent.olean ../../nonlinear_repair_2026_11_15/lean/RepairCore.lean
```

The command exited 0. Its saved output is [`lean_independent_build.log`](lean_independent_build.log), SHA-256 `1c4853e6679ac5330b10ad4746b5a170f0ee81ff68efbfa9d26d564ce1df0474`. I also copied the same source to `/tmp/RepairCoreAudit.lean`, appended `#print axioms RepairCore.recurrence` and `#print axioms RepairCore.unordered_from_sum_product`, and checked it in the same environment with `lake env lean --root=/tmp /tmp/RepairCoreAudit.lean`; it exited 0. The additional output is [`lean_independent_extra_axioms.log`](lean_independent_extra_axioms.log), SHA-256 `3bd710ae8d15b9edff483a3f0d038639d27be3018b3c691c4f04e2beb4113d3a`. All ten named theorem reports list only `propext`, `Classical.choice`, and `Quot.sound`. Source inspection found no `sorry`, `admit`, or custom `axiom` declarations.

Source SHA-256: `RepairCore.lean` `faf358eaf9f4275491e069c3ac5b5a7e1a9a4cec6dbeaa950a70c0c6793732fc`; its README `886aab177b21a6e65201b86b3bd10797fd629ebf049a2022314e3f552277f706`. Environment pins: older `lean-toolchain` SHA-256 `55e97be96000b5e9e290c9e74482e5e317861499a5540353ce845471bded8cea`; `lake-manifest.json` SHA-256 `bad9d3afb68fa72b110dca4e40a29db96889999014877c98684050ee63d094f9`. The environment is Lean 4.19.0 and Mathlib 4.19.0 at the pinned manifest revision. Independent `.olean` SHA-256 `531bb5b26e85e70e293f78e0b7215bf949df28524881f4db0f1999b808da867b` was written under `/tmp`; no old source was edited.

## Correspondence

`moment w a b n` is the algebraic power sum `w a^n+(1−w)b^n` over reals. `variance_identity` proves `m2−m1²=w(1−w)(a−b)²`, and `variance_positive` needs `0<w<1` and `a≠b`. The recurrence, recovered sum and product, and elementary polynomial root argument support `recover_two_atom`. That theorem assumes both weights lie strictly between zero and one, each model's two atoms differ, and first, second and third moments agree. It then proves atom and weight equality in the same order or after one joint swap. The second model's positivity assumptions are stronger than strictly needed to write some intermediate algebra, but they define a legitimate nondegenerate class. The checked example `w=1/3, a=1, b=4` has the displayed moments 3, 11, 43 and satisfies positivity/distinctness by elementary arithmetic; rate nonnegativity is not encoded in the theorem itself.

For a Poisson rate mixture, these rate power sums correspond to **factorial** moments `E[(Y)_n]`, not ordinary raw moments `E[Y^n]`: for example `E[Y²]=E[Λ²]+E[Λ]`. Conditional Poisson factorial-moment equality, distribution-to-moment existence, and any multivariate observation map are outside this Lean file. The finite positive-rate fixture has finite moments, so the standard separate probabilistic argument applies. `recover_two_atom` is not a proof of the arbitrary mixing-measure Laplace injection, general K recovery or statistical estimation in C01.

`repeated_weight_swap` shows that exchanging atoms at weight 1/2 preserves every power sum; the 1-versus-4 example proves the two rates differ. This formalizes indistinguishability of two labelings in one assay. It illustrates why weight-only *cross-assay* alignment cannot distinguish assignments when weights repeat, but it does not itself formalize two full assay laws or two distinct completed cross-assay joints. `distinct_weight_match` is a local contradiction from two distinct numbers both equaling one number; it is not a theorem identifying all finite-mixture matchings.

`scalar_contrast` correctly solves `l t − a = lt tt − aa` as `t=(lt/l)tt+(a−aa)/l` with nonzero linear coefficient when both slopes are nonzero. It captures the scalar affine step of proposal D, not the higher-dimensional matrix theorem, Radon–Nikodym ratios, auxiliary-label positivity, common-support mutual absolute continuity, measurable decoder inverses, or almost-everywhere statements. The multivariate Laplace/characteristic-function proof and intensity-measure identification remain separately reviewed mathematical arguments, not formalized here.

The source's README accurately describes those limitations and the absence of general K, empirical or numerical claims. A build and `#print axioms` establish the checked statements and their dependencies, not that any biological dataset satisfies their hypotheses. No blocking correspondence defect was found within this stated scope.
