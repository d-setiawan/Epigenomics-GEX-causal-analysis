# Existing-study integrity check

Checked on 2026-11-14 from the current workspace. This is a mechanical replay
of stored record counts and hashes. It does not rerun scientific fitting,
evaluation or the mathematical arguments in the historical reviews.

| Study | Completed fit records | Evaluation records | Sealed artifact hashes checked | Hash mismatches |
| --- | ---: | ---: | ---: | ---: |
| [Count extensions](../../count_extensions_2026_11_01/README.md) | 225 | 225 | 903 | 0 |
| [Selected-dimension follow-up](../../count_rank20_2026_11_01/README.md) | 54 | 54 | 219 | 0 |

Both scientific-source and supporting-source hashes match their fit seals,
and the evaluator hashes match their evaluation seals. All 279 fit records
have status `completed`. Their saved summaries contain zero equivalent signed
cross-assay moment pairs and zero confirmed pairs. The historical reports
record that CF confirmation and causal extraction were not reached. These
are previous outcomes on reused cohorts, not fresh replication or proof of
nonidentification.

The [machine-readable audit](existing_studies_audit.json) records exact study
paths and fit/evaluation seal hashes. The
[review-receipt check](historical_review_receipts.json) compares the historical
independent-review files with the earlier final-validation record. These
integrity checks do not establish biological assumptions, correctness of the
source data, or completeness of the new research program.

The repository currently has an extensive uncommitted migration. Its status
was inspected and retained; no reset, cleanup, commit or historical-artifact
rewrite was performed during this check.

## Earlier reproduction and theory records

The [earlier-study audit](historical_sturma_theory_literature_audit.json)
checked 130 files in the published Sturma reproduction manifest and 23 files
in its historical final-validation record. All match, including 60 saved
JSON fits and their 60 NPZ artifacts. The preserved independent score replay
reported 43 correct-dimension fits, 15 positive underestimates, two zero-shared
estimates and 21 passes of its fixed coefficient criterion. These are the
historical native scores and criteria, not exact-DAG success rates or new
replication results.

The earlier symbolic-verification source matches its saved record of 22
identities; this audit did not rerun the symbolic argument. The earlier
64-paper ledger's extraction hashes and both frozen theory notes also match.
Two references in an intermediate component manifest point to reviewed
drafts superseded by documented corrections. The existing final review and
reconciliation explain those changes, and all six final artifact hashes in
that reconciliation match. The audit retains both intermediate mismatches
and their resolution; it does not rewrite historical records.

The [P01 input audit](p01_schema_population_audit.json) separately verifies
the nine permitted public training arrays and their exact 30-feature
prefixes. It checks only headers and provenance for the paired reference.
The original nine-node focal outcome panel is evaluator-only; the later
30/300-feature background panels are permitted learner inputs. Different
training and holdout QC rules remain a population-comparability limitation.

## Computational environment

The [resource snapshot](resources.json) reports 20 logical CPU cores, 9.66 GB
available RAM and 573 GB free workspace storage. It was taken inside the
sandbox. Its GPU non-detection does not override the previously verified host
CUDA availability; any new GPU computation requires a current runtime check.

At the initial check, `lean`, `lake` and `elan` were absent from the command path. The usual
home Elan/Lean/Mathlib cache locations were absent, and no Lean project or
toolchain was found in the inspected repository and tool-cache locations.
This is a scoped availability check, not an exhaustive filesystem search.
The Lean worker subsequently provisioned Lean 4.19.0 and Mathlib v4.19.0
and built the [new formal project](../lean/README.md). Its proof and review
records are separate from this initial availability audit.
