# P01 independent implementation and result review

Reviewed 2026-11-14 by a worker who did not implement P01. Read the approved plan, protocol, source and prediction seals before inspecting outcomes. Independent replay code is `p01_independent_replay.py`; full machine-readable hashes and numeric results are in `P01_INDEPENDENT_RECEIPT.json`.

## Decision

Accept the numerical result as an exploratory computation of coupling intervals for the supplied empirical training marginals, with the pre-existing limitations in P01. There is no identified arithmetic, feature-index, training-input or evaluator defect in the checked artifacts. The results do not estimate confidence coverage or identify causal directions.

## Independent replay

Without importing the implementation, independently loaded all nine sealed permitted training arrays and the manifest; checked their SHA-256 values, nonnegative integral counts, complete 300 feature IDs and the specified first-30 columns. Calculated `count > 0` marginals, then verified all 8,100 unique dataset/assay/index records, both feature IDs, bounds, independence completion, covariance bounds, widths and nonnegative endpoint cells. Maximum absolute disagreement was 0. Each assay's reversed rows yielded exactly the same marginals. The procedure is deterministic and contains no random seed or generation step.

The two existing unit tests passed with `python -m unittest -v test_fit`. Fresh attempts to invoke `fit.py` and `evaluate.py` against existing artifacts failed with `FileExistsError`; hashes of predictions, seal, evaluation and summary were unchanged. This confirms refusal for these existing-output paths; it does not make multi-file output creation atomic after unexpected failures.

Only after these checks, opened the sealed biological reference. Its shape is 595×900, with 300 columns per assay in RNA, H3K27ac, H3K4me3 order, and corresponding first-30 IDs agree with prediction rows. Recomputed all 2,700 unique biological joint detections, holdout marginals, train bounds, gaps and paired-marginal arithmetic checks; maximum absolute disagreement was 0. There are 25 above the upper endpoint, 0 below, with max discrepancy 0.009183477951441161. The counts per pair are 9, 16, 0 in the declared assay order. Each paired-marginal check passes as a tautological same-sample arithmetic identity. No simulation state/graph truth is scored in the inspected evaluator.

Each dataset's 2,700 rows reproduce all reported minimum/median/maximum widths and zero-width counts, and all covariance intervals include zero by construction. These facts do not establish biological independence. The local source and protocol encode no paired read or evaluation-based selection in fitting, though this alone cannot rule out historical knowledge outside the script.

## Sealing and interpretation

SHA-256 of plan, manifest, all nine inputs, protocol, source files, prediction rows, prediction seal, evaluator file and output agrees with the corresponding seals. Local file mtimes and ctimes show plan/source before protocol, protocol before prediction, prediction before seal, and seal before evaluator output. These mutable filesystem timestamps plus code read order corroborate the recorded sequence but cannot prove first physical data access, immutable external timestamping or absence of earlier human knowledge. This is a historical empirical panel, not fresh preregistered external validation.

Observed train marginal distributions come from assay-specific selected pools; the heldout reference has all-assay QC. Thus an observed joint outside a train plug-in bound can arise from marginal or selection differences or sampling, and is not a Fréchet theorem violation. Plug-in sets are neither common-population identified sets without transport assumptions nor confidence intervals. Historical feature selection and prior holdout use prohibit nominal coverage claims.

No benchmarks, data acquisitions, new output worlds, or modifications to frozen artifacts were performed. P01 output rewrite safety under concurrent races or interrupted first creation was not exhaustively tested.
