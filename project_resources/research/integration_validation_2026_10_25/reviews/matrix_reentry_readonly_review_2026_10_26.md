# Frozen matrix reentry review

Read-only capture: 2026-10-26T02:07:56.476042+00:00. The unchanged driver supports safe, fail-closed reentry after archiving only `unrestricted_chain_0/realistic/scglue/align_high_seed1`. No code, plan, settings, or completed-fit changes are required. This review did not archive anything, run the driver, execute audits/probes, deserialize arrays/checkpoints, inspect the partial fit, or call CUDA. Root separately observed session 1169 exit 143 at 01:58:27 UTC and no surviving matrix/native processes; the termination cause is unknown.

The [evidence receipt](matrix_reentry_readonly_evidence_2026_10_26.json) verifies 16 selection and nine confirmation native configurations, checkpoint hashes, model/seal/study bindings, fixed source pins, and all 225 public export hashes. The nine existing confirmation reports bind to their fresh native fits and current analysis/package versions. The original protocol, driver, execution plan, analysis sources and runtime-library bytes retain their recorded hashes. The selected-settings file remains `9c8bf9cc…`; the confirmation plan remains `6de5226a…`. No completion or failure marker existed at capture.

In `matrix_orchestration.py.txt`, every existing candidate directory enters `public_run` for train, val and discovery before being counted as reused. These calls require complete model/export receipts, exact declared configuration, checkpoint/seal hashes, study-protocol binding, all nine export hashes and valid public embedding schemas. An incomplete directory is not a resumable checkpoint: it causes refusal. After the reviewed partial is moved intact to its archive, its original path is absent and the exact frozen job starts afresh. Original logs/command evidence and the archive mapping must remain available.

Reentry also recomputes existing interim audits and the two all-candidate selection audits. `write_new` compares canonical JSON text and refuses any difference; it never silently replaces existing reports, settings or the confirmation plan. Matching, pool selection and fixed diagnostics are deterministic under the unchanged inputs, package versions and original single-thread environment, but this review did not execute that recomputation. If exact equality fails, preserve the refusal for inspection; do not edit existing JSON to force progress. One pre-matrix scMRDR trial records historical runner `5e025d63…`; its checkpoint matches the original independent first-fit review and its unchanged protocol/config/export chain passes the current reuse requirements. Preserve that historical receipt.

`choose` reads only the predeclared selection worlds' informative seed-0 validation cohorts. Its recomputation must reproduce the existing immutable selection; it cannot use confirmation outcomes. The confirmation Cartesian product is exactly the existing 16-job plan. After the archive, the remaining seven jobs are:

| World | Regime | Method / candidate | Seeds |
| --- | --- | --- | --- |
| unrestricted_chain_0 | realistic | scGLUE / align_high | 1, restarted |
| unrestricted_chain_0 | realistic | scMRDR / baseline | 0, 1 |
| unrestricted_chain_0 | informative | scGLUE / align_high | 0, 1 |
| unrestricted_chain_0 | informative | scMRDR / baseline | 0, 1 |

On successful reentry, `matrix_execution_complete.json` should contain 25 `verified_reused`, seven `newly_executed`, planned count 32, and 16 confirmation report hashes. Those execution lists describe this invocation; they do not erase the original attempts or imply 32 fresh fits. SIGTERM bypasses the driver's `except Exception`, so absence of a failure marker does not identify the cause.

Append a distinct resume observed-start receipt binding the original observed-start hash, unchanged command/source/plan pins, archive inventory/mapping and new session/start observation. Retain `matrix_execution_observed_start.json` byte-for-byte: the baseline retry helper explicitly pins its original hash. That helper accepts completion with a 25/7 split, verifies the completion/source/selection/report bindings, and requires matrix processes stopped before either separate baseline retry. No helper change is needed. Final matrix integrity review must still check the terminated driver's receipt and all 32 fit/report bindings; completion remains a computation claim, not integration or causal certification.

