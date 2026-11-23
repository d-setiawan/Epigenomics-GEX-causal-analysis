# Preparing a bounded retry of the old baseline

_Read-only preparation · 2026-10-25 · constrained_chain_1, realistic and informative, seed 0_

An explicitly reviewed operational retry is defensible after the integration-matrix controller and every child are terminal. A momentary idle GPU between matrix fits is insufficient. This is a proposal for the parent decision, not authorization or execution of a retry.

Both failures retain the original scGLUE default configuration, seed 0, matching adapter source, eight recorded package versions, and two native scGLUE source hashes. All 17 frozen scientific sources still match the assessment lock. The world remains complete and passes declared calibration; its manifest hash is `9a0d3c2040a959189c30b745c56d0179aa82b5136bae675c0a732fd58c110b3b`. Bank/calibration hashes and prepared-training-input metadata agree with the saved fit inputs. Both unsuccessful pipeline receipts retain exit code 1 and matching log hashes. No scientific input arrays were opened for these checks.

Export recovery is ruled out. Realistic output contains only `fit_inputs.json`; informative output contains that file, an `AcceleratorError` failure during `native_fit_SCGLUE`, and one small pretraining event file. Neither has a `.dill`, optimizer `.pt`, or completed model manifest. The existing export-recovery helper requires a completed native fine-tune snapshot, retained checkpoints, and the specific post-fit missing-TensorBoard-reader failure. These runs meet none of those recovery conditions. A pretraining event file cannot supply missing trained weights or resume state.

The smallest route back to the original canonical pipeline is:

1. Predeclare one bounded retry for each named failed stage, preserving exact configuration, seed, inputs, source pins, world, assessment lock, schedule, and all failed receipts.
2. Under the global executor lock, save an immutable inventory and verified archive mapping for **only the two incomplete `primary/scglue_seed0` directories**. Do not delete their contents or move worlds/preprocessing. Removing the partial directories from their canonical destinations would be an explicit reviewed operational action; no archive or move was made during this review.
3. With canonical fit destinations absent, verify that the stage is failed only through its retained failed attempt, dependencies are complete, the lock is valid, and preprocessing caches are complete. `suite_plan.emit_commands(retry_failed=True, stage_ids=[...])` can then emit the exact old stage argv. The normal serial executor still refuses failed-stage eligibility, so a narrow reviewed one-stage wrapper must record this explicit exception using the existing safety and journaling primitives. Do not edit or monkeypatch frozen code or weaken general partial-output checks.
4. Run the original frozen `run-world` argv for one regime, with `--methods scglue scmrdr --seeds 0 --arm primary` and the same assessment lock. Keep the outer global lock; let the unchanged nested fit commands acquire the GPU lock. Holding the same GPU lock in the parent throughout `run-world` risks deadlock. Record the exact process-only runtime and synchronous-reporting environment without changing scientific parameters.
5. Stop on a new failure. On success, verify all scheduled model/export/report artifacts and append a new attempt receipt for the same frozen stage. The archived originals and old failed receipts remain evidence. Attempts are not additional independent worlds or unique planned fits.

The isolated runtime alone already failed in the informative attempt. Later healthy development fits with synchronous reporting make a bounded retry plausible, but do not prove that the baseline failure is fixed or that a repeated GPU fit will be bitwise identical. The resulting `fit_inputs` fingerprint must match the archived original; runtime differences remain explicitly recorded.

The [preparation record](constrained_chain_1_retry_preparation_readonly_2026_10_25.json) includes original argv, configurations, input/source/version hashes, failed receipt/log hashes, partial-file inventories, and required future guards. It does not revalidate every protected binary or the complete historical development-release evidence; the unchanged native preflight remains required at execution.

No retry wrapper was created, no file moved or archived, and no fit, CUDA operation, or protected-array read occurred. New integration-matrix settings and outcomes stay separate; no causal discovery on new settings is authorized by this recipe.
