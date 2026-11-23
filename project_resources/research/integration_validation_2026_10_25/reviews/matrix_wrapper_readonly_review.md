# Matrix wrapper review and observed launch

_Read-only review · 2026-10-25 · the running one-off driver was preserved unchanged_

The reviewed plan contains exactly the 16 declared selection fit jobs. The selected-only confirmation product contains 16 conditions. Complete cache reuse goes through the native model/export checks; existing partial output fails rather than being retried or overwritten. Native fit children acquire the existing global and GPU locks, so the outer command's `gpu=False` avoids double locking. Selection uses the unchanged all-candidate rule; discovery state access requires selected mode after the settings file is frozen.

The wrapper checks source, protocol, and planned runtime-library hashes **before fits**. It does not check every audit/selection boundary, bind its own hash internally, or compare the final execution-plan hash with an initial in-memory plan hash. Hashing a runtime library alone also does not demonstrate that CUDA loaded it. These were provenance observations, not detected mismatches or scientific-rule failures.

The driver had already started when the wrapper review returned. It was left unchanged. The external [observed-launch receipt](../../../../pipelines/simulation/integration_recoverability/runs/integration_validation/20261025_v1/matrix_execution_observed_start.json) records the exact command and reviewed driver, plan, protocol, and analysis-source hashes. Its command explicitly sets the isolated runtime through `LD_PRELOAD`, `CUDA_LAUNCH_BLOCKING=1`, and one-thread CPU settings. This reviewer independently checked that the current driver/plan/protocol and three analysis-source hashes match that receipt. No claim is made that stronger internal guards were installed.

The reviewed driver hash is `ef0ec55384b25b607f814ac51f4893b3bdefd4ad24333d1afc80ee3da0087128`; the plan hash is `d70d1c34c672305bda2ad69854685a1606a8855990512f3ff573cc98c03a1375`. Independent end-of-matrix integrity verification against the observed-launch receipt and every fit/report binding is still required before accepting completion. The observed start is a historical snapshot, not proof of current liveness or successful completion.

The [review record](matrix_wrapper_readonly_review.json) stores the compared hashes, launch-receipt hash, schedule checks, and limits. No driver, plan, core source, model, or process was modified; no CUDA operation or protected scientific-data read was performed for this wrapper review.
