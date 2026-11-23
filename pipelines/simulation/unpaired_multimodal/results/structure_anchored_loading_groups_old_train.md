# Structure-anchored assay-local loading groups (old TRAIN diagnostic)

> Historical prototype. The production successor freezes formal state
> membership before loading selection and uses a `0.85` lower-cosine plus
> `0.10` competing-parent margin. On the complete already-open
> `33001–33024` diagnostic block it recovered 119/144 P/S groups with zero
> false groups (precision 1.0, recall 0.8264). See the August 30 revision in
> [the v4 experiment log](channel_aware_boss_v4_experiment_log.md) and the
> content-addressed audit under
> `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/structure_state_reporter_gate_train_diagnostic_v1/`.

## Scope and validity

This is a bounded, post-truth-known development diagnostic on the already-used
TRAIN worlds 30001, 30018, 30019, 30021, and 30024. No DEV, LOCKED, or fresh
world was opened. Candidate generation used only frozen observable CA-BOSS v4
artifacts: the typed direct graph, assay-local loading candidates, and their
multi-view empirical-null evidence. The global candidate freeze was written
before this prototype was scored against the structure truth.

The estimand is **assay-local shared-state pair support**. An emitted pair is
not a direct edge, does not add either feature to a cross-assay state
membership set, and does not claim exact cross-assay axis correspondence.

## Frozen rule

For each chromatin assay, require exactly two typed direct candidates with a
common collider center and two distinct parents. Among candidate pairs joining
one collider parent to one feature outside the collider motif, require:

- lower absolute loading cosine at least 0.65;
- positive median minimum partial-state fit;
- median minimum loading stability at least 0.80;
- eligible-view support at least 0.50;
- empirical-null p-value at most 0.25 in at least half of eligible views;
- relative best-cosine margin at least 0.05.

Emit at most one pair per assay and abstain on missing, failed, or ambiguous
evidence. These thresholds and all five candidate files were frozen before the
scorer was run.

## Post-freeze result

| TRAIN seed | Selected | True | False | P/S recall | Complete |
|---:|---:|---:|---:|---:|:---:|
| 30001 | 5 | 5 | 0 | 5/6 | no |
| 30018 | 4 | 4 | 0 | 4/6 | no |
| 30019 | 3 | 3 | 0 | 3/6 | no |
| 30021 | 6 | 6 | 0 | 6/6 | yes |
| 30024 | 4 | 4 | 0 | 4/6 | no |

Across the five worlds the rule selected 22 pairs, all 22 true, from 30 truth
pairs: micro precision 1.00 and micro recall 0.733. It has zero false pairs in
5/5 worlds, reaches the existing 0.80 recall gate in 2/5, and is complete in
1/5.

Relative to the preceding unanchored loading layer, seed 30001 improves from
4/6 true plus one false pair to 5/6 true and zero false pairs. Seed 30018 stays
at 4/6 with zero false pairs. Thus the rule rescues seed 30001 across the 0.80
recall gate, but it does not rescue both former 4/6 worlds and does not make
either one strictly complete. Used as a wholesale replacement, it also lowers
recall in seeds 30019 (5/6 to 3/6) and 30024 (5/6 to 4/6). It is therefore a
useful high-confidence annotation, not a replacement for the loading layer.

The abstentions are observable. Seed 30001 H3K4me2 has only one frozen direct
edge, so the required collider anchor is unavailable. In seed 30018,
H3K27me3 and H3K4me3 each have a complete direct motif but no pair passes every
pre-margin loading gate; notably all structural candidates have zero
eligible-view support under the frozen loading evidence. Relaxing that gate
after seeing truth would be overfitting, so no further rule change was made.

## Artifacts

- Candidate/scoring root:
  `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/structure_anchored_groups_old_train_30001_30018_30019_30021_30024_v1/`
- Pre-truth freeze: `global_pretruth_freeze.json`
- Aggregate score: `scores/score_summary.json`
- Candidate generator: `pipelines/simulation/unpaired_multimodal/scripts/structure_anchored_loading_groups_v4.py`
- Separate post-freeze scorer: `pipelines/simulation/unpaired_multimodal/scripts/score_structure_anchored_loading_groups_v4.py`
