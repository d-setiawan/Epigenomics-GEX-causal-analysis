# CA-BOSS v4 cached-world development replay

## Current v4.30 layered checkpoint — 2026-09-01

CA-BOSS v4.30 preserves v4.29 state/axis admission, v4.27 feature membership,
and the v4.22 direct candidates. It adds one precision-first post-guard: a
state-route loading group must have minimum member-centroid cosine at least
0.90, and any route overlapping a complete directed collider motif is typed
`unresolved`. A direct-overlap downgrade means the shared-state and direct
explanations are not separated by the available evidence; it does not assert
that the biological association is absent.

The two-phase audit reused 151 cached worlds across seven source blocks. V4.29
had 756 TP, 10 FP, and 150 FN. V4.30 retained all 756 TP and reduced the false
groups to five, raising micro precision from 0.9869 to 0.9934 at unchanged
0.8344 recall. It selected no sentinel group, and all 151 worlds met the
0.80-precision/zero-sentinel reliability gate. Two false groups were
downgraded for direct-layer overlap and three for falling below the loading
uncertainty floor.

Five source-current production panels used selection seeds `20261020` through
`20261024`. All 60 cached evaluations passed, spanning 44 unique worlds. On
those exact rows, the v4.29 baseline was 302 TP/5 FP/58 FN and v4.30 was 302
TP/2 FP/58 FN: micro precision improved from 0.9837 to 0.9934 at unchanged
0.8389 recall. Direct precision and recall were 1.000, feature-state precision
was 1.000 with mean recall 0.9613, and no sentinel was selected.

No simulation, missing-world generation, new modality, paired observation,
intervention, or external information was used. The 0.90 threshold and all
upstream selectors were developed on this cache, so these are retrospective
development and production-wiring results, not fresh locked validation. The
machine-readable record is
`pipelines/simulation/unpaired_multimodal/results/ca_boss_v4_cached_random_v430_summary.json`; the nested
audit is
`pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_loading_structure_uncertainty_v430_full_pool/`.
The complete Evaluation suite passes 489 tests with 62 existing dependency
warnings.

## Previous v4.29 layered checkpoint — 2026-09-01

CA-BOSS v4.29 is the current cached-world candidate. It preserves the v4.22
direct selector and v4.27 state set. The v4.28 loading guard retains a group
when every member is v4.27 state-supported; v4.29 adds a second route only when
the group has a v4.27 state anchor and joins a selected cross-assay axis under
leave-one-group-out evidence. The route does not change state membership,
create an exact cross-assay pair, or claim causal direction.

The final two-phase diagnostic reused 151 cached worlds across seven source
blocks. Relative to v4.28, the new route added 20 true loading groups, zero
false groups, and zero sentinel groups. Aggregate precision changed from
0.9866 to 0.9869 and recall from 0.8124 to 0.8344. Ten false proposals from the
upstream assay-local group layer remain; the new route adds none. One inherited
world remains below 0.80 loading precision.

Five final production-path panels reused selection seeds `20261009` through
`20261013` for an exact comparison. All 60 evaluations passed. V4.29 retained
293 groups through the state route and seven through the axis route, with 29
typed abstentions. Loading recovery was 299 TP, 1 FP, and 61 FN; mean
precision was 0.9972 and mean recall 0.8306, versus 0.8111 recall under v4.28.
The state layer remained 1,429 TP, 0 FP, and 71 FN.

No simulation, missing-world generation, new modality, paired observation,
intervention, or external information was used. These are retrospective
development results, not fresh locked validation. The machine-readable record
is `pipelines/simulation/unpaired_multimodal/results/ca_boss_v4_cached_random_v429_summary.json`. The final
nested audit is
`pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_nested_loading_axis_consensus_v429_full_pool_final/`.
The complete Evaluation suite passes 480 tests with the same 62 frozen-audit
deprecation warnings.

## Historical v4.23 layered checkpoint — 2026-09-01

CA-BOSS v4.23 was the current layered candidate at this checkpoint; its direct selector remains
v4.22. It was developed and tested without simulating, copying, or modifying
any world. The replay utility
inventoried 152 already completed worlds, sampled block-balanced 12-world
panels without replacement, and reconstructed decisions from frozen observable
evidence. Phase A refused scorer/truth paths and hash-closed every observable
input and output before Phase B opened the already existing scorer snapshots.

Twenty-three final-code panels produced:

| Quantity | v4.22 result |
| --- | ---: |
| Panel-world evaluations | 276/276 exact |
| Unique existing worlds represented | 152/152 exact |
| Eligible cached-pool coverage | 152/152 (100%) |
| Directed within-assay arms, panel instances | 3,312 TP / 0 FP / 0 FN |
| Directed within-assay arms, unique worlds | 1,824 TP / 0 FP / 0 FN |
| Strict reversals | 0 |
| Micro direct precision / recall | 1.000 / 1.000 |
| Mean direct adjacencies per world | 12.0 |
| Assay-level shared-state calls | 1,064/1,064 selected |
| Worlds with all seven modalities participating | 152/152 |
| Strict state-feature members | 2,990 TP / 0 FP / 810 FN |
| Strict plus equivalence-supported members | 3,246 TP / 0 FP / 554 FN |
| New worlds or datasets created | 0 |

The 152 distinct worlds exhaust every eligible source block: development340
(16), development470 (16), and all 24 worlds in each of TRAIN390, TRAIN400,
TRAIN410, TRAIN460, and TRAIN490. Panel overlap is reported explicitly; 276 is
the number of randomized panel instances, not the number of independent
worlds. The final coverage seeds were chosen only from random panel identities
and block labels to maximize previously unseen-world coverage; no measurement,
observable statistic, or truth outcome entered that choice.

V4.23 adds one typed uncertainty layer without changing any direct decision,
strict state member, or assay-participation call. A feature is retained as a
`within_assay_state_equivalence_member` only when it passed the original
pre-redundancy selector, was demoted specifically as a redundant reporter, and
passes the already frozen world-maxT strong-core threshold `p <= 0.01`. This
adds 256/256 true reporters and zero sentinels. Supported feature recall rises
from 0.7868 to 0.8542 at precision 1.000. Each of the seven source blocks has
zero false supported members and recall between 0.8225 and 0.8700. The 1,064
typed within-assay equivalence groups prohibit pairwise expansion and do not
alter the n-ary cross-assay participation hypothesis.

Schemas v4.19 through v4.22 added four narrow observable validators after new
random panels exposed distinct failures:

- v4.19 replaces a state-loaded center only with a loading-equivalent sibling
  rotation when the residual collider topology and beta-binomial arm evidence
  improve together; this repaired seed 49003 H3K27ac;
- v4.20 replaces one state-inconsistent parent while retaining the production
  center and stable arm only when a unique same-center candidate is stable
  across aggregation resolutions; this repaired seed 46004 H3K4me1;
- v4.21 extends the state-loaded-parent rotation with a conservative topology
  route requiring empirical-null stability and cross-resolution collider
  separation; this repaired seed 39019 H3K9me3;
- v4.22 adds a final zero-edge completion only when production and unresolved
  layers are both empty, frozen rank fusion abstained for incomplete practical
  fold support, all three nodes are held-out state/Wald positive, and exactly
  one of the complete 60-star universe passes residual-topology, hierarchical
  Bernoulli, beta-binomial, and empirical-null gates. This repaired seed 40019
  H3K4me1 without activating on any other distinct world.

The seed 40019 candidate has minimum/median collider separation
`-0.0115/0.0688`, 4/5 positive-parent folds, 3/5 practical-effect folds,
hierarchical arm empirical p-value `0.40625`, outer-null compatibility
`0.28125`, minimum beta-binomial log odds `0.921`, and maximum beta arm
empirical p-value `0.421`. These values are weak individually; the route acts
only on their predeclared conjunction and uniqueness. Its output remains a
`within_assay_direct_candidate`, not proof of an intervention-level effect.

The cross-assay layer was checked independently from the direct layer. Applying
the current two-route omnibus/Holm assay-participation statistic to all 152
hash-bound membership tables selected all seven modalities in every world.
The complete decision table SHA-256 is
`54204a1084720eaadc83b3f56529b8ae9deb9a12d573971138368dce8f7afa5e`.
The existing 120-world sentinel pseudo-assay audit observed 5/120 worlds with
at least one false participation call (4.17%; nominal familywise target 10%).
Assay participation is deliberately distinct from exact feature membership.
Replaying the current subtractive strong-core guard removes the one historical
sentinel selection and gives 2,990 true, zero false, and 810 missed strict
members: precision 1.000 and recall 0.7868. No claim of complete feature
pairing is made.

Verification completed with 105/105 collider tests and 450/450 tests across
`pipelines/simulation/unpaired_multimodal/tests`. The complete suite emitted 62 scikit-learn deprecation
warnings from the unchanged frozen rank-fusion derivation audit. Compilation
and `git diff --check` were clean.

This is strong retrospective development evidence, not fresh locked-world
certification. The recovered cross-assay object is one n-ary shared-latent-
state participation/subspace hypothesis over the seven assays, kept separate
from within-assay residual candidates. It is not an exact peak-to-peak or
peak-to-gene pairing and is not interpreted as direct cell-specific cross-assay
regulation. Those directions remain unidentified from disjoint cells under the
agreed data contract.

The current complete-cache machine-readable checkpoint is
`pipelines/simulation/unpaired_multimodal/results/ca_boss_v4_cached_full_pool_v423_summary.json`; the v4.22
and earlier 142-world checkpoints are retained as immutable iteration records. Final replay
artifacts are under `pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/existing_random_panel_v422_*`,
and the deduplicated two-phase audit is under
`pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/cached_layered_contract_v423_full_pool`.

## Historical v4.2 checkpoint

### Status at that checkpoint

This is a posthoc development-only replay, not a certification result. The
current v4.2 candidate fails the newest opened TRAIN block: TRAIN400 has
21/24 reliable worlds, below the registered 22/24 requirement. It must not be
advanced unchanged to a fresh TRAIN block.

The replay used only the already opened TRAIN330, TRAIN360, TRAIN370,
TRAIN380, and TRAIN400 worlds (120 total). Phase A froze and hashed all
observable predictions before Phase B opened the corresponding scorer
snapshots. No DEV340, LOCKED450, TRAIN410, new world, paired-cell data,
external data, or additional modality was read.

## Exact block results

| Block | Prior reliable | v4.2 reliable | Full 7-assay participation | Direct TP/FP/FN/reversal | Loading TP/FP/FN | Mean density | All gates |
|---|---:|---:|---:|---:|---:|---:|---|
| TRAIN330 | 22/24 | 22/24 | 24/24 | 288/0/0/0 | 116/5/28 | 18.042 | pass |
| TRAIN360 | 24/24 | 24/24 | 24/24 | 286/2/2/0 | 128/0/16 | 18.333 | pass |
| TRAIN370 | 23/24 | 23/24 | 23/24 | 285/1/3/0 | 131/1/13 | 18.417 | pass |
| TRAIN380 | 22/24 | 23/24 | 23/24 | 286/2/2/0 | 130/1/14 | 18.458 | pass |
| TRAIN400 | 18/24 | 21/24 | 23/24 | 284/4/4/1 | 125/2/19 | 18.292 | **fail** |

All block-mean recall, precision, and density gates pass. TRAIN400 fails only
the required reliable-world count.

Strict feature membership retained zero false selected features in all five
blocks. Its mean feature recall ranged from 0.775 to 0.802. This remains a
strict feature annotation and is not conflated with the separate assay-level
participation call. Full seven-assay participation was observed in 116/120
worlds.

## Unresolved worlds

- TRAIN330/33003 and 33018 fail loading precision.
- TRAIN370/37019 fails assay participation, direct recall, and loading member
  safety.
- TRAIN380/38020 fails assay participation despite recovering its missing
  direct motif and a true loading pair.
- TRAIN400/40002 fails assay participation.
- TRAIN400/40019 fails direct recall: the zero-edge rank-fusion winner is a
  sentinel-centered false motif.
- TRAIN400/40023 retains the known strict reversal.

The decisive audit correction is TRAIN400/40019. v4.2 selects
`chr3:2469000-2469199__H3K4me1` (role N2) as the center, with
`chr3:2150250-2150449__H3K4me1` (S) and
`chr3:2381750-2381949__H3K4me1` (the true C) as outers. The true motif is
P,Q -> `chr3:2381750-2381949__H3K4me1`. Therefore the two added edges are
false, contrary to the earlier projected 286/2/2 direct aggregate. The
truth-bound aggregate is 284 TP, 4 FP, 4 FN, and one reversal.

## Exact decision changes

Relative to the latest prior repaired predictions and the former strict-
feature assay proxy:

- Assay participation adds four true calls and abstains from three true
  calls. The additions are TRAIN380/38011 H3K27ac and H3K27me3,
  TRAIN400/40003 H3K9me3, and TRAIN400/40019 H3K4me1. The abstentions are
  TRAIN370/37019 H3K9me3, TRAIN380/38020 H3K4me2, and TRAIN400/40002
  H3K9me3.
- Direct selection adds six true oriented edges and two false edges, while
  removing one false edge and one reversed edge. The false additions are the
  two TRAIN400/40019 arms.
- Loading selection adds one true pair (TRAIN380/38020) and one false pair
  (TRAIN400/40011), and removes one false pair (TRAIN400/40012).
- World reliability improves for TRAIN380/38011 and TRAIN400/40003, 40008,
  40011, and 40012. TRAIN400/40002 changes from pass to fail because the new
  replicated assay-participation test correctly abstains at 5/6 views. The
  net TRAIN400 change is 18/24 to 21/24.

## Reproduction and provenance

```bash
python pipelines/simulation/unpaired_multimodal/scripts/audit_current_repaired_train_replay_v2.py \
  --phase a \
  --out-dir pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/current_repaired_train_replay_v2_final

python pipelines/simulation/unpaired_multimodal/scripts/audit_current_repaired_train_replay_v2.py \
  --phase b \
  --out-dir pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/current_repaired_train_replay_v2_final
```

The final Phase-A manifest SHA-256 is
`da02cb7307c25d77b3f23ae486509a586071371e95d795b336d664be110a6821`.
It binds direct schema `crossfit_hierarchical_collider.v4.2`, the frozen
zero-edge model payload, all state/loading/direct source files, all observed
input files, and every Phase-A output. The detailed outputs are under
`pipelines/simulation/unpaired_multimodal/outputs/ca_boss_v4/current_repaired_train_replay_v2_final/`:

- `phase_b_world_metrics.tsv`
- `phase_b_block_summary.tsv`
- `phase_b_state_feature_outcomes.tsv`
- `phase_b_direct_edge_outcomes.tsv`
- `phase_b_loading_pair_outcomes.tsv`
- `phase_b_changed_decisions_labeled.tsv`
- `summary.json`

The simulator has no wholly null assay. Assay-participation precision is
therefore not identifiable from these five blocks; the state/depth-preserving
null and sentinel controls remain the available negative controls. This audit
supports only shared-state assay participation, strict within-assay direct
candidates, and assay-local loading support. It does not identify direct
cell-specific cross-assay residual effects from disjoint cells.
