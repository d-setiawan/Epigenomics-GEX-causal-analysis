# Project Roadmap — Monocyte Chromatin→Expression Causal Discovery

**This is the living master plan. Update it as decisions are made and steps are completed.**
Keep it in sync: when you finish an action, check its box and move a one-line note to the
Changelog; when a decision changes, edit the Decision and note why in the Changelog.

- Related docs: [README.md](README.md) · [CausalDiscovery/README.md](CausalDiscovery/README.md) ·
  [integration/README.md](integration/README.md) ·
  [CausalDiscovery/evaluation/run_audit_report.md](CausalDiscovery/evaluation/run_audit_report.md) ·
  [CausalDiscovery/evaluation/external_sources.md](CausalDiscovery/evaluation/external_sources.md) ·
  `notes/` (planning + identifiability).
- Status date: **2026-07-08**. Genome build: **GRCh38/hg38**. Cell type: **monocyte** (CD14+/classical).
  Targets: **CSF1R, CD14, IL1B**. Identifiability target: **local PAG/MAG (no causal sufficiency assumed)**.

---

## 0. Where we are

**Pipeline:** scCUT&Tag preprocessing → scGLUE `joint_v2` integration → RNA-anchor pairing →
nearby-peak matrices → PC/FCI/DAGMA → external biological support.

**Done recently (see Changelog):** graph-run audit (565 runs tiered) · post-selection sample-split
(`elastic-net-screen --graph-split-fraction`) · notes moved into repo · repo-structure cleanup ·
**ENCODE-rE2G monocyte gene-link source added** (first target-gene-aware + cell-type-aligned
validator; it already flagged the "stable CD14 edge" peak as the **TMCO6 promoter**, not CD14).

**Known problems to fix (from the audit):** 1 Mb graphs too dense; KCI slow/brittle & empty;
count-family DAGMA empty 67% of the time; kNN-averaging inflates effective n; only ENCODE (of the
old sources) is cell-type-aligned and none could validate peak→gene edges (now partly fixed by rE2G).

---

## 1. The three method questions — decisions

These were researched 2026-07-08 (three literature/DB sweeps). Decisions below; build steps in §2.

### Q1 — Have we found all validation databases? What else can validate our edges?

**Decision: triangulate peak→gene edges across THREE orthogonal axes, and validate peak-nodes /
peak-peak edges with monocyte-specific ChIP + 3D data. Count an edge "validated" when ≥2 axes agree.**
Perturbation databases are a confirmed dead end for our three genes — stop looking there.

| Need | Primary (add) | Orthogonal (add) | Already have |
| --- | --- | --- | --- |
| **peak→gene** | **Javierre BLUEPRINT PCHi-C** (`Mon`, physical contact) + **eQTL Catalogue** monocyte eQTL (genetics) | ABC (Nasser), EpiMap, FANTOM5, DICE, SEdb | **ENCODE-rE2G** (prediction) |
| **peak-node** | **BLUEPRINT monocyte histone ChIP** (GRCh38, all 6 marks) | Roadmap E029 chromHMM, UniBind | ENCODE peaks, SCREEN cCRE |
| **peak-peak** | **ChIP-Atlas** colocalization (shared TF) + **primary-monocyte Hi-C** (GSE157971 loops) | ReMap 2022 `TF:biotype` | JASPAR motifs |

Key endpoints: Javierre `https://osf.io/u8tzp/` (`PCHiC_peak_matrix_cutoff5.tsv`; filter `baitName∈targets`,
`Mon≥5`) · eQTL Catalogue `ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/` (monocyte IDs QTD000021 BLUEPRINT,
QTD000081-83 Fairfax, QTD000069 CEDAR; SuSiE credible sets) · BLUEPRINT ChIP
`http://ftp.ebi.ac.uk/pub/databases/blueprint/data/homo_sapiens/GRCh38/venous_blood/` · ChIP-Atlas
`https://chip-atlas.dbcls.jp/data/hg38/` · ReMap `remap2022_all_macs2_hg38_v1_0.bed.gz` · Roadmap E029
`egg2.wustl.edu/roadmap/...E029_18_core_K27ac...`.

Caveats: **build friction** — Javierre, DICE, Roadmap, ABC, EpiMap, FANTOM5, Hi-C are **hg19** (need a
single liftOver pass); BLUEPRINT ChIP, eQTL Catalogue, ChIP-Atlas hg38, ReMap hg38 are GRCh38-native.
BLUEPRINT caQTL/hQTL are DAC-gated at EGA (email BLUEPRINT DAC if we want them). GeneHancer/EnhancerAtlas
are not usable (license-locked / dead). Perturbation (ENCODE-rE2G CRISPR benchmark, IGVF) = **0 hits for
CSF1R/CD14/IL1B** — do not invest; keep only a standing IGVF query for future myeloid FlowFISH.

### Q2 — Is kNN-averaging the best way to aggregate from the scGLUE latent?

**Decision: NO. Move from overlapping uniform kNN-averaging to NON-OVERLAPPING metacells
(SEACells in the scGLUE latent), and for the unpaired design prefer metacell-to-metacell optimal-
transport matching. QC with mcRigor; stabilize library size; report effective sample size.**

Why: our PC/FCI/DAGMA statistics are explicit functions of `n`. Overlapping neighbor-averaging (a) reuses
cells → dependent rows → inflates `n` (`n_eff ≪ 1887`) → type-I error blow-up (denser, false graphs), and
(b) shrinks within-feature variance → kills the signal CI tests use. Non-overlapping metacells use each
cell once (`n_rows ≈ n_eff`). Regression-based GRN tools tolerate overlap; CI-test/score-based discovery
does not — so this matters more for us. Metacells aren't free: control **library-size-driven spurious
correlations** (LSMetacell idea) and **heterogeneous "dubious" metacells** (mcRigor). Accept that honest
`n` is small (~25–150) — if it's too small for FCI, that is itself a reportable result.

Tools: **SEACells** (RNA+epigenomic, hard disjoint assignments, needs only a distance metric = our latent) ·
mcRigor (QC + graining hyperparameter) · OT matching (uniPort / FigR scOptMatch precedent).

### Q3 — Which causal discovery algorithms are best?

**Decision: PRIMARY = GFCI (→ PAG, latent-aware) via py-tetrad, with rank/nonparanormal fisherz as the
default CI test and RCoT as the nonlinearity guard (retire KCI). Make all nodes continuous (VST the gene
count) and DROP count-family DAGMA. Regress out/block global confounders (donor, batch, cell-cycle) before
discovery. Report donor-level block-bootstrap edge-mark stability.**

- **Primary:** GFCI (FGES skeleton + FCI orientation → PAG). Beats plain FCI on accuracy and directly fixes
  the "too dense" problem (FCI prunes from a complete graph; GFCI's score skeleton is sparser/more accurate).
- **CI test (biggest lever):** rank-fisherz default (fast, calibrated, captures monotone nonlinearity, best
  PPV for count data at n≥500); require edges to also survive **RCoT** (RFF KCI surrogate, ~100× faster,
  linear in n) to be "nonlinearity-robust". CMIknn as an audit. This is the fix for the empty/brittle graphs.
- **Secondary (cross-check the PAG):** RFCI/FCI+ (density robustness) and latent-aware LiNGAM variants
  **RCD** / **CAM-UV** (test confounding from the functional-model side). Caveat: FCI-family & RCD/CAM-UV
  assume *sparse/local* latents; our donor/batch/cell-cycle are *global* → regress them out first.
- **Sensitivity (sufficiency-assuming, NOT identifiability):** fGES→CPDAG (isolates orientation lost to
  latents), DirectLiNGAM (on lightly-transformed, not Gaussianized, data), GOLEM/NOTEARS + Gaussian DAGMA.
- **Stability:** donor-level **block bootstrap** (200–500×); report per-edge endpoint-mark frequencies
  (`o`/`>`/`−`), lenient ≥0.5, strict ≈0.85. Optional: Causal Dropout Model test-wise deletion for sparse peaks.
- Software: **py-tetrad/Tetrad** (GFCI, RFCI, fGES, bootstrap) · causal-learn (PC/FCI/LiNGAM) · lingam
  (RCD/CAM-UV) · RCIT (RCoT) · gCastle (NOTEARS/GOLEM) · TIGRAMITE (CMIknn).

---

## 2. Actionable next steps (phased)

Ordered so decisions/foundations precede analysis. Do NOT start bulk analysis until Phase A–C land.

### Phase A — Foundations & quick wins
- [ ] Pin the environments: write `requirements-causal.txt` / `environment.yml` (scGLUE, causal-learn,
      py-tetrad+JPype+Java, SEACells, lingam, gCastle, RCIT). Currently none exist — top reproducibility risk.
- [ ] Add a `liftOver` utility (hg19↔hg38) + chain files as a shared helper (needed by several DBs and by
      any hg19 resource). Decide direction: lift our hg38 peaks → hg19 once, or lift each DB → hg38.
- [ ] Promote **rE2G to a first-class `re2g` source** in `build_node_support_table.py` (own source name +
      plot color + preset), instead of routing through `--eperturbdb-tsv`.
- [ ] Merge the current `audit-and-rigor-postselection-split` branch (or open a PR) once reviewed.

### Phase B — Validation layer (Q1) — build the triangulation
- [ ] **peak→gene, physical:** Javierre PCHi-C `Mon` loader → support template (bait=target gene, `oe`=
      interacting region; filter `Mon≥5`; liftOver).
- [ ] **peak→gene, genetics:** eQTL Catalogue monocyte eQTL loader (BLUEPRINT/Fairfax/CEDAR; use SuSiE
      credible sets); count an eQTL variant in a peak targeting the gene as support.
- [ ] **peak-node:** BLUEPRINT monocyte histone ChIP loader (GRCh38, per mark) — "is this a real monocyte
      mark". Add Roadmap E029 chromHMM active-state overlap.
- [ ] **peak-peak:** ChIP-Atlas colocalization + ReMap shared-`TF:biotype`; monocyte Hi-C loop co-membership.
- [ ] Add a `cell_type_aligned` flag to every support match and a **multi-axis "≥2 agree" edge validation**
      class; stop counting cell-type-agnostic (SCREEN/JASPAR/ePerturbDB) evidence as monocyte validation.
- [ ] Re-run support on the tracked graphs; regenerate `support_report.md` per graph with the new sources.

### Phase C — Aggregation (Q2) — replace kNN-averaging
> Decision (2026-07-08): **keep the current kNN-averaged datasets as a labeled baseline** for
> comparison; do not delete them. Add metacell/OT schemes alongside and compare (do not replace outright).
- [ ] Implement SEACells metacells in the scGLUE joint latent (per modality, non-overlapping).
- [ ] Implement metacell-to-metacell OT matching (unpaired design); keep 1:1 and kNN-avg as baselines.
- [ ] Add mcRigor QC + library-size stabilization; choose graining by mcRigor.
- [ ] Implement `n_eff` estimators (participation ratio, reuse proxy, empirical null calibration) and a
      **null-calibration false-positive test** (permute peak columns; measure realized type-I error).
- [ ] Run the 5-scheme comparison (kNN-avg, dist-weighted, SEACells, OT-1:1, metacell-OT) on null
      calibration, `n_eff`, graph density/stability, and external-support precision. Pick the winner.

### Phase D — Causal discovery upgrade (Q3)
- [ ] Wire GFCI (py-tetrad) into `causal_cli.py` as a method → PAG output; keep PC/FCI for comparison.
- [ ] Add rank-fisherz and RCoT CI tests; make rank-fisherz the default; deprecate KCI to opt-in.
- [ ] VST the gene-count node; remove count-family DAGMA presets (keep Gaussian DAGMA as sensitivity).
- [ ] Add a pre-discovery step to regress out / condition on donor, batch, cell-cycle score.
- [ ] Add donor-level block-bootstrap edge-mark stability reporting; make stability the headline output.
- [ ] (Optional) Causal Dropout Model test-wise deletion for sparse peaks.

### Phase E — Analysis (only after A–D)
- [ ] Regenerate datasets on the chosen aggregation, at 300 kb (interpretable window).
- [ ] Run GFCI (primary) + secondary/sensitivity tiers with block-bootstrap; report PAGs + stable edges.
- [ ] Validate edges through the Phase-B triangulation; re-examine promoter/target assignment for all 3
      genes (the TMCO6 catch says "trust no promoter→expr edge without target-gene evidence").
- [ ] Write up: methods, identifiability framing, validated vs plausible edges.

---

## 3. Decision log
- **2026-07-08** Identifiability target = local PAG/MAG (no causal sufficiency). *(pre-existing)*
- **2026-07-08** Q1: triangulate peak→gene across rE2G(prediction)+PCHi-C(physical)+eQTL(genetics); node/
  peak-peak via BLUEPRINT ChIP + ChIP-Atlas/Hi-C; ≥2 axes = validated; perturbation DBs dropped.
- **2026-07-08** Q2: replace kNN-averaging with non-overlapping SEACells metacells (+metacell OT); report n_eff.
- **2026-07-08** Q3: GFCI primary (PAG); rank-fisherz + RCoT CI tests, retire KCI; drop count-DAGMA; block
  global confounders; donor-level block-bootstrap stability.

## 4. Changelog
- **2026-07-08** Created this roadmap. Completed earlier today: run audit + trust tiers; post-selection
  `--graph-split-fraction`; notes/ moved into repo; empty skeleton dirs removed; cell-type alignment audit
  of the evaluation suite; **ENCODE-rE2G monocyte gene-link source added** (caught the CD14→TMCO6 mis-assignment).
