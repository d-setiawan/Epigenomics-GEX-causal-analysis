# Project Roadmap — Monocyte Chromatin→Expression Causal Discovery

**This is the living master plan. Update it as decisions are made and steps are completed.**
Keep it in sync: when you finish an action, check its box and move a one-line note to the
Changelog; when a decision changes, edit the Decision and note why in the Changelog.

- Related docs: [README.md](README.md) · [CausalDiscovery/README.md](CausalDiscovery/README.md) ·
  **[CausalDiscovery/PIPELINE.md](CausalDiscovery/PIPELINE.md)** (end-to-end driver + alternative paths) ·
  [integration/README.md](integration/README.md) ·
  [CausalDiscovery/evaluation/run_audit_report.md](CausalDiscovery/evaluation/run_audit_report.md) ·
  [CausalDiscovery/evaluation/external_sources.md](CausalDiscovery/evaluation/external_sources.md) ·
  [CausalDiscovery/evaluation/aggregation_bakeoff.md](CausalDiscovery/evaluation/aggregation_bakeoff.md) ·
  [CausalDiscovery/notes/aggregation_causal_discovery_litreview.md](CausalDiscovery/notes/aggregation_causal_discovery_litreview.md) ·
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
- [x] **Pin the environments** — `envs/` has `environment-scrna_r.yml` (recreation artifact) + pip-freeze
      records for `scrna_r` (heavy: scGLUE/causal-learn/dagma/scanpy/snapatac2/torch), `cvbench`, and base;
      `envs/README.md` maps env→stage and lists not-yet-installed Phase C/D tools.
- [x] **liftOver utility** — `CausalDiscovery/scripts/liftover.py` (pyliftover backend + UCSC fallback,
      auto-downloads chains to gitignored `Data/external/chains/`). Verified exact hg38→hg19→hg38 round-trip.
- [x] **rE2G is now a first-class `re2g` source** — `--re2g-tsv` on `graph`/`plot`/`support`; own
      `source=re2g`, `highlight_class=re2g_only`, blue plot color (`#2E86DE`). Tested on the CD14 graph.
- [ ] Merge the current `audit-and-rigor-postselection-split` branch (or open a PR) once reviewed. *(needs push authorization)*

### Phase B — Validation layer (Q1) — build the triangulation
- [x] **Generic local named-source mechanism** `--external-support-tsv SOURCE=PATH` (local analog of
      `--external-support-url`), so new validators plug in without per-source CLI plumbing.
- [x] **Efficiency fix:** hoisted `external_df.to_dict()` out of the per-node loop (was O(nodes*records);
      the 667k-row rE2G source made `support` time out — now ~18s for the two-axis CD14 run).
- [x] **peak→gene, physical:** `build_pchic_gene_link_support.py` — Javierre PCHi-C `Mon≥5`, hg19→hg38
      lifted, source `pchic`. 86 monocyte gene-links for the panel; on CD14 it corroborates 15 nodes with
      rE2G (`multi_source`). Verified end-to-end.
- [x] **peak→gene, genetics:** `build_eqtl_gene_link_support.py` — eQTL Catalogue monocyte `ge` SuSiE
      credible sets, filtered to target genes, each credible-set variant emitted as a `[pos,pos+1]`
      interval; source `eqtl` (teal). **19 variant-links: CD14=17 + CSF1R=2, both from TLR-stimulated
      monocytes (Quach_2016 Pam3CSK4/R848) — cell-type aligned but NOT resting-state aligned; IL1B has
      no monocyte credible set anywhere.** Naive monocyte datasets (BLUEPRINT/Quach/Schmiedel/Randolph)
      have 0 credible sets for the 3 targets. Completes the 3-axis suite (prediction+physical+genetics).
- [ ] **peak-node:** BLUEPRINT monocyte histone ChIP loader (GRCh38, per mark) — "is this a real monocyte
      mark". Add Roadmap E029 chromHMM active-state overlap.
- [ ] **peak-peak:** ChIP-Atlas colocalization + ReMap shared-`TF:biotype`; monocyte Hi-C loop co-membership.
- [x] **Axis-aware edge validation rule formalized** in the end-to-end driver
      (`run_gene_pipeline.py`): peak→gene edges validate only under
      `region_interval_with_matching_target_gene` (target-gene-aware); peak-peak
      "either-endpoint" overlap is demoted to `no_target_validator` (node-existence,
      not edge support). Verdict tiers: `VALIDATED_MULTI_ALGO`/`VALIDATED`/
      `recurrent_unvalidated`/`unvalidated`. Fixed a support bug where the expr node's
      quant suffix (`expr__CD14_log1p`) leaked into the gene symbol and broke every
      `same_gene` match. See [PIPELINE.md](CausalDiscovery/PIPELINE.md).
- [ ] Add an explicit `cell_type_aligned` flag column to every support match (the
      driver already ignores cell-type-agnostic SCREEN/JASPAR for edge validation).
- [ ] Re-run support on the tracked graphs; regenerate `support_report.md` per graph with the new sources.

### Phase C — Aggregation (Q2) — replace kNN-averaging
> Decision (2026-07-08): **keep the current kNN-averaged datasets as a labeled baseline** for
> comparison; do not delete them. Add metacell/OT schemes alongside and compare (do not replace outright).
- [x] **Non-overlapping metacells implemented** (native scanpy Leiden over the scGLUE joint latent —
      NOT SEACells, which force-upgrades numpy and breaks scrna_r): `metacell_assign.py` (148 metacells,
      133 complete) + `build_metacell_matching_dir.py` (emits a dataset-stage-compatible matching dir +
      pseudobulk metacell RNA h5ad; dataset stage runs unchanged). Metacell CD14 matrix = 133 honest-n rows.
- [x] **Bake-off done → winning config = metacell + α=0.01 + bootstrap stability selection** (see
      [evaluation/aggregation_bakeoff.md](evaluation/aggregation_bakeoff.md)). Metacell fixes calibration
      (α=0.01 null/expected 3.0×→1.0×) and is the only config whose single-run graph beats the null p95
      (16 > 14); its stable peak→gene edges are externally validated by rE2G + eQTL.
- [ ] (Optional refinement) metacell-to-metacell OT matching; mcRigor QC + library-size stabilization.
- [x] **null-calibration false-positive test** built (`scripts/null_calibration.py`) and run on the kNN-avg
      baseline → **the current graphs are noise-dominated** (real edges ≈ α×pairs ≈ null floor for all 3
      genes; peak→gene edges occur as often under noise as in real data). See
      [evaluation/null_calibration_findings.md](evaluation/null_calibration_findings.md). This reframes
      the plan: stability-selection + multiple-testing control are the PRIMARY fixes; metacells (honest n)
      are necessary but insufficient alone.
- [ ] Implement `n_eff` estimators (participation ratio, reuse proxy) as a complementary diagnostic.
- [x] Comparison run (baseline 1:1 vs metacell, α 0.05/0.01, single-run vs stability) via
      `null_calibration.py` + `bootstrap_stability.py`. Winner recorded above. Bootstrap **stability
      selection is the primary signal/noise separator** (real ≥0.5 edges vs null max-freq 0.12–0.22)
      and belongs in every config — fold it into the pipeline as the headline output (Phase D/E).

### Phase D — Causal discovery upgrade (Q3)
- [x] **Method zoo installed in one `causaldisco` env** (mamba; py-tetrad+JDK21, lingam, causal-learn
      0.1.4.7, gcastle) and **cross-method smoke-tested on metacell CD14** (α=0.01, non-transformed):
      **GFCI (Tetrad), FCI, and RCD (non-Gaussian, latent-aware) all independently pick the same
      eQTL-validated peak→CD14 edge** `chr5:140638860-140639949__H3K4me2`; GFCI is far sparser than FCI
      (1 vs 16 edges — the Q3 "score skeleton fixes density" claim); RCD flagged 26 latent-confounded
      entries; CAM-UV disagreed (nonlinear, small n); GIN found no latent structure. See envs/README.md.
- [x] **GFCI wired into the pipeline** (`run_gfci_tetrad.py` + `run_gene_pipeline.py --methods gfci`,
      routed to the causaldisco env/JDK21; support-compatible graph dir). GFCI is far sparser than FCI
      (1 vs 28 edges on metacell CD14). *(Wired into `run_gene_pipeline.py`, not `causal_cli.py`.)*
- [x] **Stability selection is now the pipeline headline** — `run_gene_pipeline.py` runs
      `bootstrap_stability.py` and reports the top tier = edges BOTH bootstrap-stable AND target-aware
      validated. On metacell CD14: `chr5:140638860-140639949__H3K4me2→CD14` (stability 0.70, fci∩gfci,
      eQTL, VALIDATED_MULTI_ALGO) + `chr5:140591420-140591815__H3K4me3→CD14` (0.70, rE2G). *(Row
      subsampling; upgrade to donor-level block-bootstrap once donor labels are threaded.)*
- [x] **Causal Dropout Model added + assessed** — `--indep-test zerodel_fisherz` in
      `run_pc_causallearn.py` (test-wise deletion, raw zero-mask threaded). **Verdict: not usable as a
      default** — our peaks are 70–93% zero (metacell) / 98–99% (1:1/raw), so test-wise deletion starves
      the sample (1:1 collapses to 1 edge). Keep only as an optional sensitivity check on metacell data
      with `--transform none`. See envs/README.md / commit.
- [ ] Add rank-fisherz and RCoT CI tests; make rank-fisherz the default; deprecate KCI to opt-in.
- [ ] VST the gene-count node; remove count-family DAGMA presets (keep Gaussian DAGMA as sensitivity).
- [ ] Add a pre-discovery step to regress out / condition on donor, batch, cell-cycle score.
- [ ] Upgrade stability to donor-level block-bootstrap (needs donor labels on rows).

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
- **2026-07-09** **Pipeline upgraded: GFCI + stability headline; Causal-Dropout CI added & assessed.**
  `run_gfci_tetrad.py` + `run_gene_pipeline.py --methods gfci` (causaldisco/JDK21); bootstrap
  stability-selection is now the report headline (top tier = stable AND validated). On metacell CD14 the
  top tier is the eQTL-validated `…140638860…H3K4me2→CD14` (stability 0.70, fci∩gfci) + the rE2G-validated
  `…140591420…H3K4me3→CD14`. Added `--indep-test zerodel_fisherz` (Causal Dropout Model) — assessed as
  NOT usable by default (peaks 70–99% zero → test-wise deletion starves the sample), only a niche
  sensitivity check. Commit bbc796d.
- **2026-07-09** **Phase-D method zoo installed + cross-method-validated.** One centralized `causaldisco`
  mamba env (py-tetrad + JDK 21, lingam, causal-learn, gcastle) covers all three requested directions:
  Java suite (GFCI/RFCI/FCI-Max), non-sufficiency functional (RCD/CAM-UV/ParceLiNGAM), latent-structure
  (GIN), and the Causal Dropout Model (test-wise-deletion CI, cloned). On metacell CD14 (α=0.01), GFCI +
  FCI + RCD independently converge on the same eQTL-validated peak→CD14 edge — method triangulation atop
  the data-axis triangulation. Env pinned in `envs/`.
- **2026-07-09** **Combined aggregation overhaul + bake-off → winning config chosen.** Built
  non-overlapping metacells (native scanpy Leiden; `metacell_assign.py` + `build_metacell_matching_dir.py`;
  133 honest-n rows, dataset stage unchanged) and `bootstrap_stability.py`. Bake-off (CD14):
  **metacell + α=0.01 + stability selection wins** — metacells fix calibration (α=0.01 null/expected
  3.0×→1.0×), metacell+α=0.01 is the only single-run config beating null p95 (16>14), and bootstrap
  stability cleanly separates signal (real 12–57 stable edges) from noise (null 0, max-freq 0.12–0.22).
  Metacell stable CD14 peak→gene edges externally validated by rE2G + eQTL. Stability selection is the
  primary lever and should become the pipeline's headline output. See evaluation/aggregation_bakeoff.md.
- **2026-07-09** **Null-calibration diagnostic built + run → current graphs are noise-dominated.**
  `scripts/null_calibration.py` (permute every feature column, re-run FCI, compare edge counts). On the
  300 kb kNN-avg baseline the real edge count sits inside/below the null band for all 3 genes
  (CD14 56 vs null 51.7; CSF1R 10 vs 9.2; IL1B 6 vs 12.4) — edges track α×pairs, not signal; peak→gene
  edges are noise-level. Reframes Q2: the primary fix is stability-selection + multiple-testing control,
  with non-overlapping metacells (honest n) necessary but not sufficient. SEACells CANNOT go in `scrna_r`
  (it force-upgrades numpy 1.26→2.2, breaking scGLUE/causal-learn) — use an isolated env or native
  scanpy Leiden metacells. See evaluation/null_calibration_findings.md.
- **2026-07-09** **Genetics axis added → 3-axis evaluation suite complete.**
  `build_eqtl_gene_link_support.py` loads monocyte eQTL Catalogue SuSiE credible sets (source `eqtl`,
  teal); wired into `run_gene_pipeline.py` (`--eqtl-tsv`, auto-detected). 3-axis node coverage now in
  `PANEL_SUMMARY.md`: CD14 25 target-aware nodes (pred 8 / phys 15 / **gen 4**), CSF1R 5 (5/1/1),
  IL1B 18 (5/17/0). Genetics support localises to the CD14 promoter/TMCO6-boundary peaks and every gene
  node; the discovered peak→gene edges are unchanged (eQTL variants sit in the promoter, not in those
  specific distal peaks). Cell-state caveat logged (support = TLR-stimulated monocytes).
- **2026-07-08** **End-to-end pipeline spine landed** (`CausalDiscovery/PIPELINE.md`):
  `run_gene_pipeline.py` runs pc/fci/dagma on a gene → runs the target-aware support
  suite on each → cross-algorithm consensus with axis-aware verdicts;
  `summarize_pipeline_panel.py` rolls up the panel. Ran all 3 genes on the 300 kb
  kNN-avg baseline (fisherz+rank_gaussian): CD14 has 1 `VALIDATED_MULTI_ALGO` edge
  (`chr5:140612012-140612249__H3K4me2 → CD14`, rE2G-supported, FCI∩PC); CSF1R 0 /
  IL1B 1 peak→gene edges, none target-aware validated; DAGMA empty on all three.
  Fixed a target-gene-matching bug (quant suffix leaked into the expr node's gene
  symbol). This formalizes the Phase B axis-aware edge validation rule.
- **2026-07-08** Phase B started: generic `--external-support-tsv` local named-source mechanism;
  support matching efficiency fix (per-node `to_dict` hoisted); BLUEPRINT monocyte PCHi-C physical
  peak→gene source (`build_pchic_gene_link_support.py`, source `pchic`, pink). Two-axis rE2G+PCHi-C
  triangulation demonstrated on CD14 (15 `multi_source` nodes).
- **2026-07-08** Phase A foundations landed: pinned envs (`envs/`), hg19↔hg38 `liftover.py` (round-trip
  verified), and rE2G promoted to a first-class `--re2g-tsv`/`re2g` source (blue). kNN-averaged datasets
  kept as a labeled baseline (Phase C decision).
- **2026-07-08** Created this roadmap. Completed earlier today: run audit + trust tiers; post-selection
  `--graph-split-fraction`; notes/ moved into repo; empty skeleton dirs removed; cell-type alignment audit
  of the evaluation suite; **ENCODE-rE2G monocyte gene-link source added** (caught the CD14→TMCO6 mis-assignment).
