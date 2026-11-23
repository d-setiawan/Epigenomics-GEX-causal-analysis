# Graph-Run Audit (565 saved runs)

Generated as a rigor/reproducibility pass over `pipelines/empirical/analysis/outputs/methods/`.
Per-run detail is in [`run_audit_manifest.tsv`](../outputs/run_audit_manifest.tsv) (one row per run,
tagged with a trust `tier` and `flags`).

## Trust tiers

Runs are tiered by **what data was fed to graph discovery**, because the main
statistical risk is post-selection inference (selecting peaks and then learning a
graph on the same rows).

| Tier | Runs | Meaning | Trust |
| --- | ---: | --- | --- |
| **A** unfiltered (`300kb`, `300kb_rawcounts`, `1mb_avg_k10`) | 314 | No expression-based peak selection | Clean on selection; but `1mb` graphs are far too dense to read |
| **B** structural filters (`nz001/005/010`) | 162 | `nz` filters use only nonzero fraction, not the outcome | Clean (no leakage); `nz001` barely thins density |
| **C** elastic-net screened (`elasticnet_top30_crossfit`) | 54 | Peaks selected on the **same 1887 rows** later used for the graph | **Exploratory only** — biased CI tests/scores |
| unknown | 35 | matrix path not under `outputs/datasets/` | needs path normalization |

## Key findings

1. **Post-selection leakage in the 54 screened runs.** The "crossfit" screen cross-fits
   only the *peak selection* (RepeatedKFold to compute selection frequency); it then
   keeps **all** rows in the screened matrix. Graph discovery therefore runs on the same
   samples used to pick the peaks. **Fixed:** `elastic-net-screen --graph-split-fraction`
   now holds out a disjoint graph split (see README → Screening Plan). Regenerate the
   screened datasets with a split before drawing causal conclusions from them.

2. **Count-family DAGMA is mostly returning empty graphs.** 28% of all runs (157) have
   zero edges; **128 of those are DAGMA**, and **108 of 162 count-family DAGMA runs (67%)**
   (`bernoulli_peak_nb_gene`, `nb_nb`, `gaussian_nb`) are empty. Essentially all real DAGMA
   signal comes from `gaussian_gaussian`. The raw-count mixed-family presets need debugging
   or retuning before they can be claimed as a method contribution.

3. **Density makes the 1 Mb graphs uninterpretable.** Densest runs reach 1,362 edges on
   110 nodes (`1mb_avg_k10`, PC/FCI); `nz001` still leaves ~1,245. The **300 kb** roots
   (25–43 nodes) are the only human-readable graphs.

4. **Effective sample size caveat (all tiers).** kNN-averaged pairing reuses chromatin
   cells, so `n_eff < 1887`. KCI/FCI p-values are somewhat optimistic regardless of tier.

## CD14 300 kb stability snapshot (`graph_stability/cd14_pc_fci_300kb_audit/`)

Across 14 PC+FCI runs (depths 1–5, α∈{0.05,0.1}) on the Tier-A 300 kb CD14 dataset:

- **Exactly one stable adjacency**, present in 100% of runs and consistently oriented
  as a directed edge into expression:
  `chr5:140638835-140639790 __H3K4me3  →  expr__CD14`
  (FCI orientation consistency 0.875, PC 0.833). This is a **promoter-proximal H3K4me3
  peak driving CD14 expression** — biologically exactly what you'd expect, and correctly
  oriented (mark → expression sink).
- **These KCI graphs are near-empty (mean 1 edge).** So at these settings we recover the
  clean promoter→expression edge but essentially **no peak–peak structure** — which is the
  actual scientific question. KCI on 300 kb is too conservative to surface coordination.
- **Curated-region credit gap:** the stable peak (`chr5:140638835–140639790`) sits ~5 kb
  from the CD14 TSS, just **outside** the ±1 kb curated promoter window
  (`chr5:140632700–140634700`), so `curated_expr_adjacent_peak_fraction = 0`. The one
  robust biological edge is *not* credited by the annotation — the promoter window is too
  narrow. Widen promoter windows (or add a promoter-proximal tier) in the evaluation layer.

## Cell-type alignment of the external evidence suite

Before regenerating datasets we checked whether the evaluation suite can actually do its
job for **monocyte** data (our RNA harmonizes to `monocyte`; fine types CD14-positive /
classical / non-classical). Two things were audited: (1) does the code align by cell type,
and (2) do the databases even contain monocyte / target-gene evidence.

**Code (how matching works).** `build_node_support_table.py` matches a record to a node on
**only** interval overlap + target-gene-name equality. Cell type is never a match/filter
criterion, with one exception: the ENCODE Portal query passes `biosample_ontology.term_name`
to the API (server-side filter). ePerturbDB's tissue/cell-type column is read but used only as
a display string. So SCREEN, JASPAR, and ePerturbDB matches are cell-type-agnostic — they can
credit a monocyte edge with non-monocyte (skin/breast/hESC/K562) evidence.

**Data (coverage), per source:**

| Source | Target-gene aware? | Cell-type gated? | Monocyte coverage | Can validate peak→gene for CSF1R/CD14/IL1B? |
| --- | --- | --- | --- | --- |
| ENCODE Portal (`encode_pbmc`) | No (interval-only) | **Yes** (server-side) | **Yes — 159 CD14+ monocyte GRCh38 peak beds** (PBMC: 25) | No — interval-only by the suite's own rule |
| ENCODE SCREEN cCRE | No (interval-only) | No (genome-wide registry) | agnostic | No |
| JASPAR motifs | No (motif-only) | No (sequence) | agnostic | No |
| ePerturbDB (all 3) | **Yes** (`Gene (literature)`) | No | **~0 immune rows**; CSF1R/CD14 = 0, IL1B = 0 literature hits | **No — no coverage** |

**Verdict.** The suite **can** provide cell-type-aligned **peak-node / peak-peak interval**
support via ENCODE with a monocyte biosample term (use `monocyte_encode` /
`monocyte_regulatory`, not `pbmc_encode` or the PBMC default). It **cannot** provide
cell-type-aligned **peak-to-gene edge** validation for our genes: the only target-gene-aware
source (ePerturbDB) has no monocyte content and zero `Gene (literature)` rows for
CSF1R/CD14/IL1B. So the stable CD14 promoter-H3K4me3 → expression edge is currently
**unvalidatable** by this suite.

**Options to make peak→gene validation possible (pick before over-investing in regeneration):**

1. **[DONE] Add a cell-type-aware, target-gene-aware monocyte source — ENCODE-rE2G.**
   `pipelines/empirical/evaluation/scripts/build_re2g_gene_link_support.py` fetches the CD14+ monocyte
   ENCODE-rE2G element→gene predictions (native GRCh38, `TargetGene` per element) and writes a
   support-template TSV; feed via `support --eperturbdb-tsv`. Validated on the CD14 300 kb graph:
   the "stable" `chr5:140638835-140639790__H3K4me3 → expr__CD14` edge stays unvalidated because
   rE2G links that element to **TMCO6** (its TSS is inside the peak), not CD14 — the source
   correctly flags a probable neighbor-promoter artifact. See `external_sources.md`.
2. Reframe honestly: use ENCODE-monocyte for peak-node support, and report peak→gene edges
   as **unvalidated-but-plausible** (the promoter overlap is the biology), not as DB-validated.
3. Add a `cell_type_aligned` flag to every support match and only count aligned matches toward
   the "validated" highlight class, so SCREEN/JASPAR/ePerturbDB stop silently crediting monocyte
   edges with off-target evidence.

## Suggested next actions

- Regenerate the screened datasets with `--graph-split-fraction 0.3` and re-run PC/FCI/DAGMA.
- Debug or drop the count-family DAGMA presets (item 2).
- Standardize on 300 kb for interpretability; explore mid-conservative CI settings (fisherz
  first, KCI only on small sets) to surface peak–peak edges without the 1 Mb density blowup.
- Widen curated promoter windows so promoter-proximal marks are credited.
