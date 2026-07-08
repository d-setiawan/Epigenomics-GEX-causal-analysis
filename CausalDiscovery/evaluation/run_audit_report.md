# Graph-Run Audit (565 saved runs)

Generated as a rigor/reproducibility pass over `CausalDiscovery/outputs/methods/`.
Per-run detail is in [`run_audit_manifest.tsv`](run_audit_manifest.tsv) (one row per run,
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

## Suggested next actions

- Regenerate the screened datasets with `--graph-split-fraction 0.3` and re-run PC/FCI/DAGMA.
- Debug or drop the count-family DAGMA presets (item 2).
- Standardize on 300 kb for interpretability; explore mid-conservative CI settings (fisherz
  first, KCI only on small sets) to surface peak–peak edges without the 1 Mb density blowup.
- Widen curated promoter windows so promoter-proximal marks are credited.
