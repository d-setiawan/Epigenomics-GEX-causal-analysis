# MISSION — what this project is, and why it is novel

**One-line:** recover the **local *causal architecture* of a gene's cis-regulation** — not just
which enhancers regulate a gene, but the causal relationships *among* its regulatory elements —
from single-cell multi-omic histone-mark + expression data, using formal latent-confounder-aware
causal discovery, and triangulate it against orthogonal biological evidence.

This is the "why/what's novel" master doc. Operational "how" lives in [README.md](../README.md);
the tactical plan lives in [ROADMAP.md](ROADMAP.md).

---

## 1. The central question

For a target gene, its nearby chromatin is organized into regulatory elements (enhancers,
promoters) marked by histone modifications. The standard question — *which enhancers regulate the
gene?* — is largely solved (see §3). We ask the **next** question:

> **What is the causal architecture *among* a gene's regulatory elements, and how does that
> architecture jointly control the gene — recovered from observational single-cell data, with
> latent confounders and causal insufficiency handled honestly?**

Concretely, for one gene we learn a **local causal graph** over {nearby peaks, observed
confounders, gene expression} with edge types `C→A`, `C→Y`, **`A→A`**, `A→Y` (per the original
design in [`docs/planning/local_multiome_notears_plan.md`](planning/local_multiome_notears_plan.md)).
The **`A→A` (element↔element) structure is the
novel object of study**, not a nuisance.

## 2. Why the peak↔peak edges ARE the point (not a distraction)

A recurring worry: "if we drop peak↔peak edges and only do peak→gene, there's no novelty — many
papers do peak→gene." **Correct — and that is exactly why peak↔peak is the contribution.**
Biologically, a gene's enhancers are not independent: they show **hierarchy, redundancy, synergy,
and context-dependence** (Catarino & Stark 2018 on enhancer necessity/sufficiency; Brosh et al.
2023, enhancer context-dependence at *Sox2*; super-enhancer constituent hierarchy). But that
architecture is today accessible almost only through **CRISPR/synthetic perturbation** — there is
no established way to infer it **observationally** from multi-omics.

Our earlier analysis found the raw peak↔peak edges are ~all confounded (49% bidirected, 1/243
directed) — shared latent drivers (TF programs, chromatin domains, cell-state), not direct
causation. **That is not a reason to delete them; it is the reason to model confounders.** Once we
condition on the observed confounders that are actually available (library depth — donor is not a
usable variable, see [LIMITATIONS.md](LIMITATIONS.md#l1-donor-is-not-an-available-variable)) and
demand invariance across **cell types**, the peak↔peak edges that *survive* are the credible
candidates for **element-level causal
dependence (enhancer hierarchy/co-regulation)** — the observational-causal-inference counterpart of
what CRISPR measures. Either outcome is a real finding:
- **few survive** → a gene's enhancers are largely *independent contributors co-driven by shared
  TFs* (supports the additive activity-by-contact view, argues against strong hierarchy);
- **some survive** → candidate **enhancer hierarchy/dependence discoverable without perturbation.**

The method is the contribution regardless of which biological answer emerges.

## 3. Positioning against the literature (what is NOT our novelty)

| Prior work | What it does | Why it is not this project |
| --- | --- | --- |
| **ABC / ENCODE-rE2G** (Gschwind et al. 2023, 13M enhancer-gene links, 352 cell types) | predict enhancer→gene links | peak→gene only; predictive/associational; no element↔element causal structure. *We use it as a validation axis, not a method to beat.* |
| **SCENIC+** (Bravo González-Blas et al. 2022) | enhancer-driven eGRN (TF→enhancer→gene) | TF-centric, correlational/regression; no formal causal graph, no latent-confounder handling, no element↔element architecture |
| **Dictys** (Wang et al. 2023, *Nat Methods*) | dynamic TF→gene GRN from multiome | TF-centric network model; "causal" via a stochastic model, not constraint-based discovery of the cis-element architecture |
| **SCENIC / ARACNE / co-expression** | correlational/MI GRNs | associational, not interventional-calculus causal; gene–gene, not cis-element |
| **MOFA** (Argelaguet et al. 2018) | unsupervised multi-omics factors | descriptive latent factors, not causal (we tested this — factors capture confounders, not the gene-driver) |
| **CRISPR / Perturb-seq** (Ray et al. 2025; Catarino & Stark 2018; Brosh et al. 2023) | perturb enhancers → measure necessity/hierarchy | the gold standard for element↔element causality — but requires perturbation; we target the same object *observationally* |
| **Co-accessibility / co-activity** — Cicero (Pliner 2018), JRIM (Dong 2021), **Ziyani et al. 2023** (89k enhancer–enhancer co-activities from multiome) | undirected statistical dependence between elements | **our closest contrast + strongest hook**: they **explicitly concede** they cannot separate a *direct* element–element link from co-dependence via a **shared hidden driver**; undirected, no latent-cause edge. We resolve exactly that confound. |
| **MIIC** (Verny et al. 2017, *PLoS Comput Biol*) | ancestral graph w/ **bidirected `<->` = latent common cause** on genomic data | THE precedent to own: bidirected-latent-cause is not new *in genomics* — but MIIC is **gene-level only** (no chromatin, no enhancers, no multiome) |
| **CausalCell** (Wen et al. 2023, *eLife*) | PC/GES/GSP/DAGMA on scRNA-seq | causal-structure learning in single cell — but **assumes causal sufficiency (no latent), scRNA only, no chromatin/elements** |
| **Squires & Uhler 2022** (*CLeaR*) | causal structure among **clusters sharing a latent parent** | theory foundation we *extend* to enhancer clusters; not multiome, not validated biologically |

**The gap (sharpened after the 2026-07-15 competitive scan — see
[competitive_landscape_novelty.md](../CausalDiscovery/results/competitive_landscape_novelty.md)):**
every latent-variable causal-discovery method (MIIC, CausalCell, Squires & Uhler, Luo 2025) works on
**gene–gene expression**; every enhancer-level multiome method (SCENIC+, Cicero, Ziyani…) is
**correlation/regression without latent-variable causal search**. The two literatures have **never
been joined at the element level**, and never on scRNA + histone **CUT&Tag**. So the open unit is not
"a PAG" and not "bidirected = latent cause" (both taken) but: **latent-variable causal discovery
brought to enhancer-level multiome, which *operationalizes the shared-TF confound the co-accessibility
field concedes it cannot resolve*, and validates the resulting bidirected co-regulation modules
against orthogonal axes** (for which no accepted element↔element null yet exists — itself a
contribution).

## 4. What is genuinely novel here (the claim)

1. **Object:** the *local causal architecture* of a gene's cis-regulation — element→gene **and**
   element↔element — rather than a flat list of enhancer→gene links.
2. **Method:** formal **constraint-based causal discovery** (FCI/GFCI → **PAG**, *no causal
   sufficiency assumed*), with **observed confounders as root nodes** (library depth; donor and
   cell-cycle were investigated and are not applicable — [L1](LIMITATIONS.md#l1-donor-is-not-an-available-variable)),
   **mechanistic orientation constraints** (chromatin→expression), and **invariance across cell
   types** as the route to causal claims under unavoidable insufficiency.
3. **Signal:** multi-**histone-mark** CUT&Tag (a regulatory *state*: active/poised/repressed),
   not just ATAC accessibility; aggregated into interpretable regulatory **elements**.
4. **Validation:** triangulation across **three orthogonal axes** — prediction (rE2G), genetics
   (eQTL), physical (PCHi-C) — plus the element↔element axes and cross-cell-type invariance, kept
   **independent** of the discovery priors to avoid circularity.
5. **Honesty:** a PAG (with explicit bidirected/latent-confounder edges), stability selection, and
   null calibration — the causal claims are conditional, falsifiable, and independently checked.

**Framing guardrail (from the competitive scan):** do **not** headline "our PAG distinguishes
directed/direct/bidirected edges" — that is the generic, defining output of FCI and invites "you ran
an off-the-shelf algorithm." **Concede** the machinery (FCI taxonomy; MIIC already made
bidirected-latent edges on gene data) and **relocate** the novelty to: (a) the **domain transfer** to
enhancer-level multiome + the tiered mechanistic prior that makes the PAG orientable here; (b)
**resolving the shared-latent-driver confound** the co-accessibility field names but leaves open; (c)
**validating element↔element edges**, where no accepted null exists. Cite MIIC and Squires & Uhler
up front as the lineage we extend.

## 5. Honest scope & limits

The full ledger lives in **[LIMITATIONS.md](LIMITATIONS.md)** (single source of truth — do not
restate caveats here). The headline four:

- **Observational, not interventional** — every causal claim is conditional on assumptions plus
  independent corroboration; no perturbational proof ([L6](LIMITATIONS.md#l6-observational-pag-not-dag)).
- **Causal insufficiency is permanent** — handled by PAG bidirected edges, not pretended away.
- **Small n** (≈133 metacells/gene) — hence stability selection ([L3](LIMITATIONS.md#l3-small-n-and-n_eff-is-not-reported)).
- **Local, per-gene** graphs — not a genome-wide DAG.

## 6. How the current direction serves the mission

The modeling exploration (`CausalDiscovery/results/element_level_modeling_design.md`) and the
confounder/invariance discussion converge on the design the **original plan already specified**
(`docs/planning/local_multiome_notears_plan.md`): **element-level nodes + observed confounders (as
roots) + mechanistic constraints + invariance**, discovered with FCI and validated on independent
axes. That is precisely the machinery needed to turn the confounded raw peak↔peak soup into credible
element↔element causal architecture — i.e., to deliver §1's novel object.

> **Correction (2026-07-20):** earlier drafts named *donor* invariance as this pillar. Donor is not a
> usable variable in this dataset ([L1](LIMITATIONS.md#l1-donor-is-not-an-available-variable)), so the
> invariance axis is **cross-cell-type** (monocyte↔T-cell, ICP). The T-cell suite is built; the ICP
> run is open work in [ROADMAP](ROADMAP.md) §0.5.

**Bottom line:** this project is not "peak→gene link prediction with extra steps." It is an attempt
to read the **causal wiring among a gene's regulatory elements** off observational single-cell
epigenomes — a question otherwise reserved for CRISPR — with the confounding and insufficiency
handled honestly. The peak↔peak edges are the deliverable; the confounder/invariance machinery is
what makes them believable.

---

*Key references (verify DOIs before formal citation):* **Lineage to own up front** — MIIC (bidirected
latent-cause on genomic data) Verny et al. 2017 (*PLoS Comput Biol* 13(9):e1005662); latent-factor
cluster causal discovery — Squires & Uhler 2022 (*CLeaR*, arXiv:2207.01237); tiered-FCI w/ latent
vars — Bang & Didelez 2025 (arXiv:2503.21526). **Closest contrasts** — Cicero (Pliner et al. 2018,
*Mol Cell*), enhancer co-activity Ziyani et al. 2023 (*Commun Biol*), JRIM (Dong & Zhang 2021),
CausalCell (Wen et al. 2023, *eLife*), latent-confounder GRN Luo et al. 2025 (arXiv:2501.10124).
**Validation stack** — ENCODE-rE2G Gschwind et al. 2023 (*Nature*); ABC Fulco 2019 / Nasser 2021;
eQTL Catalogue Kerimov 2021/2023; BLUEPRINT PCHi-C Javierre et al. 2016 (*Cell*); distance-matched
null BENGI, Moore 2020. **eGRN incumbents** — SCENIC+ Bravo González-Blas et al. 2023 (*Nat Methods*);
Dictys Wang et al. 2023; GRaNIE, FigR, CellOracle, Pando, DIRECT-NET, scMEGA. **Biology** — enhancer
necessity/sufficiency Catarino & Stark 2018; context-dependence Brosh et al. 2023; Perturb-seq Ray et
al. 2025; enhancer co-regulation via shared TF Malin & Hannenhalli 2013 (*NAR*). MOFA — Argelaguet et
al. 2018. Invariant causal prediction — Peters, Bühlmann & Meinshausen 2016 (*JRSS-B*). Full ranked
landscape + reviewer-objection counters:
[CausalDiscovery/results/competitive_landscape_novelty.md](../CausalDiscovery/results/competitive_landscape_novelty.md).
