# MISSION — what this project is, and why it is novel

**One-line:** recover the **local *causal architecture* of a gene's cis-regulation** — not just
which enhancers regulate a gene, but the causal relationships *among* its regulatory elements —
from single-cell multi-omic histone-mark + expression data, using formal latent-confounder-aware
causal discovery, and triangulate it against orthogonal biological evidence.

This is the "why/what's novel" master doc. Operational "how" lives in [README.md](README.md);
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
design in `notes/local_multiome_notears_plan.md`). The **`A→A` (element↔element) structure is the
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
condition on observed confounders (donor/cell-cycle/library-size) and demand invariance across
donors, the peak↔peak edges that *survive* are the credible candidates for **element-level causal
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

**The gap:** formal, latent-confounder-aware **causal discovery (constraint-based → PAG)** of the
**local cis-regulatory architecture** (including element↔element structure) from **observational
single-cell multi-omics**, is not an established approach. A direct search for
"NOTEARS/structural-causal-model on single-cell multiome" and "causal discovery among regulatory
elements" returns essentially nothing on-target.

## 4. What is genuinely novel here (the claim)

1. **Object:** the *local causal architecture* of a gene's cis-regulation — element→gene **and**
   element↔element — rather than a flat list of enhancer→gene links.
2. **Method:** formal **constraint-based causal discovery** (FCI/GFCI → **PAG**, *no causal
   sufficiency assumed*), with **observed confounders as root nodes** (donor/cell-cycle/library),
   **mechanistic orientation constraints** (chromatin→expression), and **invariance across donors**
   as the route to causal claims under unavoidable insufficiency.
3. **Signal:** multi-**histone-mark** CUT&Tag (a regulatory *state*: active/poised/repressed),
   not just ATAC accessibility; aggregated into interpretable regulatory **elements**.
4. **Validation:** triangulation across **three orthogonal axes** — prediction (rE2G), genetics
   (eQTL), physical (PCHi-C) — plus donor-invariance, kept **independent** of the discovery priors
   to avoid circularity.
5. **Honesty:** a PAG (with explicit bidirected/latent-confounder edges), stability selection, and
   null calibration — the causal claims are conditional, falsifiable, and independently checked.

## 5. Honest scope & limits

- **Observational, not interventional.** Without monocyte CRISPR, every causal claim is conditional
  on assumptions (Markov, faithfulness, the mechanistic constraints) + corroboration by independent
  evidence + invariance. We state these; we do not claim perturbational proof.
- **Causal insufficiency is permanent** — handled by PAG (bidirected edges) + confounder
  conditioning + invariance, not by pretending it away.
- **Small n** (≈133 metacells/gene) — hence stability selection, transforms, and the modeling
  exploration below.
- **Local, per-gene** graphs — not a genome-wide DAG.

## 6. How the current direction serves the mission

The modeling exploration (`CausalDiscovery/notes/element_level_modeling_design.md`) and the
confounder/invariance discussion converge on the design the **original plan already specified**
(`notes/local_multiome_notears_plan.md`): **element-level nodes + observed confounders (as roots) +
mechanistic constraints + invariance across donors**, discovered with FCI and validated on
independent axes. That is precisely the machinery needed to turn the confounded raw peak↔peak soup
into credible element↔element causal architecture — i.e., to deliver §1's novel object.

**Bottom line:** this project is not "peak→gene link prediction with extra steps." It is an attempt
to read the **causal wiring among a gene's regulatory elements** off observational single-cell
epigenomes — a question otherwise reserved for CRISPR — with the confounding and insufficiency
handled honestly. The peak↔peak edges are the deliverable; the confounder/invariance machinery is
what makes them believable.

---

*Key references (retrieved via OpenAlex, 2026-07-10; verify DOIs before formal citation):*
ENCODE-rE2G encyclopedia — Gschwind et al. 2023 (*Nature*). SCENIC+ — Bravo González-Blas et al.
2022. Dictys — Wang et al. 2023 (*Nat Methods*). MOFA — Argelaguet et al. 2018 (*Mol Syst Biol*).
Enhancer necessity/sufficiency — Catarino & Stark 2018 (*Genes Dev*). Enhancer context-dependence —
Brosh et al. 2023. Perturb-seq enhancer screens — Ray et al. 2025. Cohesin/3D enhancer-promoter —
Guckelberger & Doughty 2024. Invariant causal prediction — Peters, Bühlmann & Meinshausen 2016
(*JRSS-B*; not retrieved here). FCI/PAG causal-discovery foundations — see
`CausalDiscovery/notes/aggregation_causal_discovery_litreview.md`.
