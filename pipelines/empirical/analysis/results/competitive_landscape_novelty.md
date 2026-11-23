# Competitive landscape & novelty — where the gap actually is

**Date:** 2026-07-15. Basis for the paper's Related Work + the novelty claim in
[MISSION.md](../../../../project_resources/research/MISSION.md). Verify all DOIs before formal citation; two load-bearing claims are
flagged for re-check at the end.

## Bottom line

The method sits at the intersection of three mature-but-separate literatures. **Every individual
ingredient has a precedent; the specific combination is unoccupied.** But our *original* headline —
"a PAG distinguishes directed / direct / bidirected edges" — is the **weakest** way to state it,
because that edge taxonomy is the generic, defining output of FCI, and one paper (MIIC) already
emits bidirected "latent common cause" edges on genomic data. The novelty must be relocated to
**domain transfer + resolving a named confound + validating element–element edges.**

## The defensible gap (what is genuinely open)

1. **Domain transfer.** No one has run latent-variable constraint-based causal discovery
   (FCI/GFCI → PAG) at the **enhancer/chromatin-element level on single-cell multiome**. Every
   causal-discovery-with-latent-confounders paper (MIIC, CausalCell, Luo 2025, Squires & Uhler) is
   **gene–gene expression**; every enhancer-level multiome method (SCENIC+, Cicero, Ziyani…) is
   **correlation/regression without latent-variable causal search**. Never joined at the element
   level — and never on scRNA + histone **CUT&Tag**.
2. **Resolving the named confound.** Cicero, JRIM, and Ziyani 2023 *explicitly state* they cannot
   distinguish a direct element–element interaction from co-dependence via a shared hidden driver
   (a common TF). We turn that conceded limitation into an **explicit, discoverable bidirected edge.**
3. **Validating element–element edges.** There is **no established benchmark or accepted null for
   enhancer↔enhancer predictions** (only Lin 2022 *Science* and GLiMMIRS/Zhou 2024 test epistasis
   among *perturbed* enhancers — not predicted E–E edges vs ground truth). Treating bidirected edges
   as falsifiable co-regulation-module hypotheses and validating them is open territory.

**Net novel unit:** *"latent-variable causal discovery brought to enhancer-level multiome, which
operationalizes the shared-TF confound that co-accessibility methods concede they cannot resolve,
and subjects the resulting bidirected co-regulation modules to orthogonal validation."* Narrow but
real.

## Closest papers (ranked by threat to novelty)

1. **MIIC — Verny et al. 2017, *PLoS Comput Biol* 13(9):e1005662.** THE dangerous precedent:
   reconstructs ancestral graphs with **bidirected (↔) edges = unobserved latent common cause** from
   genomic data. But **gene-level only** (TFs/genes; qPCR, cancer expression) — no chromatin, no
   enhancers, no multiome. Establishes "bidirected = latent cause in genomics" as prior art. **Own
   this citation up front.**
2. **Ziyani, Delaneau, Ribeiro 2023, *Commun Biol* (bioRxiv 2022.10.13.511947).** Closest on
   biological target: ~89,885 **enhancer–enhancer co-activity** associations from single-cell
   multiome; method is **plain correlation, undirected**; co-active enhancers share TF profiles.
   **Explicitly names our gap as unsolved** ("could occur … due to regulation of other nearby genes …
   does not necessarily imply the two enhancers are acting together"). They see the confound; don't
   resolve it.
3. **Cicero — Pliner et al. 2018, *Mol Cell* 71(5):858.** Primary contrast: single symmetric
   **co-accessibility** score (distance-penalized graphical-LASSO over aggregated cells).
   **Undirected**; glasso removes indirect correlation through *observed* peaks but **cannot remove
   an unobserved shared driver.** Links are "putative."
4. **CausalCell — Wen et al. 2023, *eLife* 12:e81464.** Closest method-class in single cell: PC, GES,
   GSP, DAGMA on scRNA-seq. But **no FCI/PAG, no latent confounders** (assumes sufficiency), **scRNA
   only.**
5. **Malin & Hannenhalli 2013, *NAR* 41(14):6828.** Enhancer co-regulation precedent: ~314k
   correlated enhancer pairs (MI across 37 cell types), attributed to a **"common modulator" TF**,
   shared-motif SVM. Bulk DNase, correlation + motif — never models the shared TF as a *latent
   variable to recover.*
6. **SCENIC+ 2023, *Nat Methods* 20:1355** (+ GRaNIE, FigR, Dictys, CellOracle, Pando, DIRECT-NET,
   scMEGA). All infer **directed enhancer→gene** by regression/tree/correlation/motif; **direction is
   imposed by TF-motif priors + the cis-window, not learned.** None use causal-structure learning,
   model latent confounding, output bidirected/PAG, or predict enhancer↔enhancer. (CellOracle only
   *ingests* Cicero co-accessibility as input.)
7. **Squires, …, Uhler 2022, *CLeaR* (arXiv:2207.01237).** Theory analogue: causal structure among
   **clusters of variables sharing a latent parent**, motivated by GRNs. Our "latent TF induces a
   co-dependent enhancer cluster" is a domain instantiation — cite as theory foundation.
8. **Luo et al. 2025, arXiv:2501.10124.** Latent-confounder- & selection-bias-aware GRN inference —
   **gene expression only, no FCI/PAG, no chromatin.**
9. **ENCODE-rE2G/ABC** (Gschwind 2023; Fulco 2019; Nasser 2021) + **eQTL Catalogue** (Kerimov 2021/23)
   + **BLUEPRINT PCHi-C** (Javierre 2016): our exact 3-axis validation stack for enhancer→**gene**,
   scored vs a **distance-matched null** (BENGI, Moore 2020). Standard, expected.
10. **JRIM — Dong & Zhang 2021, *Brief Bioinform* 22(3):bbaa120.** Joint glasso co-accessibility
    across tissues; distinguishes direct vs indirect **among observed peaks** but **assumes
    sufficiency, undirected, no bidirected edges** — same ceiling as Cicero.

*Method foundation (not a competitor):* **Bang & Didelez 2025, arXiv:2503.21526** — constraint-based
causal discovery with **tiered background knowledge + latent variables** (general tiered-FCI): the
backbone for our chromatin-upstream-of-expression tiered prior + depth roots.

## Already done (concede these, don't claim them)

- Bidirected = latent common cause **in genomic data** → MIIC.
- Distinguishing directed/direct/bidirected edges → generic FCI output (Richardson–Spirtes ancestral
  graphs; Zhang 2008). **Do not headline this.**
- Causal-structure learning on single cell → CausalCell.
- Latent-confounder-aware GRN inference → MIIC, Luo 2025.
- Enhancer→gene links from multiome → crowded (8+ tools).
- Enhancer–enhancer co-regulation as a target w/ shared-TF interpretation → Ziyani 2023, Malin 2013.
- 3-axis enhancer→gene validation → standard.

## The most dangerous reviewer objection + counter

**Objection:** *"No methodological novelty — the edge taxonomy is generic FCI, MIIC already produced
bidirected latent-cause edges on genomic data, enhancer co-regulation via shared TFs is textbook,
co-activity is already mapped from multiome (Ziyani). You ran an off-the-shelf algorithm on a new
dataset."*

**Counter (3 moves):**
1. **Concede the machinery, relocate the novelty.** Agree the taxonomy is generic FCI and MIIC made
   bidirected edges — then show every such method is gene–gene expression, **never element-level,
   never multiome, never scRNA+histone CUT&Tag.** The contribution is the domain transfer + the
   **tiered mechanistic prior** (chromatin→expression; depth as root) that makes the PAG orientable
   here and is not portable from the gene-expression work.
2. **Point to the named-but-unsolved confound.** Cicero/JRIM/Ziyani *state* they can't separate a
   direct E–E link from a shared-latent-driver one. We are first to make that distinction **at the
   enhancer level**, turning a conceded limitation of the dominant methods into a discoverable edge.
3. **Validate what no one validates.** Enhancer→gene validation is standard; there is **no accepted
   null for enhancer↔enhancer.** Subjecting bidirected modules to orthogonal validation (shared-TF
   ChIP/motif, PCHi-C domain co-membership, distance/degree-matched nulls) converts "off-the-shelf
   output" into a tested biological claim the correlation incumbents never make.

**Guardrail:** drop "our PAG distinguishes three edge types" from the headline. Lead with *"first
latent-variable causal discovery at the enhancer level on multiome, resolving the
direct-vs-shared-latent-driver ambiguity co-accessibility methods concede, and validating the
resulting co-regulation modules."* Cite MIIC and Squires & Uhler 2022 **up front** as the lineage we
extend.

## Two claims to re-verify before submission
1. The enhancer↔enhancer **validation-null gap** (that no accepted E–E benchmark/null exists).
2. "**None** of the 8 multiome eGRN tools use causal-structure learning / model latent confounding."
