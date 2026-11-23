# Monocyte Locus Panel for One-to-One scGLUE Matching

The current monocyte panel is focused on **human** loci with direct promoter or enhancer evidence in monocyte-lineage contexts.

Each pilot graph stays intentionally small:

\[
\text{gene expression} + \text{promoter-region marks} + \text{one small number of named cis-regulatory regions}
\]

so that each locus remains tractable for a baseline PC run on the one-to-one matched pseudo-cells.

## Current panel

- `CSF1R`
  - Canonical monocyte/macrophage locus with a well-characterized promoter and the `FIRE` intronic enhancer.
  - Best current positive-control locus in this repo.

- `CD14`
  - Human monocyte marker with strong promoter evidence and validated downstream enhancer activity in human monocytes.
  - Good human-focused replacement for weaker exploratory loci.

- `IL1B`
  - Human monocyte inflammatory locus with classic promoter-centric regulation through `PU.1` and `C/EBP` family activity.
  - Best treated as a promoter-first validation locus.

- `CCR2`
  - Human monocyte chemokine receptor locus with direct promoter and `5'` UTR cis-regulatory characterization.
  - Useful for testing promoter/core-promoter structure in the matched pseudo-cell framework.

## Deprioritized loci

- `SPI1`
  - Biologically important, but the current human distal-window definition was too flat in the matched monocyte data to support a useful PC graph.

- `IRF8`
  - Potentially promising, but the pilot distal region relied on offset mapping from non-human enhancer evidence and is no longer part of the main panel.

## Why this panel

This panel favors loci where we can say both:

\[
\text{the gene is strongly monocyte-relevant}
\]

and

\[
\text{the cis-regulatory story is anchored in human literature}
\]

which makes the downstream PC results much easier to interpret against known biology.
