# CSF1R Locus Pilot

This note records the initial literature-backed `CSF1R` locus design for causal discovery in monocyte metacells.

## Why `CSF1R`

`CSF1R` is one of the best-known mononuclear-phagocyte regulators and has a classic intronic regulatory element, `FIRE`, that makes it a good first locus for a small, interpretable graph.

## Regions used in the pilot

### 1. Primary promoter

- gene: `CSF1R`
- genome build: `GRCh38`
- strand: `-`
- gene span from current repo GTF: `chr5:150053291-150113372`
- operational promoter window:

\[
\texttt{chr5:150112372-150114372}
\]

This is a simple \( \pm 1000 \) bp window around the annotated gene TSS.

### 2. FIRE intronic enhancer

- curated region:

\[
\texttt{chr5:150081538-150086859}
\]

- source: NCBI biological region `LOC111188156`
- aliases: `CSF1R promoter E2`, `fms intronic regulatory element`, `FIRE`

For the locus pilot, this region is treated as the best-defined cis-regulatory enhancer block near `CSF1R`.

## Literature anchors

1. Zhang et al. identified monocytic `CSF1R` promoter activity and transcription-factor binding at the promoter.
2. Sauter et al. characterized the conserved `FIRE` element as an essential macrophage regulatory element and orientation-specific transcribed enhancer.
3. Rojo et al. showed that deletion of `FIRE` strongly alters `CSF1R` expression and macrophage development in vivo.
4. NCBI now curates a human `CSF1R promoter/intronic regulatory region` entry with explicit genomic coordinates and `FIRE` aliasing.

## Modeling recommendation

For the first `CSF1R` causal graph, keep the variable set small:

- `expr__CSF1R`
- `promoter_primary_tss__H3K27ac`
- `promoter_primary_tss__H3K4me3`
- `promoter_primary_tss__H3K4me1`
- `fire_curated__H3K27ac`
- `fire_curated__H3K4me1`
- optionally one repressive node such as `fire_curated__H3K27me3`

This is preferable to using all nearby bins independently.
