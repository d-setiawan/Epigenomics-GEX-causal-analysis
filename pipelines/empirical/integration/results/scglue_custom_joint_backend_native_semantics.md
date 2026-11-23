# Custom Joint Graph Backend (Native Interval Semantics)

This note documents the upgraded custom graph backend used by:

- `pipelines/empirical/integration/methods/scglue/build_guidance_graph_joint_scglue.py`
- with `--backend custom`

Goal:

1. Keep memory usage stable for multi-mark joint runs.
2. Match scGLUE native interval semantics (`promoter`, `gene_body`, `combined`) as closely as possible.

---

## Why we needed this

Native `scglue.genomics.rna_anchored_prior_graph` is biologically correct, but can be memory-heavy in large joint settings because it constructs and composes large in-memory `networkx` graphs.

The custom backend now keeps the same interval logic while streaming edges to TSV, so graph construction does not require one giant graph object in RAM.

---

## High-level pipeline

For each run:

1. Load `rna_preprocessed.h5ad`.
2. Load all chromatin modalities from `chrom_preprocessed_manifest.tsv`.
3. Build gene coordinate table from GTF (or load a prebuilt table).
4. Map RNA features to genes (`gene_name` or `gene_id` mode).
5. Apply gene-region semantics (`promoter`, `gene_body`, or `combined`).
6. For each histone mark:
   - parse chromatin bins into intervals,
   - generate gene-bin pairs using windowed interval-distance semantics,
   - write directed edges (`fwd` and `rev`) directly to `guidance_edges.tsv`,
   - add self-loops.
7. Write `guidance_nodes.tsv` and `guidance_summary.json`.

---

## Native interval semantics now implemented

### 1) Gene intervals

From GTF:

- `chromStart = start - 1`
- `chromEnd = end`
- strand is preserved (`+` or `-`)

### 2) `gene_region` modes

These mirror scGLUE behavior:

1. `gene_body`:
   - use `[chromStart, chromEnd)` directly.
2. `promoter`:
   - convert each gene to strand-specific TSS 1-bp interval.
   - expand upstream by `promoter_len`.
3. `combined`:
   - use gene body and expand upstream by `promoter_len`.

Upstream is strand-aware:

- `+` strand: upstream extends toward smaller coordinates.
- `-` strand: upstream extends toward larger coordinates.

### 3) Interval distance rule

Distance between gene interval `L` and bin interval `R`:

1. `0` if they overlap.
2. Negative if `L` is left of `R`.
3. Positive if `L` is right of `R`.

A pair is connected when:

`-window_bp <= dist <= window_bp`

Stored edge distance is `abs(dist)` (same convention as scGLUE graph attributes).

---

## Weighting

Two modes:

1. `--weight-mode exp` (default):
   - `weight = alpha * exp(-dist / decay_bp)`
2. `--weight-mode power`:
   - `weight = alpha * ((dist + 1000) / 1000)^(-0.75)`
   - This mirrors scGLUE's default power-decay form (with optional `alpha` scaling).

Sign per mark follows the configured histone sign map (`+1` activator-like, `-1` repressor-like).

---

## Memory-stability design

What keeps memory stable:

1. Per-mark processing (not all marks at once).
2. Streaming edge writes to TSV as pairs are generated.
3. Optional `--max-edges-per-bin` pruning:
   - `0` means no pruning (best semantic parity),
   - `>0` keeps nearest K genes per bin.
4. Optional GraphML export is off by default (`--no-export-graphml`) because it is memory-intensive.

---

## Main outputs

Under:

- `pipelines/empirical/integration/runs/scglue_<RUN_ID>/graph/`

Files:

1. `guidance_edges.tsv`
2. `guidance_nodes.tsv`
3. `guidance_summary.json`
4. optional `guidance_graph.graphml`

`guidance_summary.json` includes:

1. global node/edge totals,
2. backend metadata (annotation mode, retained genes),
3. per-mark diagnostics:
   - used/dropped chrom features,
   - candidate/kept pairs,
   - connected gene/bin counts,
   - window/decay/alpha/sign settings.

---

## Recommended run pattern

Comprehensive semantics, no per-bin pruning:

```bash
bash pipelines/empirical/integration/methods/scglue/run_build_graph_joint.sh \
  joint_v1 /abs/path/to/gencode.v49.primary_assembly.annotation.gtf.gz \
  --backend custom \
  --gene-region promoter \
  --promoter-len 1000 \
  --window-bp 50000 \
  --weight-mode exp \
  --max-chrom-features-per-mark 15000 \
  --max-edges-per-bin 0 \
  --no-export-graphml
```

If runtime becomes too high, enable controlled pruning:

```bash
--max-edges-per-bin 30
```

