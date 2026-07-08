# scGLUE to Causal Discovery Plan

## Goal
Use scGLUE for cross-modality alignment, then run causal graph discovery for target genes with interpretable histone features.

## Key principle
Treat scGLUE as infrastructure for cell-state alignment, not the final causal variable set.

## Outputs to keep from scGLUE
1. `rna_with_glue.h5ad` and `chrom_<MARK>_with_glue.h5ad`
2. Cell embeddings (`X_glue`) for alignment/QC and covariate control
3. Feature embeddings (`feature_glue_embeddings.tsv`) for exploratory interpretation
4. `train_summary.json` and guidance summary files for provenance

## Causal inputs to build (recommended)
1. Gene expression matrix for target genes (cell or metacell level)
2. Gene-centric histone features:
   - promoter signal per mark
   - enhancer/nearby region signal per mark
3. Region-to-gene mapping table:
   - region type (promoter/enhancer/candidate)
   - distance to TSS
   - prior graph weight/sign
4. Covariates table:
   - batch/donor/depth/cell state

## Should bins be used directly?
1. Use bins for integration preprocessing and consistent feature space.
2. Do not feed all bins directly into causal discovery (too high-dimensional and hard to interpret).
3. Aggregate/select bins into biologically interpretable gene-centric features first.

## Recommended analysis strategy
1. Stage 1 (screening): mark-level graph
   - nodes: mark summaries around gene + gene expression
   - objective: identify candidate marks with strongest directional influence signals
2. Stage 2 (refinement): locus-level graph
   - nodes: promoter/enhancer loci (per mark) + gene expression
   - objective: localize likely regulatory regions for target genes

## Stability recommendations
1. Prefer metacell or pseudobulk representations for causal discovery stability.
2. Keep a strict run manifest and fixed preprocessing rules across marks.
3. Run sensitivity checks across:
   - neighborhood size / metacell settings
   - region windows
   - covariate sets

## Interpretation caveat
Causal discovery on observational single-cell multi-omics data gives causal hypotheses, not definitive causal proof. Prioritize perturbation/orthogonal validation for final claims.

## Immediate next implementation tasks
1. Create a script to build target-gene causal matrices from integrated outputs.
2. Implement promoter/enhancer feature extraction per mark.
3. Export one tidy table per target gene:
   - cells/metacells x (expression + regional histone features + covariates)
4. Add a runner that performs stage-1 screening then stage-2 refinement.
