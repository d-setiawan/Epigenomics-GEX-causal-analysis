#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
})

# -------------------------
# Args / config
# -------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript 01_demux_hto_adt.R <adt.tsv> <hto.tsv> <out_prefix>\n")
  quit(status = 1)
}

adt_path <- args[[1]]
hto_path <- args[[2]]
out_prefix <- args[[3]]

MIN_ADT <- 10
MIN_HTO <- 10

# -------------------------
# Helpers: read TSV matrix (rows=features, cols=barcodes)
# -------------------------
read_tsv_matrix <- function(path) {
  df <- read.delim(path, header = TRUE, row.names = 1, check.names = FALSE)
  m <- as.matrix(df)
  # If counts are integer-like, keep numeric anyway
  m <- Matrix(m, sparse = TRUE)
  return(m)
}

# -------------------------
# Load matrices
# -------------------------
adt <- read_tsv_matrix(adt_path)
hto <- read_tsv_matrix(hto_path)

# Ensure barcodes intersect
common_cells <- intersect(colnames(adt), colnames(hto))
if (length(common_cells) == 0) stop("No overlapping cell barcodes between ADT and HTO matrices.")

adt <- adt[, common_cells, drop = FALSE]
hto <- hto[, common_cells, drop = FALSE]

# -------------------------
# Apply raw count thresholds
# -------------------------
adt_total <- Matrix::colSums(adt)
hto_total <- Matrix::colSums(hto)

keep_counts <- (adt_total >= MIN_ADT) & (hto_total >= MIN_HTO)
cells_pass_counts <- names(which(keep_counts))

# Subset matrices to threshold-passing cells
adt2 <- adt[, cells_pass_counts, drop = FALSE]
hto2 <- hto[, cells_pass_counts, drop = FALSE]

# -------------------------
# Seurat object + HTO demux
# We use a minimal RNA assay just to host cells; ADT/HTO live in their own assays
# -------------------------
dummy_rna <- Matrix::Matrix(0, nrow = 1, ncol = ncol(hto2), sparse = TRUE)
colnames(dummy_rna) <- colnames(hto2)
rownames(dummy_rna) <- "dummy"

obj <- CreateSeuratObject(counts = dummy_rna)

# Add assays
obj[["HTO"]] <- CreateAssayObject(counts = hto2)
obj[["ADT"]] <- CreateAssayObject(counts = adt2)

# Normalize (matching your description)
# HTO often uses CLR as well; Seurat vignette uses CLR for HTO
obj <- NormalizeData(obj, assay = "HTO", normalization.method = "CLR")
obj <- NormalizeData(obj, assay = "ADT", normalization.method = "CLR")

# Demultiplex
obj <- HTODemux(obj, assay = "HTO", positive.quantile = 0.99)

md <- obj@meta.data
# Seurat outputs columns:
#  - HTO_classification (Singlet/Doublet/Negative)
#  - HTO_maxID (best tag)
#  - HTO_secondID (second best)
#  - HTO_margin, etc.

# Keep singlets only
is_singlet <- md$HTO_classification == "Singlet"
cells_singlet <- rownames(md)[is_singlet]

# Build output table for all cells that passed count thresholds (before singlet filter)
out <- data.frame(
  barcode = rownames(md),
  adt_total = as.numeric(adt_total[rownames(md)]),
  hto_total = as.numeric(hto_total[rownames(md)]),
  hto_classification = md$HTO_classification,
  donor_id = as.character(md$HTO_maxID),
  second_id = as.character(md$HTO_secondID),
  hto_margin = as.numeric(md$HTO_margin),
  stringsAsFactors = FALSE
)

# Add convenience flags
out$doublet_flag <- out$hto_classification == "Doublet"
out$negative_flag <- out$hto_classification == "Negative"
out$singlet_flag <- out$hto_classification == "Singlet"

# Write:
# 1) all cells that passed ADT/HTO counts (includes doublets/negatives)
# 2) singlet-only list of barcodes for downstream intersection
write.table(out, file = paste0(out_prefix, "_hto_adt_metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

write.table(data.frame(barcode = cells_singlet),
            file = paste0(out_prefix, "_singlet_barcodes.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("Wrote:\n")
cat("  ", paste0(out_prefix, "_hto_adt_metadata.tsv\n"))
cat("  ", paste0(out_prefix, "_singlet_barcodes.tsv\n"))