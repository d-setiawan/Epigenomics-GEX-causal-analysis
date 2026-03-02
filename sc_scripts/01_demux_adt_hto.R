suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
})

# -------------------------
# Args / config
# -------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  cat("Usage: Rscript 01_demux_adt_hto.R <adt.tsv> <hto.tsv> <out_prefix> [min_adt] [min_hto]\n")
  quit(status = 1)
}

adt_path <- args[[1]]
hto_path <- args[[2]]
out_prefix <- args[[3]]

MIN_ADT <- if (length(args) >= 4) as.numeric(args[[4]]) else 10
MIN_HTO <- if (length(args) >= 5) as.numeric(args[[5]]) else 10
if (is.na(MIN_ADT) || is.na(MIN_HTO)) {
  stop("min_adt and min_hto must be numeric.")
}

# -------------------------
# Helpers: read matrix (rows=features, cols=barcodes)
# -------------------------
read_tsv_matrix <- function(path) {
  # Primary expectation: TSV with features in col 1 and barcodes in header.
  # Some exports are space-delimited even when named "*.tsv", so retry with
  # generic whitespace splitting if tab parsing collapses to one column.
  df <- tryCatch(
    read.delim(path, header = TRUE, row.names = 1, check.names = FALSE),
    error = function(e) NULL
  )

  if (is.null(df) || ncol(df) <= 1) {
    df <- read.table(
      path,
      header = TRUE,
      row.names = 1,
      check.names = FALSE,
      sep = "",
      comment.char = "",
      quote = ""
    )
  }

  m <- as.matrix(df)
  storage.mode(m) <- "numeric"
  Matrix(m, sparse = TRUE)
}

normalize_barcodes <- function(x) {
  x <- trimws(x)
  # Normalize common 10x suffix conventions (.1 vs -1)
  x <- sub("\\.[0-9]+$", "", x)
  x <- sub("-[0-9]+$", "", x)
  x
}

# -------------------------
# Load matrices
# -------------------------
adt <- read_tsv_matrix(adt_path)
hto <- read_tsv_matrix(hto_path)

# Ensure barcodes intersect (with robust harmonization for suffix/style mismatches)
adt_bc_raw <- colnames(adt)
hto_bc_raw <- colnames(hto)

adt_bc_norm <- normalize_barcodes(adt_bc_raw)
hto_bc_norm <- normalize_barcodes(hto_bc_raw)

common_cells_norm <- intersect(adt_bc_norm, hto_bc_norm)
if (length(common_cells_norm) == 0) {
  stop("No overlapping cell barcodes between ADT and HTO matrices after barcode normalization.")
}
cat(sprintf(
  "Barcode overlap diagnostics: ADT=%d, HTO=%d, shared=%d\n",
  length(adt_bc_norm), length(hto_bc_norm), length(common_cells_norm)
))

# Use first occurrence per normalized barcode to avoid duplicate-column ambiguity.
adt_first <- !duplicated(adt_bc_norm)
hto_first <- !duplicated(hto_bc_norm)

adt <- adt[, adt_first, drop = FALSE]
hto <- hto[, hto_first, drop = FALSE]

adt_bc_norm <- adt_bc_norm[adt_first]
hto_bc_norm <- hto_bc_norm[hto_first]

adt_idx <- match(common_cells_norm, adt_bc_norm)
hto_idx <- match(common_cells_norm, hto_bc_norm)

adt <- adt[, adt_idx, drop = FALSE]
hto <- hto[, hto_idx, drop = FALSE]

colnames(adt) <- common_cells_norm
colnames(hto) <- common_cells_norm

# -------------------------
# Apply raw count thresholds
# -------------------------
adt_total <- Matrix::colSums(adt)
hto_total <- Matrix::colSums(hto)

keep_counts <- (adt_total >= MIN_ADT) & (hto_total >= MIN_HTO)
cells_pass_counts <- names(which(keep_counts))
cat(sprintf(
  "Threshold diagnostics: MIN_ADT=%d, MIN_HTO=%d, passing_cells=%d\n",
  MIN_ADT, MIN_HTO, length(cells_pass_counts)
))
if (length(cells_pass_counts) == 0) {
  stop(sprintf(
    paste0(
      "No cells pass thresholds: MIN_ADT=%d, MIN_HTO=%d. ",
      "Check ADT/HTO count distributions, barcode harmonization, or lower thresholds."
    ),
    MIN_ADT, MIN_HTO
  ))
}

# Subset matrices to threshold-passing cells
adt2 <- adt[, cells_pass_counts, drop = FALSE]
hto2 <- hto[, cells_pass_counts, drop = FALSE]

# -------------------------
# Seurat object + HTO demux
# Build object directly from HTO counts to avoid Assay5 dummy-layer edge cases.
# -------------------------
obj <- CreateSeuratObject(counts = hto2, assay = "HTO")

# Add assays
obj[["ADT"]] <- CreateAssayObject(counts = adt2)

# CLR normalize HTO then demultiplex
obj <- NormalizeData(obj, assay = "HTO", normalization.method = "CLR", margin = 2, verbose = FALSE)
obj <- HTODemux(obj, assay = "HTO", positive.quantile = 0.99, verbose = FALSE)

md <- obj@meta.data
# Seurat versions differ:
# - HTO_classification.global: Singlet/Doublet/Negative
# - HTO_classification: donor/hashtag assignment (e.g., HTO1)
if ("HTO_classification.global" %in% colnames(md)) {
  hto_class <- as.character(md$HTO_classification.global)
} else {
  hto_class <- as.character(md$HTO_classification)
}
cells_singlet <- rownames(md)[hto_class == "Singlet"]

out <- data.frame(
  barcode = rownames(md),
  adt_total = as.numeric(adt_total[rownames(md)]),
  hto_total = as.numeric(hto_total[rownames(md)]),
  hto_classification = hto_class,
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
