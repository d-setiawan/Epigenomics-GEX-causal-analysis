# Transfer verification

The November 22 handoff was tested from an export of the Git index at a
different absolute path, starting without raw inputs, generated worlds or
fitted models. A new Python 3.11.14 virtual environment was installed from the
CPU requirements. Empirical preprocessing checks used the existing Linux
`scrna_r` environment against the exported source.

| Check | Result |
| --- | --- |
| Exact-model CPU tests | 112 passed in the fresh virtual environment. |
| Matched simulation | All 46 original generator artifact hashes matched. The full 72-start fit and 144-artifact evaluation completed with the same aggregate world recovery counts. |
| Source search | Froze a follow-up against the new matched run and completed a public-count fit. The complete follow-up search was previously executed and is documented with its original receipts. |
| Empirical, integration and CA-BOSS tests | 51 passed, with four additional subtests. |
| Empirical sources | Six GEO fragment downloads matched after decompression. The 10x HDF5 and annotation tarball and the GENCODE GTF matched directly. The regenerated RNA barcode CSV matched. The full input inventory was also validated locally through the mirror route. |
| PBMC preprocessing | All six marks completed. The 24 compared files matched byte for byte across clean barcodes, peak BEDs, peak matrices and 500 bp fixed-bin matrices. All 27 workspace checks passed. |
| scMTR-seq S1 | All seven source files matched the publisher downloads. Rebuilt the full cohort and all six learner H5ADs. Their artifact hashes matched the saved cohort. Provenance referencing the new location produced new receipt hashes. |
| CA-BOSS seed 49011 | 32 of 40 discovery payloads and all 12 scorer payloads matched byte for byte. The eight differing files describe alignment or contain derived embedding/centroid values. |
| Integration bank | `linear_chain_0` and `constrained_collider_0` regenerated with byte-identical world manifests. |
| Upstream Sturma code | Fetched and detached at the documented Git commit. |

The input restorer rejected a corrupted incoming file and preserved an existing
file whose hash did not match. The source and plan files pinned by the
count-transport protocols were checked against the staged Git blobs.

The complete source-search repeat, GPU model training and installation of the
full Linux Conda lock were outside this verification. Historical artifact links
can require the original run cache. The legacy documentation hook was bypassed
for the migration commit because it reports pre-existing archive link and
orphan issues. The transfer and environment guide links were checked directly
against the Git index.

[Restore commands](../TRANSFER.md)
