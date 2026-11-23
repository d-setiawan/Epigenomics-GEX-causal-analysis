# scMTR-seq S1 count pilot

This directory prepares one human scMTR-seq sublibrary for an integration
recoverability test using RNA, H3K27ac, and H3K4me3 counts. scMTR-seq is the
measurement technology. The scConfluence pilot used the prepared counts to test integration. The
learner receives artificially disjoint training cells. Original same-cell
identities and differentiation-day labels are available only to evaluation.

## Source and identity

The selected source is human H9 endoderm differentiation in
[GSE277778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277778):
[GSM8529544](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8529544)
contains S1 DNA fragments and
[GSM8529552](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8529552)
contains its RNA matrix. GEO explicitly links the original samples. Only these
two histone fragment files, the RNA matrix/features/barcodes, and two small
feature-reference files were downloaded, totaling approximately 160 MB.
No FASTQs or other histone count libraries were acquired.

The [study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12327464/) and
[pinned author scripts](https://github.com/Rugg-Gunn-Lab/scMTR-seq_paper_scripts/tree/0469c082532680d9c9152e0a365acf8dba781fe3/figure3)
establish that the first three split-pool barcode components plus sublibrary
identify a cell. The fourth RNA component is an RT/sample tag; it must not be
treated as a separate cell or blindly stripped. For each canonical barcode,
we retain the RNA row with the greatest matrix count and require its fraction
of all observed RT-tag counts to exceed 0.5 for RNA eligibility. Valid tags
01–04 correspond to D0–D3. This matrix-count fraction approximates the authors'
read-summary criterion; it is not an exact reconstruction of their unavailable
read-summary QC. Same-day doublets and barcode collisions remain unresolved.

The feature-reference files named `*.csv.gz` actually contain gzipped XLSX
workbooks. Their endoderm rows confirm antibody tags 02=H3K4me3 and
03=H3K27ac. Extracted rows and source receipts are in [metadata](metadata/).

**Inherited selection:** the authors filtered DNA cells on at least 2,500
observations summed across six targets before exporting the mark-specific
files. We cannot undo that selection using the three permitted assays. Our
subsequent training QC uses only the assigned assay, but the source population
already reflects this six-target filter. S1 is a technical sublibrary of pooled
days, not a verified independent biological replicate.

## Frozen split and counts

The source universe contains 2,444 canonical cells present in both selected
histone files. Seed 2718 hashes canonical IDs, reserves 600 cells for evaluation,
and assigns the remainder round-robin to RNA, H3K27ac, and H3K4me3 before QC.
RNA QC requires counts ≥1,000, detected genes ≥500, a dominant valid RT tag
fraction >0.5, and mitochondrial fraction ≤5%. Histone QC requires ≥250
genomic fragments in the observed assay. Evaluation applies all three criteria
to the reserved cells; its target population is therefore the internal
complete-case subset.

| Assay | Assigned training cells | Retained training cells | Encoder features |
| --- | ---: | ---: | ---: |
| RNA | 615 | 602 | 23,383 |
| H3K27ac | 615 | 615 | 26,569 |
| H3K4me3 | 614 | 614 | 22,481 |

The paired evaluation reference contains **595 cells**. Every learner row has
an opaque assay-specific ID. Evaluation rows are independently reordered in
each assay, so matching row positions do not disclose original pairing.
No original barcode, RT tag, day, or shared cell identifier enters a learner
matrix or feature cost.

The GENCODE v49 annotation restored by the input manifest identifies GRCh38 explicitly. Of
30,234 RNA Ensembl IDs, 29,361 match after version removal; restricting to
chr1–22/X/Y yields 29,338 gene-level promoters. Each promoter is the fixed
zero-based half-open window `[TSS-2000, TSS+2000)`, clipped at coordinate zero.
Each fragment contributes its multiplicity once to every overlapping window.
Overlapping promoters can therefore share counts; the selected evaluation
panel uses nonoverlapping loci.

Median genomic fragment counts are 1,568 for H3K27ac and 714 for H3K4me3.
Median promoter counts are 197 and 253, respectively. This is a restricted
promoter model: distal elements and alternative transcript-specific promoters
are not evaluated. The newer GENCODE annotation is not claimed to reproduce
the study's original annotation exactly.

Encoder features include every annotation-matched feature detected in its own
disjoint training assay after the panel exclusions below. There is no PCA or
variance ranking. Learner `library_depth` is the sum of retained encoder
features only, preventing heldout panel counts from entering through this
quantity. Original genomic/RNA depths remain evaluator-only. Promoter sums
can double-count fragments that overlap multiple promoters and are not unique
genomic fragment totals.

## Frozen nine-node panel

Eligibility uses only the corresponding unpaired training cells: at least
10% RNA detection, 5% detection in each histone assay, and at least 20 detected
cells in every assay. Three nonoverlapping loci are selected in ascending
SHA256 order of `20261101_causal_panel_v1|` plus unversioned Ensembl ID. The
rules fail if fewer than three loci qualify; thresholds are not relaxed.

Seven genes qualify. The selected panel is:

| Gene | Ensembl ID | RNA detections | H3K27ac detections | H3K4me3 detections |
| --- | --- | ---: | ---: | ---: |
| BRD2 | ENSG00000204256 | 203 | 31 | 38 |
| EPM2AIP1 | ENSG00000178567 | 116 | 32 | 33 |
| NBPF1 | ENSG00000219481 | 115 | 31 | 31 |

All six overlapping promoter/alias features are removed from every encoder
and feature-cost universe. Panel counts are evaluator-only. The sparse
histone detections can limit the power of conditional-dependence tests;
an empty or unstable paired-reference graph must not be reported as successful
causal recovery. Real paired observations supply a dependence reference, not
a known causal graph or known direction labels.

## Files and reproduction

- `raw_s1/`: original processed downloads and individual URL/size/SHA256 receipts.
- `prepared_s1/`: canonical RNA audit and preliminary count diagnostics.
- `full_s1/`: all source-cell promoter counts, canonical identities, annotation,
  and provenance; preparation/evaluation scope only.
- `cohort_s1/learner/`: sparse count NPZs, matching opaque cell TSVs, and assay
  feature TSVs. Each assay has separate `train` and `test` files.
- `cohort_s1/evaluator/`: canonical maps, heldout pairing, day labels, original
  depths, training-only coverage evidence, and panel count arrays.
- `cohort_s1/model_inputs/{train,test}/`: minimal per-assay H5AD files built
  exclusively from checksum-verified learner inputs. Raw integer counts are
  stored in `X`; `obs` has only its opaque ID index, and `var` contains allowed
  gene/coordinate annotations with `assay|gene_id` feature IDs. The files carry
  no observation columns, `uns`, embeddings, layers, or `raw` slot. Their
  separate `manifest.json` records source/output checksums, ordered ID hashes,
  software/helper versions, and exact count/annotation roundtrip checks.
- [frozen_protocol.json](records/cohort_s1/frozen_protocol.json) records the approved
  rules; [summary.json](records/cohort_s1/summary.json) records outputs and checksums.

The evaluator's `paired_panel_counts.npz` columns follow
`panel_nodes.tsv`: assay-major order RNA, H3K27ac, H3K4me3, with the three
genes in the table's hash-selection order. Rows follow `paired_panel_rows.tsv`.
Per-assay panel arrays follow the corresponding learner row order.

From the repository root in the empirical `scrna_r` environment, use the following commands in a fresh checkout. [Environment guide](../../../../envs/README.md)

```bash
python reproduce/restore_inputs.py reference scmtr
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python pipelines/empirical/data/scmtr_seq/prepare_s1.py \
  --gtf pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz
python pipelines/empirical/data/scmtr_seq/build_cohort_s1.py
python pipelines/empirical/data/scmtr_seq/build_model_inputs_s1.py
python -m unittest discover -s pipelines/empirical/data/scmtr_seq -p 'test_*.py' -v
```

The preparation/cohort builders refuse to overwrite existing output folders.
Original downloads are reused only after receipt verification. Transient empty
HTTP responses for tiny feature-reference files were rejected and retained as
failed partials; bounded retries completed those files. There is no advertised
independent GEO publisher checksum, so recorded SHA256 values identify the
downloaded bytes; content length and complete gzip reads were also checked.

Eight unit tests cover truncation/tampering, RT-tag ambiguity, exact genomic
overlap/TSS coordinates, split/QC separation, and panel exclusion. The saved
[artifact validation](metadata/artifact_validation.json) checks exact sparse
count joins, disjoint populations, identity separation, and panel removal.
The H5AD bridge tests also reject changed inputs and check count, row-order,
annotation, and metadata roundtrips. A fresh read of the original RNA matrix
exactly reproduced the retained sparse collapse and barcode winners; see
[RNA cache reproduction](metadata/RNA_cache_reproduction.json).
On transfer, all seven source files matched, the cohort was rebuilt, and all six learner H5AD hashes matched the original artifacts. Receipts containing the new absolute location had new hashes. The historical pilot found no cross-assay adjacency in the held-out paired panel with its specified test. [Fresh-export verification](../../../../reproduce/VERIFICATION.md) · [Pilot readout](../../../../project_resources/research/scconfluence_scmtr_pilot_2026_11_01/README.md)
