# Git transfer and reproduction

Clone the handoff commit from Git, or clone its bundle on a machine without access
to the remote. The Git root is this directory, `Epigenomics-GEX-causal-analysis`.

```bash
git clone /path/to/epigenomics-handoff.bundle Epigenomics-GEX-causal-analysis
cd Epigenomics-GEX-causal-analysis
```

Git carries source, environment specifications, analysis settings, dataset
recipes, and selected research records. The data, fitted models, local
environments and generated workspaces stay outside Git. The old run IDs and
their recorded checksums refer to historical artifacts. Replays generate new
receipts in their own output directories.

## Exact-model simulation on CPU

Use Python 3.11. The historical run used Python 3.11.14 and one BLAS thread
per worker. This path needs no biological dataset or GPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r envs/requirements-replay-cpu.txt
python -m pytest -q pipelines/simulation/integration_recoverability/tests/test_count_transport*.py

bench=pipelines/simulation/integration_recoverability
python "$bench/replay_count_transport.py" transfer-input-check --stage generate
python "$bench/replay_count_transport.py" transfer-e2 --stage search
```

Use a new run ID for each command. `generate` reconstructs the nine seeded
worlds. `matched` also fits and evaluates the original comparison. `search`
then fits and evaluates the later count-informed source search. Each output is
under `runs/<run-id>/`. The complete search took about 51 minutes in the
original two-worker run, plus the matched comparison. The previous aggregate
results and the generation hashes are in the saved run records. Numerical
library and BLAS builds can affect floating-point results and optimizer paths.
[Matched experiment](pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_MATCHED.md) ·
[Source search](pipelines/simulation/integration_recoverability/COUNT_TRANSPORT_SOURCE_SEARCH.md)

## Empirical inputs

Install `curl` for public downloads. The input manifest records URLs, paths, transformations, and
the SHA-256 and byte count of each original local file. Use the same downloader
for the six CUT&Tag marks and 10x RNA (`pbmc`), GENCODE v49 (`reference`), and
the separate scMTR-seq S1 cohort (`scmtr`). The 10x barcode CSV comes from
the vendor HDF5, in its original row order. ADT/HTO tables are present in the
commit for compatibility and are also mapped to their GEO accessions.
[Input manifest](reproduce/inputs.json) ·
[GEO GSE195725](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE195725) ·
[10x RNA dataset](https://www.10xgenomics.com/datasets/10k_Human_PBMC_5p_v3_Ultima) ·
[GENCODE v49](https://www.gencodegenes.org/human/release_49.html) ·
[scMTR-seq GSE277778](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE277778)

```bash
# From the repository root. h5py is needed for the PBMC barcode file.
python reproduce/restore_inputs.py pbmc reference
python reproduce/restore_inputs.py pbmc reference --check
python reproduce/restore_inputs.py scmtr
```

If a publisher returns a truncated file or is unavailable, download the
listed file on a network that reaches the provider and place its final
(unpacked where indicated) bytes under the manifest path in a mirror directory.
Pass `--mirror /path/to/mirror`. The checksum check is the same for the
mirror. Use `--only <filename>` to retry one input.

For the empirical workflows, use the Linux environment and pip overlay described
in the environment guide. Activate the environment so `python3`, `Rscript` and
`macs3` resolve from it. The preprocessing script invokes the existing QC, peak
calling and fixed-bin workflows with the saved 500 bp bins and creates the
integration workspace. Run it in a fresh checkout with restored inputs.
[Environments](envs/README.md) ·
[Preprocessing](pipelines/empirical/data/results/preprocessing_notes.md)

```bash
bash reproduce/preprocess_pbmc.sh
export NUMBA_CACHE_DIR="$PWD/.replay-cache/numba"
export MPLCONFIGDIR="$PWD/.replay-cache/matplotlib"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
gtf=pipelines/empirical/integration/workspace/data/gene_annotation/gencode.v49.primary_assembly.annotation.gtf.gz
bash pipelines/empirical/integration/methods/scglue/run_joint_pipeline.sh transfer_scglue "$gtf"
```

For scMRDR, apply its joint pipeline to the rebuilt workspace and the same GTF.
The analysis and evaluation guides cover graph execution and the download
builders for external evidence. The scMTR-seq builders create a separate cohort
from its restored raw data. Its historical provenance is stored in `records/`
so it cannot block the builders from creating their output directories.
[scMRDR](pipelines/empirical/integration/methods/scmrdr/README.md) ·
[Analysis](pipelines/empirical/analysis/README.md) ·
[Evaluation](pipelines/empirical/evaluation/README.md) ·
[scMTR-seq cohort](pipelines/empirical/data/scmtr_seq/README.md)

## Other simulation tracks and historical runs

| Track | Inputs and replay |
| --- | --- |
| Aggregation | Run the seeded generator and experiments in the [aggregation guide](pipelines/simulation/aggregation_identifiability/README.md). |
| CA-BOSS | Rebuild the formal worlds with `multimodal_simulation_v4.py` using the seed, base cell count and background feature count from the [world inventory](reproduce/formal_worlds.tsv). Keep discovery and scorer outputs separate. The [benchmark guide](pipelines/simulation/unpaired_multimodal/README.md) links the experiment logs. |
| Integration benchmark | Restore the frozen calibration and world bank, then run the generator below. Available world IDs are in the [execution record](pipelines/simulation/integration_recoverability/IMPLEMENTATION_STATUS.md). The original suites and validation protocols are preserved in [historical recipe records](reproduce/records/integration_recoverability/). |
| Sturma replication | Fetch the upstream revision below, then follow the [replication guide](project_resources/research/sturma_pilot_2026_11_01/README.md). |

For example, to reconstruct one CA-BOSS world in a new output directory:

```bash
python pipelines/simulation/unpaired_multimodal/scripts/multimodal_simulation_v4.py \
  --seed 49011 --base-cells 1200 --n-background-features 64 \
  --out-dir pipelines/simulation/unpaired_multimodal/worlds/replay/world_seed49011
```

For the original integration worlds, the repository carries the frozen feature
summaries and exposure profiles used to calibrate the generator. They are stored
compressed, without the original count matrices or cell identifiers. Use
`calibration.py` from the original benchmark if you need to compute new profiles
from a rebuilt scGLUE joint run.

```bash
python reproduce/restore_inputs.py benchmark
python reproduce/run_integration_benchmark.py generate --world-id linear_chain_0
python reproduce/run_integration_benchmark.py generate --world-id constrained_collider_0
```

The Sturma method has a nested upstream source checkout. Restore that separately
at the pinned commit from the study protocol.

```bash
upstream=project_resources/research/sturma_pilot_2026_11_01/upstream
git clone https://github.com/NilsSturma/multi-domain-crl.git "$upstream"
git -C "$upstream" checkout --detach 578b5900590164b2ed1750fb6a7f98ae7153e220
```

Some older runs require original intermediate artifacts, absolute paths or a
particular CUDA runtime to validate their frozen receipts. Preserve those
receipts and generate new ones for relocated runs. For byte-for-byte access to
those historical outputs, keep a separate archive of the original workspace
using [the archive exclusion list](.transfer-exclude). This is also the fallback
for manual or unpublished inputs whose historical bytes cannot be fetched.
The archived GPU runtime has a checksum record and a published redistributable.
[CUDA receipt](reproduce/records/cuda-13.0.96.json) ·
[NVIDIA CUDA runtime archive](https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.0.96-archive.tar.xz)

[Clean-checkout verification](reproduce/VERIFICATION.md) records the checks and
remaining limits of the handoff.
