# scGLUE Method Track

Use this folder for scGLUE-specific training and evaluation scripts.

Expected shared inputs:
- `integration/manifests/scglue_input_manifest.tsv`
- `integration/workspace/data/rna/gex_feature_cell_matrix.h5`
- `integration/workspace/data/chromatin/<MARK>/bin_chromatin_clean.*`

Recommended outputs:
- `integration/outputs/scglue/<run_id>/`
- `integration/logs/scglue_<run_id>.log`

## Pilot preprocessing script

Script:
- `integration/methods/scglue/scripts/preprocess_pilot_scglue.py`

Wrapper:
- `integration/scripts/run_preprocess_scglue_pilot.sh`

Example:

```bash
bash integration/scripts/run_preprocess_scglue_pilot.sh H3K4me1
```

If `scglue` import fails due environment incompatibility, you can still run
preprocessing with sklearn LSI fallback:

```bash
bash integration/scripts/run_preprocess_scglue_pilot.sh H3K4me1 --chrom-lsi-backend sklearn
```

What it does:
1. Loads RNA `.h5` and one mark's chromatin bin matrix from the shared manifest.
2. RNA preprocessing: filter, HVG selection, normalize/log, PCA.
3. Chrom preprocessing: select top chromatin features, run LSI (`scglue.data.lsi` by default, fallback available).
4. Writes preprocessed AnnData files and a JSON summary.
