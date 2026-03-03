# Jianle Method Track

Use this folder for implementation of Jianle's integration method
(from `papers/IntegrationJianLe.pdf`).

Expected shared inputs are the same manifest/workspace used by scGLUE so
both methods can be compared on identical datasets.

Recommended outputs:
- `integration/outputs/jianle/<run_id>/`
- `integration/logs/jianle_<run_id>.log`

## Pilot preprocessing script

Script:
- `integration/methods/jianle/scripts/preprocess_pilot_jianle.py`

Wrapper:
- `integration/scripts/run_preprocess_jianle_pilot.sh`

Example:

```bash
bash integration/scripts/run_preprocess_jianle_pilot.sh H3K4me1
```

What it does:
1. Loads RNA `.h5` and one mark's chromatin bin matrix from the shared manifest.
2. RNA preprocessing: filter, HVG selection, normalize/log, PCA.
3. Chrom preprocessing: select top chromatin features, run TF-IDF + TruncatedSVD LSI.
4. Writes preprocessed AnnData files and a JSON summary.
