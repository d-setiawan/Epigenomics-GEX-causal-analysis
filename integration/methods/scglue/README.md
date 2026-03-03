# scGLUE Method Track

Use this folder for scGLUE-specific training and evaluation scripts.

Expected shared inputs:
- `integration/manifests/scglue_input_manifest.tsv`
- `integration/workspace/data/rna/gex_feature_cell_matrix.h5`
- `integration/workspace/data/chromatin/<MARK>/bin_chromatin_clean.*`

Recommended outputs:
- `integration/outputs/scglue/<run_id>/`
- `integration/logs/scglue_<run_id>.log`
