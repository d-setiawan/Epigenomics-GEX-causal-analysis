#!/usr/bin/env python3
"""Shared utilities for Jianle-track integration scripts."""

import sys
from pathlib import Path

# Direct script entry points put only the method directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integration_common import (
    apply_harmonization,
    build_embedding_df,
    infer_repo_root,
    load_harmonization_table,
    load_manifest_row,
    load_rna_cell_annotations,
    majority_vote,
    normalize_label_level,
    pick_rna_h5,
    read_named_column,
    resolve_path,
    strip_10x_suffix,
    write_tsv,
)
