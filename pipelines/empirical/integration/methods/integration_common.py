#!/usr/bin/env python3
"""Shared utilities for empirical integration methods."""

from __future__ import annotations

import csv
import tarfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def strip_10x_suffix(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"-\d+$", "", regex=True)


def pick_rna_h5(repo_root: Path, rna_h5: str | None) -> Path:
    if rna_h5:
        p = resolve_path(repo_root, rna_h5)
        if p is None or not p.exists():
            raise FileNotFoundError(f"RNA h5 not found: {p}")
        return p

    candidates = [
        repo_root / "pipelines/empirical/integration/workspace/data/rna/gex_feature_cell_matrix.h5",
        repo_root / "pipelines/empirical/data/raw/GexData/GEX_feature_cell_matrix_HDF5.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    gex_dir = repo_root / "pipelines/empirical/data/raw/GexData"
    for ext in ("*.h5", "*.hdf5"):
        found = sorted(gex_dir.glob(ext))
        if found:
            return found[0]
    raise FileNotFoundError("Could not find RNA h5 input. Provide --rna-h5 explicitly.")


def load_manifest_row(manifest_path: Path, mark: str) -> Dict[str, str]:
    df = pd.read_csv(manifest_path, sep="\t")
    if "mark" not in df.columns:
        raise ValueError(f"Manifest missing 'mark' column: {manifest_path}")
    hit = df.loc[df["mark"] == mark]
    if hit.empty:
        raise ValueError(f"Mark '{mark}' not found in manifest: {manifest_path}")
    return hit.iloc[0].to_dict()


def read_named_column(path: Path, preferred_col: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    if preferred_col in df.columns:
        return df[preferred_col].astype(str)
    return df.iloc[:, 0].astype(str)


def build_embedding_df(mat: np.ndarray, index: pd.Index, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i+1}" for i in range(mat.shape[1])]
    return pd.DataFrame(mat, index=index, columns=cols)


def write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_rna_cell_annotations(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r:gz") as tf:
        member = "cell_types/cell_types.csv"
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Missing {member} in {path}")
        ann = pd.read_csv(BytesIO(f.read()))
    required = {"barcode", "coarse_cell_type", "fine_cell_type"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")
    keep_cols = ["barcode", "coarse_cell_type", "fine_cell_type"]
    if "cell_count_in_model" in ann.columns:
        keep_cols.append("cell_count_in_model")
    ann = ann[keep_cols].copy()
    ann["barcode"] = ann["barcode"].astype(str)
    ann["barcode_core"] = strip_10x_suffix(ann["barcode"])
    return ann.drop_duplicates(subset=["barcode_core"])


def normalize_label_level(level: str) -> str:
    aliases = {
        "coarse_cell_type": "coarse_cell_type",
        "fine_cell_type": "fine_cell_type",
        "harmonized_coarse": "harmonized_coarse",
        "harmonized_fine": "harmonized_fine",
    }
    if level not in aliases:
        raise ValueError(f"Unsupported label level: {level}")
    return aliases[level]


def load_harmonization_table(path: Path) -> pd.DataFrame:
    harm = pd.read_csv(path, sep="\t")
    required = {"coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine"}
    missing = required - set(harm.columns)
    if missing:
        raise ValueError(f"Harmonization table missing columns: {sorted(missing)}")
    dup = harm.duplicated(subset=["coarse_cell_type", "fine_cell_type"])
    if dup.any():
        raise ValueError("Harmonization table has duplicate coarse/fine label pairs")
    if "cell_ontology_id" not in harm.columns:
        harm["cell_ontology_id"] = pd.NA
    return harm


def apply_harmonization(ann: pd.DataFrame, harm: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    merged = ann.copy()
    if harm is None:
        merged["harmonized_coarse"] = merged["coarse_cell_type"]
        merged["harmonized_fine"] = merged["fine_cell_type"]
        merged["cell_ontology_id"] = pd.NA
        return merged, {"mode": "identity", "n_unmapped_rows": 0}

    merged = merged.merge(
        harm[["coarse_cell_type", "fine_cell_type", "harmonized_coarse", "harmonized_fine", "cell_ontology_id"]],
        on=["coarse_cell_type", "fine_cell_type"],
        how="left",
    )
    unmapped = int(merged["harmonized_coarse"].isna().sum() + merged["harmonized_fine"].isna().sum())
    merged["harmonized_coarse"] = merged["harmonized_coarse"].fillna(merged["coarse_cell_type"])
    merged["harmonized_fine"] = merged["harmonized_fine"].fillna(merged["fine_cell_type"])
    return merged, {"mode": "table", "n_unmapped_rows": unmapped}


def majority_vote(labels: list[str]) -> tuple[str, float]:
    labels = [str(x) for x in labels if pd.notna(x) and str(x) != ""]
    if not labels:
        return "unknown", 0.0
    label, count = Counter(labels).most_common(1)[0]
    return label, float(count / len(labels))
