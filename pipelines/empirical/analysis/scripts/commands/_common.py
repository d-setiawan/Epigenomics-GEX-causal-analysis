"""Shared helpers for pipelines/empirical/analysis command-line scripts."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def resolve_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def sanitize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "value"


def load_panel_genes(panel_path: Path | None, *, allow_none: bool = False) -> list[str] | None:
    if panel_path is None:
        if allow_none:
            return None
        raise ValueError("A gene panel path is required.")

    panel = pd.read_csv(panel_path, sep="\t")
    if "gene" not in panel.columns:
        raise ValueError(f"Gene panel missing 'gene' column: {panel_path}")
    genes = [str(gene).strip() for gene in panel["gene"].dropna().tolist() if str(gene).strip()]
    if not genes:
        raise ValueError(f"No genes found in gene panel: {panel_path}")
    return genes


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_bed(metadata: pd.DataFrame, out_path: Path) -> None:
    if metadata.empty:
        out_path.write_text("")
        return
    required = ["chrom", "start", "end", "variable_name"]
    missing = [col for col in required if col not in metadata.columns]
    if missing:
        raise ValueError(f"Cannot write BED; metadata missing columns: {missing}")
    bed = metadata[required].copy()
    bed["score"] = 0
    bed["strand"] = metadata["strand"].astype(str) if "strand" in metadata.columns else "."
    bed.to_csv(out_path, sep="\t", header=False, index=False)
