#!/usr/bin/env python3
"""Build a repo-local integration workspace from Data/ using symlinks.

Outputs:
- integration/workspace/data/{rna,chromatin,chromsizes}/...
- integration/manifests/scglue_input_manifest.tsv
- integration/manifests/step1_data_check.tsv

This keeps source-of-truth files in Data/ while standardizing paths for
integration scripts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REQUIRED_SUFFIXES: Sequence[str] = (
    "_bin_chromatin_clean.mtx",
    "_bin_chromatin_clean_barcodes.tsv",
    "_bin_chromatin_clean_features.tsv",
    "_clean_cells.tsv",
)

OPTIONAL_SUFFIXES: Sequence[str] = (
    "_bins.bed",
    "_peaks.bed",
    "_clean_fragments.tsv.gz",
    "_chromatin_clean.mtx",
    "_chromatin_clean_barcodes.tsv",
    "_chromatin_clean_features.tsv",
)

IGNORE_DIRS = {"GexData", "chromsizes", ".git", "sandbox_outputs"}


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def symlink_force(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    # relative symlink for portability
    dst.symlink_to(src.resolve())


def detect_prefix(outputs_dir: Path) -> Optional[str]:
    files = [p.name for p in outputs_dir.iterdir() if p.is_file()]
    candidates = sorted(
        {
            name[: -len(sfx)]
            for name in files
            for sfx in REQUIRED_SUFFIXES
            if name.endswith(sfx)
        }
    )
    for prefix in candidates:
        if all(f"{prefix}{sfx}" in files for sfx in REQUIRED_SUFFIXES):
            return prefix
    return None


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def write_tsv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup integration workspace with symlinked inputs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--check-out", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or (repo_root / "Data")).resolve()
    workspace_dir = (args.workspace_dir or (repo_root / "integration" / "workspace")).resolve()
    manifest_out = (args.manifest_out or (repo_root / "integration" / "manifests" / "scglue_input_manifest.tsv")).resolve()
    check_out = (args.check_out or (repo_root / "integration" / "manifests" / "step1_data_check.tsv")).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for d in (
        workspace_dir,
        workspace_dir / "data",
        workspace_dir / "data" / "rna",
        workspace_dir / "data" / "chromatin",
        workspace_dir / "data" / "chromsizes",
        repo_root / "integration" / "manifests",
        repo_root / "integration" / "logs",
        repo_root / "integration" / "outputs",
    ):
        ensure_dir(d)

    # RNA links
    gex_dir = data_dir / "GexData"
    h5 = first_existing(
        sorted(gex_dir.glob("*.h5")) + sorted(gex_dir.glob("*.hdf5"))
    )
    sample_barcodes = first_existing(sorted(gex_dir.glob("*barcodes*.csv")))
    cell_ann = first_existing(sorted(gex_dir.glob("*cell*type*annotation*.tar.gz")))

    if h5:
        symlink_force(h5, workspace_dir / "data" / "rna" / "gex_feature_cell_matrix.h5")
    if sample_barcodes:
        symlink_force(sample_barcodes, workspace_dir / "data" / "rna" / "gex_sample_barcodes.csv")
    if cell_ann:
        symlink_force(cell_ann, workspace_dir / "data" / "rna" / "cell_type_annotation.tar.gz")

    chromsizes = first_existing(
        [data_dir / "chromsizes" / "hg38.chrom.sizes"] + sorted((data_dir / "chromsizes").glob("*.chrom.sizes"))
    )
    if chromsizes:
        symlink_force(chromsizes, workspace_dir / "data" / "chromsizes" / chromsizes.name)

    manifest_rows: List[Dict[str, str]] = []

    mark_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name not in IGNORE_DIRS
    )

    for mark_dir in mark_dirs:
        outputs = mark_dir / "outputs"
        if not outputs.exists():
            continue
        prefix = detect_prefix(outputs)
        if not prefix:
            continue

        target_dir = workspace_dir / "data" / "chromatin" / mark_dir.name
        ensure_dir(target_dir)

        symlink_force(mark_dir, target_dir / "source_mark_dir")
        symlink_force(outputs, target_dir / "source_outputs_dir")

        required_targets = {
            "bin_mtx": "bin_chromatin_clean.mtx",
            "bin_barcodes": "bin_chromatin_clean_barcodes.tsv",
            "bin_features": "bin_chromatin_clean_features.tsv",
            "clean_cells": "clean_cells.tsv",
        }

        for key, canonical_name in required_targets.items():
            suffix_map = {
                "bin_mtx": "_bin_chromatin_clean.mtx",
                "bin_barcodes": "_bin_chromatin_clean_barcodes.tsv",
                "bin_features": "_bin_chromatin_clean_features.tsv",
                "clean_cells": "_clean_cells.tsv",
            }
            src = outputs / f"{prefix}{suffix_map[key]}"
            if src.exists():
                symlink_force(src, target_dir / canonical_name)

        for suffix in OPTIONAL_SUFFIXES:
            src = outputs / f"{prefix}{suffix}"
            if src.exists():
                canonical = suffix.lstrip("_")
                symlink_force(src, target_dir / canonical)

        manifest_rows.append(
            {
                "mark": mark_dir.name,
                "prefix": prefix,
                "bin_mtx": relpath(target_dir / "bin_chromatin_clean.mtx", repo_root),
                "bin_barcodes": relpath(target_dir / "bin_chromatin_clean_barcodes.tsv", repo_root),
                "bin_features": relpath(target_dir / "bin_chromatin_clean_features.tsv", repo_root),
                "clean_cells": relpath(target_dir / "clean_cells.tsv", repo_root),
                "bins_bed": relpath(target_dir / "bins.bed", repo_root) if (target_dir / "bins.bed").exists() else "",
                "peaks_bed": relpath(target_dir / "peaks.bed", repo_root) if (target_dir / "peaks.bed").exists() else "",
            }
        )

    manifest_rows.sort(key=lambda r: r["mark"])
    write_tsv(
        manifest_out,
        ["mark", "prefix", "bin_mtx", "bin_barcodes", "bin_features", "clean_cells", "bins_bed", "peaks_bed"],
        manifest_rows,
    )

    # Step 1 check table
    check_rows: List[Dict[str, str]] = []

    def add_check(item: str, rel_path: str) -> None:
        p = repo_root / rel_path
        check_rows.append({"item": item, "path": rel_path, "status": "OK" if p.exists() else "MISSING"})

    add_check("rna_h5", "integration/workspace/data/rna/gex_feature_cell_matrix.h5")
    add_check("rna_sample_barcodes", "integration/workspace/data/rna/gex_sample_barcodes.csv")

    chrom_link = next((workspace_dir / "data" / "chromsizes").glob("*.chrom.sizes"), None)
    if chrom_link is not None:
        add_check("chromsizes", relpath(chrom_link, repo_root))
    else:
        check_rows.append({"item": "chromsizes", "path": "integration/workspace/data/chromsizes/*.chrom.sizes", "status": "MISSING"})

    for row in manifest_rows:
        for col in ("bin_mtx", "bin_barcodes", "bin_features", "clean_cells"):
            add_check(f"{row['mark']}:{col}", row[col])

    write_tsv(check_out, ["item", "path", "status"], check_rows)

    print(f"Workspace ready: {workspace_dir}")
    print(f"Manifest: {manifest_out}")
    print(f"Data check: {check_out}")
    print(f"Marks discovered: {len(manifest_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
