#!/usr/bin/env python3
"""Bootstrap a label harmonization TSV from the RNA annotation tarball."""

from __future__ import annotations

import argparse
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd


def infer_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (repo_root / p)


def load_annotation(path: Path) -> pd.DataFrame:
    with tarfile.open(path, "r:gz") as tf:
        member = "cell_types/cell_types.csv"
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"Missing {member} in {path}")
        df = pd.read_csv(BytesIO(f.read()))
    required = {"coarse_cell_type", "fine_cell_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Build a harmonization template from RNA cell type annotations")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument(
        "--rna-annotation-tar",
        default="integration/workspace/data/rna/cell_type_annotation.tar.gz",
        help="Tar.gz containing cell_types/cell_types.csv",
    )
    p.add_argument(
        "--out-tsv",
        default="integration/manifests/rna_label_harmonization.tsv",
        help="Output TSV to edit and reuse during validation",
    )
    p.add_argument("--force", action="store_true", help="Overwrite output TSV if it already exists")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ann_path = resolve_path(repo_root, args.rna_annotation_tar)
    out_path = resolve_path(repo_root, args.out_tsv)

    if out_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {out_path} (pass --force to overwrite)")

    df = load_annotation(ann_path)
    tpl = (
        df[["coarse_cell_type", "fine_cell_type"]]
        .drop_duplicates()
        .sort_values(["coarse_cell_type", "fine_cell_type"], kind="stable")
        .reset_index(drop=True)
    )
    tpl["harmonized_coarse"] = tpl["coarse_cell_type"]
    tpl["harmonized_fine"] = tpl["fine_cell_type"]
    tpl["cell_ontology_id"] = ""
    tpl["notes"] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tpl.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote: {out_path}")
    print("Edit harmonized_coarse / harmonized_fine / cell_ontology_id as needed, then reuse the TSV in validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
