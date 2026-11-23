#!/usr/bin/env python3
"""Make isolated scGLUE inputs without loading the source dense RNA X.

This is a storage-only preparation, not feature selection or preprocessing.
All source HDF5 components except X are copied. Destination X is a local HDF5
hard link to its copied layers/counts: X therefore contains RAW COUNTS, not the
source's potentially normalized/scaled values. Training must explicitly use
layers/counts and the unchanged X_pca (RNA) / X_lsi (chromatin) representations.
Do not treat the destination as an interchangeable input for X-based analyses.

The helper never opens a source for writing. It hashes counts, representations,
and identity arrays in bounded chunks before/after copying, checks source file
identity/size/mtime/ctime for concurrent changes, and writes a completion
manifest last. It intentionally does not read or hash source X values. Thus
source preservation is checked for the protected datasets plus filesystem
metadata, not claimed as a whole-file content hash of the unread dense X.
Existing output directories, including empty directories/symlinks, are refused.
On failure a partial output is left for inspection, never silently reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


MARKS = ("H3K27ac", "H3K27me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9me3")
CHUNK_BYTES = 8 * 1024 * 1024
COMPLETION_FILE = "biology_input_preparation.json"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _frame(digest, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def hash_h5_object(obj, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Hash logical dtype/shape/attributes/content, independent of compression.

    Fixed-width values use exact C-order bytes. Variable-length strings use
    length-framed UTF-8/byte values. pipelines/empirical/dataset/group path names are excluded so a
    copied counts group can be compared with the destination X hard link.
    """
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()

    def visit(item):
        kind = "dataset" if isinstance(item, h5py.Dataset) else "group"
        metadata = {"kind": kind, "attrs": {k: _jsonable(v) for k, v in sorted(item.attrs.items())}}
        if isinstance(item, h5py.Dataset):
            metadata.update(shape=item.shape, dtype=item.dtype.str)
        _frame(digest, json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        if isinstance(item, h5py.Group):
            for name in sorted(item):
                if not isinstance(item.get(name, getlink=True), h5py.HardLink):
                    raise ValueError(f"Only local HDF5 hard links are supported: {item.name}/{name}")
                _frame(digest, name.encode())
                visit(item[name])
            return
        if item.dtype.hasobject and h5py.check_string_dtype(item.dtype) is None:
            raise ValueError(f"Unsupported non-string variable-length dataset: {item.name}")
        if item.shape is None:
            return
        if item.ndim == 0:
            chunks = (item[()],)
        else:
            row_bytes = max(1, math.prod(item.shape[1:]) * item.dtype.itemsize)
            rows = max(1, chunk_bytes // row_bytes)
            chunks = (item[start:start + rows] for start in range(0, item.shape[0], rows))
        for chunk in chunks:
            array = np.asarray(chunk)
            if item.dtype.hasobject:
                for value in array.ravel(order="C"):
                    _frame(digest, value if isinstance(value, bytes) else str(value).encode())
            else:
                digest.update(array.tobytes(order="C"))

    visit(obj)
    return digest.hexdigest()


def _stat(path: Path) -> dict:
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _shape(obj) -> tuple[int, ...]:
    return tuple(int(size) for size in (obj.shape if isinstance(obj, h5py.Dataset) else obj.attrs["shape"]))


def _index_path(handle, axis: str) -> str:
    frame = handle[axis]
    name = _text(frame.attrs.get("_index", "_index"))
    path = f"{axis}/{name}"
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise ValueError(f"Missing dataset index: {path}")
    return path


def _inspect(handle, rep: str) -> dict:
    for key in ("obs", "var", "layers/counts", f"obsm/{rep}", "X"):
        if key not in handle:
            raise ValueError(f"Missing required H5AD component: {key}")
    if _text(handle.attrs.get("encoding-type", "")) != "anndata":
        raise ValueError("Expected anndata-encoded H5AD")
    obs_path, var_path = _index_path(handle, "obs"), _index_path(handle, "var")
    n_obs, n_vars = len(handle[obs_path]), len(handle[var_path])
    for path in (obs_path, var_path):
        names = [_text(value) for value in handle[path][...]]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate identities in {path}")
    if _shape(handle["layers/counts"]) != (n_obs, n_vars):
        raise ValueError("Count dimensions do not match cell/feature identities")
    if _shape(handle["X"]) != (n_obs, n_vars):
        raise ValueError("Source X dimensions do not match counts")
    rep_shape = handle[f"obsm/{rep}"].shape
    if len(rep_shape) != 2 or rep_shape[0] != n_obs:
        raise ValueError("Representation dimensions do not match cell identities")
    return {"n_cells": n_obs, "n_features": n_vars, "representation": rep,
            "n_representation_dimensions": rep_shape[1],
            "protected_paths": ["layers/counts", f"obsm/{rep}", obs_path, var_path]}


def _resolve_chrom(source_dir: Path, row: dict) -> tuple[Path, bool]:
    expected_name = f"chrom_{row['mark']}_preprocessed.h5ad"
    recorded = Path(row["chrom_h5ad"])
    if recorded.name != expected_name:
        raise ValueError(f"Unexpected chromatin filename for {row['mark']}: {recorded.name}")
    candidate = source_dir / expected_name
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing expected source-local input: {candidate}")
    # Existing absolute entries must agree with the expected local input. A
    # stale entry may be repaired ONLY through this exact directory/basename.
    if recorded.is_absolute() and recorded.exists() and recorded.resolve() != candidate.resolve():
        raise ValueError(f"Manifest names a different existing file: {recorded}")
    return candidate.resolve(), not (recorded.is_absolute() and recorded == candidate.resolve())


def _copy_one(source: Path, destination: Path, rep: str, chunk_bytes: int) -> dict:
    before_stat = _stat(source)
    with h5py.File(source, "r") as src:
        inspected = _inspect(src, rep)
        paths = inspected.pop("protected_paths")
        before_hashes = {path: hash_h5_object(src[path], chunk_bytes) for path in paths}
        source_x = {"encoding": _text(src["X"].attrs.get("encoding-type", "array")),
                    "shape": list(_shape(src["X"])), "values_read": False}
        with h5py.File(destination, "x") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for key in src:
                if key != "X":
                    if not isinstance(src.get(key, getlink=True), h5py.HardLink):
                        raise ValueError(f"Nonlocal source root link: {key}")
                    src.copy(key, dst, name=key)
            dst["X"] = dst["layers/counts"]
            # Persist a warning in the H5AD itself as well as the sidecar. Extra
            # root attributes are ignored by AnnData readers, not added to X.
            dst.attrs["biology-preparation-x-semantics"] = "raw counts; local hard link to layers/counts"
            dst.attrs["biology-preparation-source"] = str(source)
            dst.flush()
            after_hashes = {path: hash_h5_object(dst[path], chunk_bytes) for path in paths}
            if before_hashes != after_hashes:
                raise RuntimeError(f"Protected values changed while copying {source.name}")
            if dst["X"].id != dst["layers/counts"].id:
                raise RuntimeError("Destination X is not a local counts hard link")
        # Re-read protected source arrays; source X is deliberately never read.
        source_after_hashes = {path: hash_h5_object(src[path], chunk_bytes) for path in paths}
    if before_hashes != source_after_hashes or before_stat != _stat(source):
        raise RuntimeError(f"Source changed during preparation: {source}")
    return {**inspected, "source": str(source), "destination": str(destination),
            "source_file_stat": before_stat, "source_unchanged_checked": True,
            "source_x": source_x, "protected_sha256": before_hashes,
            "destination_matches_source": True, "destination_x_hardlinks_counts": True}


def prepare_inputs(source_dir: Path, out_dir: Path, chunk_bytes: int = CHUNK_BYTES) -> dict:
    source_dir = Path(source_dir).resolve(strict=True)
    raw_output = Path(out_dir).absolute()
    if os.path.lexists(raw_output):
        raise FileExistsError(f"Refusing existing output directory: {raw_output}")
    out_dir = raw_output.resolve()
    if source_dir == out_dir or source_dir in out_dir.parents:
        raise ValueError("Output must not be inside the source preprocessing directory")
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    manifest_path = source_dir / "chrom_preprocessed_manifest.tsv"
    manifest_stat = _stat(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    with manifest_path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {"mark", "chrom_h5ad"}.issubset(fields):
        raise ValueError("Chromatin manifest requires mark and chrom_h5ad columns")
    if len(rows) != 6 or {row["mark"] for row in rows} != set(MARKS):
        raise ValueError("Expected exactly six unique agreed histone marks")
    inputs = []
    for row in rows:
        source, repaired = _resolve_chrom(source_dir, row)
        inputs.append((row, source, repaired))
    rna_source = source_dir / "rna_preprocessed.h5ad"
    if not rna_source.is_file():
        raise FileNotFoundError(rna_source)
    # Read only shapes and identity metadata before creating any destination.
    for source, rep in [(rna_source, "X_pca")] + [(source, "X_lsi") for _, source, _ in inputs]:
        with h5py.File(source, "r") as handle:
            _inspect(handle, rep)
    out_dir.mkdir(parents=True, exist_ok=False)
    records = [_copy_one(rna_source, out_dir / rna_source.name, "X_pca", chunk_bytes)]
    corrected_rows = []
    for row, source, repaired in inputs:
        destination = out_dir / source.name
        record = _copy_one(source, destination, "X_lsi", chunk_bytes)
        record.update(mark=row["mark"], original_manifest_path=row["chrom_h5ad"],
                      manifest_path_repaired=repaired)
        records.append(record)
        corrected_rows.append({**row, "chrom_h5ad": str(destination)})
    if manifest_bytes != manifest_path.read_bytes() or manifest_stat != _stat(manifest_path):
        raise RuntimeError("Source manifest changed during preparation")
    # Also detect changes to an earlier source while later files were copied.
    for record in records:
        if record["source_file_stat"] != _stat(Path(record["source"])):
            raise RuntimeError(f"Source changed during preparation: {record['source']}")
    with (out_dir / "chrom_preprocessed_manifest.tsv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(corrected_rows)
    summary = {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir), "output_dir": str(out_dir),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hash_chunk_bytes": chunk_bytes, "files": records,
        "x_semantics": "RAW COUNTS: local hard link to copied layers/counts; source X values omitted",
        "required_training_configuration": {"rna_layer": "counts", "chrom_layer": "counts",
                                             "rna_rep": "X_pca", "chrom_rep": "X_lsi"},
        "memory_rationale": "Do not materialize or copy dense scaled source RNA X; retain sparse counts and unchanged PCA/LSI",
        "not_performed": ["feature selection", "normalization", "LSI/PCA recomputation", "training", "source X content hashing"],
        "source_preservation_scope": "Read-only HDF5 handles; counts/representation/identity hashes before and after; file identity, size, mtime and ctime checks; no whole-file source hash",
        "copied_components": "All root HDF5 components and attributes except X; X replaced by local counts hard link; two biology-preparation provenance attributes added",
    }
    serialized_summary = json.dumps(summary, indent=2) + "\n"
    with (out_dir / COMPLETION_FILE).open("x") as stream:
        stream.write(serialized_summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hash-chunk-bytes", type=int, default=CHUNK_BYTES)
    args = parser.parse_args()
    result = prepare_inputs(args.source_dir, args.out_dir, args.hash_chunk_bytes)
    print(f"Prepared {len(result['files'])} isolated inputs: {args.out_dir / COMPLETION_FILE}")
    print("Destination X contains raw counts; train using counts layers and unchanged PCA/LSI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
