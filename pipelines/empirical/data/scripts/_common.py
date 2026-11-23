"""Shared helpers for scCUT&Tag preprocessing scripts."""

from __future__ import annotations

import csv
import inspect
from pathlib import Path


def normalize_barcode(bc: str) -> str:
    bc = bc.strip()
    if "." in bc and bc.rsplit(".", 1)[-1].isdigit():
        bc = bc.rsplit(".", 1)[0]
    if "-" in bc and bc.rsplit("-", 1)[-1].isdigit():
        bc = bc.rsplit("-", 1)[0]
    return bc


def load_chrom_sizes(path: Path) -> dict[str, int]:
    chrom_sizes: dict[str, int] = {}
    with path.open("r") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            chrom = parts[0]
            try:
                size = int(parts[1])
            except ValueError:
                continue
            if size > 0:
                chrom_sizes[chrom] = size
    if not chrom_sizes:
        raise ValueError(f"No valid chromosome sizes found in {path}")
    return chrom_sizes


def load_barcode_whitelist(path: Path, *, include_suffix_variants: bool = False) -> set[str]:
    barcodes: set[str] = set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "barcode" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'barcode' column in {path}")
        for row in reader:
            raw = row["barcode"].strip()
            norm = normalize_barcode(raw)
            if not norm:
                continue
            if include_suffix_variants:
                if raw:
                    barcodes.add(raw)
                barcodes.add(norm)
                barcodes.add(f"{norm}-1")
                barcodes.add(f"{norm}.1")
            else:
                barcodes.add(norm)
    if not barcodes:
        raise ValueError(f"No barcodes found in {path}")
    return barcodes


def call_supported(func, **kwargs):
    sig = inspect.signature(func)
    supported = {key: value for key, value in kwargs.items() if key in sig.parameters}
    return func(**supported)


def import_with_whitelist(snap, fragments, chrom_sizes, whitelist, sorted_by_barcode):
    if hasattr(snap.pp, "import_fragments"):
        fn = snap.pp.import_fragments
        return call_supported(
            fn,
            fragment_file=fragments,
            chrom_sizes=chrom_sizes,
            chrom_size=chrom_sizes,
            whitelist=whitelist,
            min_num_fragments=1,
            sorted_by_barcode=sorted_by_barcode,
        )

    if hasattr(snap.pp, "import_data"):
        fn = snap.pp.import_data
        return call_supported(
            fn,
            fragment_file=fragments,
            chrom_sizes=chrom_sizes,
            chrom_size=chrom_sizes,
            whitelist=whitelist,
            min_num_fragments=1,
            sorted_by_barcode=sorted_by_barcode,
            low_memory=True,
        )

    raise RuntimeError("SnapATAC2 import function not found. Expected pp.import_fragments or pp.import_data.")
