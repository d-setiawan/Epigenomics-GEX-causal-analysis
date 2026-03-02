#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def normalize_barcode(bc: str) -> str:
    bc = bc.strip()
    if "." in bc and bc.rsplit(".", 1)[-1].isdigit():
        bc = bc.rsplit(".", 1)[0]
    if "-" in bc and bc.rsplit("-", 1)[-1].isdigit():
        bc = bc.rsplit("-", 1)[0]
    return bc


def read_fragment_counts(path: Path):
    counts = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "barcode" not in reader.fieldnames:
            raise ValueError(f"Missing 'barcode' column in {path}")

        frag_col = None
        for cand in ("n_cuttag_fragments", "n_fragments"):
            if cand in reader.fieldnames:
                frag_col = cand
                break
        if frag_col is None:
            raise ValueError(
                f"Missing fragment count column in {path}; expected 'n_cuttag_fragments' or 'n_fragments'"
            )

        for row in reader:
            bc = normalize_barcode(row["barcode"])
            if not bc:
                continue
            try:
                n = int(float(row[frag_col]))
            except (TypeError, ValueError):
                continue
            if bc not in counts or n > counts[bc]:
                counts[bc] = n
    return counts


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build final scCUT&Tag clean-cell metadata by combining HTO/ADT demux metadata "
            "with per-cell fragment depth."
        )
    )
    p.add_argument("hto_adt_metadata_tsv", help="Output from 01_demux_adt_hto.R (*_hto_adt_metadata.tsv)")
    p.add_argument("fragment_counts_tsv", help="Output from 02_count_fragments.py")
    p.add_argument("out_prefix", help="Output prefix for clean metadata/barcode files")
    p.add_argument(
        "--min-fragments",
        type=int,
        default=100,
        help="Minimum scCUT&Tag fragments required per cell (default: 100)",
    )
    p.add_argument(
        "--allow-missing-donor",
        action="store_true",
        help="Keep singlets even when donor_id is NA/empty",
    )
    return p.parse_args()


def donor_is_present(value: str) -> bool:
    if value is None:
        return False
    v = value.strip().lower()
    return v not in {"", "na", "nan", "none"}


def to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "t", "1", "yes", "y"}


def main():
    args = parse_args()

    md_path = Path(args.hto_adt_metadata_tsv)
    frag_path = Path(args.fragment_counts_tsv)

    frag_counts = read_fragment_counts(frag_path)

    clean_rows = []
    with md_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        needed = {"barcode", "hto_classification", "donor_id"}
        missing = needed.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required metadata columns in {md_path}: {sorted(missing)}")

        for row in reader:
            raw_bc = row.get("barcode", "")
            bc = normalize_barcode(raw_bc)
            if not bc:
                continue

            cls = str(row.get("hto_classification", "")).strip()
            if cls != "Singlet":
                continue

            donor_id = row.get("donor_id", "")
            if (not args.allow_missing_donor) and (not donor_is_present(donor_id)):
                continue

            n_frag = frag_counts.get(bc)
            if n_frag is None or n_frag < args.min_fragments:
                continue

            out = dict(row)
            out["barcode"] = bc
            out["n_cuttag_fragments"] = str(n_frag)

            # Keep a normalized singlet flag for downstream pipelines.
            if "singlet_flag" in out:
                out["singlet_flag"] = "True" if to_bool(out["singlet_flag"]) else "False"

            clean_rows.append(out)

    if clean_rows:
        clean_rows.sort(key=lambda r: (str(r.get("donor_id", "")), r["barcode"]))

    metadata_out = Path(f"{args.out_prefix}_clean_cells.tsv")
    barcode_out = Path(f"{args.out_prefix}_clean_barcodes.tsv")
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original metadata columns order and append fragment depth.
    base_columns = list(clean_rows[0].keys()) if clean_rows else [
        "barcode",
        "adt_total",
        "hto_total",
        "hto_classification",
        "donor_id",
        "second_id",
        "hto_margin",
        "doublet_flag",
        "negative_flag",
        "singlet_flag",
        "n_cuttag_fragments",
    ]

    if "n_cuttag_fragments" not in base_columns:
        base_columns.append("n_cuttag_fragments")

    with metadata_out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=base_columns,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(clean_rows)

    with barcode_out.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["barcode"])
        for row in clean_rows:
            writer.writerow([row["barcode"]])

    print(
        f"Wrote {metadata_out} ({len(clean_rows)} cells) and {barcode_out} with "
        f"min_fragments={args.min_fragments}."
    )


if __name__ == "__main__":
    main()
