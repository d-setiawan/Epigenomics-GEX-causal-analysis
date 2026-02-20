#!/usr/bin/env python3
import argparse
import csv
import gzip
import random
from pathlib import Path


def write_feature_by_cell_tsv(path: Path, features, barcodes, generator):
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["feature"] + barcodes)
        for feat in features:
            row = [feat]
            row.extend(generator(feat, bc) for bc in barcodes)
            w.writerow(row)


def main():
    p = argparse.ArgumentParser(description="Generate toy scCUT&Tag test data for pipeline validation.")
    p.add_argument("out_dir")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_a = 20
    n_b = 20
    n_doublet = 2
    n_negative = 2

    barcodes = []
    labels = {}  # A/B/Doublet/Negative
    for i in range(1, n_a + 1):
        bc = f"CELLA{i:03d}-1"
        barcodes.append(bc)
        labels[bc] = "A"
    for i in range(1, n_b + 1):
        bc = f"CELLB{i:03d}-1"
        barcodes.append(bc)
        labels[bc] = "B"
    for i in range(1, n_doublet + 1):
        bc = f"CELLD{i:03d}-1"
        barcodes.append(bc)
        labels[bc] = "Doublet"
    for i in range(1, n_negative + 1):
        bc = f"CELLN{i:03d}-1"
        barcodes.append(bc)
        labels[bc] = "Negative"

    # ADT matrix
    adt_features = ["ADT_CD3", "ADT_CD4", "ADT_CD14", "ADT_CD19"]

    def adt_value(_feat, bc):
        lab = labels[bc]
        if lab in {"A", "B", "Doublet"}:
            return random.randint(15, 60)
        return random.randint(8, 20)

    adt_path = out_dir / "toy_ADT_counts.tsv"
    write_feature_by_cell_tsv(adt_path, adt_features, barcodes, adt_value)

    # HTO matrix
    hto_features = ["HTO_A", "HTO_B"]

    def hto_value(feat, bc):
        lab = labels[bc]
        if lab == "A":
            return random.randint(160, 250) if feat == "HTO_A" else random.randint(0, 4)
        if lab == "B":
            return random.randint(160, 250) if feat == "HTO_B" else random.randint(0, 4)
        if lab == "Doublet":
            return random.randint(140, 220)
        return random.randint(0, 3)

    hto_path = out_dir / "toy_HTO_counts.tsv"
    write_feature_by_cell_tsv(hto_path, hto_features, barcodes, hto_value)

    # Peaks BED
    peaks = [
        ("chr1", 1000, 1500, "peak_1"),
        ("chr1", 2000, 2500, "peak_2"),
        ("chr1", 3000, 3500, "peak_3"),
        ("chr2", 1000, 1600, "peak_4"),
        ("chr2", 2400, 2900, "peak_5"),
    ]
    peaks_path = out_dir / "toy_peaks.bed"
    with peaks_path.open("w") as f:
        for chrom, start, end, name in peaks:
            f.write(f"{chrom}\t{start}\t{end}\t{name}\n")

    # Fragments
    frag_path = out_dir / "toy_fragments.tsv.gz"
    with gzip.open(frag_path, "wt") as f:
        for bc in barcodes:
            lab = labels[bc]
            if lab in {"A", "B"}:
                n = 160
            elif lab == "Doublet":
                n = 170
            else:
                n = 40

            for _ in range(n):
                chrom, start, end, _name = random.choice(peaks)
                frag_start = random.randint(start, end - 60)
                frag_end = frag_start + random.randint(40, 120)
                f.write(f"{chrom}\t{frag_start}\t{frag_end}\t{bc}\n")

            # Add a few off-peak fragments.
            for _ in range(10):
                chrom = random.choice(["chr1", "chr2"])
                frag_start = random.randint(5000, 7000)
                frag_end = frag_start + random.randint(40, 120)
                f.write(f"{chrom}\t{frag_start}\t{frag_end}\t{bc}\n")

    # Mock step-1 metadata to test downstream scripts when R/Seurat is unavailable.
    md_path = out_dir / "toy_hto_adt_metadata.tsv"
    with md_path.open("w", newline="") as f:
        fieldnames = [
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
        ]
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        w.writeheader()
        for bc in barcodes:
            lab = labels[bc]
            if lab == "A":
                cls = "Singlet"
                donor = "HTO_A"
                second = "HTO_B"
                margin = "6.0"
            elif lab == "B":
                cls = "Singlet"
                donor = "HTO_B"
                second = "HTO_A"
                margin = "6.0"
            elif lab == "Doublet":
                cls = "Doublet"
                donor = "HTO_A"
                second = "HTO_B"
                margin = "0.2"
            else:
                cls = "Negative"
                donor = "NA"
                second = "NA"
                margin = "0.0"
            w.writerow(
                {
                    "barcode": bc,
                    "adt_total": random.randint(40, 160),
                    "hto_total": random.randint(4, 400),
                    "hto_classification": cls,
                    "donor_id": donor,
                    "second_id": second,
                    "hto_margin": margin,
                    "doublet_flag": str(cls == "Doublet"),
                    "negative_flag": str(cls == "Negative"),
                    "singlet_flag": str(cls == "Singlet"),
                }
            )

    print("Wrote toy dataset:")
    print(f"  {adt_path}")
    print(f"  {hto_path}")
    print(f"  {frag_path}")
    print(f"  {peaks_path}")
    print(f"  {md_path} (mock metadata for testing steps 2-5)")


if __name__ == "__main__":
    main()
