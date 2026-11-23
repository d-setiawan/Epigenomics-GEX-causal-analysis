#!/usr/bin/env python3
import argparse
from pathlib import Path

from _common import call_supported, import_with_whitelist, load_barcode_whitelist, load_chrom_sizes


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build clean scCUT&Tag cell-by-peak matrix using SnapATAC2, then export MatrixMarket files."
        )
    )
    p.add_argument("fragments_tsv_gz", help="fragments.tsv or fragments.tsv.gz")
    p.add_argument("peaks_bed", help="BED file with peaks (chr start end [name])")
    p.add_argument("chrom_sizes_tsv", help="2-column chromosome sizes file: chrom<TAB>size")
    p.add_argument("clean_barcodes_tsv", help="TSV with barcode column (from step 03)")
    p.add_argument("out_prefix", help="Output prefix")
    p.add_argument(
        "--sorted-by-barcode",
        action="store_true",
        help="Set if fragments file is already sorted by barcode for faster import",
    )
    return p.parse_args()


def export_matrix_market(adata, out_prefix: Path):
    import scipy.sparse as sp
    from scipy.io import mmwrite

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    mtx_out = Path(f"{out_prefix}_chromatin_clean.mtx")
    cells_out = Path(f"{out_prefix}_chromatin_clean_barcodes.tsv")
    feat_out = Path(f"{out_prefix}_chromatin_clean_features.tsv")

    x = adata.X
    if not sp.issparse(x):
        x = sp.csr_matrix(x)

    mmwrite(str(mtx_out), x.tocoo())

    with cells_out.open("w") as w:
        w.write("barcode\n")
        for bc in adata.obs_names:
            w.write(f"{bc}\n")

    with feat_out.open("w") as w:
        w.write("feature\n")
        for feat in adata.var_names:
            w.write(f"{feat}\n")

    print(f"Wrote {mtx_out} ({x.shape[0]} cells x {x.shape[1]} features, nnz={x.nnz})")
    print(f"Wrote {cells_out}")
    print(f"Wrote {feat_out}")


def main():
    args = parse_args()

    try:
        import snapatac2 as snap
    except ImportError as e:
        raise SystemExit(
            "SnapATAC2 is required. Install it first, e.g. `pip install snapatac2`."
        ) from e

    chrom_sizes = load_chrom_sizes(Path(args.chrom_sizes_tsv))
    whitelist = sorted(load_barcode_whitelist(Path(args.clean_barcodes_tsv), include_suffix_variants=True))

    adata = import_with_whitelist(
        snap=snap,
        fragments=args.fragments_tsv_gz,
        chrom_sizes=chrom_sizes,
        whitelist=whitelist,
        sorted_by_barcode=args.sorted_by_barcode,
    )

    peak_mat = call_supported(
        snap.pp.make_peak_matrix,
        adata=adata,
        peak_file=args.peaks_bed,
        inplace=False,
    )
    if peak_mat is None:
        peak_mat = adata

    export_matrix_market(peak_mat, Path(args.out_prefix))


if __name__ == "__main__":
    main()
