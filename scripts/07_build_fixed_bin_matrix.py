#!/usr/bin/env python3
import argparse
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


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build clean scCUT&Tag cell-by-bin matrix using SnapATAC2, then export MatrixMarket files."
        )
    )
    p.add_argument("fragments_tsv_gz", help="fragments.tsv or fragments.tsv.gz")
    p.add_argument("chrom_sizes_tsv", help="2-column chromosome sizes file: chrom<TAB>size")
    p.add_argument("clean_barcodes_tsv", help="TSV with barcode column (from step 03)")
    p.add_argument("out_prefix", help="Output prefix")
    p.add_argument("--bin-size", type=int, default=5000, help="Fixed bin size in bp (default: 5000)")
    p.add_argument(
        "--sorted-by-barcode",
        action="store_true",
        help="Set if fragments file is already sorted by barcode for faster import",
    )
    return p.parse_args()


def load_chrom_sizes(path: Path):
    chrom_sizes = {}
    with path.open("r") as f:
        for line in f:
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


def load_whitelist(path: Path):
    barcodes = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "barcode" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'barcode' column in {path}")
        for row in reader:
            bc = row["barcode"].strip()
            if bc:
                barcodes.add(bc)
                norm = normalize_barcode(bc)
                if norm:
                    barcodes.add(norm)
                    barcodes.add(f"{norm}-1")
                    barcodes.add(f"{norm}.1")
    if not barcodes:
        raise ValueError(f"No barcodes found in {path}")
    return sorted(barcodes)


def call_supported(func, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
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


def parse_feature_to_bed(feature: str):
    # Expected region format from SnapATAC2 var names: chr:start-end
    if ":" not in feature or "-" not in feature:
        return None
    chrom, pos = feature.split(":", 1)
    start_s, end_s = pos.split("-", 1)
    try:
        start = int(start_s)
        end = int(end_s)
    except ValueError:
        return None
    return chrom, start, end


def export_matrix_market(adata, out_prefix: Path):
    import scipy.sparse as sp
    from scipy.io import mmwrite

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    mtx_out = Path(f"{out_prefix}_bin_chromatin_clean.mtx")
    cells_out = Path(f"{out_prefix}_bin_chromatin_clean_barcodes.tsv")
    feat_out = Path(f"{out_prefix}_bin_chromatin_clean_features.tsv")
    bed_out = Path(f"{out_prefix}_bins.bed")

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

    with bed_out.open("w") as w:
        for feat in adata.var_names:
            parsed = parse_feature_to_bed(str(feat))
            if parsed is None:
                continue
            chrom, start, end = parsed
            w.write(f"{chrom}\t{start}\t{end}\t{feat}\n")

    print(f"Wrote {mtx_out} ({x.shape[0]} cells x {x.shape[1]} bins, nnz={x.nnz})")
    print(f"Wrote {cells_out}")
    print(f"Wrote {feat_out}")
    print(f"Wrote {bed_out}")


def main():
    args = parse_args()
    if args.bin_size <= 0:
        raise ValueError("--bin-size must be > 0")

    try:
        import snapatac2 as snap
    except ImportError as e:
        raise SystemExit(
            "SnapATAC2 is required. Install it first, e.g. `pip install snapatac2`."
        ) from e

    chrom_sizes = load_chrom_sizes(Path(args.chrom_sizes_tsv))
    whitelist = load_whitelist(Path(args.clean_barcodes_tsv))

    adata = import_with_whitelist(
        snap=snap,
        fragments=args.fragments_tsv_gz,
        chrom_sizes=chrom_sizes,
        whitelist=whitelist,
        sorted_by_barcode=args.sorted_by_barcode,
    )

    tile = call_supported(
        snap.pp.add_tile_matrix,
        adata=adata,
        bin_size=args.bin_size,
        inplace=False,
    )
    if tile is None:
        tile = adata

    export_matrix_market(tile, Path(args.out_prefix))


if __name__ == "__main__":
    main()
