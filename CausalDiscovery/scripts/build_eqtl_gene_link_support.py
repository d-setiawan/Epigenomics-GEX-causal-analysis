#!/usr/bin/env python3
"""Build a monocyte GENETICS peak->gene support table from eQTL Catalogue
SuSiE fine-mapped credible sets.

This is the GENETICS axis of the triangulation, complementing ENCODE-rE2G
(prediction) and BLUEPRINT PCHi-C (physical). Logic: eQTL Catalogue fine-maps
each gene's cis-eQTL to a 95% credible set of variants. A variant that lies
INSIDE one of our CUT&Tag peaks and whose credible set targets the edge's gene
is genetic evidence that the peak harbours regulatory sequence controlling that
gene -- an independent, orthogonal reason to believe a peak->gene edge.

eQTL Catalogue is GRCh38-native (no liftOver). `variant` is `chr5_140633688_A_G`
(chrom_pos_ref_alt). We keep only monocyte `ge` datasets and only credible sets
for the target genes, emitting one row per credible-set variant:

    chrom  start  end  gene  support_label  external_id  biosample  notes

`start,end = pos,pos+1` so the SNP overlaps any peak that contains it. Feed it as
the local named source `eqtl`:

    causal_cli.py support --graph-dir <dir> --external-support-tsv eqtl=<output.tsv>

IMPORTANT cell-state caveat: for CD14/CSF1R/IL1B the only monocyte credible sets
in eQTL Catalogue come from TLR-STIMULATED monocytes (Quach_2016 Pam3CSK4/R848),
not resting monocytes. That is still monocyte-lineage and cell-type aligned, but
NOT resting-state aligned -- the `condition` is written into every biosample/notes
field so the caller can see it. IL1B has no monocyte credible set at all.
"""
from __future__ import annotations

import argparse
import gzip
import io
import urllib.request
from pathlib import Path

METADATA_URL = (
    "https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/"
    "master/tabix/tabix_ftp_paths.tsv"
)
# Target gene symbol -> Ensembl gene id (GRCh38).
DEFAULT_GENE_MAP = {
    "CD14": "ENSG00000170458",
    "CSF1R": "ENSG00000182578",
    "IL1B": "ENSG00000125538",
}
CACHE_DIR = Path("Data/external/eqtl")


def fetch_bytes(url: str, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def load_text(raw: bytes) -> io.TextIOBase:
    if raw[:2] == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding="utf-8")
    return io.StringIO(raw.decode("utf-8"))


def monocyte_datasets(timeout: int) -> list[dict[str, str]]:
    """Return monocyte `ge` datasets (dataset_id, study, condition, cs_url)."""
    raw = fetch_bytes(METADATA_URL, timeout)
    fh = load_text(raw)
    header = fh.readline().rstrip("\n").split("\t")
    idx = {c: i for i, c in enumerate(header)}
    out = []
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[idx["quant_method"]] != "ge":
            continue
        if "monocyte" not in f[idx["sample_group"]].lower():
            continue
        out.append({
            "dataset_id": f[idx["dataset_id"]],
            "study": f[idx["study_label"]],
            "sample_group": f[idx["sample_group"]],
            "condition": f[idx["condition_label"]],
            "cs_url": f[idx["ftp_cs_path"]],
        })
    return out


def cached_download(url: str, dataset_id: str, cache_dir: Path, timeout: int) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset_id}.credible_sets.tsv.gz"
    if path.exists():
        return path.read_bytes()
    print(f"[eqtl] downloading {dataset_id} credible sets", flush=True)
    raw = fetch_bytes(url, timeout)
    path.write_bytes(raw)
    return raw


def parse_variant(variant: str) -> tuple[str, int] | None:
    # chr5_140633688_A_G  (chrom_pos_ref_alt)
    parts = variant.split("_")
    if len(parts) < 2:
        return None
    chrom = parts[0]
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    try:
        return chrom, int(parts[1])
    except ValueError:
        return None


def build(genes: dict[str, str], datasets: list[dict[str, str]],
          min_pip: float, cache_dir: Path, timeout: int) -> list[dict]:
    ens_to_symbol = {v: k for k, v in genes.items()}
    rows: list[dict] = []
    for ds in datasets:
        raw = cached_download(ds["cs_url"], ds["dataset_id"], cache_dir, timeout)
        fh = load_text(raw)
        header = fh.readline().rstrip("\n").split("\t")
        col = {c: i for i, c in enumerate(header)}
        n_ds = 0
        for line in fh:
            f = line.rstrip("\n").split("\t")
            gene_id = f[col["gene_id"]].split(".")[0]
            if gene_id not in ens_to_symbol:
                continue
            try:
                pip = float(f[col["pip"]])
            except (ValueError, KeyError):
                pip = 0.0
            if pip < min_pip:
                continue
            pv = parse_variant(f[col["variant"]])
            if pv is None:
                continue
            chrom, pos = pv
            symbol = ens_to_symbol[gene_id]
            rsid = f[col["rsid"]] if "rsid" in col else ""
            cs_id = f[col["cs_id"]] if "cs_id" in col else ""
            cs_size = f[col["cs_size"]] if "cs_size" in col else ""
            biosample = f"{ds['study']} monocyte ({ds['condition']})"
            rows.append({
                "chrom": chrom,
                "start": pos,
                "end": pos + 1,
                "gene": symbol,
                "support_label": "eqtl_susie",
                "external_id": rsid or f[col["variant"]],
                "biosample": biosample,
                "notes": (f"eQTL Catalogue SuSiE credible set; dataset={ds['dataset_id']}; "
                          f"study={ds['study']}; condition={ds['condition']}; pip={pip:.4f}; "
                          f"cs_id={cs_id}; cs_size={cs_size}; variant={f[col['variant']]}"),
            })
            n_ds += 1
        if n_ds:
            print(f"[eqtl] {ds['dataset_id']} ({ds['condition']}): {n_ds} credible-set variants for targets", flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-tsv", type=Path,
                   default=CACHE_DIR / "eqtl_mon_gene_links.tsv")
    p.add_argument("--gene-map", default=None,
                   help="Override targets, e.g. 'CD14=ENSG00000170458,CSF1R=ENSG00000182578'")
    p.add_argument("--datasets", default=None,
                   help="Comma-separated QTD ids to restrict to (default: all monocyte ge datasets).")
    p.add_argument("--naive-only", action="store_true",
                   help="Keep only naive (unstimulated) monocyte datasets. NB: yields 0 hits for the 3 targets.")
    p.add_argument("--min-pip", type=float, default=0.0,
                   help="Minimum SuSiE PIP to keep a credible-set variant (default 0.0 = full 95%% credible set).")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--timeout", type=int, default=180)
    a = p.parse_args()

    genes = DEFAULT_GENE_MAP.copy()
    if a.gene_map:
        genes = {}
        for kv in a.gene_map.split(","):
            k, v = kv.split("=")
            genes[k.strip().upper()] = v.strip()

    datasets = monocyte_datasets(a.timeout)
    if a.naive_only:
        datasets = [d for d in datasets if d["condition"].lower() == "naive"]
    if a.datasets:
        keep = {x.strip() for x in a.datasets.split(",")}
        datasets = [d for d in datasets if d["dataset_id"] in keep]
    print(f"[eqtl] {len(datasets)} monocyte ge datasets; targets={list(genes)}")

    rows = build(genes, datasets, a.min_pip, a.cache_dir, a.timeout)

    import pandas as pd
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "gene",
                                     "support_label", "external_id", "biosample", "notes"])
    a.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out_tsv, sep="\t", index=False)
    print(f"\n[eqtl] wrote {len(df)} variant-links -> {a.out_tsv}")
    if not df.empty:
        print(df.groupby(["gene", "biosample"]).size().to_string())
    else:
        print("[eqtl] WARNING: no credible-set variants for the targets in the selected datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
