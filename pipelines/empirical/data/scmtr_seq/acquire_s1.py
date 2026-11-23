"""Acquire only the declared human scMTR-seq S1 processed inputs."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

BASE = Path(__file__).resolve().parent
SAMPLE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8529nnn"
SERIES = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE277nnn/GSE277778/suppl"
FILES = {
    "H3K27ac.fragments.tsv.gz": (f"{SAMPLE}/GSM8529544/suppl/GSM8529544_HIST_S1_H3K27ac.fragments.tsv.gz", 54136502),
    "H3K4me3.fragments.tsv.gz": (f"{SAMPLE}/GSM8529544/suppl/GSM8529544_HIST_S1_H3K4me3.fragments.tsv.gz", 24116580),
    "RNA.matrix.mtx.gz": (f"{SAMPLE}/GSM8529552/suppl/GSM8529552_Endo.S1.matrix.mtx.gz", 80973053),
    "RNA.barcodes.tsv.gz": (f"{SAMPLE}/GSM8529552/suppl/GSM8529552_Endo.S1.barcodes.tsv.gz", None),
    "RNA.features.tsv.gz": (f"{SAMPLE}/GSM8529552/suppl/GSM8529552_Endo.S1.features.tsv.gz", None),
    "feature_reference.DNA.csv.gz": (f"{SERIES}/GSE277778_feature_reference.DNA.csv.gz", None),
    "feature_reference.RNA.csv.gz": (f"{SERIES}/GSE277778_feature_reference.RNA.csv.gz", None),
}


def acquire(name, url, expected_size):
    directory = BASE / "raw_s1"
    directory.mkdir(exist_ok=True)
    destination = directory / name
    receipt = destination.with_suffix(destination.suffix + ".json")
    if destination.exists() or receipt.exists():
        record = json.loads(receipt.read_text())
        with destination.open("rb") as handle:
            current_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if (record["url"] != url or destination.stat().st_size != record["bytes"]
                or current_sha256 != record["sha256"]):
            raise ValueError(f"Existing input changed: {destination}")
        return record
    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as output:
        headers = dict(response.headers)
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            output.write(chunk)
            digest.update(chunk)
    size = partial.stat().st_size
    announced = headers.get("Content-Length")
    if not size or (expected_size is not None and size != expected_size) or (announced and size != int(announced)):
        raise ValueError(f"Download length mismatch; partial retained: {partial}")
    partial.rename(destination)
    record = {"url": url, "path": str(destination), "bytes": size, "sha256": digest.hexdigest(),
              "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "headers": headers, "publisher_checksum_available": False}
    with receipt.open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return record


if __name__ == "__main__":
    for name, (url, size) in FILES.items():
        record = acquire(name, url, size)
        print(json.dumps({k: record[k] for k in ("path", "bytes", "sha256")}), flush=True)
