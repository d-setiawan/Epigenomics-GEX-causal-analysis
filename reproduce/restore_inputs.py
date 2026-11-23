"""Restore published inputs to their pipeline paths and verify exact SHA-256 pins."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verify(path, item):
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"Wrong byte count for {path} (expected {item['bytes']})")
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")


def restore(root, item, check_only, mirror):
    path = root / item["path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes repository: {path}")
    if path.exists() or path.is_symlink():
        verify(path, item)
        return "verified"
    if check_only or item["transform"] == "tracked":
        raise FileNotFoundError(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=path.parent) as temp:
        pending = Path(temp) / path.name
        if mirror is not None and (mirror / item["path"]).is_file():
            shutil.copyfile(mirror / item["path"], pending)
        elif item["transform"] == "rna_barcodes":
            import h5py

            with h5py.File(root / item["source"], "r") as matrix, pending.open("wb") as output:
                for barcode in matrix["matrix/barcodes"]:
                    output.write(b"GRCh38," + barcode + b"\r\n")
        else:
            if "source" in item:
                payload = root / item["source"]
            else:
                payload = Path(temp) / "download"
                subprocess.run(
                    ["curl", "-fsSL", "--connect-timeout", "20", "--max-time", "600",
                     "--retry", "2", "--retry-all-errors", "--output", str(payload),
                     item["url"]],
                    check=True,
                )
            if item["transform"] == "gzip":
                with gzip.open(payload, "rb") as source, pending.open("wb") as output:
                    shutil.copyfileobj(source, output)
            elif item["transform"] == "copy":
                shutil.copyfile(payload, pending)
            else:
                raise ValueError(f"Unknown transformation: {item['transform']}")
        verify(pending, item)
        # Atomic publication without replacing a file produced by another process.
        try:
            path.hardlink_to(pending)
        except FileExistsError:
            verify(path, item)
    return "restored"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", choices=["pbmc", "reference", "scmtr", "benchmark", "all"])
    parser.add_argument("--check", action="store_true", help="Verify local inputs without downloading")
    parser.add_argument("--mirror", type=Path, help="Optional directory of original files with the same relative paths")
    parser.add_argument("--only", action="append", help="Restrict to a filename, repeatable")
    args = parser.parse_args()
    inventory = json.loads((ROOT / "reproduce/inputs.json").read_text())
    items = [item for item in inventory if "all" in args.groups or item["group"] in args.groups]
    if args.only:
        items = [item for item in items if Path(item["path"]).name in args.only]
        missing = set(args.only) - {Path(item["path"]).name for item in items}
        if missing:
            parser.error(f"No matching input in selected groups: {', '.join(sorted(missing))}")
    failures = []
    for item in items:
        try:
            state = restore(ROOT, item, args.check, args.mirror)
        except (OSError, ValueError, ImportError, EOFError, subprocess.CalledProcessError) as error:
            failures.append((item["path"], str(error)))
        else:
            print(f"{state}: {item['path']}", flush=True)
    if failures:
        parser.exit(1, "\n".join(f"{path}: {error}" for path, error in failures) + "\n")


if __name__ == "__main__":
    main()
