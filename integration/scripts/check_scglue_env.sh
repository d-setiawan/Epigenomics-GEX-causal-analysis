#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash integration/scripts/check_scglue_env.sh
#   bash integration/scripts/check_scglue_env.sh <expected_conda_env>

EXPECTED_ENV="${1:-}"

printf "=== scGLUE Environment Check ===\n"
printf "Date: %s\n" "$(date)"
printf "Working dir: %s\n\n" "$(pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "[FAIL] conda not found in PATH"
  exit 1
fi

echo "[OK] conda found: $(command -v conda)"
ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
if [[ -n "$ACTIVE_ENV" ]]; then
  echo "[OK] Active conda env: $ACTIVE_ENV"
else
  echo "[WARN] No active conda env detected (CONDA_DEFAULT_ENV is empty)"
fi

if [[ -n "$EXPECTED_ENV" ]]; then
  if [[ "$ACTIVE_ENV" == "$EXPECTED_ENV" ]]; then
    echo "[OK] Active env matches expected env: $EXPECTED_ENV"
  else
    echo "[WARN] Expected env '$EXPECTED_ENV' but active env is '${ACTIVE_ENV:-<none>}'"
  fi
fi

echo
python - <<'PY'
import importlib.util
import platform
import sys

try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:  # pragma: no cover
    version = None
    PackageNotFoundError = Exception

required = [
    ("scglue", "scglue"),
    ("scanpy", "scanpy"),
    ("anndata", "anndata"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("networkx", "networkx"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
]

optional = [
    ("umap", "umap-learn"),
    ("h5py", "h5py"),
    ("yaml", "PyYAML"),
]


def dist_version(dist_name: str) -> str:
    if version is None:
        return "unknown"
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "unknown"


print("=== Python Runtime ===")
print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")
print(f"Platform:          {platform.platform()}")
print()

missing = []
print("=== Required Python Packages ===")
for import_name, dist_name in required:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[FAIL] {import_name}: not installed")
        missing.append(import_name)
    else:
        print(f"[OK]   {import_name}: {dist_version(dist_name)}")

print()
print("=== Optional Python Packages ===")
for import_name, dist_name in optional:
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[WARN] {import_name}: not installed")
    else:
        print(f"[OK]   {import_name}: {dist_version(dist_name)}")

print()
if missing:
    print("=== RESULT: FAIL ===")
    print("Missing required packages:", ", ".join(missing))
    raise SystemExit(2)
else:
    print("=== RESULT: PASS ===")
PY

echo
printf "=== External CLI Tools (optional but useful) ===\n"
for tool in bedtools macs3 samtools; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "[OK] $tool: $(command -v "$tool")"
  else
    echo "[WARN] $tool not found"
  fi
done

echo
printf "Environment check complete.\n"
