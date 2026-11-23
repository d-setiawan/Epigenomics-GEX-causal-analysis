"""Invoke the historical integration runner with the active Python environment."""
from pathlib import Path
import sys


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "pipelines/simulation/integration_recoverability"))
    import run_benchmark

    run_benchmark.PYTHON = sys.executable
    run_benchmark.main()
