"""Serial CPU scoring of the six fixed fits after their sealed exports exist."""

import json
from pathlib import Path
import subprocess
import sys
import time

import causal_reliability_pilot as simulation
import causal_reliability_pilot_empirical as empirical


def main():
    scripts = Path(__file__).resolve().parent
    jobs = []
    for world in simulation.WORLDS:
        for seed in (0, 1):
            directory = simulation.BASE / f"runs/integration_validation/20261101_scconfluence_v1/{world}/seed{seed}"
            command = [sys.executable, str(scripts / "causal_reliability_pilot.py"), "method", "--world", world,
                       "--seed", str(seed), "--directory", str(directory)]
            jobs.append((f"{world}_scconfluence_seed{seed}", directory, "discovery", simulation.ROOT, command))
    for seed in (0, 1):
        directory = empirical.REPO / f"pipelines/empirical/integration/runs/20261101_scmtr_scconfluence_v1/seed{seed}"
        command = [sys.executable, str(scripts / "causal_reliability_pilot_empirical.py"), "method",
                   "--seed", str(seed), "--directory", str(directory)]
        jobs.append((f"scmtr_scconfluence_seed{seed}", directory, "test", empirical.ROOT, command))
    for name, directory, cohort, root, command in jobs:
        receipt = root / f"{name}_execution.json"
        if receipt.exists():
            raise FileExistsError(f"Existing execution receipt; inspect before resuming: {receipt}")
        export = directory / cohort / "encoding_manifest.json"
        print(f"Waiting for declared sealed export: {name}", flush=True)
        waiting = time.monotonic()
        while not export.exists() and not (directory / "failure.json").exists():
            if time.monotonic() - waiting > 10800:
                raise TimeoutError(f"No declared export after three hours: {name}")
            time.sleep(5)
        if not export.exists():
            simulation.write(receipt, {"status": "unavailable_fit_failure", "failure": simulation.pin(directory / "failure.json")})
            print(f"Fit failed before export: {name}", flush=True)
            continue
        start = time.monotonic()
        with (root / f"{name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = {"status": "complete" if result.returncode == 0 else "failed", "command": command,
                  "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - start,
                  "export": simulation.pin(export), "log": simulation.pin(root / f"{name}.log")}
        simulation.write(receipt, record)
        print(json.dumps({"job": name, "status": record["status"], "seconds": record["elapsed_seconds"]}), flush=True)
        if result.returncode:
            raise RuntimeError(f"Scoring failed; inspect {root / (name + '.log')}")


if __name__ == "__main__":
    main()
