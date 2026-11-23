"""Wait for each declared validation export, then run its fixed CPU state readout."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PYTHON = Path("/home/dgsetiawan/anaconda3/envs/scrna_r/bin/python")
SCORER = REPO / "pipelines/empirical/integration/methods/scconfluence/state_readout.py"


def main():
    plan = json.loads((HERE / "execution_plan.json").read_text())
    jobs = [j for j in plan["jobs"] if j["name"] in ("linear_chain_0", "constrained_collider_0")]
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(LD_PRELOAD=plan["runtime"]["path"], OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLCONFIGDIR="/tmp/ir_mpl_cache")
    receipts = []
    for job in jobs:
        directory = Path(job["output"])
        execution = HERE / f"{job['name']}_seed{job['seed']}_execution.json"
        required = [directory / name for name in
                    ("model_manifest.json", "train/encoding_manifest.json", "val/encoding_manifest.json")]
        failed = False
        while not all(p.exists() for p in required):
            if execution.exists() and not json.loads(execution.read_text())["complete"]:
                receipts.append({"world": job["name"], "seed": job["seed"], "status": "not_scored",
                                 "reason": "Declared fit/export failed; no replacement or pool reduction",
                                 "execution_receipt": str(execution)})
                failed = True
                break
            time.sleep(10)
        if failed:
            continue
        command = [str(PYTHON), str(SCORER), "score", "--world", job["name"], "--seed", str(job["seed"])]
        log = HERE / "logs" / f"{job['name']}_seed{job['seed']}_state.log"
        log.parent.mkdir(exist_ok=True)
        started = time.time()
        print(f"START state {job['name']} seed{job['seed']}", flush=True)
        with log.open("x") as handle:
            code = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        result = {"world": job["name"], "seed": job["seed"], "exit_code": code,
                  "status": "complete" if code == 0 else "failed", "command": command,
                  "seconds": time.time() - started, "log": str(log),
                  "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        receipts.append(result)
        with (HERE / f"{job['name']}_seed{job['seed']}_state_execution.json").open("x") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(f"DONE state {job['name']} seed{job['seed']} exit={code}", flush=True)
    result = {"status": "complete" if all(r["status"] == "complete" for r in receipts) else "incomplete",
              "execution_plan_sha256": hashlib.sha256((HERE / "execution_plan.json").read_bytes()).hexdigest(),
              "watcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "readouts": receipts}
    with (HERE / "state_execution_results.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
