"""Execute the six declared fits serially, preserving inputs, logs and failures."""

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = REPO / "pipelines/simulation/integration_recoverability"
METHOD = REPO / "pipelines/empirical/integration/methods/scconfluence"
sys.path.insert(0, str(METHOD))
import pilot_adapter as adapter

PYTHON = REPO / "envs/runtime/scconfluence_pilot_20261101/bin/python"
RUNTIME = Path("/tmp/integration_followup_runtime_20261031/libcudart.so.13.0.96")
RUNTIME_SHA = "96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0"


def jobs():
    result = []
    for name in ("linear_chain_0", "constrained_collider_0", "scmtr_s1"):
        empirical = name == "scmtr_s1"
        inputs = (REPO / "pipelines/empirical/data/scmtr_seq/cohort_s1/model_inputs" if empirical
                  else BASE / "runs/worlds" / name / "observed/informative")
        parent = (REPO / "pipelines/empirical/integration/runs/20261101_scmtr_scconfluence_v1" if empirical
                  else BASE / "runs/integration_validation/20261101_scconfluence_v1" / name)
        selection = (None if empirical else BASE / "runs/extended" / name /
                     "informative/primary/scglue_preprocessing/train")
        for seed in (0, 1):
            result.append({"name": name, "seed": seed, "inputs": str(inputs),
                           "output": str(parent / f"seed{seed}"),
                           "selection": str(selection) if selection else None,
                           "cohorts": ["test"] if empirical else ["val", "discovery"]})
    return result


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    assert PYTHON.is_file() and adapter.sha256(RUNTIME) == RUNTIME_SHA
    declared = jobs()
    assert len(declared) == len({j["output"] for j in declared}) == 6
    assert not any(Path(j["output"]).exists() for j in declared), "Never overwrite an attempted fit"
    sources = [Path(__file__), METHOD / "pilot_adapter.py", METHOD / "loss_patch.py",
               METHOD / "upstream/UPSTREAM.json", BASE / "gene_features.py"]
    pins = {str(p): adapter.sha256(p) for p in sources}
    for job in declared:
        job["configuration"] = asdict(adapter.Config(seed=job["seed"]))
        job["input_records"] = {c: adapter.input_records(Path(job["inputs"]) / c)
                                for c in ["train", *job["cohorts"]]}
        if job["selection"]:
            job["selection_records"] = adapter.input_records(job["selection"])
    assert json.loads((HERE / "gpu_api_check.json").read_text())["status"] == "passed"
    plan = {"jobs": declared, "sources": pins, "runtime": {"path": str(RUNTIME), "sha256": RUNTIME_SHA},
            "python": str(PYTHON), "created_unix": time.time(),
            "scope": "Fixed six fits, no tuning; existing simulated worlds and one real sublibrary"}
    lock = (BASE / "runs/gpu_training.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    adapter.write_new(HERE / "execution_plan.json", plan)
    shutil.copyfile("/tmp/scconfluence_preflight_20261101.json", HERE / "input_preflight.json")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("LD_LIBRARY_PATH", None)
    env.update(LD_PRELOAD=str(RUNTIME), CUDA_LAUNCH_BLOCKING="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR="/tmp/ir_mpl_cache",
               NUMBA_CACHE_DIR="/tmp/ir_numba_cache")
    receipts = []
    for job in declared:
        assert all(adapter.sha256(p) == sha for p, sha in pins.items()), "Implementation changed"
        for records in [*job["input_records"].values(), job.get("selection_records", {})]:
            assert all(adapter.sha256(v["path"]) == v["sha256"] for v in records.values())
        common = [str(PYTHON), str(METHOD / "pilot_adapter.py")]
        fit = common + ["fit", "--input-dir", str(Path(job["inputs"]) / "train"),
                        "--output", job["output"], "--seed", str(job["seed"]), "--device", "cuda"]
        if job["selection"]:
            fit += ["--selection-dir", job["selection"]]
        commands = [("fit", fit)] + [(c, common + ["encode", "--input-dir", str(Path(job["inputs"]) / c),
                     "--output", job["output"], "--cohort", c, "--device", "cuda"]) for c in job["cohorts"]]
        result = {"name": job["name"], "seed": job["seed"], "output": job["output"], "steps": []}
        for stage, command in commands:
            log = HERE / "logs" / f"{job['name']}_seed{job['seed']}_{stage}.log"
            log.parent.mkdir(exist_ok=True)
            start = time.time()
            print(f"START {job['name']} seed{job['seed']} {stage}", flush=True)
            with log.open("x") as handle:
                code = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            result["steps"].append({"stage": stage, "command": command, "log": str(log),
                                     "exit_code": code, "started_unix": start, "seconds": time.time() - start})
            print(f"DONE {job['name']} seed{job['seed']} {stage} exit={code}", flush=True)
            if code:
                break
        result["complete"] = len(result["steps"]) == len(commands) and all(s["exit_code"] == 0 for s in result["steps"])
        adapter.write_new(HERE / f"{job['name']}_seed{job['seed']}_execution.json", result)
        receipts.append(result)
    adapter.write_new(HERE / "execution_results.json", receipts)
    return 0 if all(r["complete"] for r in receipts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
