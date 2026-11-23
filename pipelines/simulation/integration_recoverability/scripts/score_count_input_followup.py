"""Score the frozen count-input study with its observed native GPU-mask mutation."""

import argparse
import ast
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path

import count_input_experiment as experiment

PROTOCOL_SHA256 = "89469c1bbd4bb5d21403affa9cfddd4e45e14b45d949bf55154e86f256a14390"
RUNNER_SHA256 = "c053f76034939ef5a9aa681c31e8ae74a20002695b7d5164c49ee6fb14c841b3"
AUTODEVICE_SOURCE = Path(importlib.metadata.distribution("scglue").locate_file("scglue/models/nn.py")).resolve()
AUTODEVICE_SHA256 = "d4b02529a95aee43710e33b188e3b46dff50ff1e90177b0fcd275c4904474680"


def prepare():
    """Declare the sole compatibility change before any state labels are read."""
    digest, read = experiment.glue.file_hash, experiment.validation.read
    protocol_path = experiment.ROOT / "protocol.json"
    runner_path = Path(experiment.__file__).resolve()
    if digest(protocol_path) != PROTOCOL_SHA256 or digest(runner_path) != RUNNER_SHA256:
        raise ValueError("Original protocol or count-input runner changed")
    initial = read(protocol_path)["inherited_loader_environment"]
    if initial.get("CUDA_VISIBLE_DEVICES") is not None:
        raise ValueError("Only the recorded null-to-0 GPU-mask mutation is eligible")
    if any(os.environ.get(key) != value for key, value in initial.items()):
        raise ValueError("Run under the original initial loader environment before checked_protocol")
    original = experiment.checked_protocol()
    expected = deepcopy(original)
    expected["inherited_loader_environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    if digest(AUTODEVICE_SOURCE) != AUTODEVICE_SHA256:
        raise ValueError("Installed native autodevice source changed")
    source = AUTODEVICE_SOURCE.read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "autodevice")
    first = experiment.ROOT / "fits" / experiment.WORLDS[0] / experiment.MODES[0]
    observation_path = first / "experiment.json"
    observation = read(observation_path)
    expected_runtime = {**expected["runtime_flags"], **expected["inherited_loader_environment"]}
    if (observation.get("status") != "complete" or observation["world"] != experiment.WORLDS[0]
            or observation["mode"] != experiment.MODES[0]
            or observation["protocol_sha256"] != PROTOCOL_SHA256
            or observation["runtime_environment"] != expected_runtime
            or observation["validation_fit_sha256"] != digest(first / "validation_fit.json")):
        raise ValueError("First completed runtime observation does not support this exact amendment")
    amendment = {
        "schema_version": 1, "status": "declared_analysis_only_amendment",
        "original_protocol": {"path": str(protocol_path), "sha256": PROTOCOL_SHA256},
        "original_runner": {"path": str(runner_path), "sha256": RUNNER_SHA256},
        "scoring_runner": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "native_autodevice": {"path": str(AUTODEVICE_SOURCE), "sha256": AUTODEVICE_SHA256,
                              "function_start_line": function.lineno, "function_end_line": function.end_lineno,
                              "mutation_line": 189},
        "first_runtime_observation": {"path": str(observation_path), "sha256": digest(observation_path),
                                      "validation_fit_sha256": observation["validation_fit_sha256"],
                                      "runtime_environment": observation["runtime_environment"]},
        "change": {"field": "inherited_loader_environment.CUDA_VISIBLE_DEVICES", "from": None, "to": "0"},
        "initial_loader_environment": initial,
        "expected_post_autodevice_environment": expected["inherited_loader_environment"],
        "reason": "Installed scGLUE autodevice selects GPU0 and sets CUDA_VISIBLE_DEVICES='0'; the completed receipt records that post-selection value rather than the initially unset variable.",
        "scope": "Amends only the in-memory scorer expectation. All original load_fit guards remain; protocol, training code, fitted outputs, controller, original reports and evaluator failures are preserved. No retraining or causal scoring.",
    }
    path = experiment.ROOT / "runtime_scoring_amendment.json"
    experiment.validation.write_new(path, amendment)
    return expected, path


def evaluate(world):
    if world not in experiment.WORLDS:
        raise ValueError("Unplanned development world")
    expected, amendment = prepare()
    runs, train_runs, evidence = {}, {}, {}
    for mode in experiment.MODES:
        runs[mode], evidence[mode] = experiment.load_fit(world, mode, "val", expected)
        train_runs[mode], _ = experiment.load_fit(world, mode, "train", expected)
    report = experiment.evaluate_views(world, runs, train_runs, evidence)
    report["protocol_sha256"] = PROTOCOL_SHA256
    report["runtime_scoring_amendment"] = {"path": str(amendment), "sha256": experiment.glue.file_hash(amendment)}
    report["scoring_source_sha256"] = experiment.glue.file_hash(Path(__file__))
    destination = experiment.ROOT / "reports_runtime_corrected" / f"{world}.json"
    experiment.validation.write_new(destination, report)
    return {"report": str(destination), "worst_pair_gap": {
        mode: result["worst_pair_gap"] for mode, result in report["results"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "evaluate"))
    parser.add_argument("--world", choices=experiment.WORLDS)
    args = parser.parse_args()
    if args.action == "plan":
        _, path = prepare()
        result = {"amendment": str(path), "sha256": experiment.glue.file_hash(path)}
    else:
        if args.world is None:
            parser.error("evaluate requires --world")
        result = evaluate(args.world)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
