"""Isolated development fits using frozen transforms; never generate or score."""

from dataclasses import asdict, fields
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import scglue_adapter as glue

ASSAYS = glue.ASSAYS
COHORTS = ("train", "val", "discovery")


def read(path):
    return json.loads(Path(path).read_text())


def _configuration(config_type, overrides, seed, device):
    if type(seed) is not int or seed < 0 or device not in {"cpu", "cuda"}:
        raise ValueError("Require a nonnegative integer seed and explicit cpu/cuda device")
    if not isinstance(overrides, dict):
        raise ValueError("Overrides must be an object")
    defaults = asdict(config_type())
    allowed = {field.name for field in fields(config_type)} - {"seed", "device"}
    if set(overrides) - allowed:
        raise ValueError(f"Unknown or protected overrides: {sorted(set(overrides) - allowed)}")
    for key, value in overrides.items():
        default = defaults[key]
        valid = type(value) is type(default)
        if isinstance(default, float):
            valid = type(value) in {int, float} and math.isfinite(value)
        elif isinstance(default, tuple):
            valid = isinstance(value, (list, tuple)) and all(type(item) is int for item in value)
        if not valid:
            raise ValueError(f"Invalid override type or nonfinite value for {key}")
    config = config_type(**overrides, seed=seed, device=device)
    config.validate()
    values = asdict(config)
    if any(isinstance(value, float) and not math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite model configuration")
    if values["learning_rate"] <= 0 or any(value < 0 for key, value in values.items() if key.startswith("lam_")):
        raise ValueError("Require positive learning rate and nonnegative loss weights")
    return config


def _input_record(path, **extra):
    return {"path": str(Path(path).resolve()), "sha256": glue.file_hash(Path(path)), **extra}


def _scmrdr_fit(wrapper, observed, transform, output, config, label):
    gene = wrapper._reuse_gene_manifest(transform, wrapper._paths(observed / "train"),
                                       wrapper.GeneConfig(), (), 2000)
    features = gene["feature_names"]
    data = {cohort: {assay: wrapper.transform_gene_features(
        observed / cohort / f"{assay}.h5ad", assay, gene, limit=2000 if cohort == "train" else None)
        for assay in ASSAYS} for cohort in ("train", "val")}
    record = {"schema_version": 1, "method": "scmrdr_paper_faithful", "label": label,
              "seed": config.seed, "status": "running", "config": asdict(config),
              "gene_transform_dir": str(transform.resolve()),
              "gene_manifest_sha256": glue.file_hash(transform / "manifest.json"),
              "feature_count": len(features), "train_limit": 2000,
              "adapter_source_sha256": glue.file_hash(Path(wrapper.__file__)),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "torch", "scanpy", "anndata", "scikit-misc")},
              "runner_source_sha256": glue.file_hash(Path(__file__)),
              "train_inputs": {}, "validation_inputs": {}}
    for cohort, key in (("train", "train_inputs"), ("val", "validation_inputs")):
        record[key] = {assay: _input_record(
            observed / cohort / f"{assay}.h5ad", retained_cells=len(values["cell_ids"]),
            dropped_cell_ids=values["dropped_cell_ids"].tolist()) for assay, values in data[cohort].items()}
    output.mkdir(parents=True)
    wrapper.write_new(output / "input_manifest.json", record)
    model = wrapper.fit({a: x["counts"] for a, x in data["train"].items()},
                        {a: x["availability"] for a, x in data["train"].items()}, features,
                        output / "model", config=config,
                        validation_counts_by_modality={a: x["counts"] for a, x in data["val"].items()})
    import numpy as np
    for assay, values in data["train"].items():
        encoded = model.encode(values["counts"], assay, features)
        path = output / "train" / f"{assay}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, cell_ids=values["cell_ids"], shared=encoded["shared"],
                                private=encoded["specific"])
    record.update(status="complete", checkpoint_sha256=glue.file_hash(output / "model/checkpoint.pt"))
    wrapper.write_new(output / "manifest.json", record)
    del model, data
    for cohort in ("val", "discovery"):
        wrapper.encode_run(output, observed / cohort, cohort,
                           limit=1000 if cohort == "discovery" else None, device=config.device)
    return record


def fit_candidate(base, world_id, regime, method, candidate, overrides, *, seed=0, device="cuda", output):
    """Fit one predeclared alternative; the caller serializes benchmark resources."""
    for value in (world_id, candidate):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Require safe world and candidate identifiers")
    if regime not in {"realistic", "informative"} or method not in {"scglue", "scmrdr"}:
        raise ValueError("Unsupported regime or method")
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing candidate output: {output}")
    wrapper = importlib.import_module("scmrdr_adapter") if method == "scmrdr" else glue
    config_type = wrapper.SCmRDRConfig if method == "scmrdr" else glue.ScGLUEConfig
    config = _configuration(config_type, overrides, seed, device)
    bank_path = base / "runs/worlds/extended_bank_manifest.json"
    specs = [entry for entry in read(bank_path)["specifications"] if entry["world_id"] == world_id]
    world = base / "runs/worlds" / world_id
    manifest_path = world / "manifest.json"
    if len(specs) != 1 or specs[0]["split"] != "development":
        raise ValueError("Integration validation requires an existing development world")
    manifest = read(manifest_path)
    if (manifest.get("status") != "complete" or manifest.get("split") != "development"
            or manifest.get("world_id") != world_id or regime not in manifest.get("regimes", ())
            or manifest.get("count_calibration_status") != "passed_declared_checks"):
        raise ValueError("Require a complete calibration-passed development world and regime")
    primary = base / "runs/extended" / world_id / regime / "primary"
    if output.is_relative_to(world) or output.is_relative_to(primary):
        raise ValueError("Candidate output must be separate from frozen worlds and primary runs")
    observed = world / "observed" / regime
    inputs = {f"{cohort}/{assay}": _input_record(observed / cohort / f"{assay}.h5ad")
              for cohort in COHORTS for assay in ASSAYS}
    label = f"validation_{method}_{candidate}_seed{seed}"
    if method == "scglue":
        prepared = primary / "scglue_preprocessing"
        preprocessing = importlib.import_module("preprocessing")
        for cohort in COHORTS:
            cached = read(prepared / ("manifest.json" if cohort == "train" else f"cohorts/{cohort}/manifest.json"))
            if cached.get("status") != "complete":
                raise ValueError("Incomplete frozen preprocessing")
            for assay in ASSAYS:
                if cached["inputs"][assay] != inputs[f"{cohort}/{assay}"]:
                    raise ValueError("Cached preprocessing belongs to different observed inputs")
            preprocessing._validate_saved_outputs(cached)
        exported = glue.fit_prepared(prepared, prepared / "guidance.graphml.gz", output,
                                     config=config, label=label, cohorts=COHORTS)
        fitted = read(output / "model_manifest.json")
        fit_manifest = output / "model_manifest.json"
        transform = {"path": str(prepared), **fitted["inputs"]["transformations"]}
        retention = exported["retention"]
        training = fitted["training_history"]
        sources = fitted["inputs"]["package_provenance"]
    else:
        transform_dir = primary / "gene_transform"
        fitted = _scmrdr_fit(wrapper, observed, transform_dir, output, config, label)
        fit_manifest = output / "manifest.json"
        transform = _input_record(transform_dir / "manifest.json")
        retention = {f"{cohort}/{assay}": entry for cohort, key in
                     (("train", "train_inputs"), ("val", "validation_inputs"))
                     for assay, entry in fitted[key].items()}
        for assay, entry in read(output / "discovery/encoding_manifest.json")["inputs"].items():
            retention[f"discovery/{assay}"] = entry
        training = read(output / "model/fit_summary.json")
        sources = training["source"]
    result = {"schema_version": 1, "status": "complete", "world_id": world_id, "regime": regime,
              "method": method, "candidate": candidate, "label": label, "seed": seed, "device": device,
              "output": str(output), "configuration": asdict(config), "overrides": overrides,
              "cohorts": list(COHORTS), "inputs": inputs, "transform": transform,
              "world_manifest": _input_record(manifest_path), "bank_manifest": _input_record(bank_path),
              "fit_manifest": _input_record(fit_manifest), "checkpoint_sha256": fitted["checkpoint_sha256"],
              "retention": retention, "training": training, "source_provenance": sources,
              "files": {f"{cohort}/{assay}.npz": glue.file_hash(output / cohort / f"{assay}.npz")
                        for cohort in COHORTS for assay in ASSAYS},
              "adapter_source_sha256": glue.file_hash(Path(wrapper.__file__)),
              "runner_source_sha256": glue.file_hash(Path(__file__)),
              "scope": "Development public cohorts only; no truth, private cohorts, or causal evaluation"}
    result = json.loads(json.dumps(result, allow_nan=False))
    glue.write_json_new(output / "validation_fit.json", result)
    return result
