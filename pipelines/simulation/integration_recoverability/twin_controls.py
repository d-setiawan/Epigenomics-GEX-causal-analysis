"""Exact ambiguity controls using existing worlds and one frozen integration.

No model is fitted here. Public embeddings are deliberately reused byte for
byte after checking equality of all public assay inputs. Private views may be
encoded with the saved model solely for evaluator-side retention probes. A
different private prediction score is not evidence of distinguishing the
identical unpaired inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ASSAYS = ("RNA", "H3K27ac", "H3K4me3")
PUBLIC_COHORTS = ("train", "val", "discovery")
PRIVATE_COHORTS = ("probe", "assessment")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_new(path: Path, value: Any) -> None:
    value = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != value:
            raise FileExistsError(f"Refusing to change existing twin artifact: {path}")
        return
    with path.open("x") as handle:
        handle.write(value)


def copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    """Independent copies prevent accidental mutation of the frozen source."""
    digest = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != digest:
            raise FileExistsError(f"Existing counterpart differs from source: {destination}")
    else:
        shutil.copy2(source, destination)
    if file_hash(destination) != digest:
        raise IOError(f"Copy checksum differs: {destination}")
    return {"source": str(source.resolve()), "destination": str(destination.resolve()),
            "sha256": digest, "mode": "independent_byte_copy"}


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _checked_manifest_file(root: Path, manifest: Mapping[str, Any], relative: str) -> str:
    expected = manifest["files"][relative]
    if file_hash(root / relative) != expected:
        raise ValueError(f"World artifact changed after freeze: {root / relative}")
    return expected


def validate_twins(base_world: Path, counterpart_world: Path, regime: str) -> dict[str, Any]:
    """Prove exact learner-visible equality, not approximate statistical similarity."""
    roots = (Path(base_world).resolve(), Path(counterpart_world).resolve())
    manifests = [read_json(root / "manifest.json") for root in roots]
    truths = [read_json(root / "truth/manifest.json") for root in roots]
    for root, manifest in zip(roots, manifests):
        _checked_manifest_file(root, manifest, "truth/manifest.json")
    for manifest, truth in zip(manifests, truths):
        if manifest.get("status") != "complete" or truth["spec"]["design"] != "extended":
            raise ValueError("The ambiguity control requires two complete extended worlds")
        if truth["spec"].get("motif") != "ambiguity" or regime not in manifest["regimes"]:
            raise ValueError("Not an ambiguity world with the requested count regime")
    specs = [truth["spec"] for truth in truths]
    if [float(spec["rho"]) for spec in specs] != [0.0, 0.7]:
        raise ValueError("Use the declared rho=0 base and rho=0.7 counterpart, in that order")
    invariant_specs = [{key: value for key, value in spec.items() if key not in {"rho", "world_id"}} for spec in specs]
    if invariant_specs[0] != invariant_specs[1]:
        raise ValueError("Ambiguity specifications differ beyond rho and world ID")
    if truths[0]["candidates"] != truths[1]["candidates"]:
        raise ValueError("Scored feature identities, roles, or order differ across twins")
    public_keys = [{key for key in manifest["files"] if key.startswith("observed/")} for manifest in manifests]
    if public_keys[0] != public_keys[1]:
        raise ValueError("Public artifact sets differ across twins")
    required = {f"observed/{regime}/{cohort}/{assay}.h5ad" for cohort in PUBLIC_COHORTS for assay in ASSAYS}
    if not required <= public_keys[0]:
        raise ValueError("Missing train, validation, or discovery public assay inputs")
    public = {}
    for relative in sorted(public_keys[0]):
        hashes = [_checked_manifest_file(root, manifest, relative) for root, manifest in zip(roots, manifests)]
        if hashes[0] != hashes[1]:
            raise ValueError(f"Learner-visible twin inputs are not byte-identical: {relative}")
        public[relative] = hashes[0]
    private = {}
    for cohort in PRIVATE_COHORTS:
        for assay in ASSAYS:
            relative = f"private_inputs/{regime}/{cohort}/{assay}.h5ad"
            private[relative] = [_checked_manifest_file(root, manifest, relative)
                                 for root, manifest in zip(roots, manifests)]
    joint_audits = {}
    for cohort in (*PUBLIC_COHORTS, *PRIVATE_COHORTS):
        relative = f"truth/{cohort}.npz"
        for root, manifest in zip(roots, manifests):
            _checked_manifest_file(root, manifest, relative)
        arrays = [load_arrays(root / relative) for root in roots]
        for key in ("cell_ids", "assigned_assay", "state", "candidate_names", "candidate_assays", "candidate_roles"):
            if not np.array_equal(arrays[0][key], arrays[1][key]):
                raise ValueError(f"Twin {cohort} {key} differs; not the same sampled populations")
        if cohort in PUBLIC_COHORTS:
            for assay in ASSAYS:
                rows = np.flatnonzero(arrays[0]["assigned_assay"] == assay)
                columns = np.flatnonzero(arrays[0]["candidate_assays"] == assay)
                for key in ("activities", f"counts_{regime}"):
                    if not np.array_equal(arrays[0][key][np.ix_(rows, columns)], arrays[1][key][np.ix_(rows, columns)]):
                        raise ValueError(f"Twin observed {cohort}/{assay} {key} differs")
        changed = not np.array_equal(arrays[0]["activities"], arrays[1]["activities"])
        joint_audits[cohort] = {"joint_activities_differ": changed,
                               "changed_activity_entries": int(np.count_nonzero(arrays[0]["activities"] != arrays[1]["activities"]))}
    if not joint_audits["assessment"]["joint_activities_differ"]:
        raise ValueError("Hidden paired assessment activities do not differ")
    edges = [set(map(tuple, truth["biological_dag_edges"])) for truth in truths]
    if edges[0] == edges[1]:
        raise ValueError("The declared hidden biological structures do not differ")
    return {
        "status": "verified_exact_public_equivalence", "base_world": str(roots[0]),
        "counterpart_world": str(roots[1]), "regime": regime, "rho": [0.0, 0.7],
        "world_manifest_sha256": [file_hash(root / "manifest.json") for root in roots],
        "truth_manifest_sha256": [file_hash(root / "truth/manifest.json") for root in roots],
        "public_files": public, "private_files": private, "joint_audits": joint_audits,
        "biological_edges_base_only": sorted(map(list, edges[0] - edges[1])),
        "biological_edges_counterpart_only": sorted(map(list, edges[1] - edges[0])),
        "interpretation": "Identifiability stress test: identical unpaired observations cannot distinguish these joint structures",
        "independent_biological_trials": 1,
    }


def _module(name: str):
    local = str(Path(__file__).resolve().parent)
    if local not in sys.path:
        sys.path.insert(0, local)
    return importlib.import_module(name)


def validate_base_run(base_world: Path, base_run: Path, method: str, regime: str,
                      base_preprocessing: Path | None) -> dict[str, Any]:
    """Verify the fit lineage before exposing any counterpart private inputs."""
    if method == "scglue":
        if base_preprocessing is None:
            raise ValueError("scGLUE requires the frozen --base-preprocessing directory")
        model = read_json(base_run / "model_manifest.json")
        prep = read_json(base_preprocessing / "manifest.json")
        if model.get("status") != "complete" or prep.get("status") != "complete":
            raise ValueError("scGLUE model and preprocessing must be complete")
        for assay in ASSAYS:
            expected = file_hash(base_world / "observed" / regime / "train" / f"{assay}.h5ad")
            if prep["inputs"][assay]["sha256"] != expected:
                raise ValueError("scGLUE preprocessing did not fit on the base world's public training counts")
            if file_hash(base_preprocessing / "train" / f"{assay}.h5ad") != model["inputs"]["training_files"][assay]["sha256"]:
                raise ValueError("Prepared scGLUE training data do not match the saved model")
        if file_hash(Path(model["inputs"]["guidance"]["path"])) != model["inputs"]["guidance"]["sha256"]:
            raise ValueError("Fitted guidance changed")
        checkpoint = base_run / "scglue_model.dill"
        label = model["label"]
        receipt_path = base_run / "embedding_manifest.json"
        manifest_paths = [base_run / "model_manifest.json", base_preprocessing / "manifest.json", receipt_path]
    elif method == "scmrdr":
        model = read_json(base_run / "manifest.json")
        if model.get("status") != "complete" or model.get("method") != "scmrdr_paper_faithful":
            raise ValueError("A complete paper-faithful scMRDR model is required")
        for section, cohort in (("train_inputs", "train"), ("validation_inputs", "val")):
            if not model.get(section):
                raise ValueError("Extended scMRDR control requires separate recorded train/validation inputs")
            for assay in ASSAYS:
                expected = file_hash(base_world / "observed" / regime / cohort / f"{assay}.h5ad")
                if model[section][assay]["sha256"] != expected:
                    raise ValueError(f"scMRDR {cohort} did not use the base world's public counts")
        gene_path = Path(model["gene_transform_dir"]) / "manifest.json"
        if file_hash(gene_path) != model["gene_manifest_sha256"]:
            raise ValueError("Frozen scMRDR gene transformation changed")
        checkpoint = base_run / "model/checkpoint.pt"
        label = model["label"]
        receipt_path = base_run / "public_embedding_manifest.json"
        manifest_paths = [base_run / "manifest.json", gene_path, receipt_path]
    else:
        raise ValueError("Unsupported integration method")
    if file_hash(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError("The frozen model checkpoint changed")
    if not receipt_path.exists():
        raise ValueError(f"Missing immutable public-embedding receipt: {receipt_path}; current hashes alone do not prove checkpoint lineage")
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("checkpoint_sha256") != model["checkpoint_sha256"]:
        raise ValueError("Public-embedding receipt does not identify the complete frozen model")
    public_embeddings = {}
    for cohort in PUBLIC_COHORTS:
        for assay in ASSAYS:
            path = base_run / cohort / f"{assay}.npz"
            arrays = load_arrays(path)
            ids, shared = arrays["cell_ids"], arrays["shared"]
            if len(ids) != len(set(map(str, ids))) or shared.ndim != 2 or len(shared) != len(ids) or not np.isfinite(shared).all():
                raise ValueError("Invalid public embeddings in frozen base run")
            relative = str(path.relative_to(base_run))
            digest = file_hash(path)
            if receipt["files"].get(relative) != digest:
                raise ValueError(f"Public embedding differs from its frozen output receipt: {relative}")
            public_embeddings[relative] = digest
    return {"label": label, "seed": model["seed"], "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": file_hash(checkpoint), "public_embedding_files": public_embeddings,
            "manifests": {str(path.resolve()): file_hash(path) for path in manifest_paths}}


def _encode_private_scglue(base_run: Path, preprocessing: Path, counterpart: Path,
                          output: Path, regime: str, device: str) -> tuple[Path, Path]:
    prepared = output / "counterpart_preprocessing"
    for source in sorted((preprocessing / "transforms").rglob("*")):
        if source.is_file():
            copy_verified(source, prepared / source.relative_to(preprocessing))
    prep = _module("preprocessing")
    for cohort in PRIVATE_COHORTS:
        paths = {assay: counterpart / "private_inputs" / regime / cohort / f"{assay}.h5ad" for assay in ASSAYS}
        prep.transform_preprocessing(paths, preprocessing, out_dir=prepared, cohort=cohort)
    encoded = output / "private_inference"
    _module("scglue_adapter").encode_prepared(base_run, prepared, encoded,
                                             cohorts=PRIVATE_COHORTS, device=device)
    return encoded, prepared / "representations"


def _encode_private_scmrdr(base_run: Path, counterpart: Path, output: Path,
                          regime: str, device: str) -> tuple[Path, None]:
    encoded = output / "private_inference"
    manifest = read_json(base_run / "manifest.json")
    checkpoint = encoded / "model/checkpoint.pt"
    copy_verified(base_run / "model/checkpoint.pt", checkpoint)
    gene_root = encoded / "gene_transform"
    copy_verified(Path(manifest["gene_transform_dir"]) / "manifest.json", gene_root / "manifest.json")
    manifest.update(gene_transform_dir=str(gene_root.resolve()),
                    frozen_model_reuse={"base_run": str(base_run.resolve()),
                                        "base_manifest_sha256": file_hash(base_run / "manifest.json"),
                                        "training_performed_here": False})
    write_new(encoded / "manifest.json", manifest)
    for cohort in PRIVATE_COHORTS:
        input_root = counterpart / "private_inputs" / regime / cohort
        stage_path = encoded / f"verified_{cohort}.json"
        inputs = {assay: file_hash(input_root / f"{assay}.h5ad") for assay in ASSAYS}
        fingerprint = content_hash({"inputs": inputs, "checkpoint": file_hash(checkpoint), "device": device})
        if stage_path.exists():
            previous = read_json(stage_path)
            if previous["input_fingerprint"] != fingerprint:
                raise ValueError("Cached private inference used different inputs")
            if any(file_hash(encoded / relative) != digest for relative, digest in previous["files"].items()):
                raise ValueError("Cached private inference arrays changed")
        else:
            _module("scmrdr_adapter").encode_run(encoded, input_root, cohort, device=device)
            files = {f"{cohort}/{assay}.npz": file_hash(encoded / cohort / f"{assay}.npz") for assay in ASSAYS}
            write_new(stage_path, {"input_fingerprint": fingerprint, "files": files})
    return encoded, None


def prepare_counterpart(base_world: Path, counterpart_world: Path, base_run: Path, output: Path, *,
                        method: str, regime: str, base_preprocessing: Path | None = None,
                        device: str = "cuda") -> dict[str, Any]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Specify cpu or cuda explicitly")
    base_world, counterpart_world, base_run, output = [Path(path).resolve() for path in (base_world, counterpart_world, base_run, output)]
    base_preprocessing = Path(base_preprocessing).resolve() if base_preprocessing is not None else None
    audit = validate_twins(base_world, counterpart_world, regime)
    lineage = validate_base_run(base_world, base_run, method, regime, base_preprocessing)
    source_names = ["twin_controls.py", "scglue_adapter.py", "preprocessing.py"] if method == "scglue" else ["twin_controls.py", "scmrdr_adapter.py", "gene_features.py"]
    inputs = {"twin_audit": audit, "base_lineage": lineage, "method": method, "device": device,
              "source_sha256": {name: file_hash(Path(__file__).with_name(name)) for name in source_names}}
    fingerprint = content_hash(inputs)
    report_path = output / "twin_control.json"
    if report_path.exists():
        previous = read_json(report_path)
        if previous["input_fingerprint"] != fingerprint:
            raise FileExistsError("Twin control output uses different frozen inputs")
        if any(file_hash(output / path) != digest for path, digest in previous["files"].items()):
            raise ValueError("Completed twin-control output changed")
        return previous
    if output.exists() and any(output.iterdir()) and not (output / "input_manifest.json").exists():
        raise FileExistsError("Twin output is nonempty and is not a resumable twin-control directory")
    write_new(output / "input_manifest.json", {"input_fingerprint": fingerprint, "inputs": inputs})
    if method == "scglue":
        private, native = _encode_private_scglue(base_run, base_preprocessing, counterpart_world, output, regime, device)
    else:
        private, native = _encode_private_scmrdr(base_run, counterpart_world, output, regime, device)
    embeddings = output / "embeddings"
    copies, files = {}, {}
    for cohort in (*PUBLIC_COHORTS, *PRIVATE_COHORTS):
        source_root = base_run if cohort in PUBLIC_COHORTS else private
        for assay in ASSAYS:
            relative = f"{cohort}/{assay}.npz"
            copies[relative] = copy_verified(source_root / relative, embeddings / relative)
            files[relative] = file_hash(embeddings / relative)
            if cohort in PUBLIC_COHORTS and files[relative] != lineage["public_embedding_files"][relative]:
                raise RuntimeError("Public embedding reuse is not byte-identical")
    if file_hash(Path(lineage["checkpoint_path"])) != lineage["checkpoint_sha256"]:
        raise RuntimeError("Post-fit private encoding changed the original checkpoint")
    manifest = {"status": "complete", "label": lineage["label"],
                "method": "scmrdr_paper_faithful" if method == "scmrdr" else method,
                "seed": lineage["seed"], "input_fingerprint": fingerprint, "inputs": inputs,
                "files": files, "copies": copies, "training_performed_here": False,
                "public_embedding_policy": "Verified byte-identical reuse, not a second optimization or independent trial",
                "private_embedding_policy": "Frozen post-fit diagnostic encoding only; never used for matching"}
    write_new(embeddings / "embedding_manifest.json", manifest)
    result = {"status": "complete", "input_fingerprint": fingerprint, "inputs": inputs,
              "counterpart_embedding_directory": str(embeddings),
              "counterpart_preprocessing_embedding_directory": str(native) if native is not None else None,
              "training_performed_here": False, "public_embeddings_byte_identical": True,
              "interpretation": audit["interpretation"],
              "files": {str(path.relative_to(output)): file_hash(path)
                        for path in sorted(output.rglob("*")) if path.is_file() and path != report_path}}
    write_new(report_path, result)
    return result


def compare_evaluations(base_world: Path, counterpart_world: Path, base_report: Path,
                        counterpart_report: Path, *, regime: str) -> dict[str, Any]:
    """Compare the same unpaired outputs scored under the two private truths."""
    audit = validate_twins(base_world, counterpart_world, regime)
    reports = [read_json(base_report), read_json(counterpart_report)]
    for report, world in zip(reports, (base_world, counterpart_world)):
        if Path(report["inputs"]["world"]).resolve() != Path(world).resolve() or report["inputs"]["regime"] != regime:
            raise ValueError("Evaluation report belongs to another world or count regime")
        if report["inputs"]["world_manifest_sha256"] != file_hash(Path(world) / "manifest.json"):
            raise ValueError("World changed after evaluation")
    for key in ("configuration", "pool_seed", "matching_seed", "cohort", "evaluation_source_sha256"):
        if reports[0]["inputs"][key] != reports[1]["inputs"][key]:
            raise ValueError(f"Twin evaluations used different {key}")
    provenances = [report["inputs"]["provenance"] for report in reports]
    if not provenances[0] or set(provenances[0]) != set(provenances[1]):
        raise ValueError("Twin evaluations require the same integration labels")
    for label in provenances[0]:
        original, reused = (provenance[label] for provenance in provenances)
        for assay in ASSAYS:
            if original["files"][assay]["sha256"] != reused["files"][assay]["sha256"]:
                raise ValueError(f"Twin public embeddings are not byte-identical: {label}/{assay}")
        copied = reused["manifest"]
        if copied.get("training_performed_here") is not False:
            raise ValueError("Counterpart embeddings lack frozen-model reuse provenance")
        if copied["inputs"]["base_lineage"]["checkpoint_sha256"] != original["manifest"]["checkpoint_sha256"]:
            raise ValueError("Counterpart did not reuse the same frozen integration checkpoint")
    if reports[0].get("common_pools", {}).get("cell_ids") != reports[1].get("common_pools", {}).get("cell_ids"):
        raise ValueError("Twin evaluations did not use the same public candidate pools")
    labels = [{key for key, value in report["results"].items() if "assignment" in value} for report in reports]
    if labels[0] != labels[1] or not labels[0]:
        raise ValueError("Twin evaluations lack matching method/reference sets")
    comparisons, any_failed = {}, False
    for label in sorted(labels[0]):
        first, second = (report["results"][label] for report in reports)
        if first["assignment"]["cell_ids"] != second["assignment"]["cell_ids"]:
            raise ValueError(f"Same public embeddings produced different cell assignments: {label}")
        rows = {}
        for scale in ("activities", "counts"):
            discoveries = [first[scale], second[scale]]
            if [item["status"] for item in discoveries] != ["ok", "ok"]:
                rows[scale] = {"status": "not_evaluable", "reason": "At least one corresponding discovery failed",
                               "discovery_statuses": [item["status"] for item in discoveries]}
                any_failed = True
                continue
            same = content_hash(discoveries[0]["edges"]) == content_hash(discoveries[1]["edges"])
            if not same:
                raise ValueError(f"Identical observed {scale} readouts gave different graph output: {label}")
            metric_key = "activity_graph_metrics" if scale == "activities" else "count_graph_metrics"
            scores = [result[metric_key].get("exact_cross_assay_stress_test") for result in (first, second)]
            rows[scale] = {"status": "ok", "graph_output_identical": True,
                           "base_truth_stress_score": scores[0], "counterpart_truth_stress_score": scores[1]}
        comparisons[label] = {"matching_identical": True, "scales": rows}
    return {"status": "completed_with_failed_discovery" if any_failed else "completed",
            "twin_audit": audit, "comparisons": comparisons,
            "report_sha256": [file_hash(base_report), file_hash(counterpart_report)],
            "interpretation": "Public-input equality is guaranteed by construction; different private truth scores are not discrimination",
            "ambiguity_set_metric": {"status": "not_applicable", "reason": "Integrators/discovery methods return no justified ambiguity-set output"},
            "success_criterion": "No false certainty that identical learner-visible inputs distinguish the two hidden structures"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "compare"):
        command = sub.add_parser(name)
        command.add_argument("--base-world", type=Path, required=True)
        command.add_argument("--counterpart-world", type=Path, required=True)
        command.add_argument("--regime", choices=("realistic", "informative"), required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--base-run", type=Path, required=True)
            command.add_argument("--base-preprocessing", type=Path)
            command.add_argument("--method", choices=("scglue", "scmrdr"), required=True)
            command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name == "compare":
            command.add_argument("--base-report", type=Path, required=True)
            command.add_argument("--counterpart-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_counterpart(args.base_world, args.counterpart_world, args.base_run, args.output,
                                     method=args.method, regime=args.regime,
                                     base_preprocessing=args.base_preprocessing, device=args.device)
    elif args.command == "validate":
        result = validate_twins(args.base_world, args.counterpart_world, args.regime)
        write_new(args.output, result)
    else:
        result = compare_evaluations(args.base_world, args.counterpart_world, args.base_report,
                                     args.counterpart_report, regime=args.regime)
        write_new(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "counterpart_embedding_directory": result.get("counterpart_embedding_directory")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
