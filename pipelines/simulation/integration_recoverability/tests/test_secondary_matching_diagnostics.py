"""Tiny array/receipt tests only. Discovery is ALWAYS mocked; no real CI/fits."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/secondary_matching_diagnostics.py"
spec = importlib.util.spec_from_file_location("secondary_matching_unit", SCRIPT)
secondary = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = secondary
spec.loader.exec_module(secondary)
ev = secondary.ev


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(ev._jsonable(value), sort_keys=True))
    return ev._sha256(path)


def fake_discovery(values, names, *, method, scale, config, **kwargs):
    return {"status": "ok", "method": method, "scale": scale, "names": list(names),
            "edges": [], "tests": [{"audit_only_fake": True, "seed": config.seed}],
            "configuration": asdict(config), "n_rows": len(values)}


@pytest.fixture
def frozen_fixture(tmp_path, monkeypatch):
    base = tmp_path / "pipelines/simulation/integration_recoverability"
    base.mkdir(parents=True)
    protocol = base / "SECONDARY_DIAGNOSTICS_PROTOCOL.md"
    protocol.write_text("Fixed tiny unit-test protocol, not a scientific run.\n")
    schedule_path = base / "runs/suite/schedule_v1.json"
    monkeypatch.setattr(secondary, "BASE", base)
    monkeypatch.setattr(secondary, "REPO", tmp_path)
    monkeypatch.setattr(secondary, "PROTOCOL", protocol)
    monkeypatch.setattr(secondary, "SCHEDULE", schedule_path)
    monkeypatch.setattr(ev, "discover", fake_discovery)
    monkeypatch.setattr(secondary.operator, "assert_no_external_jobs", lambda _: None)
    for filename in ("scglue_adapter.py", "scmrdr_adapter.py", "gene_features.py"):
        write(base / filename, b"# immutable fake model source\n")
    paper = tmp_path / "pipelines/empirical/integration/methods/scmrdr/paper_faithful"
    write(paper / "adapter.py", b"# fake paper adapter\n")
    write(paper / "upstream/model.py", b"# fake upstream model\n")
    cit = tmp_path / "installed/causallearn/utils/cit.py"
    write(cit, b"# no backend loaded by these fixtures\n")
    backend = {"cit_source_path": str(cit), "cit_source_sha256": ev._sha256(cit),
               "configuration": asdict(ev.DiscoveryConfig()), "version": "fake", "fallback": "none"}
    monkeypatch.setattr(ev, "backend_provenance", lambda config: {**backend, "configuration": asdict(config)})
    bank, calibration = base / "runs/worlds/bank.json", base / "runs/calibration.json"
    write(bank, {})
    write(calibration, {})
    schedule = {"base_directory": str(base), "repository": str(tmp_path), "runs_directory": str(base / "runs"),
                "stages": [], "inputs": {"bank_path": str(bank), "bank_sha256": ev._sha256(bank),
                "calibration_path": str(calibration), "calibration_sha256": ev._sha256(calibration),
                "source_pins": secondary.suite.source_pins(base, tmp_path)}}
    schedule["schedule_hash"] = secondary.suite.canonical(schedule)
    write(schedule_path, schedule)
    world = base / "runs/worlds/linear_chain_0"
    root = base / "runs/extended/linear_chain_0/informative/primary"
    report_path = root / "evaluation_scglue_scmrdr_seeds0/report.json"
    candidates = [{"name": f"{assay}_{role}", "assay": assay, "role": role, "feature_id": f"f{index}_{role}"}
                  for index, assay in enumerate(ev.ASSAYS) for role in ("core", "extra", "sentinel")]
    n = 24
    ids = np.array([f"{assay}_{i:03d}" for assay in ev.ASSAYS for i in range(n)])
    counts = np.arange(len(ids) * 9).reshape(-1, 9) % 13
    raw_truth = {"cell_ids": ids, "assigned_assay": np.repeat(ev.ASSAYS, n),
                 "state": np.column_stack((np.arange(len(ids)) % n, (np.arange(len(ids)) % n) ** 2)),
                 "activities": np.arange(len(ids) * 9, dtype=float).reshape(-1, 9) / 31,
                 "exposures": np.ones((len(ids), 3)), "counts_informative": counts,
                 "candidate_names": np.array([item["name"] for item in candidates]),
                 "candidate_assays": np.array([item["assay"] for item in candidates]),
                 "candidate_roles": np.array([item["role"] for item in candidates]),
                 "candidate_feature_ids": np.array([item["feature_id"] for item in candidates])}
    truth = ev.validate_truth(raw_truth, "informative")
    files, validation_cohorts = {}, {}
    for cohort in ("discovery", "assessment", "probe"):
        truth_path = world / "truth" / f"{cohort}.npz"
        ev._save_npz_new(truth_path, **raw_truth)
        files[str(truth_path.relative_to(world))] = ev._sha256(truth_path)
        entries = {}
        for assay in ev.ASSAYS:
            source = world / ("observed" if cohort == "discovery" else "private_inputs") / "informative" / cohort / f"{assay}.h5ad"
            digest = write(source, f"unit counts: {cohort}/{assay}".encode())
            files[str(source.relative_to(world))] = digest
            entries[assay] = {"path": str(source), "sha256": digest}
        validation_cohorts[cohort] = {"truth_sha256": ev._sha256(truth_path), "assays": entries}
    truth_manifest = {"candidates": candidates, "biological_dag_edges": [[candidates[0]["name"], candidates[3]["name"]]]}
    truth_manifest_path = world / "truth/manifest.json"
    files["truth/manifest.json"] = write(truth_manifest_path, truth_manifest)
    world_manifest = {"status": "complete", "world_id": world.name, "design": "extended", "split": "development", "files": files}
    world_manifest["manifest_hash"] = secondary.canonical(world_manifest)
    write(world / "manifest.json", world_manifest)
    write(world / "truth/observation_validation_informative.json", {
        "status": "ok", "world": str(world), "regime": "informative", "cohorts": validation_cohorts,
        "truth_manifest_sha256": ev._sha256(truth_manifest_path)})
    pools = {assay: ids[truth["assigned_assay"] == assay] for assay in ev.ASSAYS}
    arrays = {assay: {"cell_ids": pool, "shared": np.column_stack((np.arange(n), np.arange(n) ** 2))}
              for assay, pool in pools.items()}
    provenance, runs = {}, {}
    for label in secondary.METHODS:
        glue = label == secondary.METHODS[0]
        directory = root / ("scglue_seed0" if glue else "scmrdr_seed0")
        checkpoint = directory / ("scglue_model.dill" if glue else "model/checkpoint.pt")
        checkpoint_hash = write(checkpoint, f"not a model: {label}".encode())
        public_files = {}
        for cohort in ("train", "val", "discovery"):
            for assay in ev.ASSAYS:
                path = directory / cohort / f"{assay}.npz"
                ev._save_npz_new(path, **arrays[assay])
                public_files[f"{cohort}/{assay}.npz"] = ev._sha256(path)
        model = {"status": "complete", "seed": 0, "label": label,
                 "method": "scglue" if glue else "scmrdr_paper_faithful", "checkpoint_sha256": checkpoint_hash}
        if glue:
            prepared = root / "scglue_preprocessing"
            write(prepared / "manifest.json", {"status": "complete"})
            model["inputs"] = {"adapter_sha256": ev._sha256(base / "scglue_adapter.py"), "transformations": {"files": {}}}
            model_path = directory / "model_manifest.json"
            write(model_path, model)
            exported = {"status": "complete", "label": label, "checkpoint_sha256": checkpoint_hash,
                        "inputs": {"model_manifest_sha256": ev._sha256(model_path)}, "files": public_files}
            write(directory / "embedding_manifest.json", exported)
        else:
            gene_dir = root / "gene_transform"
            write(gene_dir / "manifest.json", {"excluded_features": [], "source_sha256": ev._sha256(base / "gene_features.py")})
            model.update(config={"seed": 0}, gene_transform_dir=str(gene_dir), gene_manifest_sha256=ev._sha256(gene_dir / "manifest.json"),
                         adapter_source_sha256=ev._sha256(base / "scmrdr_adapter.py"))
            model_path = directory / "manifest.json"
            write(model_path, model)
            write(directory / "model/fit_summary.json", {"method": model["method"], "config": model["config"],
                "source": {"adapter_sha256": ev._sha256(paper / "adapter.py"), "vendored_sha256": {"model.py": ev._sha256(paper / "upstream/model.py")}}})
            for cohort in ("train", "val", "discovery"):
                identity = {"checkpoint_sha256": checkpoint_hash, "fit_manifest_sha256": ev._sha256(model_path)}
                if cohort != "train":
                    audit_path = directory / cohort / "encoding_manifest.json"
                    write(audit_path, {"checkpoint_sha256": checkpoint_hash, "post_fit_only": True})
                    identity["encoding_manifest_sha256"] = ev._sha256(audit_path)
                write(directory / cohort / "artifact_receipt.json", {"status": "complete", "identity": identity,
                    "files": {f"{assay}.npz": public_files[f"{cohort}/{assay}.npz"] for assay in ev.ASSAYS}})
        write(directory / "public_embedding_manifest.json", {"status": "complete", "checkpoint_sha256": checkpoint_hash, "files": public_files})
        loaded_label, run, recorded = ev._load_embedding_run(directory, "discovery")
        assert loaded_label == label
        provenance[label], runs[label] = recorded, run
    _, pool_audit = ev.select_common_pools(runs, pools)
    results = {}
    paired = secondary.paired_readout(truth, pools, "RNA")
    for arm in secondary.ARMS:
        if arm == secondary.ARMS[0]:
            arrays_to_save = paired
            values = paired
            assignment = None
        else:
            assignment = (ev.match_one_to_one(pools, random_seed=3141) if arm == "random_reference" else ev.match_one_to_one(pools, arrays))
            values = ev.assemble_readout(assignment, truth)
            arrays_to_save = {scale: values[scale] for scale in secondary.SCALES}
            arrays_to_save.update(candidate_names=truth["candidate_names"], **{f"cell_ids_{assay}": cell_ids for assay, cell_ids in assignment["cell_ids"].items()})
        ev._save_npz_new(secondary.primary_readout_path(report_path, arm), **arrays_to_save)
        results[arm] = {scale: fake_discovery(values[scale], truth["candidate_names"], method="fci", scale=scale, config=ev.DiscoveryConfig()) for scale in secondary.SCALES}
        if assignment is not None:
            results[arm]["assignment"] = assignment
    results["independent_paired_assessment"] = deepcopy(results[secondary.ARMS[0]])
    results["independent_activities_with_state"] = {"discovery": fake_discovery(truth["activities"], truth["candidate_names"],
                                                                                method="pc", scale="activities", config=ev.DiscoveryConfig())}
    inputs = {"world": str(world), "world_manifest_sha256": ev._sha256(world / "manifest.json"),
              "truth_sha256": ev._sha256(world / "truth/discovery.npz"), "truth_manifest_sha256": ev._sha256(truth_manifest_path),
              "configuration": asdict(ev.DiscoveryConfig()), "regime": "informative", "cohort": "discovery",
              "pool_seed": 2718, "matching_seed": 3141, "provenance": provenance,
              "evaluation_source_sha256": ev._sha256(Path(ev.__file__)), "backend_provenance": backend,
              "extra_inputs": {str(root / "scglue_preprocessing/representations" / filename): "missing"
                               for filename in ("manifest.json", "embedding_manifest.json")}}
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    report = {"status": "completed", "inputs": inputs, "input_fingerprint": fingerprint, "failed_integrations": [],
              "results": results, "common_pools": {**pool_audit, "cell_ids": pools}, "candidates": candidates,
              "graphical_truth": {key: {"status": "ok", "edges": []} for key in ("activity_projection", "count_measurement_pag")},
              "probes": {"status": "fake_context_only"}, "preprocessing_probes": {"status": "fake_context_only"}, "limitations": []}
    write(report_path, report)
    write(report_path.parent / "input_manifest.json", {"inputs": inputs, "input_fingerprint": fingerprint})
    return {"report_path": report_path, "world": world, "output": base / "runs/secondary/unit", "truth": truth, "pools": pools}


def make_frozen(fixture):
    result = secondary.freeze(fixture["report_path"], fixture["output"])
    return result["freeze_sha256"]


def test_freeze_does_no_ci_and_binds_sources_models_readouts(frozen_fixture, monkeypatch):
    monkeypatch.setattr(ev, "discover", lambda *a, **kw: pytest.fail("Freeze must not run CI"))
    sha = make_frozen(frozen_fixture)
    receipt = secondary.verify_frozen(frozen_fixture["output"], sha)
    assert receipt["conditions"] == list(secondary.CONDITIONS)
    assert len(receipt["models"]) == 2
    for model in receipt["models"].values():
        assert receipt["file_sha256"][model["checkpoint"]] == model["checkpoint_sha256"]
    assert str(secondary.PROTOCOL) in receipt["file_sha256"]
    assert "momentchi2" in receipt["packages"]
    assert any("momentchi2" in path and path.endswith(".py") for path in receipt["file_sha256"])
    assert len(receipt["absent_optional_representation_manifests"]) == 2
    for arm in secondary.ARMS:
        assert str(secondary.primary_readout_path(frozen_fixture["report_path"], arm)) in receipt["file_sha256"]
    with pytest.raises(FileExistsError):
        make_frozen(frozen_fixture)


def test_all_conditions_use_own_anchor_panel_and_fixed_pool_marginals(frozen_fixture, monkeypatch):
    seen = []
    def fake(values, names, **kwargs):
        seen.append((kwargs["config"].seed, kwargs["scale"], np.array(values)))
        return fake_discovery(values, names, **kwargs)
    monkeypatch.setattr(ev, "discover", fake)
    sha = make_frozen(frozen_fixture)
    report = secondary.run(frozen_fixture["output"], sha)
    assert report["status"] == "secondary_complete"
    assert len(seen) == 40 and {seed for seed, _, _ in seen} == {0, 1, 2}
    assert "input_fingerprint" not in report and report["schema"] == "secondary_matching_report_v1"
    for condition in secondary.CONDITIONS:
        entry = report["conditions"][condition["id"]]
        paired_path = entry["results"][secondary.ARMS[0]]["readout"]["readout"]["path"]
        saved = ev._load_npz(Path(paired_path))
        truth, pools = frozen_fixture["truth"], frozen_fixture["pools"]
        rows = np.array([list(truth["cell_ids"]).index(cell) for cell in pools[condition["anchor"]]])
        expected = {"counts": truth["counts"][rows], "activities": truth["activities"][rows],
                    "cell_ids": pools[condition["anchor"]], "candidate_names": truth["candidate_names"]}
        assert all(np.array_equal(saved[key], expected[key]) for key in expected)
        if condition["anchor"] != "RNA":
            rna_rows = np.array([list(truth["cell_ids"]).index(cell) for cell in pools["RNA"]])
            assert not np.array_equal(saved["activities"], truth["activities"][rna_rows])
            assert not np.array_equal(saved["cell_ids"], pools["RNA"])
        for arm in secondary.ARMS[1:]:
            result = entry["results"][arm]["readout"]
            assert result["fixed_pool_marginals"]["status"] == "verified_exact_marginals"
            assert result["assignment"]["anchor"] == condition["anchor"]
        controls = entry["contextual_controls"]
        expected_status = "cached_primary_context" if condition["ci_seed"] == 0 else "not_run"
        assert all(value["status"] == expected_status for value in controls.values())
    assert len(list((frozen_fixture["output"] / "stages").glob("*/result.json"))) == 60
    monkeypatch.setattr(ev, "discover", lambda *a, **kw: pytest.fail("Completed cache must not rerun CI"))
    assert secondary.run(frozen_fixture["output"], sha)["status"] == "secondary_complete"


def test_rna_repeats_preserve_learned_state_assignments_but_change_random(frozen_fixture):
    context = secondary.audit_primary(frozen_fixture["report_path"])
    for arm in ("true_state_reference", *secondary.METHODS):
        for condition in secondary.CONDITIONS[2:]:
            result = secondary.condition_assignment(condition, arm, context)
            assert secondary.canonical(result) == secondary.canonical(context["report"]["results"][arm]["assignment"])
    one = secondary.condition_assignment(secondary.CONDITIONS[2], "random_reference", context)
    two = secondary.condition_assignment(secondary.CONDITIONS[3], "random_reference", context)
    assert not np.array_equal(one["cell_ids"]["H3K27ac"], two["cell_ids"]["H3K27ac"])


def test_learned_matcher_receives_no_truth_or_counts(frozen_fixture, monkeypatch):
    context = secondary.audit_primary(frozen_fixture["report_path"])
    original = ev.match_one_to_one
    def checked(pools, representations=None, **kwargs):
        assert representations is not None
        assert all(set(value) == {"cell_ids", "shared"} for value in representations.values())
        return original(pools, representations, **kwargs)
    monkeypatch.setattr(ev, "match_one_to_one", checked)
    for arm in secondary.METHODS:
        secondary.condition_assignment(secondary.CONDITIONS[0], arm, context)


@pytest.mark.parametrize("change", ["source", "protocol", "checkpoint", "primary_readout", "observation_validation"])
def test_input_mutations_rejected_before_any_ci(frozen_fixture, monkeypatch, change):
    sha = make_frozen(frozen_fixture)
    receipt = secondary.read(frozen_fixture["output"] / "freeze.json")
    paths = {"source": secondary.BASE / "gene_features.py", "protocol": secondary.PROTOCOL,
             "checkpoint": Path(receipt["models"][secondary.METHODS[0]]["checkpoint"]),
             "primary_readout": secondary.primary_readout_path(frozen_fixture["report_path"], secondary.ARMS[0]),
             "observation_validation": frozen_fixture["world"] / "truth/observation_validation_informative.json"}
    paths[change].write_bytes(paths[change].read_bytes() + b"\nchanged")
    monkeypatch.setattr(ev, "discover", lambda *a, **kw: pytest.fail("Mutation must block CI"))
    with pytest.raises(ValueError, match="changed"):
        secondary.run(frozen_fixture["output"], sha)


@pytest.mark.parametrize("key,value", [("regime", "realistic"), ("cohort", "assessment"), ("matching_seed", 7)])
def test_reject_other_protocol_choices(frozen_fixture, key, value):
    report = secondary.read(frozen_fixture["report_path"])
    report["inputs"][key] = value
    write(frozen_fixture["report_path"], report)
    with pytest.raises(ValueError, match="configuration"):
        make_frozen(frozen_fixture)


def test_reject_assessment_world_before_loading_truth(frozen_fixture):
    report = secondary.read(frozen_fixture["report_path"])
    report["inputs"]["world"] = str(frozen_fixture["world"].with_name("linear_chain_2"))
    write(frozen_fixture["report_path"], report)
    with pytest.raises(ValueError, match="Assessment/other worlds"):
        make_frozen(frozen_fixture)


def test_missing_export_manifest_fails_closed(frozen_fixture):
    report = secondary.read(frozen_fixture["report_path"])
    model = Path(report["inputs"]["provenance"][secondary.METHODS[0]]["directory"])
    (model / "embedding_manifest.json").rename(model / "removed_manifest_for_unit_test.json")
    with pytest.raises(FileNotFoundError):
        make_frozen(frozen_fixture)


def test_numerical_failure_stays_none_and_is_not_retried(frozen_fixture, monkeypatch):
    def failing(values, names, **kwargs):
        result = fake_discovery(values, names, **kwargs)
        if kwargs["scale"] == "counts":
            result.update(status="failed", edges=None, error_type="FloatingPointError", error="unit-test numerical failure")
        return result
    monkeypatch.setattr(ev, "discover", failing)
    sha = make_frozen(frozen_fixture)
    report = secondary.run(frozen_fixture["output"], sha)
    assert report["status"] == "secondary_complete_with_failures" and len(report["failures"]) == 20
    for condition in report["conditions"].values():
        for arm in condition["results"].values():
            assert arm["counts"]["discovery"]["edges"] is None
            assert arm["counts"]["graph_metrics"]["status"] == "not_applicable"
    monkeypatch.setattr(ev, "discover", lambda *a, **kw: pytest.fail("Numerical failures must never be retried"))
    assert secondary.run(frozen_fixture["output"], sha)["status"] == "secondary_complete_with_failures"


def test_explicit_interruption_resume_keeps_journal_and_finished_stage(frozen_fixture, monkeypatch):
    calls = []
    def interrupted(values, names, **kwargs):
        calls.append(kwargs["scale"])
        if len(calls) == 2:
            raise secondary.ExecutionInterrupted("unit interruption, not numerical failure")
        return fake_discovery(values, names, **kwargs)
    monkeypatch.setattr(ev, "discover", interrupted)
    sha = make_frozen(frozen_fixture)
    with pytest.raises(secondary.ExecutionInterrupted):
        secondary.run(frozen_fixture["output"], sha)
    finished = next((frozen_fixture["output"] / "stages").glob("*/result.json"))
    before = finished.read_bytes()
    with pytest.raises(ValueError, match="explicit --resume"):
        secondary.run(frozen_fixture["output"], sha)
    monkeypatch.setattr(ev, "discover", fake_discovery)
    report = secondary.run(frozen_fixture["output"], sha, resume=True)
    assert report["status"] == "secondary_complete" and finished.read_bytes() == before
    attempts = list((frozen_fixture["output"] / "attempts").glob("*/result.json"))
    assert {secondary.read(path)["status"] for path in attempts} == {"interrupted", "secondary_complete"}
    assert len(list((frozen_fixture["output"] / "attempts").glob("*/interruption_record.json"))) == 1


def test_changed_completed_secondary_readout_refused(frozen_fixture):
    sha = make_frozen(frozen_fixture)
    secondary.run(frozen_fixture["output"], sha)
    path = next((frozen_fixture["output"] / "readouts").rglob("*.npz"))
    path.write_bytes(path.read_bytes() + b"mutation")
    with pytest.raises(ValueError, match="changed after completion"):
        secondary.run(frozen_fixture["output"], sha)


@pytest.mark.parametrize("filename", ["secondary_report.json", "secondary_artifact_manifest.json"])
def test_interrupted_final_sealing_resumes_without_any_ci(frozen_fixture, monkeypatch, filename):
    original = secondary.operator.durable_json
    interrupted = []
    def stop_after_complete_write(path, value):
        original(path, value)
        if Path(path).name == filename and not interrupted:
            interrupted.append(str(path))
            raise secondary.ExecutionInterrupted("unit sealing interruption")
    monkeypatch.setattr(secondary.operator, "durable_json", stop_after_complete_write)
    sha = make_frozen(frozen_fixture)
    with pytest.raises(secondary.ExecutionInterrupted):
        secondary.run(frozen_fixture["output"], sha)
    original_bytes = (frozen_fixture["output"] / filename).read_bytes()
    monkeypatch.setattr(ev, "discover", lambda *a, **kw: pytest.fail("Sealing resume must reuse every completed graph"))
    assert secondary.run(frozen_fixture["output"], sha, resume=True)["status"] == "secondary_complete"
    assert (frozen_fixture["output"] / filename).read_bytes() == original_bytes
    assert secondary.run(frozen_fixture["output"], sha)["status"] == "secondary_complete"


def test_changed_hbe_version_blocks_cache(frozen_fixture, monkeypatch):
    sha = make_frozen(frozen_fixture)
    original = secondary.importlib.metadata.version
    monkeypatch.setattr(secondary.importlib.metadata, "version", lambda key: "changed" if key == "momentchi2" else original(key))
    with pytest.raises(ValueError, match="Software version changed: momentchi2"):
        secondary.verify_frozen(frozen_fixture["output"], sha)


def test_preprocessing_optional_absence_is_also_pinned(frozen_fixture):
    sha = make_frozen(frozen_fixture)
    receipt = secondary.read(frozen_fixture["output"] / "freeze.json")
    write(receipt["absent_optional_representation_manifests"][0], {"new": "not previously produced"})
    with pytest.raises(ValueError, match="absence changed"):
        secondary.verify_frozen(frozen_fixture["output"], sha)


def test_shared_suite_lock_blocks_secondary_execution(frozen_fixture):
    sha = make_frozen(frozen_fixture)
    with secondary.operator.global_lock(secondary.BASE / "runs/suite"):
        with pytest.raises(RuntimeError, match="global lock"):
            secondary.run(frozen_fixture["output"], sha)


def test_marginal_validation_rejects_changed_value(frozen_fixture):
    context = secondary.audit_primary(frozen_fixture["report_path"])
    assignment = secondary.condition_assignment(secondary.CONDITIONS[0], secondary.METHODS[0], context)
    values = ev.assemble_readout(assignment, context["truth"])
    values["activities"][0, 0] += 10
    with pytest.raises(ValueError, match="feature marginal"):
        secondary.verify_marginals(values, context)
