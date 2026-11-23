"""Tiny CPU fixtures: no native fitting, CUDA, checkpoints, or benchmark data."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE / "scripts"), str(BASE)]
SPEC = importlib.util.spec_from_file_location("count_input_experiment_unit", BASE / "scripts/count_input_experiment.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def prepared():
    return {a: SimpleNamespace(layers={"counts": np.ones((3, 4))}, n_vars=4,
                              var_names=np.array([f"{a}_{i}" for i in range(4)]),
                              obsm={experiment.glue.REPRESENTATIONS[a]: np.ones((3, 2))})
            for a in experiment.ASSAYS}


@pytest.fixture
def native(monkeypatch):
    configure = Mock()
    fake = SimpleNamespace(models=SimpleNamespace(configure_dataset=configure))
    monkeypatch.setitem(sys.modules, "scglue", fake)
    monkeypatch.setattr(experiment.glue, "package_provenance", lambda: {"native": "unchanged"})
    return fake


@pytest.mark.parametrize("mode", experiment.MODES)
@pytest.mark.parametrize("fail", [False, True])
def test_scoped_mode_forwards_native_call_and_restores_on_failure(native, monkeypatch, mode, fail):
    data, graph, config, directory = prepared(), object(), object(), Path("unused")
    configure = native.models.configure_dataset
    def fit(values, received_graph, received_config, received_directory):
        assert (values, received_graph, received_config, received_directory) == (data, graph, config, directory)
        domains, networks = {}, {}
        for assay, value in values.items():
            native.models.configure_dataset(value, "NB", use_layer="counts", use_rep=experiment.glue.REPRESENTATIONS[assay])
            rep = configure.call_args.kwargs["use_rep"]
            width = value.obsm[rep].shape[1] if rep else value.n_vars
            domains[assay] = {"use_rep": rep, "rep_dim": width if rep else None,
                              "use_layer": "counts", "prob_model": "NB"}
            networks[assay] = SimpleNamespace(linear_0=SimpleNamespace(in_features=width))
        if fail:
            raise RuntimeError("native failure")
        return SimpleNamespace(domains=domains, net=SimpleNamespace(x2u=networks)), {"training_sequence": ["pretrain", "balance", "fine-tune"]}
    encode = Mock(return_value={"shared": np.ones((3, 2))})
    monkeypatch.setattr(experiment.glue, "fit_native", fit)
    monkeypatch.setattr(experiment.glue, "encode_frozen", encode)
    originals = experiment.glue.fit_native, experiment.glue.encode_frozen, experiment.glue.package_provenance
    try:
        with experiment.input_mode(mode):
            provenance = experiment.glue.package_provenance()
            assert provenance["native"] == "unchanged"
            assert provenance["input_ablation"]["mode"] == mode
            assert provenance["input_ablation"]["source_sha256"] == experiment.glue.file_hash(Path(experiment.__file__))
            model, audit = experiment.glue.fit_native(data, graph, config, directory)
            assert audit["training_sequence"] == ["pretrain", "balance", "fine-tune"]
            assert audit["actual_native_encoders"]["RNA"]["input_width"] == (2 if mode == "compressed" else 4)
            experiment.glue.encode_frozen(model, "RNA", data["RNA"], {}, batch_size=2)
    except RuntimeError as error:
        assert fail and str(error) == "native failure"
    else:
        assert not fail
    assert (experiment.glue.fit_native, experiment.glue.encode_frozen, experiment.glue.package_provenance) == originals
    assert native.models.configure_dataset is configure
    assert configure.call_count == len(experiment.ASSAYS)


@pytest.mark.parametrize("operation", ["fit", "encode"])
def test_zero_library_rejected_before_native_fit_or_encoding(native, monkeypatch, operation):
    data = prepared()
    data["RNA"].layers["counts"][0] = 0
    fit, encode = Mock(), Mock()
    monkeypatch.setattr(experiment.glue, "fit_native", fit)
    monkeypatch.setattr(experiment.glue, "encode_frozen", encode)
    with experiment.input_mode("native_counts"), pytest.raises(ValueError, match="positive observed libraries"):
        if operation == "fit":
            experiment.glue.fit_native(data, None, None, Path("unused"))
        else:
            experiment.glue.encode_frozen(None, "RNA", data["RNA"], {}, batch_size=2)
    fit.assert_not_called()
    encode.assert_not_called()


@pytest.mark.parametrize("change", ["representation", "width", "layer", "likelihood"])
def test_actual_native_configuration_must_match_count_mode(change):
    data = prepared()["RNA"]
    config = {"use_rep": None, "rep_dim": None, "use_layer": "counts", "prob_model": "NB"}
    width = 4
    if change == "width":
        width = 2
    else:
        key, value = {"representation": ("use_rep", "X_pca"), "layer": ("use_layer", None),
                      "likelihood": ("prob_model", "Normal")}[change]
        config[key] = value
    model = SimpleNamespace(domains={"RNA": config}, net=SimpleNamespace(x2u={
        "RNA": SimpleNamespace(linear_0=SimpleNamespace(in_features=width))}))
    with pytest.raises(ValueError):
        experiment.native_encoder_record(model, {"RNA": data}, "native_counts")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path / "study"
    monkeypatch.setattr(experiment, "ROOT", root)
    source = str(Path(experiment.__file__).resolve())
    spec = {"configuration": asdict(experiment.glue.ScGLUEConfig(lam_align=.2)),
            "analysis_sources": {source: experiment.glue.file_hash(Path(source))},
            "runtime_flags": dict(experiment.RUNTIME_FLAGS),
            "inherited_loader_environment": {"LD_LIBRARY_PATH": None, "CUDA_VISIBLE_DEVICES": ""}}
    write(root / "protocol.json", spec)
    directory = root / "fits/linear_chain_0/native_counts"
    write(directory / "scglue_model.dill", b"opaque dummy checkpoint, never loaded")
    checkpoint = experiment.glue.file_hash(directory / "scglue_model.dill")
    manifest = {"status": "complete", "configuration": spec["configuration"], "checkpoint_sha256": checkpoint,
                "training_audit": {"encoder_mode": "native_counts"}, "inputs": {"package_provenance": {
                    "input_ablation": {"mode": "native_counts", "source": source, "source_sha256": spec["analysis_sources"][source]}}}}
    for cohort in experiment.fitting.COHORTS:
        for assay in experiment.ASSAYS:
            write(directory / cohort / f"{assay}.npz", b"opaque dummy export, never decoded")
    seal = {"status": "complete", "checkpoint_sha256": checkpoint, "fit_manifest": {},
            "files": {f"{c}/{a}.npz": experiment.glue.file_hash(directory / c / f"{a}.npz")
                      for c in experiment.fitting.COHORTS for a in experiment.ASSAYS}}
    receipt = {"status": "complete", "mode": "native_counts", "world": "linear_chain_0",
               "protocol_sha256": experiment.glue.file_hash(root / "protocol.json"),
               "runtime_environment": {**spec["runtime_flags"], **spec["inherited_loader_environment"]}}
    def refresh():
        write(directory / "model_manifest.json", manifest)
        seal["fit_manifest"]["sha256"] = experiment.glue.file_hash(directory / "model_manifest.json")
        write(directory / "validation_fit.json", seal)
        receipt["validation_fit_sha256"] = experiment.glue.file_hash(directory / "validation_fit.json")
        write(directory / "experiment.json", receipt)
    refresh()
    loader = Mock(return_value=(None, {a: {"cell_ids": np.array(["unit"]), "shared": np.zeros((1, 2))}
                                      for a in experiment.ASSAYS}, {"files": {}}))
    monkeypatch.setattr(experiment.validation.evaluation, "_load_embedding_run", loader)
    return SimpleNamespace(spec=spec, directory=directory, manifest=manifest, seal=seal,
                           receipt=receipt, refresh=refresh, loader=loader)


def test_valid_fit_seals_checked_before_embedding_load(sealed):
    views, evidence = experiment.load_fit("linear_chain_0", "native_counts", "val", sealed.spec)
    assert set(views) == set(experiment.ASSAYS) and evidence["experiment_sha256"]
    sealed.loader.assert_called_once_with(sealed.directory, "val")


@pytest.mark.parametrize("change", ["source", "source_hash", "mode", "runtime", "export", "checkpoint",
                                    "model_status", "seal_status", "seal_checkpoint", "world"])
def test_resealed_semantic_mismatch_rejected_before_loading(sealed, change):
    ablation = sealed.manifest["inputs"]["package_provenance"]["input_ablation"]
    if change == "source":
        ablation["source"] = "/tmp/other_runner.py"
    elif change == "source_hash":
        ablation["source_sha256"] = "different"
    elif change == "mode":
        ablation["mode"] = "compressed"
    elif change == "runtime":
        sealed.receipt["runtime_environment"]["CUDA_LAUNCH_BLOCKING"] = "0"
    elif change == "export":
        write(sealed.directory / "val/RNA.npz", b"changed export")
    elif change == "checkpoint":
        write(sealed.directory / "scglue_model.dill", b"changed checkpoint")
    elif change == "model_status":
        sealed.manifest["status"] = "failed"
    elif change == "seal_status":
        sealed.seal["status"] = "failed"
    elif change == "seal_checkpoint":
        sealed.seal["checkpoint_sha256"] = "unrelated checkpoint"
    else:
        sealed.receipt["world"] = "different_world"
    sealed.refresh()
    with pytest.raises(ValueError):
        experiment.load_fit("linear_chain_0", "native_counts", "val", sealed.spec)
    sealed.loader.assert_not_called()


@pytest.mark.parametrize("fail", [False, True])
def test_fit_boundary_preserves_public_scope_and_never_seals_failure(tmp_path, monkeypatch, native, fail):
    base, root = tmp_path / "benchmark", tmp_path / "study"
    monkeypatch.setattr(experiment, "BASE", base)
    monkeypatch.setattr(experiment, "ROOT", root)
    runtime = tmp_path / "libcudart.so"
    write(runtime, b"dummy runtime, never loaded")
    monkeypatch.setattr(experiment, "RUNTIME", runtime)
    monkeypatch.setattr(experiment, "RUNTIME_SHA256", experiment.glue.file_hash(runtime))
    flags = {**experiment.RUNTIME_FLAGS, "LD_PRELOAD": str(runtime)}
    monkeypatch.setattr(experiment, "RUNTIME_FLAGS", flags)
    for key, value in flags.items():
        monkeypatch.setenv(key, value)
    spec = {"runtime_flags": flags}
    write(root / "protocol.json", spec)
    monkeypatch.setattr(experiment, "checked_protocol", lambda: spec)
    primary = base / "runs/extended/linear_chain_0/informative/primary/scglue_preprocessing"
    entries = {}
    for assay in experiment.ASSAYS:
        path = primary / f"{assay}.json"
        write(path, {key: [f"{assay}_0", f"{assay}_1"] for key in
                     ("normalization_feature_ids", "encoder_feature_ids", "reconstruction_feature_ids")})
        entries[assay] = {"transform_manifest": str(path)}
    write(primary / "manifest.json", {"assays": entries})
    monkeypatch.setattr(experiment.operator.suite, "load_schedule", lambda _: {"test_schedule": True})
    guard = Mock()
    monkeypatch.setattr(experiment.operator, "assert_no_external_jobs", guard)
    def fit(received_base, world, regime, method, mode, overrides, *, seed, device, output):
        assert (received_base, world, regime, method, mode, overrides, seed, device) == (
            base, "linear_chain_0", "informative", "scglue", "native_counts", {"lam_align": .2}, 0, "cuda")
        output.mkdir(parents=True)
        if fail:
            write(output / "failure.json", {"status": "failed"})
            raise RuntimeError("native failure")
        write(output / "validation_fit.json", {"status": "complete"})
        assert experiment.glue.package_provenance()["input_ablation"]["mode"] == mode
        return {"configuration": {"lam_align": .2}}
    monkeypatch.setattr(experiment.fitting, "fit_candidate", fit)
    original_open = Path.open
    def public_only(path, *args, **kwargs):
        assert not {"truth", "private_inputs", "probe", "assessment"} & set(path.parts)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", public_only)
    output = root / "fits/linear_chain_0/native_counts"
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            experiment.fit_one("linear_chain_0", "native_counts")
        assert (output / "failure.json").exists() and not (output / "experiment.json").exists()
    else:
        assert experiment.fit_one("linear_chain_0", "native_counts")["runtime_environment"]["LD_PRELOAD"] == str(runtime)
        with pytest.raises(FileExistsError):
            experiment.fit_one("linear_chain_0", "native_counts")
    guard.assert_called_once_with({"test_schedule": True})


def test_state_certificates_handle_both_nonanchor_permutations(monkeypatch, tmp_path):
    ev = experiment.validation.evaluation
    states = np.array([[-2., 1.], [0., -2.], [3., 2.], [1., 4.]])
    permutations = dict(zip(experiment.ASSAYS, ([0, 1, 2, 3], [2, 0, 3, 1], [1, 3, 0, 2])))
    ordered = {a: states[p] for a, p in permutations.items()}
    def views(cohort, factor=1):
        rows = np.array([3, 0, 2, 1])
        return {a: {"cell_ids": np.array([f"{a}_{cohort}_{i}" for i in rows]), "shared": factor * ordered[a][rows]}
                for a in experiment.ASSAYS}
    calls = []
    def truth(_, __, cohort, *, states=False):
        assert cohort in {"train", "val"}
        calls.append("states" if states else "identities")
        result = {"cell_ids": np.array([f"{a}_{cohort}_{i}" for a in experiment.ASSAYS for i in range(4)]),
                  "assigned_assay": np.repeat(experiment.ASSAYS, 4)}
        if states:
            result["state"] = np.concatenate(list(ordered.values()))
        return result
    def match(*args, **kwargs):
        assert "states" not in calls
        calls.append("match")
        return original_match(*args, **kwargs)
    original_match = ev.match_one_to_one
    monkeypatch.setattr(ev, "match_one_to_one", match)
    monkeypatch.setattr(experiment.validation, "protocol", lambda: {})
    monkeypatch.setattr(experiment.validation, "development_world", lambda *_: (tmp_path, {}))
    monkeypatch.setattr(experiment.validation, "truth_fields", truth)
    monkeypatch.setattr(experiment.validation, "public_run", lambda *args, **_: (views(args[6]), {}))
    monkeypatch.setattr(experiment.validation, "reference_geometry", lambda _, __, pools: (ordered, {}))
    monkeypatch.setattr(experiment.glue, "file_hash", lambda _: "fixture hash; no file access")
    monkeypatch.setattr(ev, "discover", Mock(side_effect=AssertionError("causal discovery forbidden")))
    runs = {mode: views("val", factor) for mode, factor in zip(experiment.MODES, (1, -2))}
    train = {mode: views("train", factor) for mode, factor in zip(experiment.MODES, (1, -2))}
    report = experiment.evaluate_views("linear_chain_0", runs, train, {})
    for row in report["results"].values():
        assert row["worst_pair_gap"] == 0 and len(row["pairwise_state_transport"]) == 3
        assert all(v["matched_squared_cost"] == 0 for v in row["pairwise_state_transport"].values())
        assert len(row["source_state_transfer"]) == 9
    assignment = report["assignments"]["compressed"]["cell_ids"]
    for assay in experiment.ASSAYS[1:]:
        assert assignment[assay] != report["pools"]["val"][assay]
    assert calls.index("states") > max(i for i, value in enumerate(calls) if value == "match")
    ev.discover.assert_not_called()
