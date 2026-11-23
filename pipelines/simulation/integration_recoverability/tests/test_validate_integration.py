"""Study boundaries and scoring contracts; no model training or new worlds."""

import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_integration as validation


def spec():
    return {"study_id": "fixture", "protocol_sha256": "hash", "source_pins": {},
        "selection_worlds": ["linear_chain_0"], "selection_regimes": ["informative"],
        "selection_seeds": [0], "confirmation_worlds": ["constrained_chain_0"],
        "confirmation_regimes": ["realistic", "informative"], "confirmation_seeds": [0, 1],
        "methods": {m: {"baseline": {}, "alternative": {}} for m in ("scglue", "scmrdr")},
        "output_root": "runs/integration_validation/fixture", "pool_seed": 2718, "knn": 2}


def test_private_cohort_rejected_before_any_archive_access():
    with patch.object(validation.np, "load", side_effect=AssertionError("private read")):
        for cohort in ("probe", "assessment", "../assessment"):
            with pytest.raises(ValueError, match="prohibited"):
                validation.truth_fields(Path("unused"), {}, cohort, states=True)


def test_truth_reader_only_requests_identity_and_state_arrays():
    requested = []
    data = {"cell_ids": np.array(["r", "a", "m"]),
            "assigned_assay": np.array(validation.ASSAYS), "state": np.eye(3)}
    class Archive:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def __getitem__(self, key):
            requested.append(key)
            return data[key]
    with patch.object(validation.np, "load", return_value=Archive()), \
         patch.object(validation.suite_plan, "digest", return_value="hash"):
        result = validation.truth_fields(Path("unused"), {"files": {"truth/val.npz": "hash"}}, "val", states=True)
    assert set(requested) == set(data)
    assert set(result) == set(data)


def test_audit_rejects_undeclared_role_before_data_access():
    with patch.object(validation, "development_world", side_effect=AssertionError("data read")):
        for world, regime, seed, cohort, mode in (
            ("linear_chain_1", "informative", 0, "val", "baseline"),
            ("linear_chain_0", "realistic", 0, "val", "baseline"),
            ("linear_chain_0", "informative", 1, "val", "baseline"),
            ("linear_chain_0", "informative", 0, "assessment", "baseline"),
            ("constrained_chain_0", "informative", 0, "discovery", "all")):
            with pytest.raises(ValueError):
                validation.audit(spec(), world, regime, seed, cohort, mode)


def test_candidate_paths_reject_traversal_unknowns_and_invalid_seeds():
    for world, method, candidate, seed in (
        ("../escape", "scglue", "baseline", 0),
        ("linear_chain_0", "other", "baseline", 0),
        ("linear_chain_0", "scglue", "../escape", 0),
        ("linear_chain_0", "scglue", "baseline", True),
        ("linear_chain_0", "scglue", "baseline", -1)):
        with pytest.raises(ValueError):
            validation.candidate_directory(spec(), world, "informative", method, candidate, seed)


def test_output_is_idempotent_but_never_overwrites_changed_data(tmp_path):
    path = tmp_path / "report.json"
    validation.write_new(path, {"a": 1})
    validation.write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        validation.write_new(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_protocol_rejects_changed_sources_and_unsafe_outputs(tmp_path):
    config = spec()
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(config))
    with patch.object(validation.suite_plan, "source_pins", autospec=True, return_value={"changed": "source"}):
        with pytest.raises(ValueError, match="sources changed"):
            validation.protocol(path)
    config["output_root"] = "/tmp/outside_validation"
    path.write_text(json.dumps(config))
    with patch.object(validation.suite_plan, "source_pins", autospec=True, return_value={}):
        with pytest.raises(ValueError, match="dedicated"):
            validation.protocol(path)


def test_production_protocol_preserves_frozen_source_pins():
    actual = validation.protocol()
    assert len(actual["source_pins"]) == 17
    assert set(actual["methods"]) == {"scglue", "scmrdr"}


def test_changed_export_seal_is_rejected_before_loading_embeddings(tmp_path):
    config = spec()
    config["defaults"] = {"scmrdr": {"seed": 0}}
    with patch.object(validation, "BASE", tmp_path):
        directory = validation.candidate_directory(config, "linear_chain_0", "informative", "scmrdr", "baseline", 0)
        checkpoint = directory / "model/checkpoint.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture checkpoint")
        checkpoint_hash = validation.suite_plan.digest(checkpoint)
        model = directory / "manifest.json"
        validation.write_new(model, {"status": "complete", "config": {"seed": 0}, "checkpoint_sha256": checkpoint_hash})
        seal_path = directory / "validation_fit.json"
        seal = {"status": "complete", "checkpoint_sha256": checkpoint_hash,
                "fit_manifest": {"sha256": validation.suite_plan.digest(model)}}
        validation.write_new(seal_path, seal)
        validation.write_new(directory / "study.json", {"protocol_sha256": "hash",
            "validation_fit_sha256": validation.suite_plan.digest(seal_path)})
        seal_path.write_text(json.dumps({**seal, "changed": True}))
        with patch.object(validation.evaluation, "_load_embedding_run", side_effect=AssertionError("embedding load")):
            with pytest.raises(ValueError, match="seal changed"):
                validation.public_run(config, "linear_chain_0", "informative", "scmrdr", "baseline", 0, "val")


def test_fixed_source_transfer_detects_modality_specific_sign_flip():
    states = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    targets = {a: states.copy() for a in validation.ASSAYS}
    embeddings = copy.deepcopy(targets)
    embeddings["H3K27ac"] *= -1
    result = validation.state_probes(embeddings, targets, embeddings, targets)
    assert result["RNA->RNA"]["mean_r2"] > .9
    assert result["H3K27ac->H3K27ac"]["mean_r2"] > .9
    assert result["RNA->H3K27ac"]["mean_r2"] < 0
    assert all(not row["target_refit"] for row in result.values())


def test_selected_settings_must_match_protocol_and_do_not_add_controls(tmp_path):
    with patch.object(validation, "BASE", tmp_path):
        path = tmp_path / spec()["output_root"] / "selected_settings.json"
        validation.write_new(path, {"protocol_sha256": "hash", "selected": {"scglue": "alternative"}})
        assert validation.selected_candidates(spec(), "selected", ["scglue"]) == {"scglue": ["alternative"]}
        changed = {**spec(), "protocol_sha256": "different"}
        with pytest.raises(ValueError, match="another protocol"):
            validation.selected_candidates(changed, "selected", ["scglue"])


def test_interim_candidate_selection_rejects_unknown_methods_and_candidates():
    assert validation.selected_candidates(spec(), "alternative", ["scmrdr"]) == {"scmrdr": ["alternative"]}
    for mode, methods in (("unknown", ["scmrdr"]), ("all", []),
                          ("all", ["other"]), ("baseline", ["scmrdr", "scmrdr"])):
        with pytest.raises(ValueError):
            validation.selected_candidates(spec(), mode, methods)


@pytest.mark.parametrize("mode,methods,result_count,pool_size", [
    ("baseline", ("scglue", "scmrdr"), 2, 4),
    ("alternative", ("scmrdr",), 1, 3),
    ("all", ("scglue", "scmrdr"), 4, 3)])
def test_audit_uses_reference_eligibility_without_causal_discovery(tmp_path, mode, methods, result_count, pool_size):
    states = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    def truth(_, __, cohort, *, states=False):
        ids = np.array([f"{a}_{cohort}_{i}" for a in validation.ASSAYS for i in range(4)])
        result = {"cell_ids": ids, "assigned_assay": np.repeat(validation.ASSAYS, 4)}
        if states:
            result["state"] = np.tile(np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]]), (3, 1))
        return result
    def public(*args, **kwargs):
        cohort = args[6]
        n = 3 if kwargs.get("reference") and mode != "baseline" else 4
        return {a: {"cell_ids": np.array([f"{a}_{cohort}_{i}" for i in range(n)]),
                    "shared": states[:n].copy()} for a in validation.ASSAYS}, {}
    def reference(_, __, pools):
        return {a: states[:len(pools[a])].copy() for a in pools}, {}
    with patch.object(validation, "BASE", tmp_path), \
         patch.object(validation, "development_world", return_value=(tmp_path, {})), \
         patch.object(validation, "truth_fields", side_effect=truth), \
         patch.object(validation, "public_run", side_effect=public), \
         patch.object(validation, "reference_geometry", side_effect=reference), \
         patch.object(validation.suite_plan, "digest", return_value="hash"), \
         patch.object(validation.evaluation, "discover", side_effect=AssertionError("causal test")), \
         patch.object(validation.evaluation, "evaluate_world", side_effect=AssertionError("causal evaluator")):
        path, report = validation.audit(spec(), "linear_chain_0", "informative", 0, "val", mode, methods)
    assert path.exists() and report["reference_only"] is (mode == "baseline")
    assert len(report["results"]) == result_count
    assert all(row["selected"] == pool_size for row in report["pool_audit"]["val"]["by_assay"].values())
    assert set(report["pool_reference_evidence"]["val"]) == {"scglue", "scmrdr"}
    for row in report["results"].values():
        assert row["worst_pair_gap"] == 0
        assert len(row["pairwise_state_transport"]) == 3
        assert len(row["source_state_transfer"]) == 9
        assert len(row["pre_assignment_hubness"]) == 6
