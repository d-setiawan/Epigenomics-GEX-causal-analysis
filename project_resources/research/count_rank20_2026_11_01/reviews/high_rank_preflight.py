"""Deterministic high-rank algebra/API fixtures; no project observations."""
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

STUDY = Path(__file__).resolve().parents[1]
SOURCE = STUDY/"extension_estimators.py"
spec = importlib.util.spec_from_file_location("rank20_estimators",SOURCE)
estimator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(estimator)


def main():
    started = time.monotonic()
    k,p = 20,300
    rotation = np.eye(k)
    for j in range(k-1):
        angle = (j+1)*np.pi/80
        turn = np.eye(k)
        turn[j,j] = turn[j+1,j+1] = np.cos(angle)
        turn[j,j+1],turn[j+1,j] = -np.sin(angle),np.sin(angle)
        rotation = rotation @ turn
    skew = np.linspace(.4,2.3,k)
    tensor = np.einsum("ir,jr,kr,r->ijk",rotation,rotation,rotation,skew,optimize=True)
    learned,info = estimator._orient_tensor(tensor,seed=0)
    assert learned is not None
    row,col = linear_sum_assignment(-np.abs(rotation.T @ learned))
    maximum_alignment_error = float(np.max(np.abs(1-np.abs(rotation.T @ learned)[row,col])))
    assert maximum_alignment_error < 1e-10
    assert info["orientation_fit_residual"] < 1e-10
    np.testing.assert_allclose(learned.T @ learned,np.eye(k),atol=1e-12)

    # A fixed indicator table checks dimensions/numerics only. It is not a
    # generated biological world or a claim of independent-source recovery.
    counts = np.tile(2100*np.vstack([np.zeros((1,k)),np.eye(k)]),(1,p//k))
    outputs = []
    for orientation in ["c3","c3_shrink","pgf"]:
        fitted = estimator.fit_counts(counts,1.,k,orientation=orientation)
        assert fitted["status"] == "ok",fitted["reason"]
        assert fitted["mixing"].shape == (p,k)
        assert fitted["unmixing"].shape == (k,p)
        assert fitted["white_third_tensor"].shape == (k,k,k)
        np.testing.assert_allclose(fitted["unmixing"] @ fitted["mixing"],np.eye(k),atol=1e-10)
        np.testing.assert_allclose(fitted["source_variance"],1,atol=1e-10)
        assert np.isfinite(fitted["orientation_residual"])
        if orientation == "pgf":
            assert fitted["generalized_covariance_matrices"].shape == (8,k,k)
        outputs.append(dict(orientation=orientation,status=fitted["status"],orientation_residual=fitted["orientation_residual"]))
    invalid = estimator.fit_counts(counts,1.,k,theta=.5,retention=.5)
    assert invalid["status"] == "abstained"
    assert "not positive" in invalid["reason"]
    assert invalid["mixing"].shape == (p,0)
    hashes = {name:hashlib.sha256((STUDY/name).read_bytes()).hexdigest() for name in
              ["protocol.json","apply_extensions.py","extension_estimators.py","evaluate_extensions.py"]}
    original = STUDY.parent/"count_extensions_2026_11_01"
    assert all(hashes[name] == hashlib.sha256((original/name).read_bytes()).hexdigest() for name in
               ["apply_extensions.py","extension_estimators.py","evaluate_extensions.py"])
    protocol = json.loads((STUDY/"protocol.json").read_text())
    assert protocol["rank"]["cap"] == 20 and protocol["expected_arm_records"] == 54
    receipt = dict(status="passed",date="2026-11-01",k=k,p=p,fixture_rows=len(counts),
                   maximum_exact_tensor_alignment_error=maximum_alignment_error,
                   exact_tensor_residual=info["orientation_fit_residual"],fit_shapes_and_numerics=outputs,
                   invalid_requested_mode=dict(status=invalid["status"],reason=invalid["reason"]),
                   source_sha256=hashes,fixture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   elapsed_seconds=time.monotonic()-started,
                   scope="Exact rotated tensor and deterministic indicator-count API fixture only; no project training/test arrays, labels, state references, or scientific performance evaluated.",
                   interpretation="Supports the computational rank extension; does not establish source recovery on the project data. The prior cap-only exclusions cannot predict this higher-rank refit.")
    (STUDY/"reviews/high_rank_preflight.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
