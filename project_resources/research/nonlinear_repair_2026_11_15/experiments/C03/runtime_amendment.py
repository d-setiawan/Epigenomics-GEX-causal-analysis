"""Admit the reviewed scoring-import runtime without changing frozen C03 code."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admitted_protocol(expected, actual, allowed_pools):
    if actual.get("threadpools") != allowed_pools:
        raise ValueError("Runtime libraries differ from the exact reviewed admission")
    normalized = dict(actual, threadpools=expected["threadpools"])
    if normalized != expected:
        raise ValueError("A protocol field other than the admitted runtime changed")
    return normalized


def main():
    approval = json.loads((HERE / "runtime_amendment_approval.json").read_text())
    required = {"runtime_amendment.py", "runtime_admission.json", "test_runtime_amendment.py"}
    if approval.get("approved") is not True or set(approval.get("source_sha256", {})) != required:
        raise ValueError("Exact scientific runtime-amendment approval required")
    for name, digest in approval["source_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Runtime-amendment source or admission changed: " + name)
    admission = json.loads((HERE / "runtime_admission.json").read_text())
    for name, key in (("protocol.json", "protocol_sha256"), ("fit_seal.json", "fit_seal_sha256")):
        if sha(HERE / name) != admission[key] or approval.get(key) != admission[key]:
            raise ValueError("Original protocol/fit binding changed")
    for path, digest in admission["library_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Admitted runtime library bytes changed: " + path)
    expected = json.loads((HERE / "protocol.json").read_text())
    for name, digest in expected["source_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Original source changed before import: " + name)
    import evaluate
    import run

    actual = run.protocol()
    admitted_protocol(expected, actual, admission["allowed_threadpools"])
    if any((HERE / name).exists() for name in (
        "evaluation", "evaluation_seal.json", "runtime_evaluation_receipt.json",
        "runtime_evaluation_seal.json",
    )):
        raise FileExistsError("An evaluation or admission receipt already exists")
    receipt = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": admission["protocol_sha256"],
        "fit_seal_sha256": admission["fit_seal_sha256"],
        "amendment_approval_sha256": sha(HERE / "runtime_amendment_approval.json"),
        "amendment_source_sha256": approval["source_sha256"],
        "actual_protocol_after_scoring_imports": actual,
        "original_controller_threadpools": expected["threadpools"],
        "library_sha256": admission["library_sha256"],
        "normalization": "Only the exact admitted threadpool list is normalized to the controller list for the original evaluator guard; all other fields compare exactly.",
        "numerical_evaluator_source_changed": False,
    }
    run.write(HERE / "runtime_evaluation_receipt.json", receipt)

    # SciPy scoring imports load libraries absent from the controller snapshot.
    def checked_protocol():
        return admitted_protocol(expected, run.protocol(), admission["allowed_threadpools"])

    evaluate.protocol = checked_protocol
    evaluate.main()
    run.write(HERE / "runtime_evaluation_seal.json", {
        "evaluation_seal_sha256": sha(HERE / "evaluation_seal.json"),
        "runtime_evaluation_receipt_sha256": sha(HERE / "runtime_evaluation_receipt.json"),
        "amendment_approval_sha256": sha(HERE / "runtime_amendment_approval.json"),
        "fit_seal_sha256": sha(HERE / "fit_seal.json"),
    })


if __name__ == "__main__":
    main()
