"""Small diagnostic-contract tests; these are not simulation benchmark worlds."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("label_investigation", SCRIPTS / "investigate_label_transfer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiagnosticContractTests(unittest.TestCase):
    def test_vote_tie_uses_neighbor_order(self):
        self.assertEqual(module.vote(["B", "T"]), ("B", .5))
        self.assertEqual(module.vote(["T", "B"]), ("T", .5))

    def test_low_confidence_remains_unknown(self):
        self.assertEqual(module.vote(["B", "T", "M"]), ("unknown", 1 / 3))

    def test_majority_confidence(self):
        self.assertEqual(module.vote(["M", "M", "B", "T"]), ("M", .5))

    def test_nonfinite_representation_reported(self):
        result = module.neighbor_conservation(np.array([[np.nan]]), np.array([1.]))
        self.assertFalse(result["available"])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="biology-contract-") as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserve me")
            result = subprocess.run([sys.executable, str(SCRIPTS / "investigate_label_transfer.py"),
                                     "--root", str(Path(directory) / "missing_inputs"),
                                     "--out", str(output)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(output.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
