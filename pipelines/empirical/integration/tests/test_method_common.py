"""Integration helper imports survive direct scripts and repository relocation."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[4]
METHODS = Path("pipelines/empirical/integration/methods")
IMPORT_CHECK = """
from pathlib import Path
import sys

root = Path(sys.argv[1])
method = sys.argv[2]
sys.path.insert(0, str(root / "pipelines/empirical/integration/methods" / method))
import _common
import integration_common

assert _common.infer_repo_root() == root
for name in (
    "apply_harmonization", "build_embedding_df", "infer_repo_root",
    "load_harmonization_table", "load_manifest_row", "load_rna_cell_annotations",
    "majority_vote", "normalize_label_level", "pick_rna_h5", "read_named_column",
    "resolve_path", "strip_10x_suffix", "write_tsv",
):
    assert getattr(_common, name) is getattr(integration_common, name), name
if method == "scglue":
    for name in (
        "load_graph", "subset_adata_to_graph", "drop_nonfinite_obs_by_rep",
        "patch_scglue_scheduler_verbose_compat",
    ):
        assert callable(getattr(_common, name)), name
"""


class MethodCommonCompatibilityTests(unittest.TestCase):
    def test_imports_and_root_from_outside_original_and_relocated_repo(self):
        with tempfile.TemporaryDirectory(prefix="integration-imports-") as directory:
            outside = Path(directory)
            relocated = outside / "relocated repository"
            for relative in ("integration_common.py", "scglue/_common.py", "scmrdr/_common.py"):
                destination = relocated / METHODS / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / METHODS / relative, destination)
            for root in (REPO_ROOT, relocated):
                for method in ("scglue", "scmrdr"):
                    with self.subTest(root=root, method=method):
                        result = subprocess.run(
                            [sys.executable, "-I", "-B", "-c", IMPORT_CHECK, str(root), method],
                            cwd=outside, capture_output=True, text=True, timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
