import pytest

from project_resources.research.provenance import rel, remap_legacy


@pytest.mark.parametrize(("old", "current"), [
    ("Data/external/re2g/links.tsv", "pipelines/empirical/data/raw/external/re2g/links.tsv"),
    ("Data/raw/GexData/counts.h5", "pipelines/empirical/data/raw/GexData/counts.h5"),
    ("CausalDiscovery/outputs/matrix.tsv", "pipelines/empirical/analysis/outputs/matrix.tsv"),
    ("CausalDiscovery/evaluation/edges.tsv", "pipelines/empirical/evaluation/outputs/edges.tsv"),
    ("Evaluation/scripts/evaluate.py", "pipelines/empirical/evaluation/scripts/evaluate.py"),
    ("Integration/outputs/scglue_joint_v2/train/model.dill", "pipelines/empirical/integration/runs/scglue_joint_v2/train/model.dill"),
    ("Integration/manifests/inputs.tsv", "pipelines/empirical/integration/manifests/inputs.tsv"),
    ("pipelines/empirical/data/external/links.tsv", "pipelines/empirical/data/raw/external/links.tsv"),
    ("pipelines/empirical/analysis/evaluation/edges.tsv", "pipelines/empirical/evaluation/outputs/edges.tsv"),
    ("integration/manifests/inputs.tsv", "pipelines/empirical/integration/manifests/inputs.tsv"),
    ("unknown/matrix.tsv", "unknown/matrix.tsv"),
])
def test_legacy_paths_are_read_in_current_layout(old, current):
    assert remap_legacy(old) == current
    assert remap_legacy(current) == current


def test_old_absolute_paths_work_after_relocation():
    old = "/another-host/Epigenomics-GEX-causal-analysis/CausalDiscovery/outputs/matrix.tsv"
    assert remap_legacy(rel(old)) == "pipelines/empirical/analysis/outputs/matrix.tsv"
