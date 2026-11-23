from pathlib import Path

import pytest

from pipelines.empirical.integration.scripts.setup_integration_workspace import symlink_force


def test_workspace_links_survive_relocation(tmp_path):
    original = tmp_path / "original"
    source = original / "data" / "counts.tsv"
    source.parent.mkdir(parents=True)
    source.write_text("cell\tcount\na\t3\n")
    link = original / "integration" / "workspace" / "counts.tsv"
    symlink_force(source, link)
    assert not link.readlink().is_absolute()

    moved = tmp_path / "different-project-name"
    original.rename(moved)
    assert (moved / link.relative_to(original)).read_text() == "cell\tcount\na\t3\n"


@pytest.mark.parametrize("directory", [False, True])
def test_workspace_does_not_delete_real_data(tmp_path, directory):
    source = tmp_path / "source"
    source.write_text("new")
    target = tmp_path / "target"
    if directory:
        target.mkdir()
        saved = target / "annotation.gtf"
    else:
        saved = target
    saved.write_text("irreplaceable")

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        symlink_force(source, target)
    assert saved.read_text() == "irreplaceable"


def test_workspace_replaces_stale_link(tmp_path):
    source = tmp_path / "source"
    source.write_text("counts")
    target = tmp_path / "target"
    target.symlink_to(Path("missing-old-input"))
    symlink_force(source, target)
    assert target.read_text() == "counts"
    assert not target.readlink().is_absolute()
