from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import assay_local_state_embedding_v4 as local_state  # noqa: E402
import cell_state_loading_groups as loading_groups  # noqa: E402
import shared_state_subspace_v4 as subspace  # noqa: E402


def _backgrounds(
    *, rank: int, n_cells: int = 180, n_features: int = 48
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    cell_ids: dict[str, np.ndarray] = {}
    for index, modality in enumerate(sorted(local_state.EXPECTED_MODALITIES)):
        rng = np.random.default_rng(10_000 + index + 100 * rank)
        state = rng.normal(size=(n_cells, rank))
        loadings = rng.normal(scale=1.2, size=(rank, n_features))
        logits = 1.5 * state @ loadings + rng.normal(
            scale=0.02, size=(n_cells, n_features)
        )
        if rank >= 3:
            logits -= logits.max(axis=1, keepdims=True)
            matrix = np.exp(logits)
            matrix = 1_000.0 * matrix / matrix.sum(axis=1, keepdims=True)
        else:
            matrix = np.exp(logits)
        matrices[modality] = matrix
        depths[modality] = matrix.sum(axis=1)
        cell_ids[modality] = np.asarray(
            [f"{modality}_{cell}" for cell in range(n_cells)]
        )
    return matrices, depths, cell_ids


def _config() -> local_state.AssayLocalStateEmbeddingConfig:
    return local_state.AssayLocalStateEmbeddingConfig(parallel_repetitions=39)


def test_rank_three_embedding_is_common_width_and_assay_local() -> None:
    matrices, depths, cell_ids = _backgrounds(rank=3)
    result = local_state.build_assay_local_state_embedding(
        matrices, depths, cell_ids, config=_config()
    )
    expected_columns = [
        "modality",
        "cell",
        "integration__dim0",
        "integration__dim1",
        "integration__dim2",
        "integration_uncertainty",
    ]
    assert list(result.embedding.columns) == expected_columns
    assert len(result.embedding) == 7 * 180
    assert np.isfinite(
        result.embedding.filter(regex=r"integration").to_numpy(dtype=float)
    ).all()
    provenance = result.provenance
    assert provenance["formal_state_dimension"] == 3
    assert provenance["common_coordinate_width"] == 3
    assert provenance["zero_padding_used"] is False
    assert provenance["cross_assay_alignment_used"] is False
    assert provenance["parallel_rank_affects_dimension_selection"] is False
    boundary = provenance["information_boundary"]
    assert boundary["truth_or_scorer_used"] is False
    assert boundary["cell_rows_compared_across_assays"] is False
    assert boundary["feature_columns_compared_across_assays"] is False
    assert boundary["same_assay_cells_used_for_fold_procrustes"] is True
    assert all(
        diagnostic["minimum_fold_supported_rank"] >= 3
        for diagnostic in provenance["assay_diagnostics"].values()
    )


def test_parallel_rank_below_three_abstains_without_padding() -> None:
    matrices, depths, cell_ids = _backgrounds(
        rank=2, n_cells=100, n_features=24
    )
    with pytest.raises(
        local_state.AssayLocalStateAbstention,
        match="parallel_rank_below_predeclared_three",
    ):
        local_state.build_assay_local_state_embedding(
            matrices, depths, cell_ids, config=_config()
        )


def test_writer_consumes_only_background_archives_and_hash_locks(
    tmp_path: Path,
) -> None:
    matrices, depths, cell_ids = _backgrounds(rank=3)
    discovery = tmp_path / "world" / "discovery"
    discovery.mkdir(parents=True)
    for modality in sorted(matrices):
        np.savez_compressed(
            discovery / f"background_matrix__{modality}.npz",
            matrix=matrices[modality],
            background_depth=depths[modality],
            cell_ids=cell_ids[modality],
        )
    # A malformed focal file proves the formal writer never opens focal values.
    (discovery / "observed_focal__rna.tsv").write_text("poison\n")
    output = tmp_path / "formal_embedding"
    manifest = local_state.write_assay_local_state_embedding(
        discovery, output
    )
    assert manifest["schema_version"] == (
        local_state.FROZEN_EVIDENCE_SCHEMA_VERSION
    )
    assert manifest["formal_state_dimension"] == 3
    assert manifest["zero_padding_used"] is False
    assert manifest["all_assays_parallel_rank_support_formal_dimension"] is True
    assert set(manifest["input_file_sha256"]) == {
        f"background_matrix__{modality}.npz"
        for modality in local_state.EXPECTED_MODALITIES
    }
    embedding = pd.read_csv(output / local_state.EMBEDDING_NAME, sep="\t")
    assert list(embedding.filter(regex=r"^integration__dim").columns) == [
        "integration__dim0",
        "integration__dim1",
        "integration__dim2",
    ]
    manifest_path = output / local_state.MANIFEST_NAME
    sidecar = manifest_path.with_suffix(".sha256")
    assert sidecar.read_text().split()[0] == local_state.file_sha256(manifest_path)
    provenance = json.loads((output / local_state.PROVENANCE_NAME).read_text())
    assert provenance["information_boundary"]["truth_or_scorer_used"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        local_state.write_assay_local_state_embedding(
            discovery, output
        )

    with pytest.raises(ValueError, match="frozen embedding config"):
        local_state.write_assay_local_state_embedding(
            discovery,
            tmp_path / "nonformal_embedding",
            config=_config(),
        )


def test_shared_subspace_accepts_and_rechecks_formal_local_freeze(
    tmp_path: Path,
) -> None:
    matrices, depths, cell_ids = _backgrounds(rank=3)
    discovery = tmp_path / "world" / "discovery"
    discovery.mkdir(parents=True)
    for modality in sorted(matrices):
        background_path = discovery / f"background_matrix__{modality}.npz"
        np.savez_compressed(
            background_path,
            matrix=matrices[modality],
            background_depth=depths[modality],
            cell_ids=cell_ids[modality],
        )
        n_cells = len(cell_ids[modality])
        focal = pd.DataFrame(
            {
                "cell": cell_ids[modality],
                "modality": modality,
                "observed_background_library_size": depths[modality],
                "observed_library_size": depths[modality] + 1.0,
                (
                    "expr__gene"
                    if modality == "rna"
                    else f"chr1:1-2__{modality.removeprefix('chrom_')}"
                ): (
                    np.linspace(0.0, 1.0, n_cells)
                    if modality == "rna"
                    else np.arange(n_cells) % 2
                ),
            }
        )
        focal.to_csv(
            discovery / f"observed_focal__{modality}.tsv", sep="\t", index=False
        )
    output = tmp_path / "formal_embedding"
    local_state.write_assay_local_state_embedding(discovery, output)
    inputs = loading_groups.load_loading_inputs(
        discovery, output / local_state.EMBEDDING_NAME
    )
    evidence = subspace._validate_safe_embedding_freeze(inputs)
    assert evidence["layout"] == "assay_local_state_embedding_v4"
    assert evidence["formal_state_dimension"] == 3
    assert evidence["upstream_algorithm_source_sha256"] == {
        "assay_local_state_embedding_v4.py": local_state.file_sha256(
            Path(local_state.__file__)
        )
    }

    # The manifest is not a license to swap the observed background bytes.
    target = discovery / "background_matrix__rna.npz"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="background hashes disagree"):
        subspace._validate_safe_embedding_freeze(inputs)
