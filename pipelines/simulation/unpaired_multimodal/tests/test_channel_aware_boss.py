from __future__ import annotations

from pathlib import Path
import sys
import types

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CAUSAL_SCRIPTS = Path(__file__).resolve().parents[4] / "pipelines/empirical/analysis" / "scripts"
for path in (SCRIPTS, CAUSAL_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_channel_aware_boss import candidate_grid
from channel_aware_boss import canonical_edge, merge_layers
from consensus_multimodal_causal_runs import consensus_graph, wilson_interval
from score_multimodal_causal_graph import score
from run_gfci_tetrad import run_gfci


def semantics_fixture() -> pd.DataFrame:
    rows = [
        ("expr__SIMG_log1p", "target_gene"),
        ("chr1:1000-1499__H3K27ac", "slow_state_dominated_direct_gene_cause"),
        ("chr1:2000-2499__H3K27ac", "shared_driver_peer_no_direct_gene_edge"),
        ("chr1:3000-3499__H3K27ac", "same_mark_child_of_peak0"),
        ("chr1:4000-4499__H3K27ac", "state_associated_not_direct"),
        ("chr1:5000-5499__H3K27ac", "null"),
        ("chr1:6000-6499__H3K27ac", "null"),
        ("chr1:7000-7499__H3K4me3", "fast_residual_dominated_direct_gene_cause"),
        ("chr1:8000-8499__H3K4me3", "same_mark_child_of_peak0"),
        ("chr1:9000-9499__H3K4me3", "shared_driver_member_a"),
        ("chr1:10000-10499__H3K4me3", "shared_driver_member_b"),
        ("chr1:11000-11499__H3K4me3", "null"),
        ("chr1:12000-12499__H3K4me3", "null"),
    ]
    return pd.DataFrame(rows, columns=["node", "role"])


def edge(left: str, right: str, endpoint_left: str = "CIRCLE", endpoint_right: str = "CIRCLE") -> dict[str, str]:
    return {
        "node1": left,
        "endpoint1": endpoint_left,
        "node2": right,
        "endpoint2": endpoint_right,
    }


def test_frozen_v3_configuration() -> None:
    candidate = next(item for item in candidate_grid(17) if item["name"] == "channel_final_v3")
    config = candidate["config"]
    assert config.state_penalty == 1.0
    assert config.state_starts == 20
    assert config.state_use_bes is True
    assert config.state_boss_alpha == 0.05
    assert config.state_marginal_min_abs_r == 0.20
    assert config.within_sample_prior == 5.0
    assert config.sparse_alpha == 0.25
    assert config.within_profile_clusters == 20
    assert config.within_profile_min_abs_r == 0.60
    assert config.within_profile_alpha == 0.05


def test_attainable_contract_perfect_motif_and_null_allowance() -> None:
    semantics = semantics_fixture()
    role = dict(zip(semantics["role"], semantics["node"]))
    ac_cause = role["slow_state_dominated_direct_gene_cause"]
    gene = role["target_gene"]
    ac_child = semantics.loc[
        (semantics["role"] == "same_mark_child_of_peak0")
        & semantics["node"].str.endswith("__H3K27ac"),
        "node",
    ].item()
    me_child = semantics.loc[
        (semantics["role"] == "same_mark_child_of_peak0")
        & semantics["node"].str.endswith("__H3K4me3"),
        "node",
    ].item()
    estimated = pd.DataFrame(
        [
            edge(ac_cause, gene, "CIRCLE", "ARROW"),
            edge(ac_cause, ac_child),
            edge(role["fast_residual_dominated_direct_gene_cause"], me_child),
            edge(ac_cause, role["shared_driver_peer_no_direct_gene_edge"]),
            edge(role["shared_driver_member_a"], role["shared_driver_member_b"]),
        ]
    )
    metrics, pairs = score(estimated, semantics)
    assert metrics["world_success"] == 1
    assert metrics["sentinel_f1"] == 1.0
    assert pairs.loc[pairs["truth_class"] == "null", "present"].sum() == 0

    null_nodes = semantics.loc[semantics["role"] == "null", "node"].tolist()
    one_null = pd.concat([estimated, pd.DataFrame([edge(null_nodes[0], gene)])], ignore_index=True)
    assert score(one_null, semantics)[0]["world_success"] == 1
    two_nulls = pd.concat(
        [one_null, pd.DataFrame([edge(null_nodes[1], gene)])], ignore_index=True
    )
    assert score(two_nulls, semantics)[0]["world_success"] == 0


def test_consensus_requires_two_of_three_and_abstains_without_endpoint_majority(
    tmp_path: Path,
) -> None:
    frames = [
        pd.DataFrame(
            [
                {**edge("A", "B", "TAIL", "ARROW"), "evidence_layers": "boss"},
                {**edge("C", "D"), "evidence_layers": "single"},
                {**edge("E", "F", "TAIL", "ARROW"), "evidence_layers": "profile"},
            ]
        ),
        pd.DataFrame(
            [
                {**edge("A", "B"), "evidence_layers": "marginal"},
                {**edge("E", "F", "TAIL", "ARROW"), "evidence_layers": "profile"},
            ]
        ),
        pd.DataFrame(
            [{**edge("E", "F"), "evidence_layers": "sparse"}]
        ),
    ]
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"edges_{index}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths.append(path)

    consensus = consensus_graph(paths, 2 / 3).set_index(["node1", "node2"])
    assert ("C", "D") not in consensus.index
    assert consensus.loc[("A", "B"), "selection_count"] == 2
    assert consensus.loc[("A", "B"), "endpoint1"] == "CIRCLE"
    assert consensus.loc[("A", "B"), "endpoint2"] == "CIRCLE"
    assert consensus.loc[("E", "F"), "endpoint1"] == "TAIL"
    assert consensus.loc[("E", "F"), "endpoint2"] == "ARROW"

    lower, upper = wilson_interval(37, 40)
    assert 0.80 < lower < 0.81
    assert 0.97 < upper < 0.98


def test_typed_merge_abstains_except_for_explicit_gene_sink_prior() -> None:
    gene = "expr__SIMG_log1p"
    ac_peak = "chr1:1000-1499__H3K27ac"
    ac_peer = "chr1:2000-2499__H3K27ac"
    state = {canonical_edge(gene, ac_peak): {"layers": ["state_boss"]}}
    within = {canonical_edge(ac_peak, ac_peer): {"layers": ["within_bdeu_boss"]}}
    merged = merge_layers(state, within, impose_gene_sink=True)

    cross = merged.loc[merged["cross_channel"]].iloc[0]
    gene_endpoint = cross["endpoint1"] if cross["node1"] == gene else cross["endpoint2"]
    assert gene_endpoint == "ARROW"
    assert cross["orientation_source"] == "gene_sink_prior"
    assert cross["interpretation"] == "matched_state_adjacency"

    same_assay = merged.loc[~merged["cross_channel"]].iloc[0]
    assert same_assay["endpoint1"] == "CIRCLE"
    assert same_assay["endpoint2"] == "CIRCLE"
    assert same_assay["interpretation"] == "within_assay_adjacency"


def test_bfci_alias_calls_installed_boss_fci_api(monkeypatch) -> None:
    class FakeSearch:
        latest: "FakeSearch | None" = None

        def __init__(self, frame: pd.DataFrame):
            self.frame = frame
            self.algorithm = ""
            self.kwargs = {}
            FakeSearch.latest = self

        def use_sem_bic(self, **kwargs) -> None:
            self.score_kwargs = kwargs

        def use_fisher_z(self, **kwargs) -> None:
            self.test_kwargs = kwargs

        def run_boss_fci(self, **kwargs) -> None:
            self.algorithm = "boss_fci"
            self.kwargs = kwargs

        def get_string(self) -> str:
            return "Graph Edges:\n"

    tools_module = types.ModuleType("pytetrad.tools")
    tools_module.TetradSearch = types.SimpleNamespace(TetradSearch=FakeSearch)
    package = types.ModuleType("pytetrad")
    package.tools = tools_module
    monkeypatch.setitem(sys.modules, "pytetrad", package)
    monkeypatch.setitem(sys.modules, "pytetrad.tools", tools_module)

    run_gfci(
        pd.DataFrame({"peak": [0.0, 1.0, 2.0], "gene": [0.0, 2.0, 4.0]}),
        alpha=0.05,
        penalty=1.0,
        algo="bfci",
        max_depth=2,
        max_disc_path_length=4,
    )
    assert FakeSearch.latest is not None
    assert FakeSearch.latest.algorithm == "boss_fci"
    assert FakeSearch.latest.kwargs == {
        "depth": 2,
        "max_disc_path_length": 4,
        "complete_rule_set_used": True,
        "guarantee_pag": False,
    }
