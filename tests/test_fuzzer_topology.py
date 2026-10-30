"""Tests for fabricpc.fuzzer.topology: the random DAG topology stage.

Out-degree limits (SR-F-04.3, SR-F-04.4) are not tested: GeneratorConfig has
no out-degree setting yet (open question in docs/fuzzer/stage_contract.md).
"""

import graphlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fabricpc.fuzzer import GeneratorConfig, generate, to_fabricpc
from fabricpc.fuzzer.pipeline import DEFAULT_STAGES
from fabricpc.fuzzer.seeding import graph_rng
from fabricpc.fuzzer.topology import (
    TOPOLOGY_MODELS,
    TopologyError,
    is_chain,
    random_dag_topology,
    skeleton_issues,
)


def draw(config, n=50, seed=0):
    config.validate()
    return [random_dag_topology(config, graph_rng(seed, i)) for i in range(n)]


def in_degrees(sk):
    deg = {v: 0 for v in (*sk.hidden, sk.output)}
    for _, t in sk.edges:
        deg[t] += 1
    return deg


def assert_valid(sk, config):
    assert skeleton_issues(sk, config) == []


def test_is_the_default_topology_stage():
    assert DEFAULT_STAGES["topology"] is random_dag_topology


# ------------------------------------------------- SR-F-03.3, SR-F-03.5
@pytest.mark.parametrize("num_nodes", [(0, 0), (1, 1), (2, 5), (4, 9)])
def test_hidden_count_inside_range_bounds_included(num_nodes):
    """SR-F-03.3: hidden-node count is within num_nodes, both ends reachable."""
    cfg = GeneratorConfig(num_nodes=num_nodes)
    counts = {len(sk.hidden) for sk in draw(cfg, n=200)}
    assert counts <= set(range(num_nodes[0], num_nodes[1] + 1))
    assert num_nodes[0] in counts and num_nodes[1] in counts


@pytest.mark.parametrize(
    "num_nodes,num_edges", [((3, 3), (4, 4)), ((3, 3), (6, 6)), ((2, 6), (5, 9))]
)
def test_edge_count_inside_range_bounds_included(num_nodes, num_edges):
    """SR-F-03.5: edge count is within num_edges, both ends reachable."""
    cfg = GeneratorConfig(num_nodes=num_nodes, num_edges=num_edges)
    counts = {len(sk.edges) for sk in draw(cfg, n=300)}
    assert counts <= set(range(num_edges[0], num_edges[1] + 1))
    assert num_edges[0] in counts and num_edges[1] in counts


def test_impossible_edge_range_raises():
    # Passes the config's coarse check (<= 10 edges) but the exact limit is
    # 1 + 2 + 3 + 3 = 9 forward edges for 3 hidden nodes.
    cfg = GeneratorConfig(num_nodes=(3, 3), num_edges=(10, 10), in_degree=(1, 3))
    cfg.validate()
    with pytest.raises(TopologyError):
        random_dag_topology(cfg, graph_rng(0, 0))


# --------------------------------------------------------------- SR-F-04.2
@pytest.mark.parametrize("in_degree", [(1, 1), (1, 2), (1, 4), (2, 2), (2, 3)])
@pytest.mark.parametrize("cycles", [False, True])
def test_in_degree_limits(in_degree, cycles):
    """SR-F-04.2: every hidden node and the output has in-degree in range."""
    if in_degree[0] > 1 and not cycles:
        pytest.skip("min in-degree > 1 needs cycles (rejected by config)")
    cfg = GeneratorConfig(num_nodes=(3, 6), in_degree=in_degree, allow_cycles=cycles)
    seen = set()
    for sk in draw(cfg, n=100):
        for v, d in in_degrees(sk).items():
            assert in_degree[0] <= d <= in_degree[1], (v, d, sk)
            seen.add(d)
    assert in_degree[1] in seen


def test_structural_rules():
    cfg = GeneratorConfig(num_nodes=(0, 8), in_degree=(1, 3))
    for sk in draw(cfg, n=200):
        assert_valid(sk, cfg)
        assert all(t != sk.input for _, t in sk.edges)
        assert all(s != t for s, t in sk.edges)
        assert len(set(sk.edges)) == len(sk.edges)


# ----------------------------------------------------------------- cycles
def test_acyclic_when_cycles_off():
    cfg = GeneratorConfig(num_nodes=(2, 8), in_degree=(1, 4), allow_cycles=False)
    for sk in draw(cfg, n=200):
        preds = {v: set() for v in sk.all_nodes}
        for s, t in sk.edges:
            preds[t].add(s)
        tuple(graphlib.TopologicalSorter(preds).static_order())  # no CycleError


def test_feedback_edges_when_cycles_on():
    cfg = GeneratorConfig(num_nodes=(2, 6), in_degree=(1, 3), allow_cycles=True)
    cyclic = 0
    for sk in draw(cfg, n=100):
        assert_valid(sk, cfg)
        preds = {v: set() for v in sk.all_nodes}
        for s, t in sk.edges:
            preds[t].add(s)
        try:
            tuple(graphlib.TopologicalSorter(preds).static_order())
        except graphlib.CycleError:
            cyclic += 1
    assert cyclic > 50


def test_cyclic_skeletons_build_in_fabricpc():
    cfg = GeneratorConfig(
        num_graphs=5,
        seed=3,
        num_nodes=(2, 4),
        allow_cycles=True,
        input_shape=(4,),
        output_shape=(2,),
        hidden_sizes=(4,),
    )
    specs = generate(cfg)
    assert any(s.is_cyclic for s in specs)
    for s in specs:
        to_fabricpc(s)


# ---------------------------------------------------- non-sequential shape
def test_not_a_chain_when_settings_allow_more():
    cfg = GeneratorConfig(num_nodes=(1, 6), in_degree=(1, 3))
    assert not any(is_chain(sk) for sk in draw(cfg, n=200))


def test_chain_when_settings_force_it():
    cfg = GeneratorConfig(num_nodes=(1, 5), in_degree=(1, 1))
    assert all(is_chain(sk) for sk in draw(cfg, n=50))


def test_has_branches_and_skips():
    cfg = GeneratorConfig(num_nodes=(3, 6), in_degree=(1, 3))
    skips = branches = 0
    for sk in draw(cfg, n=100):
        order = sk.all_nodes
        pos = {v: i for i, v in enumerate(order)}
        skips += any(pos[t] - pos[s] > 1 for s, t in sk.edges)
        out = {}
        for s, _ in sk.edges:
            out[s] = out.get(s, 0) + 1
        branches += any(d > 1 for d in out.values())
    assert skips > 50 and branches > 50


@pytest.mark.parametrize("model", TOPOLOGY_MODELS)
def test_every_model_satisfies_rules(model):
    cfg = GeneratorConfig(
        num_nodes=(1, 7), in_degree=(1, 3), constraints={"topology_model": model}
    )
    for sk in draw(cfg, n=100):
        assert_valid(sk, cfg)


def test_unknown_model_rejected():
    cfg = GeneratorConfig(constraints={"topology_model": "small_world_2"})
    with pytest.raises(TopologyError, match="topology_model"):
        random_dag_topology(cfg, graph_rng(0, 0))


# --------------------------------------------------------------- SR-F-06.2
@pytest.mark.parametrize("cycles", [False, True])
def test_same_seed_same_skeletons(cycles):
    """SR-F-06.2: the same seed gives identical skeletons."""
    cfg = GeneratorConfig(num_nodes=(2, 7), in_degree=(1, 3), allow_cycles=cycles)
    assert draw(cfg, seed=42) == draw(cfg, seed=42)
    assert draw(cfg, seed=42) != draw(cfg, seed=43)


def test_batch_of_50_is_structurally_distinct():
    """A batch of 50 has (almost) no repeats, and the pipeline removes the rest."""
    cfg = GeneratorConfig(num_nodes=(3, 8), in_degree=(1, 3))
    sks = draw(cfg, n=50, seed=9)
    assert len(set(sks)) >= 48
    specs = generate(
        cfg.with_overrides(
            num_graphs=50,
            seed=9,
            input_shape=(4,),
            output_shape=(2,),
            hidden_sizes=(4,),
        )
    )
    assert len({s.fingerprint() for s in specs}) == 50


# --------------------------------------------------- property-based check
@st.composite
def configs(draw_):
    lo = draw_(st.integers(0, 6))
    hi = draw_(st.integers(lo, lo + 5))
    cycles = draw_(st.booleans())
    d_lo = draw_(st.integers(1, 2)) if cycles else 1
    d_hi = draw_(st.integers(d_lo, 4))
    num_edges = None
    if draw_(st.booleans()):
        e_lo = draw_(st.integers(lo + 1, lo + 1 + 3 * hi))
        num_edges = (e_lo, draw_(st.integers(e_lo, e_lo + 8)))
    model = draw_(st.sampled_from(TOPOLOGY_MODELS))
    return GeneratorConfig(
        num_nodes=(lo, hi),
        in_degree=(d_lo, d_hi),
        num_edges=num_edges,
        allow_cycles=cycles,
        constraints={"topology_model": model},
    )


@settings(max_examples=60, deadline=None)
@given(cfg=configs(), seed=st.integers(0, 2**31 - 1))
def test_property_valid_configs_give_valid_skeletons(cfg, seed):
    try:
        cfg.validate()
    except Exception:
        return  # config layer rejects it; nothing to generate
    try:
        sks = [random_dag_topology(cfg, graph_rng(seed, i)) for i in range(20)]
    except TopologyError:
        # Only allowed when no hidden count in range can meet the edge range.
        assert cfg.num_edges is not None
        return
    for sk in sks:
        assert_valid(sk, cfg)
