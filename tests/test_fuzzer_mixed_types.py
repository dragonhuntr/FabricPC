"""Property tests for graphs that mix node types.

Covers SR-F-02.3, SR-F-09.1, SR-F-13.2, SR-F-13.3 and SR-NF-DC-03.

As in Section 6.5 of the report, Hypothesis only draws the generator settings
and the fuzzer's own sampler (topology, then typed_assign) builds each graph.
When a property fails, Hypothesis shrinks the settings, so the failure is
reported on the smallest graph that still fails. The fuzzer itself never
imports Hypothesis.
"""

from __future__ import annotations
from fabricpc.fuzzer.validity import validate_spec

import math

import jax
import numpy as np
import pytest
from hypothesis import HealthCheck, find, given, note, reject, settings
from hypothesis import strategies as st

import fabricpc.nodes
from fabricpc.fuzzer import ConfigError, GeneratorConfig
from fabricpc.fuzzer.assign import typed_assign
from fabricpc.fuzzer.build import to_fabricpc
from fabricpc.fuzzer.topology import TopologyError, random_dag_topology
from fabricpc.graph_initialization import initialize_params

DENSE_TYPES = (
    "Linear",
    "LinearExplicitGrad",
    "LinearResidual",
    "SkipConnection",
    "IdentityNode",
    "StorkeyHopfield",
)
PROPERTY_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# --------------------------------------------------------------- strategies
@st.composite
def mixed_configs(draw):
    """Generator settings with a random mix of node types.

    Every field shrinks toward a smaller graph: fewer hidden nodes, fewer
    incoming edges, fewer node types, no cycles.
    """
    node_types = draw(st.lists(st.sampled_from(DENSE_TYPES), min_size=1, unique=True))
    n_hidden = draw(st.integers(1, 6))
    cfg = GeneratorConfig(
        num_nodes=(n_hidden, n_hidden),
        in_degree=(1, draw(st.integers(1, 3))),
        node_types=tuple(node_types),
        allow_cycles=draw(st.booleans()),
        input_shape=(8,),
        output_shape=(3,),
        hidden_sizes=(8, 16),
        seed=draw(st.integers(0, 2**16)),
    )
    try:
        cfg.validate()
    except ConfigError:
        reject()
    return cfg


def sample(cfg: GeneratorConfig):
    """Run the fuzzer's own topology and assign stages for ``cfg``."""
    rng = np.random.default_rng(cfg.seed)
    try:
        skeleton = random_dag_topology(cfg, rng)
    except TopologyError:
        reject()
    spec = typed_assign(skeleton, cfg, rng)
    # The fuzzer only returns graphs that pass validation (SR-F-10.1) and
    # retries the rest, so the properties are about those. Training is
    # skipped here to keep the test fast.
    if not validate_spec(spec, train_steps=0).ok:
        reject()
    return spec


@st.composite
def mixed_graphs(draw):
    cfg = draw(mixed_configs())
    return cfg, sample(cfg)


# ------------------------------------------------------------------ helpers
def report(spec) -> None:
    """SR-F-13.3: put the graph in the failure report.

    Hypothesis prints notes only for the final, shrunk failing example.
    """
    note(f"shrunk graph:\n{spec.summary()}")


def is_public_fabricpc_class(cls) -> bool:
    """True if ``cls`` is exported by FabricPC's public node module."""
    name = cls.__name__
    return name in fabricpc.nodes.__all__ and getattr(fabricpc.nodes, name) is cls


def shape_mismatches(spec, params) -> list:
    """SR-F-09.1: edges whose source output size isn't what the target takes.

    A weighted edge's input size is the first dimension of the weight FabricPC
    made for it. An edge with no weight feeds the target unchanged, so its
    input size is the target's own size.
    """
    bad = []
    for e in spec.edges:
        source_size = math.prod(spec.node(e.source).shape)
        weight = params.nodes[e.target].weights.get(e.key)
        if weight is not None:
            input_size = weight.shape[0]
        else:
            input_size = math.prod(spec.node(e.target).shape)
        if source_size != input_size:
            bad.append(
                f"{e.key}: source gives {source_size}, target takes {input_size}"
            )
    return bad


def has_type(spec, node_type: str) -> bool:
    return any(n.node_type == node_type for n in spec.nodes_with_role("hidden"))


# --------------------------------------------------------------- properties
@PROPERTY_SETTINGS
@given(mixed_graphs())
def test_mixed_type_graphs_keep_their_properties(case):
    cfg, spec = case
    report(spec)

    # SR-F-02.3: every non-clamped node has a permitted node type.
    for node in spec.nodes_with_role("hidden"):
        assert node.node_type in cfg.node_types, node.name

    structure = to_fabricpc(spec)
    params = initialize_params(structure, jax.random.PRNGKey(0))

    # SR-F-09.1: on every edge, the source's output fits the target's input.
    assert shape_mismatches(spec, params) == []

    # SR-NF-DC-03: every node is a class from fabricpc.nodes.
    for name, node in structure.nodes.items():
        assert is_public_fabricpc_class(type(node)), name


def test_sr_nf_dc_03_every_node_class_in_1000_graphs():
    """SR-NF-DC-03 measure: check the class of every node in 1,000 graphs."""
    cfg = GeneratorConfig(
        num_nodes=(1, 6),
        in_degree=(1, 3),
        node_types=DENSE_TYPES,
        allow_cycles=True,
        input_shape=(8,),
        output_shape=(3,),
        hidden_sizes=(8, 16),
    )
    rng = np.random.default_rng(0)
    seen = set()
    checked = 0
    while checked < 1000:
        spec = typed_assign(random_dag_topology(cfg, rng), cfg, rng)
        if not validate_spec(spec, train_steps=0).ok:
            continue  # the fuzzer would discard this one and retry
        for name, node in to_fabricpc(spec).nodes.items():
            assert is_public_fabricpc_class(type(node)), f"graph {checked}, node {name}"
            seen.add(type(node).__name__)
        checked += 1
    # The batch really was mixed: every dense type showed up.
    assert set(DENSE_TYPES) <= seen


# ---------------------------------------------------------------- shrinking
def test_sr_f_13_2_failing_case_shrinks_to_smallest_graph():
    """Treat every graph with a StorkeyHopfield as failing. Hypothesis must
    shrink that failure to the smallest such graph: one hidden node."""
    _, spec = find(
        mixed_graphs(),
        lambda case: has_type(case[1], "StorkeyHopfield"),
        settings=settings(database=None, max_examples=500),
    )
    hidden = spec.nodes_with_role("hidden")
    assert [n.node_type for n in hidden] == ["StorkeyHopfield"]


def test_sr_f_13_3_shrunk_graph_is_in_the_failure_report():
    @settings(database=None, max_examples=200, deadline=None)
    @given(mixed_graphs())
    def fails_on_hopfield(case):
        _, spec = case
        report(spec)
        assert not has_type(spec, "StorkeyHopfield")

    with pytest.raises(AssertionError) as exc:
        fails_on_hopfield()
    failure_report = "\n".join(getattr(exc.value, "__notes__", []))
    assert "shrunk graph:" in failure_report
    assert "StorkeyHopfield" in failure_report
