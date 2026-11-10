"""Tests for fabricpc.fuzzer.experiments.baselines: sequential baselines."""

import json

import pytest

from fabricpc.fuzzer import GeneratorConfig, generate, to_fabricpc
from fabricpc.fuzzer.experiments.baselines import (
    describe_architecture,
    layer_sizes,
    sequential_baseline,
)
from fabricpc.fuzzer.serialize import dumps, loads
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec


def dag_spec():
    """A non-sequential graph not produced by any pipeline stage."""
    nodes = (
        NodeSpec("x", "IdentityNode", (6,), "input"),
        NodeSpec("a", "Linear", (8,), activation="tanh"),
        NodeSpec("b", "Linear", (5,), activation="sigmoid"),
        NodeSpec("y", "Linear", (3,), "output", "softmax", energy="cross_entropy"),
    )
    edges = (
        EdgeSpec("x", "a"),
        EdgeSpec("x", "b"),
        EdgeSpec("a", "b"),
        EdgeSpec("b", "y"),
        EdgeSpec("a", "y"),
    )
    return GraphSpec(nodes, edges, {"graph_index": 4})


def small_config(**kw):
    base = dict(
        num_graphs=3,
        seed=11,
        num_nodes=(1, 3),
        input_shape=(6,),
        output_shape=(3,),
        hidden_sizes=(4, 8),
    )
    base.update(kw)
    return GeneratorConfig(**base)


def test_baseline_keeps_input_and_output_nodes():
    spec = dag_spec()
    base = sequential_baseline(spec)
    assert base.input_node == spec.input_node
    assert base.output_node == spec.output_node


def test_baseline_is_a_chain():
    base = sequential_baseline(dag_spec())
    names = base.topological_order()
    assert [e.key for e in base.edges] == [
        f"{s}->{t}:in" for s, t in zip(names[:-1], names[1:])
    ]
    assert all(base.in_degree(n) == 1 for n in names[1:])
    assert not base.is_cyclic


def test_baseline_mirrors_hidden_widths_in_topological_order():
    base = sequential_baseline(dag_spec())
    assert layer_sizes(base) == [6, 8, 5, 3]
    hidden = [n for n in base.nodes if n.role == "hidden"]
    assert [n.activation for n in hidden] == ["tanh", "sigmoid"]
    assert all(n.node_type == "Linear" for n in hidden)


def test_baseline_overrides():
    base = sequential_baseline(dag_spec(), hidden_sizes=[16], activation="relu")
    assert layer_sizes(base) == [6, 16, 3]
    direct = sequential_baseline(dag_spec(), hidden_sizes=[])
    assert [e.key for e in direct.edges] == ["x->y:in"]
    with pytest.raises(ValueError):
        sequential_baseline(dag_spec(), hidden_sizes=[0])


def test_baseline_metadata_links_back_to_graph():
    spec = dag_spec()
    base = sequential_baseline(spec)
    assert base.metadata["baseline_kind"] == "sequential"
    assert base.metadata["baseline_of"] == spec.fingerprint()
    assert base.metadata["graph_index"] == 4


def test_baseline_of_cyclic_graph_is_acyclic():
    spec = dag_spec()
    cyclic = GraphSpec(spec.nodes, spec.edges + (EdgeSpec("b", "a"),), {"unroll": 2})
    assert cyclic.is_cyclic
    base = sequential_baseline(cyclic)
    assert not base.is_cyclic
    assert len(base.nodes) == len(spec.nodes)


def test_baseline_goes_through_same_build_and_serialize_path():
    base = sequential_baseline(dag_spec())
    structure = to_fabricpc(base)
    assert set(structure.nodes) == set(base.node_names)
    assert loads(dumps(base)) == base


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_baseline_for_every_pipeline_graph(seed):
    for spec in generate(small_config(seed=seed)):
        base = sequential_baseline(spec)
        to_fabricpc(base)
        assert base.input_node == spec.input_node
        assert base.output_node == spec.output_node


def test_describe_architecture_records_sizes_and_connections():
    """SR-F-12.9: baseline layer sizes and connections are recorded."""
    desc = describe_architecture(sequential_baseline(dag_spec()))
    assert [layer["shape"] for layer in desc["layers"]] == [[6], [8], [5], [3]]
    assert desc["connections"] == ["x->b0:in", "b0->b1:in", "b1->y:in"]
    assert json.loads(json.dumps(desc)) == desc
