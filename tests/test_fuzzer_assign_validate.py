"""Tests for the assign stage (typed_assign) and validate stage (validate_spec)."""

from __future__ import annotations

import numpy as np
import pytest

from fabricpc.fuzzer import (
    ConfigError,
    EdgeSpec,
    GeneratorConfig,
    GraphSpec,
    NodeSpec,
    Skeleton,
    generate,
)
from fabricpc.fuzzer import registry
from fabricpc.fuzzer.assign import SHAPE_PRESERVING_SLOTS, typed_assign, type_fits
from fabricpc.fuzzer.pipeline import DEFAULT_STAGES
from fabricpc.fuzzer.topology import random_dag_topology
from fabricpc.fuzzer.validity import validate_spec

DENSE_TYPES = (
    "Linear",
    "LinearExplicitGrad",
    "LinearResidual",
    "SkipConnection",
    "IdentityNode",
    "StorkeyHopfield",
)
# Small shapes keep the smoke training fast.
CFG = GeneratorConfig(
    num_nodes=(3, 5),
    in_degree=(1, 3),
    node_types=DENSE_TYPES,
    input_shape=(8,),
    output_shape=(3,),
    hidden_sizes=(8, 16),
)


def _assign(seed, config=CFG):
    rng = np.random.default_rng(seed)
    return typed_assign(random_dag_topology(config, rng), config, rng)


# ------------------------------------------------------------------ assign
def test_is_the_default_stage():
    assert DEFAULT_STAGES["assign"] is typed_assign
    assert DEFAULT_STAGES["validate"] is validate_spec


def test_same_seed_same_graph():
    assert _assign(3).fingerprint() == _assign(3).fingerprint()


def test_seeds_give_different_graphs():
    assert len({_assign(s).fingerprint() for s in range(10)}) > 1


@pytest.mark.parametrize("seed", range(25))
def test_assign_follows_config_and_slots(seed):
    spec = _assign(seed)
    spec.check()
    assert spec.input_node.shape == CFG.input_shape
    out = spec.output_node
    assert out.shape == CFG.output_shape
    assert (out.activation, out.energy) == (CFG.output_activation, CFG.output_energy)
    for n in spec.nodes_with_role("hidden"):
        assert n.node_type in CFG.node_types
        assert n.activation in CFG.hidden_activations
    for e in spec.edges:
        tgt = spec.node(e.target)
        assert e.slot in registry.node_class(tgt.node_type).get_slots()
        if (tgt.node_type, e.slot) in SHAPE_PRESERVING_SLOTS:
            assert spec.node(e.source).shape == tgt.shape


def test_skip_connection_gets_its_required_skip_edge():
    cfg = GeneratorConfig(
        num_nodes=(3, 5), in_degree=(1, 3), node_types=("SkipConnection", "Linear")
    )
    for seed in range(20):
        spec = _assign(seed, cfg)
        for n in spec.nodes:
            if n.node_type == "SkipConnection":
                assert "skip" in {e.slot for e in spec.in_edges(n.name)}


def test_single_input_types_only_get_one_edge():
    assert type_fits("StorkeyHopfield", 1)
    assert not type_fits("StorkeyHopfield", 2)
    assert not type_fits("SkipConnection", 1)
    assert not type_fits("ConvNode", 1)


def test_unroll_is_passed_through_for_cyclic_graphs():
    cfg = GeneratorConfig(unroll=3)
    skeleton = Skeleton(
        hidden=("h0", "h1"),
        edges=(("input", "h0"), ("h0", "h1"), ("h1", "h0"), ("h1", "output")),
    )
    spec = typed_assign(skeleton, cfg, np.random.default_rng(0))
    assert spec.is_cyclic and spec.metadata["unroll"] == 3


def test_ignores_unknown_constraints_and_global_rng():
    cfg = GeneratorConfig(node_types=DENSE_TYPES, constraints={"future_key": 1})
    np.random.seed(123)
    before = np.random.random()
    np.random.seed(123)
    _assign(0, cfg)
    assert np.random.random() == before


# ---------------------------------------------------------------- validate
@pytest.mark.parametrize("seed", range(6))
def test_assigned_graphs_pass(seed):
    result = validate_spec(_assign(seed))
    assert result.ok, result.issues


def test_cyclic_graphs_generate():
    cfg = GeneratorConfig(
        num_graphs=3,
        seed=1,
        num_nodes=(3, 4),
        in_degree=(1, 3),
        allow_cycles=True,
        node_types=DENSE_TYPES,
        input_shape=(8,),
        output_shape=(3,),
        hidden_sizes=(8, 16),
    )
    specs = generate(cfg)
    assert len(specs) == 3


def _base():
    nodes = [
        NodeSpec("input", "IdentityNode", (8,), role="input"),
        NodeSpec("h0", "Linear", (16,), activation="relu"),
        NodeSpec(
            "output",
            "Linear",
            (3,),
            role="output",
            activation="softmax",
            energy="cross_entropy",
        ),
    ]
    edges = [EdgeSpec("input", "h0", "in"), EdgeSpec("h0", "output", "in")]
    return nodes, edges


def _issue(nodes, edges):
    result = validate_spec(GraphSpec(tuple(nodes), tuple(edges)))
    assert not result.ok
    return result.issues[0]


def test_base_graph_passes():
    nodes, edges = _base()
    assert validate_spec(GraphSpec(tuple(nodes), tuple(edges))).ok


def test_rejects_edge_into_input():
    nodes, edges = _base()
    issue = _issue(nodes, edges + [EdgeSpec("h0", "input", "in")])
    assert (issue.check, issue.where) == ("structure", "h0->input:in")


def test_rejects_node_without_incoming_edge():
    nodes, edges = _base()
    nodes.insert(2, NodeSpec("h1", "Linear", (4,), activation="relu"))
    issue = _issue(nodes, edges + [EdgeSpec("h1", "output", "in")])
    assert (issue.check, issue.where) == ("structure", "h1")


def test_rejects_island_unreachable_from_input():
    nodes, edges = _base()
    nodes += [NodeSpec("a", "Linear", (4,)), NodeSpec("b", "Linear", (4,))]
    edges += [EdgeSpec("a", "b", "in"), EdgeSpec("b", "a", "in")]
    issue = _issue(nodes, edges)
    assert issue.check == "structure" and issue.where in {"a", "b"}


def test_rejects_unknown_type():
    nodes, edges = _base()
    nodes[1] = NodeSpec("h0", "Banana", (16,), activation="relu")
    issue = _issue(nodes, edges)
    assert (issue.check, issue.where) == ("registry", "h0")


def test_rejects_bad_slot():
    nodes, edges = _base()
    edges[0] = EdgeSpec("input", "h0", "skip")
    issue = _issue(nodes, edges)
    assert (issue.check, issue.where) == ("slot", "input->h0:skip")


def test_rejects_two_edges_into_single_input_slot():
    nodes, edges = _base()
    nodes[1] = NodeSpec("h0", "StorkeyHopfield", (8,))
    nodes.insert(1, NodeSpec("h1", "Linear", (8,)))
    edges += [EdgeSpec("input", "h1", "in"), EdgeSpec("h1", "h0", "in")]
    issue = _issue(nodes, edges)
    assert (issue.check, issue.where) == ("slot", "h0")


@pytest.mark.parametrize(
    "node_type, slot",
    [("SkipConnection", "skip"), ("IdentityNode", "in"), ("StorkeyHopfield", "in")],
)
def test_rejects_shape_mismatch_on_summing_slot(node_type, slot):
    nodes, edges = _base()
    nodes[1] = NodeSpec("h0", node_type, (16,))
    edges[0] = EdgeSpec("input", "h0", slot)
    issue = _issue(nodes, edges)
    assert (issue.check, issue.where) == ("shape", f"input->h0:{slot}")


def test_shape_mismatches_really_break_training():
    """Without the shape check, these graphs fail only when training starts."""
    from fabricpc.fuzzer.validity import _check_trains

    for node_type, slot in sorted(SHAPE_PRESERVING_SLOTS):
        if node_type == "SkipConnection":
            continue  # needs both slots; covered by the parametrized test above
        nodes, edges = _base()
        nodes[1] = NodeSpec("h0", node_type, (16,))
        edges[0] = EdgeSpec("input", "h0", slot)
        if node_type == "LinearResidual":
            nodes.insert(1, NodeSpec("h1", "Linear", (16,)))
            edges += [EdgeSpec("input", "h1", "in"), EdgeSpec("h1", "h0", "in")]
        assert _check_trains(GraphSpec(tuple(nodes), tuple(edges)), 1), node_type


def test_rejects_cross_entropy_without_softmax():
    nodes, edges = _base()
    nodes[2] = NodeSpec(
        "output",
        "Linear",
        (3,),
        role="output",
        activation="relu",
        energy="cross_entropy",
    )
    issue = _issue(nodes, edges)
    assert (issue.check, issue.where) == ("compat", "output")


def test_train_steps_zero_skips_training():
    nodes, edges = _base()
    assert validate_spec(GraphSpec(tuple(nodes), tuple(edges)), train_steps=0).ok


@pytest.mark.parametrize("node_type", ["ConvNode", "MaxPool", "AvgPool"])
def test_config_rejects_types_assign_cannot_build(node_type):
    with pytest.raises(ConfigError, match=node_type) as exc:
        GeneratorConfig(node_types=("Linear", node_type)).validate()
    assert exc.value.param == "node_types"


@pytest.mark.parametrize(
    "energy, activation",
    [("bernoulli", "softmax"), ("cross_entropy", "relu"), ("kl_divergence", "tanh")],
)
def test_config_rejects_impossible_output_pairs(energy, activation):
    with pytest.raises(ConfigError, match="needs output_activation") as exc:
        GeneratorConfig(output_energy=energy, output_activation=activation).validate()
    assert exc.value.param == "output_activation"
