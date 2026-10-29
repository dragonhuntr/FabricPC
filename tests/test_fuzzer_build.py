"""Tests for fabricpc.fuzzer.build: specs become trainable FabricPC graphs."""

import dataclasses

import jax
import jax.numpy as jnp
import optax
import pytest
from hypothesis import given, settings

from conftest import ListLoader, max_param_diff
from fuzzer_strategies import graph_specs
from fabricpc.fuzzer.build import BuildError, to_fabricpc
from fabricpc.fuzzer.serialize import dumps, loads
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import train


def small_spec(feedback=False):
    nodes = (
        NodeSpec("input", "IdentityNode", (6,), "input"),
        NodeSpec("h0", "Linear", (8,), activation="tanh"),
        NodeSpec("h1", "Linear", (5,), activation="relu", config={"use_bias": False}),
        NodeSpec(
            "output",
            "Linear",
            (3,),
            "output",
            activation="softmax",
            energy="cross_entropy",
        ),
    )
    edges = [
        EdgeSpec("input", "h0"),
        EdgeSpec("h0", "h1"),
        EdgeSpec("input", "h1"),
        EdgeSpec("h1", "output"),
    ]
    if feedback:
        edges.append(EdgeSpec("h1", "h0"))
    return GraphSpec(nodes, tuple(edges), {"unroll": 2})


def batches(key, in_dim=6, n_classes=3, n=2, bs=4):
    out = []
    for _ in range(n):
        kx, ky, key = jax.random.split(key, 3)
        x = jax.random.normal(kx, (bs, in_dim))
        y = jax.nn.one_hot(jax.random.randint(ky, (bs,), 0, n_classes), n_classes)
        out.append({"x": x, "y": y})
    return ListLoader(out)


def train_once(spec, seed=0):
    structure = to_fabricpc(spec)
    key = jax.random.PRNGKey(seed)
    params = initialize_params(structure, key)
    in_dim = spec.input_node.shape[0]
    n_classes = spec.output_node.shape[0]
    result = train(
        params,
        structure,
        batches(key, in_dim, n_classes),
        optax.sgd(1e-2),
        {"num_epochs": 1},
        key,
        verbose=False,
    )
    return structure, result


def test_builds_with_expected_topology():
    structure = to_fabricpc(small_spec())
    assert set(structure.nodes) == {"input", "h0", "h1", "output"}
    assert set(structure.edges) == {e.key for e in small_spec().edges}
    assert structure.task_map == {"x": "input", "y": "output"}
    assert structure.nodes["h1"].node_info.in_degree == 2


def test_cyclic_spec_builds_with_unroll():
    structure = to_fabricpc(small_spec(feedback=True))
    assert structure.config["unroll"] == 2
    assert len(structure.schedule) > len(structure.nodes)


@pytest.mark.parametrize("feedback", [False, True])
def test_trains_to_finite_params(feedback):
    _, result = train_once(small_spec(feedback))
    leaves = jax.tree_util.tree_leaves(result.params)
    assert leaves and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)


def test_reloaded_spec_trains_identically():
    """Saving and reloading a graph reproduces the exact same training run."""
    spec = small_spec()
    _, a = train_once(spec, seed=3)
    _, b = train_once(loads(dumps(spec)), seed=3)
    assert max_param_diff(a.params, b.params) == 0.0


def test_unknown_node_type_is_localized():
    spec = small_spec()
    nodes = list(spec.nodes)
    nodes[1] = dataclasses.replace(nodes[1], node_type="Lineer")
    with pytest.raises(BuildError, match="Did you mean 'Linear'") as exc:
        to_fabricpc(dataclasses.replace(spec, nodes=tuple(nodes)))
    assert exc.value.where == "h0"
    assert exc.value.stage == "node"


def test_bad_node_kwarg_is_localized():
    spec = small_spec()
    nodes = list(spec.nodes)
    nodes[2] = dataclasses.replace(nodes[2], config={"not_a_real_option": 1})
    with pytest.raises(BuildError) as exc:
        to_fabricpc(dataclasses.replace(spec, nodes=tuple(nodes)))
    assert exc.value.where == "h1"


def test_bad_slot_is_localized():
    spec = small_spec()
    bad = dataclasses.replace(
        spec, edges=spec.edges + (EdgeSpec("h0", "output", "skip"),)
    )
    with pytest.raises(BuildError, match="non-existent slot") as exc:
        to_fabricpc(bad)
    assert exc.value.where == "h0->output:skip"
    assert exc.value.stage == "graph"


def test_malformed_spec_fails_at_spec_stage():
    spec = small_spec()
    bad = dataclasses.replace(spec, edges=spec.edges + (EdgeSpec("h0", "h0"),))
    with pytest.raises(BuildError) as exc:
        to_fabricpc(bad)
    assert exc.value.stage == "spec"
    assert exc.value.where == "h0->h0:in"


def test_json_lists_become_tuples_for_node_config():
    """Conv kernel sizes are saved as JSON lists but FabricPC expects tuples."""
    nodes = (
        NodeSpec("input", "IdentityNode", (8, 8, 1), "input"),
        NodeSpec("conv", "ConvNode", (8, 8, 2), config={"kernel_size": [3, 3]}),
        NodeSpec("output", "Linear", (3,), "output", config={"flatten_input": True}),
    )
    spec = GraphSpec(nodes, (EdgeSpec("input", "conv"), EdgeSpec("conv", "output")))
    structure = to_fabricpc(loads(dumps(spec)))
    assert structure.nodes["conv"].node_info.node_config["kernel_size"] == (3, 3)


@settings(max_examples=15, deadline=None)
@given(graph_specs(max_hidden=4))
def test_every_well_formed_linear_spec_builds(spec):
    structure = to_fabricpc(spec)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    assert set(params.nodes) == set(spec.node_names)
