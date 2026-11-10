"""Tests for fabricpc.fuzzer.factory: GraphSpec -> experiment model factory."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
from conftest import max_param_diff

from fabricpc.core.inference import InferenceSGD
from fabricpc.fuzzer import as_model_factory, to_fabricpc
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec
from fabricpc.training import make_train_step

INPUT_DIM = 6
HIDDEN_DIM = 8
CLASSES = 3


def spec():
    """A small linear graph that trains quickly."""
    return GraphSpec(
        nodes=(
            NodeSpec("input", "IdentityNode", (INPUT_DIM,), role="input"),
            NodeSpec("h0", "Linear", (HIDDEN_DIM,), activation="tanh"),
            NodeSpec(
                "output",
                "Linear",
                (CLASSES,),
                role="output",
                activation="softmax",
                energy="cross_entropy",
            ),
        ),
        edges=(EdgeSpec("input", "h0"), EdgeSpec("h0", "output")),
    )


def test_sr_f_11_factory_matches_experiment_signature():
    """SR-F-11: a fuzzed graph exposes the A/B harness's model-factory shape."""
    factory = as_model_factory(spec())
    params, structure = factory(jax.random.PRNGKey(0))
    assert set(params.nodes) == {"input", "h0", "output"}
    assert set(structure.nodes) == {"input", "h0", "output"}
    assert structure.task_map == {"x": "input", "y": "output"}
    assert factory.spec is not None
    assert factory.__name__.startswith("factory_")


def test_sr_f_11_factory_reuses_structure_and_seeds_params():
    """SR-F-11: the structure is static; the key changes only the weights."""
    factory = as_model_factory(spec())
    params_a, structure_a = factory(jax.random.PRNGKey(0))
    params_b, structure_b = factory(jax.random.PRNGKey(0))
    params_c, structure_c = factory(jax.random.PRNGKey(1))
    assert structure_a is structure_b is structure_c
    assert max_param_diff(params_a, params_b) == 0.0
    assert max_param_diff(params_a, params_c) > 0.0


def test_sr_f_11_factory_inference_override_reaches_structure():
    """SR-F-11: the caller can match the experiment's inference settings."""
    inference = InferenceSGD(eta_infer=0.2, infer_steps=3)
    _, structure = as_model_factory(spec(), inference=inference)(jax.random.PRNGKey(0))
    assert structure.config["inference"] is inference


def test_sr_f_11_factory_output_trains_one_step():
    """SR-F-11: the factory's output survives a real FabricPC training step."""
    factory = as_model_factory(spec())
    key = jax.random.PRNGKey(0)
    params, structure = factory(key)
    optimizer = optax.sgd(1e-2)
    opt_state = optimizer.init(params)
    step = make_train_step(structure, optimizer)

    kx, ky, kstep = jax.random.split(key, 3)
    x = jax.random.normal(kx, (4, INPUT_DIM))
    labels = jax.random.randint(ky, (4,), 0, CLASSES)
    batch = {"x": x, "y": jax.nn.one_hot(labels, CLASSES)}

    params, opt_state, metrics, _ = step(params, opt_state, batch, kstep)
    assert math.isfinite(float(metrics["energy"]))
    for leaf in jax.tree_util.tree_leaves(params):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_sr_f_11_factory_builds_same_structure_as_to_fabricpc():
    """SR-F-11: the adapter adds no hidden graph changes."""
    direct = to_fabricpc(spec())
    _, via_factory = as_model_factory(spec())(jax.random.PRNGKey(0))
    assert set(via_factory.nodes) == set(direct.nodes)
    assert set(via_factory.edges) == set(direct.edges)
    assert via_factory.task_map == direct.task_map
