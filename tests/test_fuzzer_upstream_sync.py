"""Upstream-sync suite: generation must keep working against FabricPC main.

This replaces evaluation as the useful end-to-end check (sponsor meeting,
2026-11-03). The suite uses a fixed list of seeds and configs, generates each
batch, builds every graph with the current FabricPC, and survives one real
training step. It also pins the generated fingerprints in a checked-in golden
file, so a change in generation output has to be accepted deliberately:

    pytest tests/test_fuzzer_upstream_sync.py -q
    pytest tests/test_fuzzer_upstream_sync.py -q --update-golden

Requirements: SR-F-14.1, SR-F-14.2, UR-NF-DE-01.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from fabricpc.fuzzer import ClampedNode, GeneratorConfig, generate, to_fabricpc
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import make_train_step

GOLDEN = Path(__file__).parent / "golden" / "fuzzer_fingerprints.json"
SEEDS = (0, 1, 42)
BATCH = 4

CONFIGS = {
    "affine": GeneratorConfig(
        num_graphs=2,
        num_nodes=(2, 4),
        in_degree=(1, 2),
        node_types=("Linear",),
        input_shape=(6,),
        output_shape=(3,),
        hidden_sizes=(8, 16),
    ),
    "mixed": GeneratorConfig(
        num_graphs=2,
        num_nodes=(2, 3),
        in_degree=(1, 2),
        node_types=("Linear", "LinearResidual", "IdentityNode", "StorkeyHopfield"),
        input_shape=(6,),
        output_shape=(3,),
        hidden_sizes=(8,),
    ),
    # Requires Part 2's clamped nodes.
    "clamped": GeneratorConfig(
        num_graphs=2,
        num_nodes=(2, 3),
        in_degree=(1, 2),
        out_degree=(1, 2),
        node_types=("Linear",),
        input_shape=(6,),
        output_shape=(3,),
        hidden_sizes=(8,),
        clamped_nodes=(
            ClampedNode("input", "IdentityNode", (6,), 0, 2),
            ClampedNode("output", "Linear", (3,), 2, 0),
        ),
    ),
}


def _shape_of(spec, role: str) -> tuple[int, ...]:
    return next(node.shape for node in spec.nodes if node.role == role)


def _batch_for(spec, key) -> dict:
    kx, ky, kstep = jax.random.split(key, 3)
    x = jax.random.normal(kx, (BATCH, *_shape_of(spec, "input")))
    classes = _shape_of(spec, "output")[0]
    labels = jax.random.randint(ky, (BATCH,), 0, classes)
    return {"x": x, "y": jax.nn.one_hot(labels, classes)}


def _run_one_step(spec, key) -> None:
    structure = to_fabricpc(spec)
    params = initialize_params(structure, key)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step = make_train_step(structure, optimizer)
    batch = _batch_for(spec, key)
    params, opt_state, metrics, _ = step(params, opt_state, batch, key)
    assert math.isfinite(float(metrics["energy"]))
    for leaf in jax.tree_util.tree_leaves(params):
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _actual_fingerprints() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name, config in CONFIGS.items():
        for seed in SEEDS:
            specs = generate(config.with_overrides(seed=seed))
            result[f"{name}:seed={seed}"] = [spec.fingerprint() for spec in specs]
    return result


def _diff(expected: dict, actual: dict) -> list[str]:
    lines = []
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            lines.append(f"{key}: expected {expected.get(key)}, got {actual.get(key)}")
    return lines


# --------------------------------------------------------------- SR-F-14.1
def test_sr_f_14_1_every_generated_graph_builds_and_trains_one_step():
    """SR-F-14.1: fixed seeds/configs all survive one FabricPC training step."""
    for name, config in CONFIGS.items():
        for seed in SEEDS:
            specs = generate(config.with_overrides(seed=seed))
            assert len(specs) == config.num_graphs, (name, seed)
            for index, spec in enumerate(specs):
                key = jax.random.PRNGKey(seed * 1000 + index)
                _run_one_step(spec, key)


# --------------------------------------------------------------- SR-F-14.2
def test_sr_f_14_2_fingerprints_match_checked_in_golden(update_golden):
    """SR-F-14.2: a fixed seed list pins the generated graphs."""
    actual = _actual_fingerprints()
    if update_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        return
    assert GOLDEN.exists(), (
        f"missing golden file {GOLDEN}; create it with "
        "`pytest tests/test_fuzzer_upstream_sync.py --update-golden`"
    )
    expected = json.loads(GOLDEN.read_text())
    differences = _diff(expected, actual)
    assert not differences, (
        "Fuzzer generation output changed for "
        f"{len(differences)} seed(s):\n  "
        + "\n  ".join(differences)
        + "\nIf the change is intentional, rerun with --update-golden and "
        "review the diff."
    )


# ------------------------------------------------------------ UR-NF-DE-01
def test_ur_nf_de_01_golden_mismatch_message_is_readable():
    """UR-NF-DE-01: a changed seed is caught and explained, not silently passed."""
    expected = {"affine:seed=0": ["abc"]}
    actual = {"affine:seed=999": ["def"]}
    differences = _diff(expected, actual)
    assert differences == [
        "affine:seed=0: expected ['abc'], got None",
        "affine:seed=999: expected None, got ['def']",
    ]
