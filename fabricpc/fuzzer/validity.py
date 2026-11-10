"""Validate stage (Hai): decide whether a GraphSpec builds AND trains.

:func:`validate_spec` replaces :func:`fabricpc.fuzzer.pipeline.check_builds`.

Checks run cheapest first. The first check that finds problems returns all of
them, each naming the node or edge at fault (UR-F-11):

==========  ===========================================================
check       catches
==========  ===========================================================
structure   malformed spec (``GraphSpec.check``), nodes unreachable
            from input, output unreachable
registry    unknown node type, activation or energy name
compat      output energy that needs a specific activation
slot        non-existent slot, too many edges on a single-input slot,
            empty required slot
shape       shape mismatch on a slot that adds inputs directly
build       FabricPC rejects the graph (``to_fabricpc`` raises)
train       a short training run fails or produces NaN/inf
==========  ===========================================================

FabricPC itself only notices shape mismatches once training starts, with an
error that doesn't name a node; the ``shape`` check names the exact edge.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.fuzzer import registry
from fabricpc.fuzzer.assign import SHAPE_PRESERVING_SLOTS
from fabricpc.fuzzer.build import BuildError, to_fabricpc
from fabricpc.fuzzer.config import ENERGY_NEEDS_ACTIVATION
from fabricpc.fuzzer.spec import GraphSpec, SpecError
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import make_train_step

#: Output energies that only make sense with particular output activations.

SMOKE_BATCH = 8
SMOKE_STEPS = 2


def validate_spec(spec: GraphSpec, *, train_steps: int = SMOKE_STEPS):
    """Return ``ValidationResult.passed()`` only if ``spec`` builds and trains.

    Args:
        spec: The graph to check.
        train_steps: Training steps in the smoke run. ``0`` skips training and
            only checks that the graph builds (faster, but misses NaNs).
    """
    from fabricpc.fuzzer.pipeline import ValidationResult

    checks = [
        _check_structure,
        _check_registry,
        _check_slots,
        _check_shapes,
        _check_builds,
    ]
    for check in checks:
        issues = check(spec)
        if issues:
            return ValidationResult.failed(*issues)
    if train_steps > 0:
        issues = _check_trains(spec, train_steps)
        if issues:
            return ValidationResult.failed(*issues)
    return ValidationResult.passed()


def _issue(message: str, where: Optional[str], check: str):
    from fabricpc.fuzzer.pipeline import ValidationIssue

    return ValidationIssue(message=message, where=where, check=check)


# ------------------------------------------------------------------ checks
def _check_structure(spec: GraphSpec) -> list:
    try:
        spec.check()
    except SpecError as exc:
        return [_issue(str(exc), exc.where, "structure")]

    issues = []
    has_input = {e.target for e in spec.edges}
    for n in spec.nodes:
        if n.name != spec.input_node.name and n.name not in has_input:
            issues.append(_issue("node has no incoming edge", n.name, "structure"))
    if issues:
        return issues

    reachable = _reachable_from(spec.input_node.name, spec)
    for n in spec.nodes:
        if n.name not in reachable:
            issues.append(_issue("not reachable from input", n.name, "structure"))
    return issues


def _reachable_from(start: str, spec: GraphSpec) -> set:
    out = defaultdict(list)
    for e in spec.edges:
        out[e.source].append(e.target)
    seen, queue = {start}, deque([start])
    while queue:
        for nxt in out[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _check_registry(spec: GraphSpec) -> list:
    issues = []
    for n in spec.nodes:
        if n.node_type not in registry.NODE_TYPES:
            issues.append(
                _issue(f"unknown node type {n.node_type!r}", n.name, "registry")
            )
        if n.activation is not None and n.activation not in registry.ACTIVATIONS:
            issues.append(
                _issue(f"unknown activation {n.activation!r}", n.name, "registry")
            )
        if n.energy is not None and n.energy not in registry.ENERGIES:
            issues.append(_issue(f"unknown energy {n.energy!r}", n.name, "registry"))
    if issues:
        return issues

    for n in spec.nodes:
        needs = ENERGY_NEEDS_ACTIVATION.get(n.energy)
        if needs and n.activation not in needs:
            issues.append(
                _issue(
                    f"energy {n.energy!r} needs activation {sorted(needs)}, "
                    f"got {n.activation!r}",
                    n.name,
                    "compat",
                )
            )
    return issues


def _check_slots(spec: GraphSpec) -> list:
    issues = []
    per_slot = defaultdict(list)
    for e in spec.edges:
        slots = registry.node_class(spec.node(e.target).node_type).get_slots()
        if e.slot not in slots:
            issues.append(
                _issue(
                    f"slot {e.slot!r} does not exist; options: {sorted(slots)}",
                    e.key,
                    "slot",
                )
            )
        per_slot[(e.target, e.slot)].append(e)
    for n in spec.nodes:
        if n.name == spec.input_node.name:
            continue  # the input is clamped to data; its slots stay empty
        for slot_name, slot in registry.node_class(n.node_type).get_slots().items():
            used = per_slot[(n.name, slot_name)]
            if not slot.is_multi_input and len(used) > 1:
                issues.append(
                    _issue(
                        f"slot {slot_name!r} takes one edge, got {len(used)}",
                        n.name,
                        "slot",
                    )
                )
            if slot.require_connected and not used:
                issues.append(
                    _issue(f"slot {slot_name!r} needs an edge", n.name, "slot")
                )
    return issues


def _check_shapes(spec: GraphSpec) -> list:
    issues = []
    for e in spec.edges:
        src, tgt = spec.node(e.source), spec.node(e.target)
        if (tgt.node_type, e.slot) in SHAPE_PRESERVING_SLOTS and src.shape != tgt.shape:
            issues.append(
                _issue(
                    f"{tgt.node_type}.{e.slot} adds its inputs directly, so "
                    f"source shape {tuple(src.shape)} must equal {tuple(tgt.shape)}",
                    e.key,
                    "shape",
                )
            )
    return issues


def _check_builds(spec: GraphSpec) -> list:
    try:
        to_fabricpc(spec)
    except BuildError as exc:
        return [_issue(str(exc), exc.where, f"build/{exc.stage}")]
    return []


def _check_trains(spec: GraphSpec, steps: int) -> list:
    # validate() gets no rng, so the key is derived from the spec itself:
    # the same graph always gets the same smoke run, and no fixed key is used.
    key = jax.random.PRNGKey(int(spec.fingerprint()[:8], 16))
    k_init, k_x, k_y, k_step = jax.random.split(key, 4)
    try:
        structure = to_fabricpc(spec)
        params = initialize_params(structure, k_init)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        step = make_train_step(structure, optimizer)
        batch = _smoke_batch(spec, k_x, k_y)
        for _ in range(steps):
            params, opt_state, metrics, _ = step(params, opt_state, batch, k_step)
            energy = float(metrics["energy"])
            if not np.isfinite(energy):
                return [_issue(f"training energy became {energy}", None, "train")]
    except Exception as exc:  # noqa: BLE001 - any training crash rejects the graph
        return [_issue(f"{type(exc).__name__}: {exc}", None, "train")]

    for name, node_params in params.nodes.items():
        leaves = jax.tree_util.tree_leaves(node_params)
        if any(not bool(jnp.all(jnp.isfinite(x))) for x in leaves):
            return [_issue("parameters became NaN/inf", name, "train")]
    return []


def _smoke_batch(spec: GraphSpec, k_x, k_y) -> dict:
    x = jax.random.normal(k_x, (SMOKE_BATCH, *spec.input_node.shape))
    out = spec.output_node
    if out.activation == "softmax":
        labels = jax.random.randint(k_y, (SMOKE_BATCH,), 0, out.shape[-1])
        y = jax.nn.one_hot(labels, out.shape[-1])
    elif out.activation == "sigmoid":
        y = jax.random.bernoulli(k_y, 0.5, (SMOKE_BATCH, *out.shape)).astype(
            jnp.float32
        )
    else:
        y = jax.random.normal(k_y, (SMOKE_BATCH, *out.shape))
    return {"x": x, "y": y}


__all__: List[str] = ["validate_spec", "ENERGY_NEEDS_ACTIVATION"]
