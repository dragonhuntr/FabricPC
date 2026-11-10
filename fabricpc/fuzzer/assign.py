"""Assign stage (Hai): give a Skeleton node types, shapes, activations and slots.

:func:`typed_assign` replaces :func:`fabricpc.fuzzer.pipeline.linear_assign`.

- Hidden node types come from ``config.node_types``, filtered to the types
  whose slots can take that node's number of incoming edges.
- Slots come from each FabricPC node class's own ``get_slots()``, so a newly
  registered node type gets correct slots without changes here.
- Slots that add their inputs directly (no weight matrix) force the source and
  target to share a shape. Union-find groups those nodes; input and output
  shapes come from config and every other group gets a size from
  ``config.hidden_sizes``.

All randomness comes from ``rng`` (UR-F-08).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from fabricpc.fuzzer import registry
from fabricpc.fuzzer.config import UNSUPPORTED_TYPES, GeneratorConfig
from fabricpc.fuzzer.spec import (
    ROLE_HIDDEN,
    ROLE_INPUT,
    ROLE_OUTPUT,
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    Skeleton,
)

#: ``(node_type, slot)`` pairs whose inputs are added straight into the node,
#: so the source shape must equal the target shape. Every other slot goes
#: through a weight matrix and accepts any source shape.
SHAPE_PRESERVING_SLOTS = frozenset(
    {
        ("IdentityNode", "in"),
        ("SkipConnection", "in"),
        ("SkipConnection", "skip"),
        ("LinearResidual", "skip"),
        ("StorkeyHopfield", "in"),
    }
)

#: Fewest incoming edges for a type to be worth choosing. A SkipConnection
#: needs one edge for its required "skip" slot and one for "in".
MIN_IN_DEGREE = {"SkipConnection": 2}

INPUT_TYPE = "IdentityNode"
OUTPUT_TYPE = "Linear"


def typed_assign(
    skeleton: Skeleton, config: GeneratorConfig, rng: np.random.Generator
) -> GraphSpec:
    """Give every node a type, shape and activation and every edge a slot.

    Args:
        skeleton: Hidden node names and directed edges from the topology stage.
        config: Generator settings.
        rng: The only source of randomness.

    Returns:
        A GraphSpec. It can still be invalid (for example an unavoidable shape
        clash inside a cycle); the validate stage catches that and the pipeline
        retries. ``metadata["unroll"]`` is set so validation builds cyclic
        graphs with ``config.unroll``.
    """
    names = list(skeleton.all_nodes)
    sources_of: Dict[str, List[str]] = defaultdict(list)
    for src, tgt in skeleton.edges:
        sources_of[tgt].append(src)

    types = _pick_types(skeleton, sources_of, config, rng)
    edges = _pick_slots(names, sources_of, types, rng)
    shapes = _pick_shapes(skeleton, names, types, edges, config, rng)

    nodes = [
        NodeSpec(
            name=skeleton.input,
            node_type=types[skeleton.input],
            shape=shapes[skeleton.input],
            role=ROLE_INPUT,
        )
    ]
    for name in skeleton.hidden:
        nodes.append(
            NodeSpec(
                name=name,
                node_type=types[name],
                shape=shapes[name],
                role=ROLE_HIDDEN,
                activation=str(rng.choice(config.hidden_activations)),
            )
        )
    nodes.append(
        NodeSpec(
            name=skeleton.output,
            node_type=types[skeleton.output],
            shape=shapes[skeleton.output],
            role=ROLE_OUTPUT,
            activation=config.output_activation,
            energy=config.output_energy,
        )
    )
    return GraphSpec(
        nodes=tuple(nodes), edges=tuple(edges), metadata={"unroll": config.unroll}
    )


def type_fits(node_type: str, in_degree: int) -> bool:
    """Whether ``node_type``'s slots can take ``in_degree`` incoming edges."""
    if node_type in UNSUPPORTED_TYPES:
        return False
    try:
        cls = registry.node_class(node_type)
    except Exception:
        return True  # unknown names are reported by validate, not hidden here
    slots = cls.get_slots().values()
    capacity = sum(float("inf") if s.is_multi_input else 1 for s in slots)
    required = sum(1 for s in slots if s.require_connected)
    minimum = max(required, MIN_IN_DEGREE.get(node_type, 1))
    return minimum <= in_degree <= capacity


def _pick_types(skeleton, sources_of, config, rng) -> Dict[str, str]:
    clamped = {node.role: node for node in config.clamped_nodes}
    types = {skeleton.input: INPUT_TYPE, skeleton.output: OUTPUT_TYPE}
    if "input" in clamped:
        types[skeleton.input] = clamped["input"].node_type
    if "output" in clamped:
        types[skeleton.output] = clamped["output"].node_type
    for name in skeleton.hidden:
        k = len(sources_of[name])
        feasible = [t for t in config.node_types if type_fits(t, k)]
        pool = feasible or list(config.node_types)  # validate will reject it
        types[name] = str(rng.choice(pool))
    return types


def _slot_names(node_type: str) -> List[str]:
    try:
        return list(registry.node_class(node_type).get_slots())
    except Exception:
        return ["in"]


def _pick_slots(names, sources_of, types, rng) -> List[EdgeSpec]:
    edges: List[EdgeSpec] = []
    for tgt in names:
        srcs = sources_of[tgt]
        if not srcs:
            continue
        slot_names = _slot_names(types[tgt])
        if slot_names == ["in"]:
            edges += [EdgeSpec(s, tgt, "in") for s in srcs]
            continue

        # Multi-slot nodes ("in" + "skip"): fill required slots first, give the
        # "in" slot at least one edge, then spread the rest at random.
        specs = registry.node_class(types[tgt]).get_slots()
        order = [srcs[i] for i in rng.permutation(len(srcs))]
        chosen: Dict[str, str] = {}
        for slot in [s for s in slot_names if specs[s].require_connected]:
            if order:
                chosen[order.pop(0)] = slot
        if "in" in specs and "in" not in chosen.values() and order:
            chosen[order.pop(0)] = "in"
        for src in order:
            open_slots = [
                s
                for s in slot_names
                if specs[s].is_multi_input or s not in chosen.values()
            ]
            chosen[src] = str(rng.choice(open_slots))
        edges += [EdgeSpec(s, tgt, chosen[s]) for s in srcs]
    return edges


def _pick_shapes(
    skeleton, names, types, edges, config, rng
) -> Dict[str, Tuple[int, ...]]:
    parent = {n: n for n in names}
    fixed = {
        skeleton.input: tuple(config.input_shape),
        skeleton.output: tuple(config.output_shape),
    }

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def group_fixed(root):
        found = {fixed[n] for n in fixed if find(n) == root}
        return found.pop() if len(found) == 1 else None

    for i, e in enumerate(edges):
        if (types[e.target], e.slot) not in SHAPE_PRESERVING_SLOTS:
            continue
        a, b = find(e.source), find(e.target)
        if a == b:
            continue
        fa, fb = group_fixed(a), group_fixed(b)
        if fa and fb and fa != fb:
            # Joining would force two different fixed shapes to match. A
            # LinearResidual skip edge can go through its weighted slot instead;
            # anything else is left for validate to report.
            if types[e.target] == "LinearResidual":
                edges[i] = EdgeSpec(e.source, e.target, "in")
            continue
        parent[a] = b

    group_shape: Dict[str, Tuple[int, ...]] = {}
    for n, shape in fixed.items():
        group_shape.setdefault(find(n), shape)
    for n in names:
        root = find(n)
        if root not in group_shape:
            group_shape[root] = (int(rng.choice(config.hidden_sizes)),)
    return {n: group_shape[find(n)] for n in names}
