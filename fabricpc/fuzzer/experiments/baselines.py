"""Sequential baselines for generated graphs.

Every generated graph is compared against a plain feedforward chain that has
the same input and output nodes::

    input -> b0 -> b1 -> ... -> output

The baseline is itself a :class:`~fabricpc.fuzzer.spec.GraphSpec`, built from
the pipeline's own :func:`~fabricpc.fuzzer.pipeline.chain_topology` skeleton
shape and ``Linear`` hidden nodes like
:func:`~fabricpc.fuzzer.pipeline.linear_assign`. Generated graphs and
baselines therefore go through the same :func:`~fabricpc.fuzzer.build.to_fabricpc`
build path and serialize the same way.

Nothing here depends on which stage produced the generated spec. Only its
input node, output node and hidden-node count/widths are read.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from fabricpc.fuzzer.spec import (
    ROLE_HIDDEN,
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    Skeleton,
)

BASELINE_KIND = "sequential"
BASELINE_PREFIX = "b"


def _hidden_in_order(spec: GraphSpec) -> List[NodeSpec]:
    """Hidden nodes in topological order (declaration order if the graph is cyclic)."""
    order = spec.topological_order() or spec.node_names
    hidden = {n.name: n for n in spec.nodes_with_role(ROLE_HIDDEN)}
    return [hidden[name] for name in order if name in hidden]


def _flat_width(shape: Sequence[int]) -> int:
    return int(math.prod(shape))


def sequential_baseline(
    spec: GraphSpec,
    hidden_sizes: Optional[Sequence[int]] = None,
    activation: Optional[str] = None,
) -> GraphSpec:
    """Build the sequential baseline for ``spec``.

    The baseline keeps ``spec``'s input and output nodes unchanged. By default
    it has one ``Linear`` hidden layer per hidden node of ``spec``, taken in
    topological order, with the same (flattened) width and activation, so the
    two models have comparable size and differ mainly in their connections.

    Args:
        spec: The generated graph.
        hidden_sizes: Override the hidden layer widths. ``[]`` gives a direct
            ``input -> output`` model.
        activation: Activation for every hidden layer. ``None`` copies each
            mirrored node's activation (or ``"relu"`` when it has none, or when
            ``hidden_sizes`` is given).

    Returns:
        A chain ``GraphSpec`` whose metadata records ``baseline_kind``,
        ``baseline_of`` (the generated spec's fingerprint) and, when present,
        ``graph_index``.
    """
    mirrored = _hidden_in_order(spec)
    if hidden_sizes is None:
        sizes = [_flat_width(n.shape) for n in mirrored]
        activations = [activation or n.activation or "relu" for n in mirrored]
    else:
        sizes = [int(s) for s in hidden_sizes]
        if any(s < 1 for s in sizes):
            raise ValueError(f"hidden_sizes must be positive, got {list(sizes)}")
        activations = [activation or "relu"] * len(sizes)

    inp, out = spec.input_node, spec.output_node
    names = tuple(f"{BASELINE_PREFIX}{i}" for i in range(len(sizes)))
    chain = (inp.name, *names, out.name)
    skeleton = Skeleton(
        hidden=names,
        edges=tuple(zip(chain[:-1], chain[1:])),
        input=inp.name,
        output=out.name,
    )

    hidden_nodes = tuple(
        NodeSpec(
            name=name,
            node_type="Linear",
            shape=(size,),
            role=ROLE_HIDDEN,
            activation=act,
        )
        for name, size, act in zip(names, sizes, activations)
    )
    metadata: Dict[str, Any] = {
        "baseline_kind": BASELINE_KIND,
        "baseline_of": spec.fingerprint(),
    }
    for key in ("graph_index", "unroll"):
        if key in spec.metadata:
            metadata[key] = spec.metadata[key]

    baseline = GraphSpec(
        nodes=(inp, *hidden_nodes, out),
        edges=tuple(EdgeSpec(s, t, "in") for s, t in skeleton.edges),
        metadata=metadata,
    )
    baseline.check()
    return baseline


def describe_architecture(spec: GraphSpec) -> Dict[str, Any]:
    """Layer sizes and connections of ``spec`` as plain JSON-safe data.

    Used to record what each evaluated model looked like (SR-F-12.9), for the
    baseline and the generated graph alike.

    Returns:
        ``{"layers": [{"name", "node_type", "shape", "activation", "role"}, ...],
        "connections": ["src->dst:slot", ...]}`` with layers in topological
        order when the graph is acyclic.
    """
    order = spec.topological_order() or spec.node_names
    layers = []
    for name in order:
        n = spec.node(name)
        layers.append(
            {
                "name": n.name,
                "node_type": n.node_type,
                "shape": list(n.shape),
                "activation": n.activation,
                "role": n.role,
            }
        )
    return {"layers": layers, "connections": [e.key for e in spec.edges]}


def layer_sizes(spec: GraphSpec) -> List[int]:
    """Flattened width of every node, in the order of :func:`describe_architecture`."""
    order = spec.topological_order() or spec.node_names
    return [_flat_width(spec.node(name).shape) for name in order]
