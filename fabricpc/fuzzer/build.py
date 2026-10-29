"""Turn a ``GraphSpec`` into a FabricPC ``GraphStructure``.

This is the single bridge from the fuzzer's plain-data world to FabricPC
objects. Errors raised here are wrapped in :class:`BuildError`, which records
the node or edge involved when it can be identified, so a failing generated
graph can be traced back to its cause.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from fabricpc.core.inference import InferenceBase, InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_assembly.graph_construction import GraphStructure
from fabricpc.nodes import NodeBase
from fabricpc.fuzzer import registry
from fabricpc.fuzzer.spec import GraphSpec, NodeSpec, SpecError

DEFAULT_UNROLL = 2


class BuildError(RuntimeError):
    """FabricPC rejected a spec.

    Attributes:
        where: Node name or edge key implicated, or ``None`` if unknown.
        stage: ``"spec"``, ``"node"`` or ``"graph"``: which step failed.
    """

    def __init__(self, message: str, where: Optional[str] = None, stage: str = "graph"):
        self.where = where
        self.stage = stage
        prefix = f"[{where}] " if where else ""
        super().__init__(prefix + message)


def _tuplify(value: Any) -> Any:
    """JSON stores tuples as lists; FabricPC node configs expect tuples."""
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuplify(v) for k, v in value.items()}
    return value


def build_node(node: NodeSpec) -> NodeBase:
    """Instantiate the FabricPC node described by ``node``.

    Raises:
        BuildError: Unknown type/activation/energy, or the node constructor
            rejected the arguments.
    """
    try:
        cls = registry.node_class(node.node_type)
        kwargs: Dict[str, Any] = {k: _tuplify(v) for k, v in node.config.items()}
        activation = registry.make_activation(node.activation, node.activation_params)
        energy = registry.make_energy(node.energy, node.energy_params)
        if activation is not None:
            kwargs["activation"] = activation
        if energy is not None:
            kwargs["energy"] = energy
        return cls(shape=tuple(node.shape), name=node.name, **kwargs)
    except Exception as exc:
        raise BuildError(
            f"{type(exc).__name__}: {exc}", where=node.name, stage="node"
        ) from exc


def to_fabricpc(
    spec: GraphSpec,
    inference: Optional[InferenceBase] = None,
    unroll: Optional[int] = None,
) -> GraphStructure:
    """Build the FabricPC graph described by ``spec``.

    Args:
        spec: The graph to build. ``spec.check()`` is run first.
        inference: Inference algorithm; defaults to
            ``InferenceSGD(eta_infer=0.05, infer_steps=20)``.
        unroll: Cycle traversals for cyclic graphs. Defaults to
            ``spec.metadata["unroll"]`` or 2. Ignored for acyclic graphs.

    Returns:
        A ``GraphStructure`` ready for ``initialize_params`` and ``train``.

    Raises:
        BuildError: The spec is malformed or FabricPC rejected it.
    """
    try:
        spec.check()
    except SpecError as exc:
        raise BuildError(str(exc), where=exc.where, stage="spec") from exc

    nodes = {n.name: build_node(n) for n in spec.nodes}
    edges = []
    for e in spec.edges:
        try:
            target = nodes[e.target].slot(e.slot)
        except KeyError as exc:
            available = list(type(nodes[e.target]).get_slots())
            raise BuildError(
                f"edge targets non-existent slot {e.slot!r} on "
                f"{spec.node(e.target).node_type} node {e.target!r}; "
                f"available slots: {available}",
                where=e.key,
                stage="graph",
            ) from exc
        edges.append(Edge(source=nodes[e.source], target=target))
    if inference is None:
        inference = InferenceSGD(eta_infer=0.05, infer_steps=20)
    if spec.is_cyclic:
        unroll = unroll or int(spec.metadata.get("unroll", DEFAULT_UNROLL))
    else:
        unroll = None

    try:
        return graph(
            nodes=list(nodes.values()),
            edges=edges,
            task_map=TaskMap(
                x=nodes[spec.input_node.name], y=nodes[spec.output_node.name]
            ),
            inference=inference,
            unroll=unroll,
        )
    except Exception as exc:
        raise BuildError(
            f"{type(exc).__name__}: {exc}", where=_locate(str(exc), spec), stage="graph"
        ) from exc


_QUOTED = re.compile(r"'([^']+)'")


def _locate(message: str, spec: GraphSpec) -> Optional[str]:
    """Best-effort: find the first edge key or node name quoted in an error message."""
    edge_keys = {e.key for e in spec.edges}
    names = set(spec.node_names)
    quoted = _QUOTED.findall(message)
    for q in quoted:
        if q in edge_keys:
            return q
    for q in quoted:
        if q in names:
            return q
    return None
