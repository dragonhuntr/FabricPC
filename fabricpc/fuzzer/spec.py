"""Framework-neutral description of a generated graph.

``GraphSpec`` is the contract between the fuzzer's stages:

- the topology generator decides which nodes exist and how they connect,
- the typing/validity stage fills in node types, shapes and edge slots,
- the evaluator turns a spec into a FabricPC graph and trains it,
- the serializer saves and reloads it so a result can be shared and rerun.

A spec holds only plain data (strings, ints, tuples, JSON-safe dicts) so it can
be saved as JSON and compared or hashed without importing JAX. Converting a
spec into a FabricPC ``GraphStructure`` lives in :mod:`fabricpc.fuzzer.build`.
"""

from __future__ import annotations

import copy
import graphlib
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

ROLE_INPUT = "input"
ROLE_HIDDEN = "hidden"
ROLE_OUTPUT = "output"
ROLES = (ROLE_INPUT, ROLE_HIDDEN, ROLE_OUTPUT)


INPUT_NAME = "input"
OUTPUT_NAME = "output"


class SpecError(ValueError):
    """A ``GraphSpec`` is structurally malformed.

    Attributes:
        where: The node name or edge key at fault, or ``None`` when the
            problem concerns the graph as a whole.
    """

    def __init__(self, message: str, where: Optional[str] = None):
        self.where = where
        prefix = f"[{where}] " if where else ""
        super().__init__(prefix + message)


@dataclass(frozen=True)
class NodeSpec:
    """One node of a generated graph.

    Attributes:
        name: Unique node name.
        node_type: Key in :data:`fabricpc.fuzzer.registry.NODE_TYPES`,
            e.g. ``"Linear"``.
        shape: Output shape excluding the batch dimension.
        role: ``"input"``, ``"hidden"`` or ``"output"``. The input node is
            clamped to data (``TaskMap.x``) and the output node to targets
            (``TaskMap.y``).
        activation: Key in ``ACTIVATIONS``, or ``None`` for the node type's
            own default.
        activation_params: Keyword arguments for the activation constructor.
        energy: Key in ``ENERGIES``, or ``None`` for the node type's default.
        energy_params: Keyword arguments for the energy constructor.
        config: Extra node constructor keyword arguments (JSON-safe), for
            example ``{"use_bias": False}``.
    """

    name: str
    node_type: str
    shape: Tuple[int, ...]
    role: str = ROLE_HIDDEN
    activation: Optional[str] = None
    activation_params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    energy: Optional[str] = None
    energy_params: Mapping[str, Any] = field(default_factory=dict, hash=False)
    config: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict. Optional fields at their default are omitted."""
        out: Dict[str, Any] = {
            "name": self.name,
            "node_type": self.node_type,
            "shape": list(self.shape),
            "role": self.role,
        }
        if self.activation is not None:
            out["activation"] = self.activation
        if self.activation_params:
            out["activation_params"] = copy.deepcopy(dict(self.activation_params))
        if self.energy is not None:
            out["energy"] = self.energy
        if self.energy_params:
            out["energy_params"] = copy.deepcopy(dict(self.energy_params))
        if self.config:
            out["config"] = copy.deepcopy(dict(self.config))
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NodeSpec":
        """Build a ``NodeSpec`` from :meth:`to_dict` output.

        Raises:
            SpecError: A required key is missing or an unknown key is present.
        """
        _check_keys(data, required={"name", "node_type", "shape"}, cls=cls)
        return cls(
            name=data["name"],
            node_type=data["node_type"],
            shape=tuple(data["shape"]),
            role=data.get("role", ROLE_HIDDEN),
            activation=data.get("activation"),
            activation_params=dict(data.get("activation_params", {})),
            energy=data.get("energy"),
            energy_params=dict(data.get("energy_params", {})),
            config=dict(data.get("config", {})),
        )


@dataclass(frozen=True)
class EdgeSpec:
    """A directed edge from ``source`` into ``target``'s input ``slot``."""

    source: str
    target: str
    slot: str = "in"

    @property
    def key(self) -> str:
        """Edge key in FabricPC's format: ``"source->target:slot"``."""
        return f"{self.source}->{self.target}:{self.slot}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict."""
        return {"source": self.source, "target": self.target, "slot": self.slot}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EdgeSpec":
        """Build an ``EdgeSpec`` from :meth:`to_dict` output."""
        _check_keys(data, required={"source", "target"}, cls=cls)
        return cls(
            source=data["source"], target=data["target"], slot=data.get("slot", "in")
        )


@dataclass(frozen=True)
class GraphSpec:
    """A complete generated graph: nodes, edges and provenance metadata.

    ``metadata`` records how the graph was produced (master seed, graph index,
    per-graph seed, generator config, fuzzer version). It is saved with the
    graph but excluded from :meth:`fingerprint` and from structural equality
    checks, so two runs that produce the same structure compare equal.
    """

    nodes: Tuple[NodeSpec, ...]
    edges: Tuple[EdgeSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))

    # ------------------------------------------------------------------ lookup
    def node(self, name: str) -> NodeSpec:
        """Return the node called ``name``.

        Raises:
            KeyError: No node has that name.
        """
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(f"No node named {name!r}")

    @property
    def node_names(self) -> Tuple[str, ...]:
        """Node names in declaration order."""
        return tuple(n.name for n in self.nodes)

    def nodes_with_role(self, role: str) -> Tuple[NodeSpec, ...]:
        """All nodes with the given role, in declaration order."""
        return tuple(n for n in self.nodes if n.role == role)

    @property
    def input_node(self) -> NodeSpec:
        """The single input node (assumes :meth:`check` passed)."""
        return self.nodes_with_role(ROLE_INPUT)[0]

    @property
    def output_node(self) -> NodeSpec:
        """The single output node (assumes :meth:`check` passed)."""
        return self.nodes_with_role(ROLE_OUTPUT)[0]

    def in_edges(self, name: str) -> Tuple[EdgeSpec, ...]:
        """Edges whose target is ``name``."""
        return tuple(e for e in self.edges if e.target == name)

    def out_edges(self, name: str) -> Tuple[EdgeSpec, ...]:
        """Edges whose source is ``name``."""
        return tuple(e for e in self.edges if e.source == name)

    def in_degree(self, name: str) -> int:
        """Number of incoming edges of ``name``."""
        return len(self.in_edges(name))

    def out_degree(self, name: str) -> int:
        """Number of outgoing edges of ``name``."""
        return len(self.out_edges(name))

    # ---------------------------------------------------------------- topology
    def predecessors(self) -> Dict[str, List[str]]:
        """Map each node name to the names of its distinct source nodes."""
        preds: Dict[str, List[str]] = {n: [] for n in self.node_names}
        for e in self.edges:
            if e.source not in preds[e.target]:
                preds[e.target].append(e.source)
        return preds

    @property
    def is_cyclic(self) -> bool:
        """True if the edges contain a directed cycle."""
        return self.topological_order() is None

    def topological_order(self) -> Optional[Tuple[str, ...]]:
        """Node names in a topological order, or ``None`` if the graph is cyclic."""
        sorter = graphlib.TopologicalSorter(self.predecessors())
        try:
            return tuple(sorter.static_order())
        except graphlib.CycleError:
            return None

    # -------------------------------------------------------------- validation
    def check(self) -> None:
        """Check structural well-formedness.

        This covers only what any consumer of a spec relies on: unique names,
        known roles, positive shapes, exactly one input and one output node,
        edges that reference existing nodes, no self-edges, no duplicate edges,
        and no edges into the input node. Whether node types, shapes and slots
        are compatible with each other is decided by the validity stage.

        Raises:
            SpecError: The first problem found, naming the node or edge at fault.
        """
        seen = set()
        for n in self.nodes:
            if not isinstance(n.name, str) or not n.name:
                raise SpecError("node name must be a non-empty string", repr(n.name))
            if n.name in seen:
                raise SpecError("duplicate node name", n.name)
            seen.add(n.name)
            if n.role not in ROLES:
                raise SpecError(f"role must be one of {ROLES}, got {n.role!r}", n.name)
            if not n.shape or any(d <= 0 for d in n.shape):
                raise SpecError(
                    f"shape must be non-empty with positive dims, got {n.shape}", n.name
                )
        for role in (ROLE_INPUT, ROLE_OUTPUT):
            count = len(self.nodes_with_role(role))
            if count != 1:
                raise SpecError(f"expected exactly one {role!r} node, found {count}")

        input_name = self.input_node.name
        keys = set()
        for e in self.edges:
            if e.source not in seen:
                raise SpecError(f"source node {e.source!r} does not exist", e.key)
            if e.target not in seen:
                raise SpecError(f"target node {e.target!r} does not exist", e.key)
            if e.source == e.target:
                raise SpecError("self-edges are not allowed", e.key)
            if e.target == input_name:
                raise SpecError("the input node cannot receive edges", e.key)
            if e.key in keys:
                raise SpecError("duplicate edge", e.key)
            keys.add(e.key)

    # ---------------------------------------------------------- serialization
    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Return a JSON-safe dict (see :mod:`fabricpc.fuzzer.serialize`)."""
        out: Dict[str, Any] = {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
        if include_metadata:
            out["metadata"] = copy.deepcopy(dict(self.metadata))
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphSpec":
        """Build a ``GraphSpec`` from :meth:`to_dict` output."""
        _check_keys(data, required={"nodes", "edges"}, cls=cls)
        return cls(
            nodes=tuple(NodeSpec.from_dict(n) for n in data["nodes"]),
            edges=tuple(EdgeSpec.from_dict(e) for e in data["edges"]),
            metadata=dict(data.get("metadata", {})),
        )

    def canonical_dict(self) -> Dict[str, Any]:
        """Structure only, with nodes and edges sorted, metadata dropped."""
        d = self.to_dict(include_metadata=False)
        d["nodes"].sort(key=lambda n: n["name"])
        d["edges"].sort(key=lambda e: (e["source"], e["target"], e["slot"]))
        return d

    def fingerprint(self) -> str:
        """SHA-256 of :meth:`canonical_dict`.

        Equal for two specs with the same nodes and edges regardless of
        declaration order or metadata. It is name-sensitive: relabeling nodes
        changes it, so it detects exact duplicates, not isomorphic graphs.
        """
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def same_structure(self, other: "GraphSpec") -> bool:
        """True if ``other`` has the same nodes and edges, ignoring order and metadata."""
        return self.canonical_dict() == other.canonical_dict()

    def with_metadata(self, **updates: Any) -> "GraphSpec":
        """Return a copy with ``updates`` merged into ``metadata``."""
        merged = dict(self.metadata)
        merged.update(updates)
        return GraphSpec(nodes=self.nodes, edges=self.edges, metadata=merged)

    def __iter__(self) -> Iterator[NodeSpec]:
        return iter(self.nodes)

    def summary(self) -> str:
        """Short human-readable description, used by the CLI."""
        lines = [
            f"nodes={len(self.nodes)} edges={len(self.edges)} "
            f"cyclic={self.is_cyclic} fingerprint={self.fingerprint()[:12]}"
        ]
        for n in self.nodes:
            preds = ", ".join(f"{e.source}:{e.slot}" for e in self.in_edges(n.name))
            act = f" act={n.activation}" if n.activation else ""
            lines.append(
                f"  {n.name:<12} {n.node_type:<16} shape={n.shape!s:<10} "
                f"{n.role:<6}{act}  <- [{preds}]"
            )
        return "\n".join(lines)


def _check_keys(data: Mapping[str, Any], required: set, cls: type) -> None:
    if not isinstance(data, Mapping):
        raise SpecError(f"{cls.__name__} must be a JSON object, got {type(data)}")
    allowed = {f for f in cls.__dataclass_fields__}
    missing = required - set(data)
    if missing:
        raise SpecError(f"{cls.__name__} is missing key(s) {sorted(missing)}")
    unknown = set(data) - allowed
    if unknown:
        raise SpecError(
            f"{cls.__name__} has unknown key(s) {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )


@dataclass(frozen=True)
class Skeleton:
    """Untyped topology produced by the topology stage.

    Attributes:
        hidden: Hidden node names, in the order they should be declared.
        edges: Directed ``(source, target)`` pairs over ``input``, ``output``
            and the hidden names. Slots are chosen later by the assign stage.
        input: Name of the input node.
        output: Name of the output node.
    """

    hidden: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    input: str = INPUT_NAME
    output: str = OUTPUT_NAME

    @property
    def all_nodes(self) -> Tuple[str, ...]:
        """Input, hidden nodes, then output."""
        return (self.input, *self.hidden, self.output)
