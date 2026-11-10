"""Generation settings: what kind of graphs the researcher wants.

A ``GeneratorConfig`` can be built in Python, loaded from a JSON or TOML file,
or adjusted from the command line. :meth:`GeneratorConfig.validate` rejects
impossible settings up front with a message naming the parameter at fault,
so the generator never starts on a request it cannot satisfy.

Example TOML file::

    num_graphs = 20
    seed = 7
    num_nodes = [4, 8]          # hidden nodes, inclusive range
    in_degree = [1, 3]
    node_types = ["Linear", "StorkeyHopfield"]
    allow_cycles = true
    input_shape = [784]
    output_shape = [10]
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from fabricpc.fuzzer import registry

IntRange = Tuple[int, int]
IntOrRange = Union[int, IntRange]

CLAMPED_ROLES = ("input", "output")


class ConfigError(ValueError):
    """An invalid generation setting.

    Attributes:
        param: Name of the offending parameter.
    """

    def __init__(self, param: str, message: str):
        self.param = param
        super().__init__(f"{param}: {message}")


def _as_range(param: str, value: Any) -> IntRange:
    """Normalize ``5`` or ``[2, 5]`` to an inclusive ``(lo, hi)`` tuple."""
    if isinstance(value, bool):
        raise ConfigError(param, f"expected an int or [min, max], got {value!r}")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = value
        if all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi)):
            return (lo, hi)
    raise ConfigError(param, f"expected an int or [min, max], got {value!r}")


def _as_str_tuple(param: str, value: Any) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise ConfigError(param, f"expected a list of strings, got {value!r}")


def _as_int_tuple(param: str, value: Any) -> Tuple[int, ...]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        return tuple(value)
    raise ConfigError(param, f"expected a list of ints, got {value!r}")


@dataclass(frozen=True)
class ClampedNode:
    """A fixed input or output node shared by every graph in a batch.

    Clamped nodes are how a batch keeps one task map: the input and output keep
    the same name, type, shape, in-degree and out-degree, so every graph can be
    trained and scored on the same data.

    Attributes:
        role: ``"input"`` or ``"output"``.
        node_type: Registry key of the FabricPC node class (for example
            ``"Linear"`` or ``"IdentityNode"``).
        shape: Shape of one sample, excluding the batch dimension.
        in_degree: Exact number of incoming edges.
        out_degree: Exact number of outgoing edges.
    """

    role: str
    node_type: str
    shape: Tuple[int, ...]
    in_degree: int
    out_degree: int

    def __post_init__(self):
        object.__setattr__(self, "shape", _as_int_tuple("clamped_nodes", self.shape))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form, with the shape as a list."""
        return {
            "role": self.role,
            "node_type": self.node_type,
            "shape": list(self.shape),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ClampedNode":
        """Build from a mapping, naming ``clamped_nodes`` on any problem."""
        if not isinstance(data, Mapping):
            raise ConfigError("clamped_nodes", f"expected an object, got {data!r}")
        allowed = {"role", "node_type", "shape", "in_degree", "out_degree"}
        missing = allowed - set(data)
        if missing:
            raise ConfigError("clamped_nodes", f"entry is missing {sorted(missing)}")
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(
                "clamped_nodes", f"entry has unknown keys {sorted(unknown)}"
            )
        return cls(**dict(data))


def _as_clamped_nodes(value: Any) -> Tuple[ClampedNode, ...]:
    """Normalize a list of entries, or a role-keyed mapping, to a tuple."""
    if isinstance(value, Mapping):
        entries = []
        for role, entry in value.items():
            if not isinstance(entry, Mapping):
                raise ConfigError("clamped_nodes", f"entry {role!r} must be an object")
            merged = dict(entry)
            merged.setdefault("role", role)
            entries.append(merged)
        value = entries
    if isinstance(value, ClampedNode):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ConfigError("clamped_nodes", f"expected a list of entries, got {value!r}")
    return tuple(
        entry if isinstance(entry, ClampedNode) else ClampedNode.from_dict(entry)
        for entry in value
    )


@dataclass(frozen=True)
class GeneratorConfig:
    """Settings that control which graphs the fuzzer generates.

    Attributes:
        num_graphs: How many distinct graphs to return.
        seed: Master seed. Together with the graph's index it fully determines
            each graph (see :mod:`fabricpc.fuzzer.seeding`).
        num_nodes: Number of hidden nodes (excluding input and output), as an
            int or inclusive ``[min, max]`` range.
        num_edges: Total number of edges, or ``None`` to let the topology
            generator decide.
        in_degree: Allowed number of incoming edges per non-input node.
        out_degree: Allowed number of outgoing edges per non-clamped (hidden)
            node, or ``None`` for no limit. The input's and output's degrees are
            fixed by ``clamped_nodes``.
        node_types: Node types the generator may use for hidden nodes.
        allow_cycles: Allow feedback edges (recurrent/cyclic graphs).
        unroll: Cycle traversals used when building a cyclic graph.
        input_shape: Shape of the input node (one data sample).
        output_shape: Shape of the output node (one target).
        hidden_sizes: Feature sizes to choose from for hidden nodes.
        hidden_activations: Activations to choose from for hidden nodes.
        output_activation: Activation of the output node.
        output_energy: Energy functional of the output node.
        max_attempts: Generation attempts per graph before giving up.
        constraints: Free-form constraints supplied by a person or by an
            automated feedback loop. Unrecognized keys are kept and passed to
            every stage, which reads the ones it understands.
        clamped_nodes: Zero or two :class:`ClampedNode` entries that fix the
            input and output node for every graph in the batch. When present,
            their shapes also set ``input_shape`` and ``output_shape``.
    """

    num_graphs: int = 1
    seed: int = 0
    num_nodes: IntOrRange = (2, 6)
    num_edges: Optional[IntOrRange] = None
    in_degree: IntOrRange = (1, 3)
    out_degree: Optional[IntOrRange] = None
    node_types: Tuple[str, ...] = ("Linear",)
    allow_cycles: bool = False
    unroll: int = 2
    input_shape: Tuple[int, ...] = (784,)
    output_shape: Tuple[int, ...] = (10,)
    hidden_sizes: Tuple[int, ...] = (64, 128, 256)
    hidden_activations: Tuple[str, ...] = ("relu", "tanh", "sigmoid")
    output_activation: str = "softmax"
    output_energy: str = "cross_entropy"
    max_attempts: int = 1000
    constraints: Mapping[str, Any] = field(default_factory=dict, hash=False)
    clamped_nodes: Tuple[ClampedNode, ...] = ()

    def __post_init__(self):
        # Normalize list/int inputs so equal settings compare equal.
        norm = {
            "num_nodes": _as_range("num_nodes", self.num_nodes),
            "in_degree": _as_range("in_degree", self.in_degree),
            "node_types": _as_str_tuple("node_types", self.node_types),
            "input_shape": _as_int_tuple("input_shape", self.input_shape),
            "output_shape": _as_int_tuple("output_shape", self.output_shape),
            "hidden_sizes": _as_int_tuple("hidden_sizes", self.hidden_sizes),
            "hidden_activations": _as_str_tuple(
                "hidden_activations", self.hidden_activations
            ),
            "constraints": dict(self.constraints),
            "clamped_nodes": _as_clamped_nodes(self.clamped_nodes),
        }
        if self.num_edges is not None:
            norm["num_edges"] = _as_range("num_edges", self.num_edges)
        if self.out_degree is not None:
            norm["out_degree"] = _as_range("out_degree", self.out_degree)
        for key, value in norm.items():
            object.__setattr__(self, key, value)
        # A clamped node's shape is the task map's shape; keep the top-level
        # fields in step instead of letting them disagree.
        for node in norm["clamped_nodes"]:
            if node.role == "input":
                object.__setattr__(self, "input_shape", node.shape)
            elif node.role == "output":
                object.__setattr__(self, "output_shape", node.shape)

    # -------------------------------------------------------------- validation
    def validate(self) -> "GeneratorConfig":
        """Check every setting and their combinations.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            ConfigError: The first invalid setting, naming the parameter.
        """
        _check_int("num_graphs", self.num_graphs, minimum=1)
        _check_int("seed", self.seed, minimum=0)
        _check_int("unroll", self.unroll, minimum=1)
        _check_int("max_attempts", self.max_attempts, minimum=1)
        if not isinstance(self.allow_cycles, bool):
            raise ConfigError(
                "allow_cycles", f"expected true/false, got {self.allow_cycles!r}"
            )

        n_lo, n_hi = self.num_nodes
        _check_range("num_nodes", self.num_nodes, minimum=0)
        _check_range("in_degree", self.in_degree, minimum=1)
        d_lo, d_hi = self.in_degree

        # Every node except the input must be able to receive in_degree[0]
        # edges from distinct other nodes. With no cycles, the k-th node in
        # topological order has only k possible sources, so the first hidden
        # node can have at most one input.
        if not self.allow_cycles and d_lo > 1:
            raise ConfigError(
                "in_degree",
                f"minimum {d_lo} is impossible without cycles: the first hidden "
                "node can only receive an edge from the input node. Use a "
                "minimum of 1 or set allow_cycles = true.",
            )
        if d_lo > n_hi + 1:
            raise ConfigError(
                "in_degree",
                f"minimum {d_lo} exceeds the number of possible sources "
                f"({n_hi + 1} with num_nodes max {n_hi})",
            )

        if self.out_degree is not None:
            _check_range("out_degree", self.out_degree, minimum=1)
            o_lo, o_hi = self.out_degree
            if not self.allow_cycles and o_lo > 1:
                raise ConfigError(
                    "out_degree",
                    f"minimum {o_lo} is impossible without cycles: the last "
                    "hidden node can only feed the output node. Use a minimum "
                    "of 1 or set allow_cycles = true.",
                )
            if o_lo > n_hi:
                raise ConfigError(
                    "out_degree",
                    f"minimum {o_lo} exceeds the number of possible targets "
                    f"({n_hi} with num_nodes max {n_hi})",
                )
            if o_hi > n_hi:
                raise ConfigError(
                    "out_degree",
                    f"maximum {o_hi} exceeds the number of possible targets "
                    f"({n_hi} with num_nodes max {n_hi})",
                )

        clamped = {node.role: node for node in self.clamped_nodes}
        output_in_hi = clamped["output"].in_degree if "output" in clamped else d_hi
        output_in_lo = (
            clamped["output"].in_degree if "output" in clamped else self.in_degree[0]
        )

        if self.num_edges is not None:
            _check_range("num_edges", self.num_edges, minimum=1)
            e_lo, e_hi = self.num_edges
            # Each hidden node and the output need at least one in-edge.
            min_needed = n_lo + output_in_lo
            if e_hi < min_needed:
                raise ConfigError(
                    "num_edges",
                    f"maximum {e_hi} is too small: with at least {n_lo} hidden "
                    f"nodes, every hidden node and the output need an incoming "
                    f"edge, so at least {min_needed} edges are required",
                )
            max_possible = n_hi * d_hi + output_in_hi
            if not self.allow_cycles:
                total = n_hi + 2
                max_possible = min(max_possible, total * (total - 1) // 2)
            if e_lo > max_possible:
                raise ConfigError(
                    "num_edges",
                    f"minimum {e_lo} exceeds the {max_possible} edges possible "
                    f"with num_nodes max {n_hi}, in_degree max {d_hi}"
                    + ("" if self.allow_cycles else " and no cycles"),
                )

        if not self.node_types:
            raise ConfigError("node_types", "at least one node type is required")
        for name in self.node_types:
            _check_registered("node_types", registry.NODE_TYPES, name)
        if not self.hidden_activations:
            raise ConfigError("hidden_activations", "at least one is required")
        for name in self.hidden_activations:
            _check_registered("hidden_activations", registry.ACTIVATIONS, name)
        _check_registered(
            "output_activation", registry.ACTIVATIONS, self.output_activation
        )
        _check_registered("output_energy", registry.ENERGIES, self.output_energy)

        for param in ("input_shape", "output_shape"):
            shape = getattr(self, param)
            if not shape or any(d <= 0 for d in shape):
                raise ConfigError(param, f"dims must be positive, got {list(shape)}")
        if not self.hidden_sizes or any(s <= 0 for s in self.hidden_sizes):
            raise ConfigError(
                "hidden_sizes", f"sizes must be positive, got {list(self.hidden_sizes)}"
            )

        if self.clamped_nodes:
            if len(self.clamped_nodes) != 2:
                raise ConfigError(
                    "clamped_nodes",
                    "expected exactly two entries, one input and one output, "
                    f"got {len(self.clamped_nodes)}",
                )
            by_role: Dict[str, ClampedNode] = {}
            for node in self.clamped_nodes:
                if node.role not in CLAMPED_ROLES:
                    raise ConfigError(
                        "clamped_nodes",
                        f"role {node.role!r} is not one of {list(CLAMPED_ROLES)}",
                    )
                if node.role in by_role:
                    raise ConfigError(
                        "clamped_nodes", f"role {node.role!r} is set twice"
                    )
                by_role[node.role] = node
                _check_registered("clamped_nodes", registry.NODE_TYPES, node.node_type)
                if not node.shape or any(d <= 0 for d in node.shape):
                    raise ConfigError(
                        "clamped_nodes",
                        f"{node.role} shape dims must be positive, "
                        f"got {list(node.shape)}",
                    )
                _check_int("clamped_nodes", node.in_degree, minimum=0)
                _check_int("clamped_nodes", node.out_degree, minimum=0)
            if set(by_role) != set(CLAMPED_ROLES):
                missing = sorted(set(CLAMPED_ROLES) - set(by_role))
                raise ConfigError("clamped_nodes", f"missing role(s) {missing}")
            input_node = by_role["input"]
            output_node = by_role["output"]
            if input_node.in_degree != 0:
                raise ConfigError(
                    "clamped_nodes",
                    "the input node cannot receive edges; set its in_degree to 0",
                )
            if input_node.out_degree < 1:
                raise ConfigError(
                    "clamped_nodes",
                    "the input node must feed at least one node; set its "
                    "out_degree to 1 or more",
                )
            if input_node.out_degree > n_hi + 1:
                raise ConfigError(
                    "clamped_nodes",
                    f"input out_degree {input_node.out_degree} exceeds the "
                    f"{n_hi + 1} possible targets",
                )
            if output_node.out_degree != 0:
                raise ConfigError(
                    "clamped_nodes",
                    "the output node cannot feed other nodes; set its "
                    "out_degree to 0",
                )
            if output_node.in_degree < 1:
                raise ConfigError(
                    "clamped_nodes",
                    "the output node must receive at least one edge; set its "
                    "in_degree to 1 or more",
                )
            if output_node.in_degree > n_hi + 1:
                raise ConfigError(
                    "clamped_nodes",
                    f"output in_degree {output_node.in_degree} exceeds the "
                    f"{n_hi + 1} possible sources",
                )

        try:
            json.dumps(self.constraints)
        except TypeError as exc:
            raise ConfigError(
                "constraints", f"must be JSON-serializable ({exc})"
            ) from None
        return self

    # ------------------------------------------------------------- conversion
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict; ranges and tuples become lists."""
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name == "clamped_nodes":
                value = [node.to_dict() for node in value]
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Mapping):
                value = json.loads(json.dumps(dict(value)))
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratorConfig":
        """Build a config from a dict, rejecting unknown keys.

        Raises:
            ConfigError: An unknown key (with a "did you mean" hint) or a value
                of the wrong type.
        """
        if not isinstance(data, Mapping):
            raise ConfigError(
                "<config>", f"expected a mapping, got {type(data).__name__}"
            )
        names = [f.name for f in dataclasses.fields(cls)]
        for key in data:
            if key not in names:
                close = difflib.get_close_matches(key, names, n=1)
                hint = f" Did you mean {close[0]!r}?" if close else ""
                raise ConfigError(
                    key, f"unknown setting.{hint} Valid settings: {names}"
                )
        return cls(**dict(data))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "GeneratorConfig":
        """Load a config from a ``.json`` or ``.toml`` file.

        Raises:
            ConfigError: Unsupported extension, unparsable file or bad setting.
            FileNotFoundError: ``path`` does not exist.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                data = json.loads(text)
            elif suffix == ".toml":
                data = tomllib.loads(text)
            else:
                raise ConfigError(
                    "<config>", f"unsupported file type {suffix!r}; use .json or .toml"
                )
        except (json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError("<config>", f"could not parse {path}: {exc}") from None
        return cls.from_dict(data)

    def to_file(self, path: Union[str, Path]) -> None:
        """Write the config as JSON (TOML writing is not in the standard library)."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    def with_overrides(self, **overrides: Any) -> "GeneratorConfig":
        """Return a copy with some settings replaced (unknown names rejected)."""
        merged = self.to_dict()
        merged.update(overrides)
        return GeneratorConfig.from_dict(merged)


def parse_override(text: str) -> Tuple[str, Any]:
    """Parse a ``key=value`` command-line override.

    The value is read as JSON when possible (``num_nodes=[3,5]``,
    ``allow_cycles=true``), otherwise as a plain string (``output_energy=gaussian``).

    Raises:
        ConfigError: ``text`` has no ``=``.
    """
    if "=" not in text:
        raise ConfigError(text, "override must look like key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _check_int(param: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(param, f"expected an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(param, f"must be >= {minimum}, got {value}")


def _check_range(param: str, rng: IntRange, minimum: int) -> None:
    lo, hi = rng
    if lo < minimum:
        raise ConfigError(param, f"minimum must be >= {minimum}, got {lo}")
    if lo > hi:
        raise ConfigError(param, f"minimum {lo} is greater than maximum {hi}")


def _check_registered(param: str, table: Mapping[str, Any], name: str) -> None:
    if name not in table:
        close = difflib.get_close_matches(name, table, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ConfigError(
            param, f"{name!r} is not registered.{hint} Options: {sorted(table)}"
        )
