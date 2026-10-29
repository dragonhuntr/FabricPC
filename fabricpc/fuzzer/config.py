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
    """

    num_graphs: int = 1
    seed: int = 0
    num_nodes: IntOrRange = (2, 6)
    num_edges: Optional[IntOrRange] = None
    in_degree: IntOrRange = (1, 3)
    node_types: Tuple[str, ...] = ("Linear",)
    allow_cycles: bool = False
    unroll: int = 2
    input_shape: Tuple[int, ...] = (784,)
    output_shape: Tuple[int, ...] = (10,)
    hidden_sizes: Tuple[int, ...] = (64, 128, 256)
    hidden_activations: Tuple[str, ...] = ("relu", "tanh", "sigmoid")
    output_activation: str = "softmax"
    output_energy: str = "cross_entropy"
    max_attempts: int = 100
    constraints: Mapping[str, Any] = field(default_factory=dict, hash=False)

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
        }
        if self.num_edges is not None:
            norm["num_edges"] = _as_range("num_edges", self.num_edges)
        for key, value in norm.items():
            object.__setattr__(self, key, value)

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

        if self.num_edges is not None:
            _check_range("num_edges", self.num_edges, minimum=1)
            e_lo, e_hi = self.num_edges
            # Each hidden node and the output need at least one in-edge.
            min_needed = n_lo + 1
            if e_hi < min_needed:
                raise ConfigError(
                    "num_edges",
                    f"maximum {e_hi} is too small: with at least {n_lo} hidden "
                    f"nodes, every hidden node and the output need an incoming "
                    f"edge, so at least {min_needed} edges are required",
                )
            max_possible = (n_hi + 1) * d_hi
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
            if isinstance(value, tuple):
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
