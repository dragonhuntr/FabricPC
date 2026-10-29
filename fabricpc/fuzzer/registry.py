"""Name-to-class lookup tables for node types, activations and energies.

A ``GraphSpec`` refers to FabricPC classes by short string names so it can be
stored as JSON. These tables resolve those names. They are the only place the
fuzzer needs to change when FabricPC gains a new node type.

Shape and slot rules for each node type (which inputs a node accepts and what
shape it needs) belong to the validity stage, not here.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, Mapping, Optional, Type

from fabricpc.core.activations import (
    ActivationBase,
    GeluActivation,
    HardTanhActivation,
    IdentityActivation,
    LeakyReLUActivation,
    ReLUActivation,
    SigmoidActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import (
    BernoulliEnergy,
    CrossEntropyEnergy,
    EnergyFunctional,
    GaussianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
    LaplacianEnergy,
)
from fabricpc.nodes import (
    AvgPool,
    ConvNode,
    IdentityNode,
    Linear,
    LinearExplicitGrad,
    LinearResidual,
    MaxPool,
    NodeBase,
    SkipConnection,
    StorkeyHopfield,
)

NODE_TYPES: Dict[str, Type[NodeBase]] = {
    "Linear": Linear,
    "LinearExplicitGrad": LinearExplicitGrad,
    "LinearResidual": LinearResidual,
    "IdentityNode": IdentityNode,
    "SkipConnection": SkipConnection,
    "StorkeyHopfield": StorkeyHopfield,
    "ConvNode": ConvNode,
    "MaxPool": MaxPool,
    "AvgPool": AvgPool,
}

ACTIVATIONS: Dict[str, Type[ActivationBase]] = {
    "identity": IdentityActivation,
    "sigmoid": SigmoidActivation,
    "tanh": TanhActivation,
    "relu": ReLUActivation,
    "leaky_relu": LeakyReLUActivation,
    "gelu": GeluActivation,
    "softmax": SoftmaxActivation,
    "hard_tanh": HardTanhActivation,
}

ENERGIES: Dict[str, Type[EnergyFunctional]] = {
    "gaussian": GaussianEnergy,
    "bernoulli": BernoulliEnergy,
    "cross_entropy": CrossEntropyEnergy,
    "laplacian": LaplacianEnergy,
    "huber": HuberEnergy,
    "kl_divergence": KLDivergenceEnergy,
}


class UnknownNameError(KeyError):
    """A node type, activation or energy name is not registered."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


def _lookup(table: Mapping[str, Any], kind: str, name: str) -> Any:
    try:
        return table[name]
    except KeyError:
        close = difflib.get_close_matches(name, table, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise UnknownNameError(
            f"Unknown {kind} {name!r}.{hint} Registered: {sorted(table)}"
        ) from None


def node_class(name: str) -> Type[NodeBase]:
    """Return the node class registered under ``name``."""
    return _lookup(NODE_TYPES, "node type", name)


def make_activation(
    name: Optional[str], params: Optional[Mapping[str, Any]] = None
) -> Optional[ActivationBase]:
    """Instantiate an activation by name, or return ``None`` for ``None``."""
    if name is None:
        return None
    return _lookup(ACTIVATIONS, "activation", name)(**dict(params or {}))


def make_energy(
    name: Optional[str], params: Optional[Mapping[str, Any]] = None
) -> Optional[EnergyFunctional]:
    """Instantiate an energy functional by name, or return ``None`` for ``None``."""
    if name is None:
        return None
    return _lookup(ENERGIES, "energy", name)(**dict(params or {}))


def register_node_type(name: str, cls: Type[NodeBase], replace: bool = False) -> None:
    """Make a custom ``NodeBase`` subclass available to specs under ``name``.

    Raises:
        TypeError: ``cls`` is not a ``NodeBase`` subclass.
        ValueError: ``name`` is taken and ``replace`` is False.
    """
    if not (isinstance(cls, type) and issubclass(cls, NodeBase)):
        raise TypeError(f"{cls!r} is not a NodeBase subclass")
    if name in NODE_TYPES and not replace:
        raise ValueError(f"Node type {name!r} is already registered")
    NODE_TYPES[name] = cls
