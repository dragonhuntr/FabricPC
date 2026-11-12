"""Expose a generated graph as an experiment model factory.

FabricPC's A/B harness (``fabricpc.experiments``) builds each arm through a
``ModelFactory``: a callable that takes a JAX ``rng_key`` and returns
``(params, structure)``. The MNIST lateral-connections example uses the same
shape with ``create_lateral_model`` and ``create_mlp_model``.

:func:`as_model_factory` adapts a fuzzed :class:`~fabricpc.fuzzer.spec.GraphSpec`
to that signature, so a generated graph can drop into an existing experiment in
place of a hand-written model.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from fabricpc.core.inference import InferenceBase
from fabricpc.core.types import GraphParams
from fabricpc.fuzzer.build import to_fabricpc
from fabricpc.fuzzer.spec import GraphSpec
from fabricpc.graph_assembly.graph_construction import GraphStructure
from fabricpc.graph_initialization import initialize_params

ModelFactory = Callable[[jax.Array], tuple[GraphParams, GraphStructure]]


def as_model_factory(
    spec: GraphSpec,
    inference: InferenceBase | None = None,
    unroll: int | None = None,
) -> ModelFactory:
    """Return a ``rng_key -> (params, structure)`` callable for ``spec``.

    The structure is built once because it does not depend on the key; the
    parameters are initialised fresh on every call, matching
    ``create_lateral_model`` in ``examples/mnist_lateral_connections.py``. The
    returned callable therefore fits ``fabricpc.experiments.ExperimentArm``
    unchanged.

    Args:
        spec: Graph to expose. It must build with :func:`to_fabricpc`.
        inference: Inference algorithm to build with. ``None`` uses
            :func:`to_fabricpc`'s default.
        unroll: Cycle traversals for cyclic graphs. ``None`` uses the spec's
            own metadata.

    Returns:
        A model factory with the experiment harness's signature.

    Raises:
        BuildError: ``spec`` is malformed or FabricPC rejects it.
    """
    structure = to_fabricpc(spec, inference=inference, unroll=unroll)
    return _GraphModelFactory(spec, structure)


class _GraphModelFactory:
    """Callable model factory that keeps its ``GraphSpec`` for debugging.

    Instances satisfy the ``ModelFactory`` protocol — call one with a JAX key
    to get ``(params, structure)`` — and expose ``spec`` for the demo's
    provenance line. ``__name__`` and ``__doc__`` are set to mimic the closure
    this class replaces, so tracebacks and repr-style debugging still name the
    graph.
    """

    def __init__(self, spec: GraphSpec, structure: GraphStructure) -> None:
        self.spec = spec
        self._structure = structure
        self.__name__ = f"factory_{spec.fingerprint()[:12]}"
        self.__doc__ = f"Model factory for fuzzed graph {spec.fingerprint()[:12]}."

    def __call__(self, rng_key: jax.Array) -> tuple[GraphParams, GraphStructure]:
        return initialize_params(self._structure, rng_key), self._structure


__all__ = ["ModelFactory", "as_model_factory"]
