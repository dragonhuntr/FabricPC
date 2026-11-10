"""End-to-end generation: settings in, validated graphs out.

The pipeline runs three pluggable stages for each graph::

    GeneratorConfig ──topology──▶ Skeleton ──assign──▶ GraphSpec ──validate──▶ ok?

- **topology** (Evan): decide which hidden nodes exist and which node feeds
  which. Returns a :class:`Skeleton` (names and directed edges only).
- **assign** (Hai): choose a node type, shape and activation for every node and
  a slot for every edge. Returns a :class:`~fabricpc.fuzzer.spec.GraphSpec`.
- **validate** (Hai): decide whether the spec is usable. Returns a
  :class:`ValidationResult`; failures name the node or edge at fault.

A graph that fails validation is retried with fresh, still reproducible,
randomness up to ``config.max_attempts`` times. Exact duplicates (same
fingerprint) are skipped so every returned graph is distinct.

The topology stage is :func:`fabricpc.fuzzer.topology.random_dag_topology`.
:func:`chain_topology` stays as the shape of a sequential baseline. The other
stage functions here (:func:`linear_assign`, :func:`check_builds`) are simple
stand-ins so the whole pipeline, CLI and tests run today. Teammates replace
them by passing their own functions to :func:`generate`, or by changing the
defaults in :data:`DEFAULT_STAGES` once their modules land.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from fabricpc.fuzzer import __version__
from fabricpc.fuzzer.config import GeneratorConfig
from fabricpc.fuzzer.seeding import graph_rng, graph_seed
from fabricpc.fuzzer.spec import (
    ROLE_HIDDEN,
    ROLE_INPUT,
    ROLE_OUTPUT,
    EdgeSpec,
    GraphSpec,
    INPUT_NAME,
    OUTPUT_NAME,
    NodeSpec,
    Skeleton,
)
from fabricpc.fuzzer.topology import TopologyError, random_dag_topology
from fabricpc.fuzzer.assign import typed_assign
from fabricpc.fuzzer.validity import validate_spec


# --------------------------------------------------------------------- types
@dataclass(frozen=True)
class ValidationIssue:
    """One reason a graph was rejected.

    Attributes:
        message: What is wrong.
        where: Node name or edge key at fault, or ``None`` for the whole graph.
        check: Short name of the check that failed, e.g. ``"builds"``.
    """

    message: str
    where: Optional[str] = None
    check: str = "unspecified"

    def __str__(self) -> str:
        loc = f"[{self.where}] " if self.where else ""
        return f"{self.check}: {loc}{self.message}"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of the validate stage. ``ok`` is True when there are no issues."""

    issues: Tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    @classmethod
    def passed(cls) -> "ValidationResult":
        return cls(())

    @classmethod
    def failed(cls, *issues: ValidationIssue) -> "ValidationResult":
        return cls(tuple(issues))


class TopologyFn(Protocol):
    def __call__(
        self, config: GeneratorConfig, rng: np.random.Generator
    ) -> Skeleton: ...


class AssignFn(Protocol):
    def __call__(
        self, skeleton: Skeleton, config: GeneratorConfig, rng: np.random.Generator
    ) -> GraphSpec: ...


ValidateFn = Callable[[GraphSpec], ValidationResult]


class GenerationError(RuntimeError):
    """A graph could not be produced within ``config.max_attempts`` tries.

    Attributes:
        index: Which graph of the batch failed.
        last_issues: Validation issues from the final attempt.
    """

    def __init__(
        self, index: int, attempts: int, last_issues: Tuple[ValidationIssue, ...]
    ):
        self.index = index
        self.last_issues = last_issues
        details = (
            "\n  ".join(str(i) for i in last_issues) or "(all attempts were duplicates)"
        )
        super().__init__(
            f"graph {index}: no valid, distinct graph after {attempts} attempts. "
            f"Last problems:\n  {details}"
        )


class PartialBatchWarning(UserWarning):
    """A run returned fewer graphs than requested but kept the valid ones."""


# ------------------------------------------------------- stand-in stages
def chain_topology(config: GeneratorConfig, rng: np.random.Generator) -> Skeleton:
    """Stand-in topology: ``input -> h0 -> h1 -> ... -> output``.

    Picks the number of hidden nodes from ``config.num_nodes``. This is also
    the shape of a sequential baseline. The default topology stage is
    :func:`fabricpc.fuzzer.topology.random_dag_topology`.
    """
    lo, hi = config.num_nodes
    n = int(rng.integers(lo, hi + 1))
    hidden = tuple(f"h{i}" for i in range(n))
    chain = (INPUT_NAME, *hidden, OUTPUT_NAME)
    return Skeleton(hidden=hidden, edges=tuple(zip(chain[:-1], chain[1:])))


def linear_assign(
    skeleton: Skeleton, config: GeneratorConfig, rng: np.random.Generator
) -> GraphSpec:
    """Stand-in assigner: every hidden node is ``Linear``, every edge uses slot ``in``.

    ``Linear`` accepts any number of inputs of any size, so any skeleton is
    valid with it. Replaced by the real typing stage, which handles every node
    type in ``config.node_types``.
    """
    nodes = [
        NodeSpec(
            name=skeleton.input,
            node_type="IdentityNode",
            shape=config.input_shape,
            role=ROLE_INPUT,
        )
    ]
    for name in skeleton.hidden:
        nodes.append(
            NodeSpec(
                name=name,
                node_type="Linear",
                shape=(int(rng.choice(config.hidden_sizes)),),
                role=ROLE_HIDDEN,
                activation=str(rng.choice(config.hidden_activations)),
            )
        )
    nodes.append(
        NodeSpec(
            name=skeleton.output,
            node_type="Linear",
            shape=config.output_shape,
            role=ROLE_OUTPUT,
            activation=config.output_activation,
            energy=config.output_energy,
        )
    )
    edges = tuple(EdgeSpec(s, t, "in") for s, t in skeleton.edges)
    return GraphSpec(nodes=tuple(nodes), edges=edges)


def check_builds(spec: GraphSpec) -> ValidationResult:
    """Baseline validator: the spec is well-formed and FabricPC builds it.

    The full validator also propagates shapes, initializes parameters and runs
    a short training step to check for non-finite values.
    """
    from fabricpc.fuzzer.build import BuildError, to_fabricpc

    try:
        to_fabricpc(spec)
    except BuildError as exc:
        return ValidationResult.failed(
            ValidationIssue(
                message=str(exc), where=exc.where, check=f"builds/{exc.stage}"
            )
        )
    return ValidationResult.passed()


DEFAULT_STAGES: Dict[str, Callable] = {
    "topology": random_dag_topology,
    "assign": typed_assign,
    "validate": validate_spec,
}


# -------------------------------------------------------------- pipeline
@dataclass
class GenerationReport:
    """Bookkeeping from a :func:`generate` call, useful for the report and CLI.

    Attributes:
        attempts: Attempts used for each graph that was produced.
        duplicates_skipped: Fingerprints rejected as duplicates.
        rejected: Every validation issue from every failed attempt.
        failed: Indices of graphs that exhausted ``max_attempts``.
        failed_checks: One check name per failed attempt (the first issue).
        warnings: Human-readable notes, including the final partial-batch line.
        requested: How many graphs the run asked for.
        created: How many valid graphs it returned.
        failure_counts: Number of rejections per check name.
    """

    attempts: List[int] = field(default_factory=list)
    duplicates_skipped: int = 0
    rejected: List[ValidationIssue] = field(default_factory=list)
    failed: List[int] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requested: int = 0
    created: int = 0
    failure_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def most_common_failure(self) -> Optional[str]:
        """Check name that rejected the most attempts, or ``None``.

        Ties are broken alphabetically so the CLI line is deterministic.
        """
        if not self.failure_counts:
            return None
        return sorted(self.failure_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def tally(self) -> None:
        """Recompute ``failure_counts`` from the per-attempt checks."""
        self.failure_counts = dict(Counter(self.failed_checks))


def generate_one(
    config: GeneratorConfig,
    index: int,
    topology: Optional[TopologyFn] = None,
    assign: Optional[AssignFn] = None,
    validate: Optional[ValidateFn] = None,
    exclude: Optional[set] = None,
    report: Optional[GenerationReport] = None,
) -> GraphSpec:
    """Generate graph ``index`` of the run described by ``config``.

    Depends only on ``config`` and ``index`` (plus ``exclude``), so a graph from
    a saved batch can be regenerated on its own from its metadata.

    Args:
        config: Validated generation settings.
        index: Position of the graph in the batch.
        topology, assign, validate: Stage overrides; ``None`` uses
            :data:`DEFAULT_STAGES`.
        exclude: Fingerprints to treat as duplicates.
        report: Optional report to append statistics to.

    Raises:
        GenerationError: No valid, distinct graph within ``config.max_attempts``.
    """
    topology = topology or DEFAULT_STAGES["topology"]
    assign = assign or DEFAULT_STAGES["assign"]
    validate = validate or DEFAULT_STAGES["validate"]
    exclude = exclude if exclude is not None else set()

    last_issues: Tuple[ValidationIssue, ...] = ()
    for attempt in range(config.max_attempts):
        rng = graph_rng(config.seed, index, attempt)
        try:
            skeleton = topology(config, rng)
        except TopologyError as exc:
            # A fresh attempt cannot fix a settings-level impossibility, and
            # retrying would multiply the topology stage's own redraw loop.
            issue = ValidationIssue(message=str(exc), check="topology")
            last_issues = (issue,)
            if report is not None:
                report.rejected.append(issue)
                report.failed_checks.append(issue.check)
            break
        spec = assign(skeleton, config, rng)
        result = validate(spec)
        if not result.ok:
            last_issues = result.issues
            if report is not None:
                report.rejected.extend(result.issues)
                report.failed_checks.append(result.issues[0].check)
            continue
        fp = spec.fingerprint()
        if fp in exclude:
            if report is not None:
                report.duplicates_skipped += 1
            continue
        if report is not None:
            report.attempts.append(attempt + 1)
        return spec.with_metadata(
            fuzzer_version=__version__,
            master_seed=config.seed,
            graph_index=index,
            graph_seed=graph_seed(config.seed, index),
            attempt=attempt,
            unroll=config.unroll,
            config=config.to_dict(),
        )
    raise GenerationError(index, config.max_attempts, last_issues)


def generate(
    config: GeneratorConfig,
    topology: Optional[TopologyFn] = None,
    assign: Optional[AssignFn] = None,
    validate: Optional[ValidateFn] = None,
    report: Optional[GenerationReport] = None,
) -> List[GraphSpec]:
    """Generate up to ``config.num_graphs`` distinct, validated graphs.

    A graph that exhausts ``config.max_attempts`` is skipped, not fatal: the
    valid graphs are returned and a :class:`PartialBatchWarning` reports how
    many were made and which check failed most often. The returned list never
    contains a graph that failed validation.

    Args:
        config: Generation settings. It is validated before any work starts.
        topology, assign, validate: Stage overrides; ``None`` uses
            :data:`DEFAULT_STAGES`.
        report: Optional report to fill in. One is created when omitted.

    Returns:
        The valid graphs, possibly fewer than requested.

    Raises:
        ConfigError: ``config`` is invalid.
    """
    config.validate()
    if report is None:
        report = GenerationReport()
    report.requested = config.num_graphs
    seen: set = set()
    specs = []
    for index in range(config.num_graphs):
        try:
            spec = generate_one(
                config, index, topology, assign, validate, exclude=seen, report=report
            )
        except GenerationError as exc:
            report.failed.append(index)
            report.warnings.append(str(exc))
            continue
        seen.add(spec.fingerprint())
        specs.append(spec)
    report.created = len(specs)
    report.tally()
    if len(specs) < config.num_graphs:
        line = (
            f"made {len(specs)} of {config.num_graphs}, most common failure: "
            f"{report.most_common_failure or 'none'}"
        )
        report.warnings.append(line)
        warnings.warn(line, PartialBatchWarning, stacklevel=2)
    return specs
