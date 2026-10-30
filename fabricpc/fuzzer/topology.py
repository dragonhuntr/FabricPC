"""Topology stage: decide which hidden nodes exist and which node feeds which.

:func:`random_dag_topology` is the pipeline's topology stage. It returns a
:class:`~fabricpc.fuzzer.pipeline.Skeleton` (names and directed edges only);
node types, shapes and input slots are chosen later by the assign stage.

How a skeleton is drawn
-----------------------
1. Pick the hidden-node count ``n`` inside ``config.num_nodes`` (only counts
   for which ``config.num_edges`` is reachable are eligible).
2. Line up ``input``, the hidden nodes and ``output`` in one order; edges in
   the drawn DAG only go forward in it, so hidden node ``h{i}`` is the
   ``i``-th node of a topological order. Every labelled DAG is reachable this
   way up to renaming.
3. Give every non-input node a target in-degree inside ``config.in_degree``,
   with the totals summing to an edge count inside ``config.num_edges``.
4. Walk the order and pick each node's forward parents among the nodes
   before it. A running check forces enough parents to come from nodes that
   do not yet feed anything, so every hidden node ends up on a path from
   ``input`` to ``output`` without breaking a degree limit.
5. With ``config.allow_cycles``, part of a hidden node's in-degree comes from
   *later* hidden nodes instead: feedback edges.

When the settings leave room for more than ``n + 1`` edges, at least ``n + 2``
are used, so the result is never a plain chain: a graph where every node has
one parent and every hidden node feeds something can only be a chain.

The parent-picking rule is selected with ``config.constraints["topology_model"]``:

- ``"random_dag"`` (default): uniform over earlier nodes.
- ``"erdos_renyi"``: in-degrees are binomial, each earlier node is a parent
  with probability ``constraints["edge_prob"]`` (default 0.5).
- ``"barabasi_albert"``: parents are chosen in proportion to their current
  out-degree plus one (preferential attachment, hub-like graphs).
- ``"watts_strogatz"``: parents are the nearest earlier nodes in the order,
  each rewired to a uniform earlier node with probability
  ``constraints["rewire_prob"]`` (default 0.2).

All randomness comes from the ``rng`` argument, so the same config, seed and
graph index always give the same skeleton.
"""

from __future__ import annotations

import graphlib
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from fabricpc.fuzzer.config import GeneratorConfig
from fabricpc.fuzzer.spec import INPUT_NAME, OUTPUT_NAME, Skeleton

TOPOLOGY_MODELS = ("random_dag", "erdos_renyi", "barabasi_albert", "watts_strogatz")
DEFAULT_MODEL = "random_dag"


class TopologyError(ValueError):
    """The settings leave no valid skeleton to draw."""


def _max_sources(position: int, n: int, cycles: bool) -> int:
    """Distinct possible parents of the node at ``position`` (1..n+1).

    Positions: 0 is ``input``, 1..n are hidden nodes in topological order,
    ``n + 1`` is ``output``. Without cycles a node can only be fed by the
    ``position`` nodes before it. With cycles a hidden node can also be fed by
    any later hidden node (``output`` never feeds back).
    """
    if not cycles or position == n + 1:
        return position
    return n  # input + the other n - 1 hidden nodes


def _degree_bounds(n: int, config: GeneratorConfig) -> List[Tuple[int, int]]:
    """Per-position ``(lo, hi)`` in-degree bounds for positions 1..n+1."""
    d_lo, d_hi = config.in_degree
    bounds = []
    for k in range(1, n + 2):
        hi = min(d_hi, _max_sources(k, n, config.allow_cycles))
        bounds.append((d_lo, hi))
    return bounds


def _edge_window(n: int, config: GeneratorConfig) -> Optional[Tuple[int, int]]:
    """Inclusive range of total edge counts possible with ``n`` hidden nodes,
    intersected with ``config.num_edges``. ``None`` when empty."""
    bounds = _degree_bounds(n, config)
    if any(lo > hi for lo, hi in bounds):
        return None
    lo = sum(b[0] for b in bounds)
    hi = sum(b[1] for b in bounds)
    if config.num_edges is not None:
        lo = max(lo, config.num_edges[0])
        hi = min(hi, config.num_edges[1])
    return (lo, hi) if lo <= hi else None


def _choose_hidden_count(
    config: GeneratorConfig, rng: np.random.Generator
) -> Tuple[int, Tuple[int, int]]:
    lo, hi = config.num_nodes
    eligible = [
        (n, w) for n in range(lo, hi + 1) if (w := _edge_window(n, config)) is not None
    ]
    if not eligible:
        raise TopologyError(
            f"no hidden-node count in num_nodes={list(config.num_nodes)} admits "
            f"num_edges={config.num_edges} with in_degree={list(config.in_degree)}"
        )
    n, window = eligible[int(rng.integers(len(eligible)))]
    return n, window


def _sample_degrees(
    bounds: Sequence[Tuple[int, int]],
    total: Optional[int],
    model: str,
    edge_prob: float,
    rng: np.random.Generator,
) -> List[int]:
    """In-degrees within ``bounds``; summing to ``total`` when given."""
    if total is None:
        degrees = []
        for k, (lo, hi) in enumerate(bounds, start=1):
            if model == "erdos_renyi":
                d = int(rng.binomial(k, edge_prob))
            else:
                d = int(rng.integers(lo, hi + 1))
            degrees.append(min(max(d, lo), hi))
        return degrees

    degrees = [lo for lo, _ in bounds]
    spare = total - sum(degrees)
    while spare > 0:
        open_ = [i for i, (_, hi) in enumerate(bounds) if degrees[i] < hi]
        i = open_[int(rng.integers(len(open_)))]
        degrees[i] += 1
        spare -= 1
    return degrees


def _pick(
    candidates: Sequence[int],
    count: int,
    model: str,
    position: int,
    out_degree: Dict[int, int],
    rewire_prob: float,
    rng: np.random.Generator,
) -> List[int]:
    """Pick ``count`` distinct parents from ``candidates`` under ``model``."""
    pool = list(candidates)
    chosen: List[int] = []
    for _ in range(count):
        if model == "barabasi_albert":
            w = np.array([out_degree.get(c, 0) + 1.0 for c in pool])
            idx = int(rng.choice(len(pool), p=w / w.sum()))
        elif model == "watts_strogatz" and rng.random() >= rewire_prob:
            idx = int(np.argmin([abs(position - c) for c in pool]))
        else:
            idx = int(rng.integers(len(pool)))
        chosen.append(pool.pop(idx))
    return chosen


def _total_target(window: Tuple[int, int], n: int, rng) -> int:
    lo, hi = window
    if n >= 1 and hi >= n + 2:
        lo = max(lo, n + 2)
    return int(rng.integers(lo, hi + 1))


def random_dag_topology(config: GeneratorConfig, rng: np.random.Generator) -> Skeleton:
    """Draw a random skeleton that satisfies every structural setting in ``config``.

    Args:
        config: Generation settings. ``num_nodes``, ``num_edges``,
            ``in_degree``, ``allow_cycles`` and ``constraints["topology_model"]``
            are read.
        rng: The only source of randomness.

    Returns:
        A :class:`Skeleton` with hidden nodes ``h0 .. h{n-1}`` (declared in
        their topological order) and directed edges over ``input``, the hidden
        nodes and ``output``.

    Raises:
        TopologyError: No skeleton satisfies the settings.
    """
    constraints = config.constraints
    model = str(constraints.get("topology_model", DEFAULT_MODEL))
    if model not in TOPOLOGY_MODELS:
        raise TopologyError(
            f"unknown topology_model {model!r}; choose one of {list(TOPOLOGY_MODELS)}"
        )
    edge_prob = float(constraints.get("edge_prob", 0.5))
    rewire_prob = float(constraints.get("rewire_prob", 0.2))
    cycles = config.allow_cycles

    n, window = _choose_hidden_count(config, rng)
    bounds = _degree_bounds(n, config)
    if config.num_edges is not None:
        total = _total_target(window, n, rng)
        degrees = _sample_degrees(bounds, total, model, edge_prob, rng)
    else:
        degrees = _sample_degrees(bounds, None, model, edge_prob, rng)
        room = [i for i, (_, hi) in enumerate(bounds) if degrees[i] < hi]
        if n >= 1 and sum(degrees) == n + 1 and room:
            degrees[room[int(rng.integers(len(room)))]] += 1

    # Split each hidden node's in-degree into forward parents (earlier nodes)
    # and feedback parents (later hidden nodes). Output is forward-only.
    forward: List[int] = []
    for k, d in enumerate(degrees, start=1):
        if not cycles or k == n + 1:
            forward.append(d)
            continue
        later = n - k
        lo, hi = max(1, d - later), min(d, k)
        forward.append(int(rng.integers(lo, hi + 1)))
    backward = [d - f for d, f in zip(degrees, forward)]
    if cycles and n >= 2 and not any(backward):
        movable = [k for k in range(1, n) if forward[k - 1] >= 2 and n - k >= 1]
        if movable:
            k = movable[int(rng.integers(len(movable)))]
            forward[k - 1] -= 1
            backward[k - 1] += 1

    # Forward pass. ``uncovered`` holds hidden positions that do not feed any
    # later node yet. Invariant after position k:
    #   |uncovered| + (hidden nodes still to come) <= forward capacity after k
    # which guarantees every hidden node gets an out-edge by the output.
    order = list(range(n + 2))
    edges: List[Tuple[int, int]] = []
    out_degree: Dict[int, int] = {}
    uncovered: Set[int] = set()
    suffix = np.cumsum(forward[::-1])[::-1].tolist() + [
        0
    ]  # suffix[i] = sum forward[i:]
    for k in range(1, n + 2):
        f = forward[k - 1]
        is_hidden = k <= n
        capacity_after = suffix[k]  # positions k+1 .. n+1
        still_to_come = max(0, n - k)
        need = len(uncovered) + int(is_hidden) + still_to_come - capacity_after
        if k == n + 1:
            need = len(uncovered)
        need = max(0, need)
        must = _pick(sorted(uncovered), need, model, k, out_degree, rewire_prob, rng)
        rest_pool = [c for c in range(k) if c not in must]
        extra = _pick(rest_pool, f - need, model, k, out_degree, rewire_prob, rng)
        for p in must + extra:
            edges.append((p, k))
            out_degree[p] = out_degree.get(p, 0) + 1
            uncovered.discard(p)
        if is_hidden:
            uncovered.add(k)

    for k in range(1, n + 1):
        if backward[k - 1]:
            later = list(range(k + 1, n + 1))
            for p in _pick(
                later, backward[k - 1], model, k, out_degree, rewire_prob, rng
            ):
                edges.append((p, k))
                out_degree[p] = out_degree.get(p, 0) + 1

    names = {0: INPUT_NAME, n + 1: OUTPUT_NAME}
    names.update({k: f"h{k - 1}" for k in range(1, n + 1)})
    hidden = tuple(names[k] for k in order[1:-1])
    return Skeleton(hidden=hidden, edges=tuple((names[s], names[t]) for s, t in edges))


def skeleton_issues(skeleton: Skeleton, config: GeneratorConfig) -> List[str]:
    """Every way ``skeleton`` breaks the topology contract (empty when valid).

    Checks hidden-node count, edge count, in-degree limits, no edges into the
    input, no self or duplicate edges, acyclicity when cycles are off, and
    that every hidden node lies on a path from input to output.
    """
    issues: List[str] = []
    inp, out = skeleton.input, skeleton.output
    hidden = list(skeleton.hidden)
    nodes = set(skeleton.all_nodes)
    n_lo, n_hi = config.num_nodes
    if not n_lo <= len(hidden) <= n_hi:
        issues.append(f"{len(hidden)} hidden nodes outside num_nodes {[n_lo, n_hi]}")
    if len(set(hidden)) != len(hidden) or {inp, out} & set(hidden):
        issues.append("hidden names are not unique or reuse input/output")
    if config.num_edges is not None:
        e_lo, e_hi = config.num_edges
        if not e_lo <= len(skeleton.edges) <= e_hi:
            issues.append(
                f"{len(skeleton.edges)} edges outside num_edges {[e_lo, e_hi]}"
            )
    if len(set(skeleton.edges)) != len(skeleton.edges):
        issues.append("duplicate edges")
    preds: Dict[str, Set[str]] = {v: set() for v in nodes}
    succs: Dict[str, Set[str]] = {v: set() for v in nodes}
    for s, t in skeleton.edges:
        if s not in nodes or t not in nodes:
            issues.append(f"edge {s}->{t} names an unknown node")
            continue
        if s == t:
            issues.append(f"self-edge on {s}")
        if t == inp:
            issues.append(f"edge {s}->{t} points into the input")
        if s == out:
            issues.append(f"edge {s}->{t} leaves the output")
        preds[t].add(s)
        succs[s].add(t)
    d_lo, d_hi = config.in_degree
    for v in [*hidden, out]:
        if not d_lo <= len(preds[v]) <= d_hi:
            issues.append(
                f"{v} has in-degree {len(preds[v])} outside in_degree {[d_lo, d_hi]}"
            )
    if not config.allow_cycles:
        try:
            tuple(graphlib.TopologicalSorter(preds).static_order())
        except graphlib.CycleError:
            issues.append("cycle found but allow_cycles is false")
    from_input = _reach(inp, succs)
    to_output = _reach(out, preds)
    for v in hidden:
        if v not in from_input or v not in to_output:
            issues.append(f"{v} is not on a path from {inp} to {out}")
    return issues


def _reach(start: str, adj: Dict[str, Set[str]]) -> Set[str]:
    seen, stack = {start}, [start]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def is_chain(skeleton: Skeleton) -> bool:
    """True when the skeleton is ``input -> h -> ... -> output`` with no extras."""
    return (
        len(skeleton.edges) == len(skeleton.hidden) + 1
        and all(
            sum(1 for _, t in skeleton.edges if t == v) == 1
            for v in (*skeleton.hidden, skeleton.output)
        )
        and all(
            sum(1 for s, _ in skeleton.edges if s == v) == 1
            for v in (skeleton.input, *skeleton.hidden)
        )
    )
