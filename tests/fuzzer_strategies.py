"""Hypothesis strategies shared by the fuzzer tests.

``graph_specs()`` draws structurally well-formed ``GraphSpec`` objects (they
pass ``GraphSpec.check()``) built only from ``Linear`` hidden nodes, so every
drawn spec also builds in FabricPC. Teammates can extend it with more node
types once the typing stage knows their shape rules.
"""

from hypothesis import strategies as st

from fabricpc.fuzzer.spec import (
    ROLE_HIDDEN,
    ROLE_INPUT,
    ROLE_OUTPUT,
    EdgeSpec,
    GraphSpec,
    NodeSpec,
)

ACTS = ["relu", "tanh", "sigmoid", "identity", "gelu"]


@st.composite
def graph_specs(draw, max_hidden: int = 6, allow_cycles: bool = True, metadata=True):
    n_hidden = draw(st.integers(min_value=0, max_value=max_hidden))
    hidden = [f"h{i}" for i in range(n_hidden)]
    order = ["input", *hidden, "output"]

    nodes = [NodeSpec("input", "IdentityNode", (draw(st.integers(1, 12)),), ROLE_INPUT)]
    for name in hidden:
        nodes.append(
            NodeSpec(
                name,
                "Linear",
                (draw(st.integers(1, 16)),),
                ROLE_HIDDEN,
                activation=draw(st.sampled_from(ACTS)),
                config=draw(st.sampled_from([{}, {"use_bias": False}])),
            )
        )
    nodes.append(
        NodeSpec(
            "output",
            "Linear",
            (draw(st.integers(1, 5)),),
            ROLE_OUTPUT,
            activation="softmax",
            energy="cross_entropy",
        )
    )

    # Every non-input node gets one in-edge from an earlier node (keeps the
    # graph connected), then optional extra forward or feedback edges.
    edges = set()
    for i in range(1, len(order)):
        src = order[draw(st.integers(0, i - 1))]
        edges.add((src, order[i]))
    extra = draw(
        st.lists(
            st.tuples(st.integers(0, len(order) - 1), st.integers(1, len(order) - 1)),
            max_size=6,
        )
    )
    for a, b in extra:
        if a == b:
            continue
        if not allow_cycles and a > b:
            a, b = b, a
            if a == 0 and b == 0:
                continue
        if order[b] == "input" or a == b:
            continue
        edges.add((order[a], order[b]))

    edge_specs = [EdgeSpec(s, t) for s, t in sorted(edges)]
    edge_specs = draw(st.permutations(edge_specs))
    node_specs = draw(st.permutations(nodes))
    meta = {}
    if metadata:
        meta = {
            "master_seed": draw(st.integers(0, 2**31)),
            "note": draw(st.text(max_size=8)),
        }
    return GraphSpec(nodes=tuple(node_specs), edges=tuple(edge_specs), metadata=meta)
