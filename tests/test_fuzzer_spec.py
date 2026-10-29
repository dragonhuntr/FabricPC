"""Tests for fabricpc.fuzzer.spec (GraphSpec data model)."""

import dataclasses

import pytest
from hypothesis import given, settings

from fuzzer_strategies import graph_specs
from fabricpc.fuzzer.spec import (
    ROLE_INPUT,
    ROLE_OUTPUT,
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    SpecError,
)


def chain(n_hidden=2):
    names = ["input"] + [f"h{i}" for i in range(n_hidden)] + ["output"]
    nodes = [NodeSpec("input", "IdentityNode", (4,), ROLE_INPUT)]
    nodes += [NodeSpec(f"h{i}", "Linear", (8,)) for i in range(n_hidden)]
    nodes += [NodeSpec("output", "Linear", (3,), ROLE_OUTPUT)]
    edges = [EdgeSpec(a, b) for a, b in zip(names[:-1], names[1:])]
    return GraphSpec(tuple(nodes), tuple(edges))


def replace_nodes(spec, nodes):
    return dataclasses.replace(spec, nodes=tuple(nodes))


def replace_edges(spec, edges):
    return dataclasses.replace(spec, edges=tuple(edges))


class TestCheck:
    def test_chain_is_well_formed(self):
        chain().check()

    def test_duplicate_node_name(self):
        spec = chain()
        bad = replace_nodes(spec, spec.nodes + (NodeSpec("h0", "Linear", (2,)),))
        with pytest.raises(SpecError, match="duplicate node name") as exc:
            bad.check()
        assert exc.value.where == "h0"

    @pytest.mark.parametrize("shape", [(), (0,), (4, -1)])
    def test_bad_shape(self, shape):
        spec = chain()
        nodes = list(spec.nodes)
        nodes[1] = NodeSpec("h0", "Linear", shape)
        with pytest.raises(SpecError, match="shape") as exc:
            replace_nodes(spec, nodes).check()
        assert exc.value.where == "h0"

    def test_unknown_role(self):
        spec = chain()
        nodes = list(spec.nodes)
        nodes[1] = NodeSpec("h0", "Linear", (8,), role="middle")
        with pytest.raises(SpecError, match="role"):
            replace_nodes(spec, nodes).check()

    @pytest.mark.parametrize("role", [ROLE_INPUT, ROLE_OUTPUT])
    def test_exactly_one_input_and_output(self, role):
        spec = chain()
        extra = NodeSpec("extra", "Linear", (2,), role=role)
        with pytest.raises(SpecError, match=f"exactly one '{role}'"):
            replace_nodes(spec, spec.nodes + (extra,)).check()

    def test_edge_to_missing_node(self):
        spec = chain()
        bad = replace_edges(spec, spec.edges + (EdgeSpec("h0", "ghost"),))
        with pytest.raises(SpecError, match="does not exist") as exc:
            bad.check()
        assert exc.value.where == "h0->ghost:in"

    def test_self_edge(self):
        spec = chain()
        with pytest.raises(SpecError, match="self-edge"):
            replace_edges(spec, spec.edges + (EdgeSpec("h0", "h0"),)).check()

    def test_edge_into_input(self):
        spec = chain()
        with pytest.raises(SpecError, match="input node cannot receive"):
            replace_edges(spec, spec.edges + (EdgeSpec("h1", "input"),)).check()

    def test_duplicate_edge(self):
        spec = chain()
        with pytest.raises(SpecError, match="duplicate edge"):
            replace_edges(spec, spec.edges + (spec.edges[0],)).check()

    def test_same_pair_different_slot_is_allowed(self):
        spec = chain()
        replace_edges(spec, spec.edges + (EdgeSpec("input", "h0", "skip"),)).check()


class TestTopology:
    def test_chain_order_and_degrees(self):
        spec = chain(3)
        assert spec.topological_order() == ("input", "h0", "h1", "h2", "output")
        assert not spec.is_cyclic
        assert spec.in_degree("input") == 0
        assert spec.out_degree("input") == 1
        assert spec.input_node.name == "input"
        assert spec.output_node.name == "output"

    def test_feedback_edge_makes_it_cyclic(self):
        spec = chain(2)
        cyclic = replace_edges(spec, spec.edges + (EdgeSpec("h1", "h0"),))
        assert cyclic.is_cyclic
        assert cyclic.topological_order() is None

    def test_edge_key_matches_fabricpc(self):
        assert EdgeSpec("a", "b", "skip").key == "a->b:skip"


class TestFingerprint:
    def test_metadata_does_not_change_fingerprint(self):
        spec = chain()
        assert spec.fingerprint() == spec.with_metadata(seed=1).fingerprint()

    def test_any_structural_change_changes_fingerprint(self):
        spec = chain()
        nodes = list(spec.nodes)
        nodes[1] = dataclasses.replace(nodes[1], shape=(9,))
        assert replace_nodes(spec, nodes).fingerprint() != spec.fingerprint()
        nodes = list(spec.nodes)
        nodes[1] = dataclasses.replace(nodes[1], activation="tanh")
        assert replace_nodes(spec, nodes).fingerprint() != spec.fingerprint()

    @settings(max_examples=60, deadline=None)
    @given(graph_specs())
    def test_order_invariant(self, spec):
        shuffled = GraphSpec(
            tuple(reversed(spec.nodes)), tuple(reversed(spec.edges)), spec.metadata
        )
        assert shuffled.fingerprint() == spec.fingerprint()
        assert shuffled.same_structure(spec)


class TestDictRoundTrip:
    @settings(max_examples=80, deadline=None)
    @given(graph_specs())
    def test_round_trip_preserves_everything(self, spec):
        spec.check()
        again = GraphSpec.from_dict(spec.to_dict())
        assert again == spec
        assert again.metadata == spec.metadata

    def test_shape_lists_become_tuples(self):
        node = NodeSpec.from_dict({"name": "a", "node_type": "Linear", "shape": [3, 4]})
        assert node.shape == (3, 4)

    def test_unknown_key_rejected(self):
        with pytest.raises(SpecError, match="unknown key"):
            NodeSpec.from_dict(
                {"name": "a", "node_type": "Linear", "shape": [3], "colour": "red"}
            )

    def test_missing_key_rejected(self):
        with pytest.raises(SpecError, match="missing key"):
            EdgeSpec.from_dict({"source": "a"})
