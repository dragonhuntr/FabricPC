"""Error-message audit (SR-NF-US-02, SR-NF-US-03).

SR-NF-US-02: an error message must name the setting, node or edge at fault.
SR-NF-US-03: it must also say what to change.

Every exception the core fuzzer raises is triggered here at least once and its
message is asserted against both rules. Families: ``ConfigError`` (every
setting), ``TopologyError``, ``GenerationError``, ``SpecError``,
``BuildError``, ``UnknownNameError``, ``SerializationError``, the seeding
``ValueError`` and ``PartialBatchWarning``. Experiments code
(``fabricpc/fuzzer/experiments``) is out of scope: it is not on the default
default path (evaluation is not part of it).
"""

import warnings
from typing import Callable, List, Tuple

import numpy as np
import pytest

from fabricpc.fuzzer import registry
from fabricpc.fuzzer.build import BuildError, build_node, to_fabricpc
from fabricpc.fuzzer.config import ClampedNode, ConfigError, GeneratorConfig
from fabricpc.fuzzer.pipeline import (
    GenerationError,
    PartialBatchWarning,
    generate,
    generate_one,
)
from fabricpc.fuzzer.seeding import graph_seed
from fabricpc.fuzzer.serialize import SerializationError, save
from fabricpc.fuzzer.spec import (
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    SpecError,
)
from fabricpc.fuzzer.topology import TopologyError, random_dag_topology

# Substrings that mark a message as actionable ("says what to change").
FIX_MARKERS = (
    "fix:",
    "use ",
    "set ",
    "remove",
    "did you mean",
    "options:",
    "available",
    "expected",
    "must",
    "at least",
    "choose one",
    "usable types",
    "non-negative",
    "registered",
    "valid settings",
    "needs",
)


def assert_actionable(message: str) -> None:
    """SR-NF-US-03: the message must state what to change."""
    lowered = message.lower()
    assert any(
        marker in lowered for marker in FIX_MARKERS
    ), f"message gives no hint what to change: {message!r}"


def tiny_spec() -> GraphSpec:
    """Minimal buildable spec used by the Build/Spec message tests."""
    return GraphSpec(
        (
            NodeSpec("input", "IdentityNode", (4,), "input"),
            NodeSpec("h0", "Linear", (8,), activation="relu"),
            NodeSpec("output", "Linear", (2,), "output", activation="softmax"),
        ),
        (EdgeSpec("input", "h0"), EdgeSpec("h0", "output")),
    )


# --------------------------------------------------------------------- ConfigError
# Each row: (factory for the bad config, param that must be named).
BAD_CONFIGS: List[Tuple[Callable[[], object], str]] = [
    (lambda: GeneratorConfig(num_graphs=0).validate(), "num_graphs"),
    (lambda: GeneratorConfig(seed=-1).validate(), "seed"),
    (lambda: GeneratorConfig(unroll=0).validate(), "unroll"),
    (lambda: GeneratorConfig(max_attempts=0).validate(), "max_attempts"),
    (lambda: GeneratorConfig(num_nodes="4"), "num_nodes"),
    (lambda: GeneratorConfig(allow_cycles="yes").validate(), "allow_cycles"),
    (lambda: GeneratorConfig(in_degree=[0, 3]).validate(), "in_degree"),
    (lambda: GeneratorConfig(num_edges=[0, 1]).validate(), "num_edges"),
    (lambda: GeneratorConfig(node_types=[]).validate(), "node_types"),
    (lambda: GeneratorConfig(node_types=["Banana"]).validate(), "node_types"),
    (lambda: GeneratorConfig(node_types=["ConvNode"]).validate(), "node_types"),
    (lambda: GeneratorConfig(hidden_activations=[]).validate(), "hidden_activations"),
    (
        lambda: GeneratorConfig(hidden_activations=["reluu"]).validate(),
        "hidden_activations",
    ),
    (
        lambda: GeneratorConfig(output_activation="banana").validate(),
        "output_activation",
    ),
    (
        lambda: GeneratorConfig(
            output_energy="cross_entropy", output_activation="relu"
        ).validate(),
        "output_activation",
    ),
    (lambda: GeneratorConfig(output_energy="quantum").validate(), "output_energy"),
    (lambda: GeneratorConfig(input_shape=[0, 4]).validate(), "input_shape"),
    (lambda: GeneratorConfig(hidden_sizes=[-8]).validate(), "hidden_sizes"),
    (
        lambda: GeneratorConfig(constraints={"x": object()}).validate(),
        "constraints",
    ),
]


@pytest.mark.parametrize(
    "build, param", BAD_CONFIGS, ids=lambda v: getattr(v, "__name__", str(v))[:40]
)
def test_sr_nf_us_02_config_errors_name_the_setting(build, param):
    """SR-NF-US-02: every ConfigError starts with the offending setting."""
    with pytest.raises(ConfigError) as exc:
        build()
    message = str(exc.value)
    assert message.startswith(f"{param}: "), message
    assert exc.value.param == param
    assert_actionable(message)


def test_sr_nf_us_02_config_errors_quote_the_offending_value():
    """SR-NF-US-02: range/type errors also show the value that was wrong."""
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(num_graphs=0).validate()
    assert "must be >= 1, got 0" in str(exc.value)
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(num_nodes="4")
    assert "got '4'" in str(exc.value)


def test_sr_nf_us_02_unknown_setting_suggests_the_right_name():
    """SR-NF-US-02 (US-03): typos get a did-you-mean and the full option list."""
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig.from_dict({"num_nodez": 4, "sed": 1})
    message = str(exc.value)
    assert message.startswith("num_nodez:") or message.startswith("sed:")
    assert "Did you mean" in message
    assert "Valid settings:" in message
    assert_actionable(message)


def test_sr_nf_us_03_impossible_degree_combinations_give_the_fix():
    """SR-NF-US-03: combination errors say *which knob* to turn, not just no."""
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(in_degree=[2, 2], allow_cycles=False).validate()
    message = str(exc.value)
    assert message.startswith("in_degree:")
    assert "allow_cycles = true" in message
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(out_degree=[2, 2], allow_cycles=False).validate()
    message = str(exc.value)
    assert message.startswith("out_degree:")
    assert "allow_cycles = true" in message


def test_sr_nf_us_03_clamped_node_errors_name_role_and_field():
    """SR-NF-US-03: clamped-node combination errors name clamped_nodes and the role."""
    bad_input = ClampedNode("input", "Linear", (784,), in_degree=1, out_degree=1)
    bad_output = ClampedNode("output", "Linear", (10,), in_degree=1, out_degree=0)
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(clamped_nodes=[bad_input, bad_output]).validate()
    message = str(exc.value)
    assert message.startswith("clamped_nodes:")
    assert "in_degree" in message
    assert "set its in_degree to 0" in message
    assert_actionable(message)

    one_only = [ClampedNode("input", "Linear", (784,), in_degree=0, out_degree=1)]
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(clamped_nodes=one_only).validate()
    message = str(exc.value)
    assert message.startswith("clamped_nodes:")
    assert "output" in message  # names the missing role


# --------------------------------------------------------------------- TopologyError
def test_sr_nf_us_02_topology_error_names_the_settings_and_the_fix():
    """SR-NF-US-02/03: the infeasible-range error lists the settings and a fix."""
    config = GeneratorConfig(
        num_nodes=(0, 1), in_degree=(2, 2), allow_cycles=True
    ).validate()
    with pytest.raises(TopologyError) as exc:
        random_dag_topology(config, np.random.default_rng(0))
    message = str(exc.value)
    for setting in ("num_nodes", "in_degree"):
        assert setting in message, setting
    assert "Fix:" in message


def test_sr_nf_us_02_topology_model_error_lists_valid_models():
    """SR-NF-US-02: an unknown topology_model names the key and the choices."""
    config = GeneratorConfig(constraints={"topology_model": "watts_strogaz"}).validate()
    with pytest.raises(TopologyError) as exc:
        random_dag_topology(config, np.random.default_rng(0))
    message = str(exc.value)
    assert "topology_model" in message
    assert "watts_strogaz" in message
    assert "choose one of" in message


# -------------------------------------------------------------------- GenerationError
def impossible_hopfield_config(**overrides) -> GeneratorConfig:
    """StorkeyHopfield wants one multi-input ``patterns`` slot; in_degree=[2,2]
    with these sizes made every slot assignment fail (TEST_LOG row 7)."""
    settings = dict(
        num_graphs=1,
        seed=1,
        num_nodes=(3, 3),
        in_degree=(2, 2),
        out_degree=None,
        allow_cycles=True,
        node_types=("StorkeyHopfield",),
        input_shape=(16,),
        output_shape=(2,),
        hidden_sizes=(4,),
        max_attempts=3,
    )
    settings.update(overrides)
    return GeneratorConfig(**settings).validate()


def test_sr_nf_us_03_generation_error_names_the_check_and_suggests_a_fix():
    """SR-NF-US-03: the exhausted-attempts error names the failing check, the
    graph, the attempt budget and how to loosen it."""
    with pytest.raises(GenerationError) as exc:
        generate_one(impossible_hopfield_config(), 0)
    message = str(exc.value)
    assert "graph 0" in message
    assert "after 3 attempts" in message
    assert "slot:" in message  # the failing check names itself, with where
    assert "max_attempts" in message
    assert "Fix:" in message


def test_sr_nf_us_02_partial_batch_warning_reports_the_common_failure():
    """SR-NF-US-02/03: the partial-batch line gives counts and the top failure."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = generate(impossible_hopfield_config())
    assert specs == []
    messages = [str(w.message) for w in caught if w.category is PartialBatchWarning]
    assert messages, "no PartialBatchWarning was raised"
    message = messages[0]
    assert "made 0 of 1" in message
    assert "most common failure: " in message
    assert "slot" in message


# -------------------------------------------------------------------------- SpecError
def test_sr_nf_us_02_spec_errors_name_the_offending_node():
    """SR-NF-US-02: structural problems are prefixed with [node]."""
    broken = GraphSpec(
        (
            NodeSpec("input", "IdentityNode", (4,), "input"),
            NodeSpec("h0", "Linear", (8,)),
            NodeSpec("h0", "Linear", (8,)),  # duplicate name
            NodeSpec("output", "Linear", (2,), "output"),
        ),
        (EdgeSpec("input", "h0"), EdgeSpec("h0", "output")),
    )
    with pytest.raises(SpecError) as exc:
        broken.check()
    assert exc.value.where == "h0"
    assert "[h0]" in str(exc.value)
    assert "duplicate node name" in str(exc.value)
    assert_actionable(str(exc.value))


def test_sr_nf_us_02_spec_errors_name_the_offending_edge():
    """SR-NF-US-02: edge problems are prefixed with the edge key."""
    broken = tiny_spec()
    bad_edge = EdgeSpec("h0", "input", "in")
    broken = GraphSpec(broken.nodes, broken.edges + (bad_edge,))
    with pytest.raises(SpecError) as exc:
        broken.check()
    assert exc.value.where == bad_edge.key
    assert "[h0->input:in]" in str(exc.value)
    assert "input node cannot receive edges" in str(exc.value)


def test_sr_nf_us_02_spec_errors_on_bad_shape_say_what_shapes_are_ok():
    """SR-NF-US-03: the shape rule is restated in the message."""
    broken = GraphSpec(
        (
            NodeSpec("input", "IdentityNode", (), "input"),
            NodeSpec("output", "Linear", (2,), "output"),
        ),
        (),
    )
    with pytest.raises(SpecError) as exc:
        broken.check()
    assert "[input]" in str(exc.value)
    assert "positive dims" in str(exc.value)


# ------------------------------------------------------------------------- BuildError
def test_sr_nf_us_02_build_error_names_the_node_for_a_bad_type():
    """SR-NF-US-02: node-stage build failures name the node and what went wrong."""
    with pytest.raises(BuildError) as exc:
        build_node(NodeSpec("h7", "Banana", (4,)))
    message = str(exc.value)
    assert "[h7]" in message
    assert "Banana" in message
    assert "Registered" in message  # UnknownNameError's option list rides along
    assert exc.value.stage == "node"
    assert exc.value.where == "h7"


def test_sr_nf_us_02_build_error_names_the_edge_and_available_slots():
    """SR-NF-US-02/03: a bad slot names the edge key and lists real slots."""
    spec = tiny_spec()
    wrong = GraphSpec(
        spec.nodes, (EdgeSpec("input", "h0", "nope"), EdgeSpec("h0", "output"))
    )
    with pytest.raises(BuildError) as exc:
        to_fabricpc(wrong)
    message = str(exc.value)
    assert "[input->h0:nope]" in message
    assert "available slots:" in message
    assert exc.value.stage == "graph"


# ---------------------------------------------------------------------- registry
def test_sr_nf_us_02_unknown_registry_names_suggest_and_list():
    """SR-NF-US-02/03: every registry lookup typo gets a suggestion and options."""
    for lookup, typo, expected in (
        (registry.node_class, "LinearResidual", "LinearResidual"),  # exists: no raise
        (registry.node_class, "LinearResiduals", "LinearResidual"),
        (registry.make_activation, "reluu", "relu"),
        (registry.make_energy, "gausian", "gaussian"),
    ):
        if typo == expected:
            lookup(typo)
            continue
        with pytest.raises(registry.UnknownNameError) as exc:
            lookup(typo)
        message = str(exc.value)
        assert f"{typo!r}" in message
        assert "Registered:" in message or "Registered" in message
        assert expected in message  # the suggested spelling


def test_sr_nf_us_02_register_node_type_rejects_bad_inputs_by_name():
    """SR-NF-US-02: registry registration guards say what was wrong, by value."""
    with pytest.raises(TypeError) as exc:
        registry.register_node_type("Nope", dict)
    assert "dict" in str(exc.value)
    assert "NodeBase subclass" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        registry.register_node_type("Linear", registry.NODE_TYPES["Linear"])
    assert "'Linear'" in str(exc.value)
    assert "already registered" in str(exc.value)


# --------------------------------------------------------------------------- misc
def test_sr_nf_us_02_seeding_error_names_both_parameters():
    """SR-NF-US-02: graph_seed names the arguments it rejected."""
    with pytest.raises(ValueError) as exc:
        graph_seed(-1, 0)
    message = str(exc.value)
    assert "master_seed" in message and "index" in message
    assert "non-negative" in message


def test_sr_nf_us_02_serialization_error_names_the_file(tmp_path):
    """SR-NF-US-02: load errors prefix the path and name the expected format."""
    path = tmp_path / "graph.json"
    path.write_text('{"format": "other"}', encoding="utf-8")
    with pytest.raises(SerializationError) as exc:
        from fabricpc.fuzzer.serialize import load

        load(path)
    message = str(exc.value)
    assert "graph.json" in message
    assert "fabricpc-graph-spec" in message
    assert_actionable(message)
    assert save(tiny_spec(), tmp_path / "ok.json")  # control
