"""Tests for fabricpc.fuzzer.config (GeneratorConfig)."""

import json

import pytest
from hypothesis import given, settings, strategies as st

from fabricpc.fuzzer.config import ConfigError, GeneratorConfig, parse_override


def test_defaults_are_valid():
    GeneratorConfig().validate()


def test_int_becomes_range_and_lists_become_tuples():
    cfg = GeneratorConfig(num_nodes=4, node_types=["Linear"], input_shape=[28, 28])
    assert cfg.num_nodes == (4, 4)
    assert cfg.node_types == ("Linear",)
    assert cfg.input_shape == (28, 28)
    assert cfg == GeneratorConfig(num_nodes=(4, 4), input_shape=(28, 28))


@pytest.mark.parametrize(
    "kwargs, param, fragment",
    [
        ({"num_graphs": 0}, "num_graphs", ">= 1"),
        ({"seed": -1}, "seed", ">= 0"),
        ({"num_nodes": (5, 2)}, "num_nodes", "greater than maximum"),
        ({"in_degree": (0, 2)}, "in_degree", ">= 1"),
        ({"in_degree": (2, 3)}, "in_degree", "impossible without cycles"),
        (
            {"num_nodes": 1, "in_degree": (3, 3), "allow_cycles": True},
            "in_degree",
            "exceeds",
        ),
        ({"num_nodes": (4, 4), "num_edges": 3}, "num_edges", "too small"),
        ({"num_nodes": 2, "in_degree": 1, "num_edges": 10}, "num_edges", "exceeds"),
        ({"node_types": []}, "node_types", "at least one"),
        ({"node_types": ["Linaer"]}, "node_types", "Did you mean 'Linear'"),
        ({"hidden_activations": ["rleu"]}, "hidden_activations", "not registered"),
        ({"output_energy": "mse"}, "output_energy", "not registered"),
        ({"input_shape": (0,)}, "input_shape", "positive"),
        ({"hidden_sizes": (64, -1)}, "hidden_sizes", "positive"),
        ({"unroll": 0}, "unroll", ">= 1"),
        ({"allow_cycles": "yes"}, "allow_cycles", "true/false"),
        ({"constraints": {"f": object()}}, "constraints", "JSON-serializable"),
    ],
)
def test_validation_names_the_parameter(kwargs, param, fragment):
    with pytest.raises(ConfigError, match=fragment) as exc:
        GeneratorConfig(**kwargs).validate()
    assert exc.value.param == param
    assert str(exc.value).startswith(f"{param}:")


@pytest.mark.parametrize("value", ["abc", [1, 2, 3], True, [1.5, 2]])
def test_bad_range_type(value):
    with pytest.raises(ConfigError) as exc:
        GeneratorConfig(num_nodes=value)
    assert exc.value.param == "num_nodes"


def test_cycles_relax_in_degree_minimum():
    GeneratorConfig(num_nodes=3, in_degree=(2, 3), allow_cycles=True).validate()


def test_unknown_key_suggests_fix():
    with pytest.raises(ConfigError, match="Did you mean 'num_nodes'") as exc:
        GeneratorConfig.from_dict({"num_node": 3})
    assert exc.value.param == "num_node"


@settings(max_examples=50, deadline=None)
@given(
    st.builds(
        GeneratorConfig,
        num_graphs=st.integers(1, 50),
        seed=st.integers(0, 2**31),
        num_nodes=st.tuples(st.integers(0, 4), st.integers(4, 10)),
        allow_cycles=st.booleans(),
        hidden_sizes=st.lists(st.integers(1, 512), min_size=1, max_size=4),
        constraints=st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
    )
)
def test_dict_round_trip(cfg):
    again = GeneratorConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    assert again.constraints == cfg.constraints


def test_json_file_round_trip(tmp_path):
    cfg = GeneratorConfig(num_graphs=3, seed=11, num_nodes=(2, 4), allow_cycles=True)
    path = tmp_path / "cfg.json"
    cfg.to_file(path)
    assert GeneratorConfig.from_file(path) == cfg


def test_toml_file(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text(
        'num_graphs = 4\nseed = 9\nnum_nodes = [3, 5]\nnode_types = ["Linear"]\n'
        "allow_cycles = true\n\n[constraints]\nmax_depth = 4\n"
    )
    cfg = GeneratorConfig.from_file(path).validate()
    assert cfg.num_nodes == (3, 5)
    assert cfg.allow_cycles is True
    assert cfg.constraints == {"max_depth": 4}


@pytest.mark.parametrize(
    "name, text", [("c.yaml", "a: 1"), ("c.json", "{oops"), ("c.toml", "= =")]
)
def test_bad_files(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    with pytest.raises(ConfigError):
        GeneratorConfig.from_file(path)


def test_with_overrides():
    cfg = GeneratorConfig(seed=1).with_overrides(seed=5, num_nodes=[1, 2])
    assert cfg.seed == 5
    assert cfg.num_nodes == (1, 2)
    with pytest.raises(ConfigError):
        GeneratorConfig().with_overrides(bogus=1)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("num_nodes=[3,6]", ("num_nodes", [3, 6])),
        ("allow_cycles=true", ("allow_cycles", True)),
        ("seed=4", ("seed", 4)),
        ("output_energy=gaussian", ("output_energy", "gaussian")),
        ('constraints={"a": 1}', ("constraints", {"a": 1})),
    ],
)
def test_parse_override(text, expected):
    assert parse_override(text) == expected


def test_parse_override_needs_equals():
    with pytest.raises(ConfigError):
        parse_override("seed")
