"""Clamped nodes (SR-F-05), out-degree limits (SR-F-04.3/04.4), partial batches.

Every test name carries the Report 1.5 requirement ID it proves.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter

import pytest

from fabricpc.fuzzer import (
    ClampedNode,
    ConfigError,
    GenerationError,
    GeneratorConfig,
    GenerationReport,
    PartialBatchWarning,
    generate,
    generate_one,
)
from fabricpc.fuzzer.cli import main
from fabricpc.fuzzer.pipeline import ValidationIssue, ValidationResult
from fabricpc.fuzzer.seeding import graph_rng
from fabricpc.fuzzer.topology import random_dag_topology, skeleton_issues

INPUT_SHAPE = (6,)
OUTPUT_SHAPE = (4,)
CLAMPED = (
    ClampedNode("input", "IdentityNode", INPUT_SHAPE, 0, 2),
    ClampedNode("output", "Linear", OUTPUT_SHAPE, 2, 0),
)


def config(**overrides) -> GeneratorConfig:
    settings = dict(
        num_graphs=2,
        seed=5,
        num_nodes=(2, 3),
        in_degree=(1, 2),
        out_degree=(1, 2),
        node_types=("Linear",),
        input_shape=INPUT_SHAPE,
        output_shape=OUTPUT_SHAPE,
        hidden_sizes=(6, 8),
    )
    settings.update(overrides)
    return GeneratorConfig(**settings).validate()


def degrees(spec) -> tuple[Counter, Counter]:
    out = Counter(edge.source for edge in spec.edges)
    into = Counter(edge.target for edge in spec.edges)
    return out, into


def hidden_names(spec) -> list[str]:
    return [node.name for node in spec.nodes if node.role == "hidden"]


# --------------------------------------------------------------- SR-F-05.1
def test_sr_f_05_1_clamped_nodes_present():
    """SR-F-05.1: every graph keeps the requested input/output node types."""
    specs = generate(config(clamped_nodes=CLAMPED))
    assert len(specs) == 2
    for spec in specs:
        nodes = {node.name: node for node in spec.nodes}
        assert nodes["input"].node_type == "IdentityNode"
        assert nodes["input"].shape == INPUT_SHAPE
        assert nodes["output"].node_type == "Linear"
        assert nodes["output"].shape == OUTPUT_SHAPE


# --------------------------------------------------------------- SR-F-05.2
def test_sr_f_05_2_clamped_nodes_identical_across_batch():
    """SR-F-05.2: one task map, so the clamped nodes do not vary per graph."""
    specs = generate(config(clamped_nodes=CLAMPED))
    fingerprints = set()
    for spec in specs:
        nodes = {node.name: node for node in spec.nodes}
        fingerprints.add(
            (
                nodes["input"].node_type,
                nodes["input"].shape,
                nodes["output"].node_type,
                nodes["output"].shape,
            )
        )
    assert len(fingerprints) == 1


# --------------------------------------------------------------- SR-F-05.3
def test_sr_f_05_3_clamped_degrees_honored():
    """SR-F-05.3: the input/output in/out degrees are exactly as requested."""
    specs = generate(config(clamped_nodes=CLAMPED))
    for spec in specs:
        out, into = degrees(spec)
        assert out["input"] == 2
        assert into["output"] == 2
        for name in hidden_names(spec):
            assert 1 <= out[name] <= 2


def test_sr_f_05_3_clamped_output_overrides_in_degree_range():
    """SR-F-05.3: a clamped output may ask for more inputs than in_degree."""
    clamped = (
        ClampedNode("input", "IdentityNode", INPUT_SHAPE, 0, 2),
        ClampedNode("output", "Linear", OUTPUT_SHAPE, 2, 0),
    )
    specs = generate(
        config(
            num_nodes=(2, 2),
            in_degree=(1, 1),
            out_degree=(1, 1),
            clamped_nodes=clamped,
        )
    )
    assert len(specs) == 2
    for spec in specs:
        _, into = degrees(spec)
        assert into["output"] == 2


# ----------------------------------------------------------- SR-F-04.3/04.4
def test_sr_f_04_3_out_degree_upper_bound():
    """SR-F-04.3: no hidden node exceeds the configured out-degree maximum."""
    cfg = config(num_nodes=(3, 6), in_degree=(1, 3), out_degree=(1, 3))
    for index in range(40):
        skeleton = random_dag_topology(cfg, graph_rng(0, index))
        assert skeleton_issues(skeleton, cfg) == []
        out = Counter(source for source, _ in skeleton.edges)
        assert max(out[node] for node in skeleton.hidden) <= 3


def test_sr_f_04_4_out_degree_lower_bound():
    """SR-F-04.4: every hidden node reaches the out-degree minimum."""
    cfg = config(
        num_nodes=(2, 4),
        in_degree=(1, 2),
        out_degree=(2, 3),
        allow_cycles=True,
    )
    for index in range(40):
        skeleton = random_dag_topology(cfg, graph_rng(0, index))
        assert skeleton_issues(skeleton, cfg) == []
        out = Counter(source for source, _ in skeleton.edges)
        assert min(out[node] for node in skeleton.hidden) >= 2


# ------------------------------------------------------- SR-NF-US-01/02/03
def test_sr_nf_us_01_out_degree_error_names_the_field():
    """SR-NF-US-01: an impossible out_degree error starts with the field name."""
    with pytest.raises(ConfigError, match=r"^out_degree:"):
        GeneratorConfig(out_degree=(2, 2), allow_cycles=False).validate()


def test_sr_nf_us_02_clamped_node_error_names_the_field():
    """SR-NF-US-02: a bad clamped node reports the clamped_nodes field."""
    with pytest.raises(ConfigError, match=r"^clamped_nodes:"):
        GeneratorConfig(
            clamped_nodes=(
                ClampedNode("input", "IdentityNode", INPUT_SHAPE, 1, 2),
                ClampedNode("output", "Linear", OUTPUT_SHAPE, 2, 0),
            )
        ).validate()
    with pytest.raises(ConfigError, match=r"^clamped_nodes:"):
        GeneratorConfig(clamped_nodes=[{"role": "input", "node_type": "Linear"}])


def test_sr_nf_us_03_new_settings_round_trip_through_json():
    """SR-NF-US-03: the new fields survive to_dict/from_dict and json."""
    cfg = config(clamped_nodes=CLAMPED, out_degree=(1, 2))
    again = GeneratorConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert json.loads(json.dumps(cfg.to_dict()))["out_degree"] == [1, 2]
    assert GeneratorConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg


# --------------------------------------------------------------- SR-F-07.2
def test_sr_f_07_2_attempt_limit_defaults_to_1000():
    """SR-F-07.2: the default attempt limit is 1000."""
    assert GeneratorConfig().max_attempts == 1000


def test_sr_f_07_2_attempt_limit_stops_generation():
    """SR-F-07.2: generation uses exactly max_attempts tries, then stops."""
    seen = []

    def always_reject(spec: object) -> ValidationResult:
        seen.append(spec)
        return ValidationResult.failed(ValidationIssue("never valid", check="always"))

    with pytest.raises(GenerationError) as caught:
        generate_one(
            config(max_attempts=3),
            0,
            validate=always_reject,
        )
    assert len(seen) == 3
    assert caught.value.index == 0
    assert caught.value.last_issues[0].check == "always"


# ----------------------------------------------------------- SR-F-07.5/07.6
def impossible_config(**overrides) -> GeneratorConfig:
    settings = dict(
        num_graphs=2,
        num_nodes=(3, 3),
        in_degree=(2, 2),
        allow_cycles=True,
        node_types=("StorkeyHopfield",),
        max_attempts=4,
    )
    settings.update(overrides)
    return config(**settings)


def test_sr_f_07_5_partial_batch_keeps_only_valid_graphs():
    """SR-F-07.5: an impossible request returns a partial batch plus a warning."""
    report = GenerationReport()
    with pytest.warns(PartialBatchWarning, match="made 0 of 2"):
        specs = generate(impossible_config(), report=report)
    assert specs == []
    assert report.requested == 2
    assert report.created == 0
    assert report.failed == [0, 1]
    assert report.warnings[-1].startswith("made 0 of 2")


def test_sr_f_07_6_most_common_failing_check():
    """SR-F-07.6: the report names the check that rejected the most attempts."""
    report = GenerationReport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PartialBatchWarning)
        generate(impossible_config(), report=report)
    assert report.most_common_failure == "slot"
    assert report.failure_counts["slot"] == 2 * 4


def test_sr_f_07_5_cli_prints_made_line(tmp_path, capsys):
    """SR-F-07.5: the CLI prints made X of N and the most common failure."""
    with pytest.warns(PartialBatchWarning, match="made 0 of 1"):
        code = main(
            [
                "generate",
                "-n",
                "1",
                "--seed",
                "1",
                "--set",
                "num_nodes=[3,3]",
                "--set",
                "in_degree=[2,2]",
                "--set",
                "allow_cycles=true",
                "--set",
                'node_types=["StorkeyHopfield"]',
                "--set",
                "max_attempts=2",
                "--out",
                str(tmp_path / "batch"),
            ]
        )
    out = capsys.readouterr().out
    assert code == 1
    assert "made 0 of 1, most common failure: slot" in out
