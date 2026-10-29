"""Tests for fabricpc.fuzzer.serialize (save / reload graphs)."""

import json

import pytest
from hypothesis import HealthCheck, given, settings

from fuzzer_strategies import graph_specs
from fabricpc.fuzzer.serialize import (
    MANIFEST_FILE,
    SCHEMA_VERSION,
    SerializationError,
    dumps,
    load,
    load_batch,
    load_manifest,
    loads,
    save,
    save_batch,
)
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec


@settings(max_examples=80, deadline=None)
@given(graph_specs())
def test_string_round_trip(spec):
    again = loads(dumps(spec))
    assert again == spec
    assert again.metadata == spec.metadata
    assert again.fingerprint() == spec.fingerprint()


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(spec=graph_specs())
def test_file_round_trip(tmp_path, spec):
    path = save(spec, tmp_path / "sub" / "g.json")
    assert load(path) == spec


def test_output_is_stable():
    """Same spec -> byte-identical file, so saved graphs diff cleanly in git."""
    assert dumps(_example()) == dumps(loads(dumps(_example())))


def _example():
    return GraphSpec(
        (
            NodeSpec("input", "IdentityNode", (4,), "input"),
            NodeSpec("h0", "Linear", (8,), activation="relu"),
            NodeSpec("output", "Linear", (3,), "output", activation="softmax"),
        ),
        (EdgeSpec("input", "h0"), EdgeSpec("h0", "output")),
        {"master_seed": 1},
    )


def test_document_envelope():
    doc = json.loads(dumps(_example()))
    assert doc["format"] == "fabricpc-graph-spec"
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["fingerprint"] == _example().fingerprint()


def test_tampered_file_detected():
    doc = json.loads(dumps(_example()))
    doc["spec"]["nodes"][1]["shape"] = [16]
    with pytest.raises(SerializationError, match="fingerprint mismatch"):
        loads(json.dumps(doc))
    # Removing the fingerprint accepts the intentional edit.
    del doc["fingerprint"]
    assert loads(json.dumps(doc)).node("h0").shape == (16,)


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d.update(format="something-else"), "not a"),
        (lambda d: d.update(schema_version=99), "unsupported schema_version"),
        (
            lambda d: d["spec"]["edges"].append({"source": "h0", "target": "nope"}),
            "malformed",
        ),
        (lambda d: d["spec"].pop("nodes"), "malformed"),
    ],
)
def test_bad_documents(mutate, fragment):
    doc = json.loads(dumps(_example()))
    doc.pop("fingerprint")
    mutate(doc)
    with pytest.raises(SerializationError, match=fragment):
        loads(json.dumps(doc))


def test_invalid_json():
    with pytest.raises(SerializationError, match="invalid JSON"):
        loads("{not json")


def test_batch_round_trip(tmp_path):
    specs = [_example(), _example().with_metadata(master_seed=2)]
    paths = save_batch(specs, tmp_path / "run", config={"seed": 1})
    assert [p.name for p in paths] == ["graph_0000.json", "graph_0001.json"]
    manifest = load_manifest(tmp_path / "run")
    assert manifest["config"] == {"seed": 1}
    assert manifest["graphs"][0]["fingerprint"] == specs[0].fingerprint()
    assert load_batch(tmp_path / "run") == specs


def test_batch_refuses_to_overwrite(tmp_path):
    save_batch([_example()], tmp_path)
    with pytest.raises(FileExistsError):
        save_batch([_example()], tmp_path)
    save_batch([_example()], tmp_path, overwrite=True)


def test_batch_detects_swapped_file(tmp_path):
    ex = _example()
    other = GraphSpec(
        ex.nodes[:2] + (NodeSpec("output", "Linear", (5,), "output"),), ex.edges
    )
    save_batch([_example()], tmp_path)
    save(other, tmp_path / "graph_0000.json")
    with pytest.raises(SerializationError, match="manifest"):
        load_batch(tmp_path)


def test_missing_manifest(tmp_path):
    with pytest.raises(SerializationError, match=MANIFEST_FILE):
        load_batch(tmp_path)
