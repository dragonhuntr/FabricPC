"""Hardening of save/load for the provisional JSON format.

The sponsor has not picked a serialization option yet, so the format stays
unchanged and the load path is hardened instead. Sub-ID numbering follows the
task brief (Report 1.5's Table 7 has not been supplied, so the mapping below
is stated per test rather than assumed):

- SR-F-11.2  malformed JSON is rejected
- SR-F-11.3  a missing field is rejected, naming the field
- SR-F-11.4  a fingerprint mismatch is rejected
- SR-F-11.5  a saved batch regenerates identically from its recorded seed
- SR-F-11.6  a regeneration mismatch produces a readable diff
- SR-F-11.8  reload is faithful and the format stays versioned
- SR-NF-LE-02  the load path never touches pickle or eval
"""

import ast
import json
from pathlib import Path

import pytest

from fabricpc.fuzzer.config import GeneratorConfig
from fabricpc.fuzzer.serialize import (
    FORMAT_NAME,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    SerializationError,
    describe_diff,
    dumps,
    load,
    load_batch,
    loads,
    regenerate_batch,
    regenerate_graph,
    save,
    save_batch,
)
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec

FUZZER_DIR = Path(__file__).resolve().parent.parent / "fabricpc" / "fuzzer"


def _tiny_config(num_graphs: int = 3) -> GeneratorConfig:
    """Small CPU-only config whose batches regenerate in about a second."""
    return GeneratorConfig(
        num_graphs=num_graphs,
        seed=5,
        num_nodes=(2, 3),
        in_degree=(1, 2),
        input_shape=(8,),
        output_shape=(2,),
        hidden_sizes=(4, 8),
        hidden_activations=("relu",),
        max_attempts=200,
    ).validate()


def _example() -> GraphSpec:
    return GraphSpec(
        (
            NodeSpec("input", "IdentityNode", (4,), "input"),
            NodeSpec("h0", "Linear", (8,), activation="relu"),
            NodeSpec("output", "Linear", (3,), "output", activation="softmax"),
        ),
        (EdgeSpec("input", "h0"), EdgeSpec("h0", "output")),
        {"master_seed": 1},
    )


# ------------------------------------------------------------------ SR-F-11.2
def test_sr_f_11_2_rejects_invalid_json(tmp_path):
    """SR-F-11.2: text that is not JSON fails with a message that says so."""
    with pytest.raises(SerializationError, match="invalid JSON"):
        loads("{not json")
    path = save(_example(), tmp_path / "g.json")
    path.write_text("[[[", encoding="utf-8")
    with pytest.raises(SerializationError) as exc:
        load(path)
    assert "invalid JSON" in str(exc.value)
    assert str(path) in str(exc.value)  # names the offending file


def test_sr_f_11_2_rejects_json_that_is_not_a_graph_document():
    """SR-F-11.2: valid JSON of the wrong shape names the expected format."""
    for text in ("[]", '"just a string"', "null", '{"format": "other"}'):
        with pytest.raises(SerializationError) as exc:
            loads(text)
        assert FORMAT_NAME in str(exc.value)


# ------------------------------------------------------------------ SR-F-11.3
def test_sr_f_11_3_rejects_missing_top_level_field_naming_it():
    """SR-F-11.3: a spec without 'nodes' or 'edges' names the missing key."""
    for key in ("nodes", "edges"):
        doc = json.loads(dumps(_example()))
        doc["spec"].pop(key)
        with pytest.raises(SerializationError) as exc:
            loads(json.dumps(doc))
        assert "missing" in str(exc.value)
        assert key in str(exc.value)


@pytest.mark.parametrize("key", ["name", "node_type", "shape"])
def test_sr_f_11_3_rejects_node_missing_required_field_naming_it(key):
    """SR-F-11.3: a node entry missing a required field names that field."""
    doc = json.loads(dumps(_example()))
    doc.pop("fingerprint")
    del doc["spec"]["nodes"][1][key]
    with pytest.raises(SerializationError) as exc:
        loads(json.dumps(doc))
    assert "missing key(s)" in str(exc.value)
    assert key in str(exc.value)


@pytest.mark.parametrize("key", ["source", "target"])
def test_sr_f_11_3_rejects_edge_missing_required_field_naming_it(key):
    """SR-F-11.3: an edge entry missing source/target names that field."""
    doc = json.loads(dumps(_example()))
    doc.pop("fingerprint")
    del doc["spec"]["edges"][0][key]
    with pytest.raises(SerializationError) as exc:
        loads(json.dumps(doc))
    assert "missing key(s)" in str(exc.value)
    assert key in str(exc.value)


# ------------------------------------------------------------------ SR-F-11.4
def test_sr_f_11_4_rejects_fingerprint_mismatch_with_guidance():
    """SR-F-11.4: a post-write edit is caught and the message says how to accept it."""
    doc = json.loads(dumps(_example()))
    doc["spec"]["nodes"][1]["shape"] = [16]
    with pytest.raises(SerializationError) as exc:
        loads(json.dumps(doc))
    assert "fingerprint mismatch" in str(exc.value)
    assert "fingerprint" in str(exc.value)
    # The guidance: removing the field declares the edit intentional.
    doc.pop("fingerprint")
    assert loads(json.dumps(doc)).node("h0").shape == (16,)


def test_sr_f_11_4_batch_detects_file_swapped_against_manifest(tmp_path):
    """SR-F-11.4: a graph file replaced under a still-old manifest is rejected."""
    first = _example()
    second = GraphSpec(
        first.nodes[:2] + (NodeSpec("output", "Linear", (5,), "output"),), first.edges
    )
    save_batch([first, second], tmp_path, config=_tiny_config(1).to_dict())
    save(second, tmp_path / "graph_0000.json")
    with pytest.raises(SerializationError) as exc:
        load_batch(tmp_path)
    assert "graph_0000.json" in str(exc.value)


# ------------------------------------------------------- SR-F-11.5 and SR-F-11.6
def test_sr_f_11_5_saved_batch_regenerates_identically_from_seed(tmp_path):
    """SR-F-11.5: regenerating from the manifest seed returns identical graphs."""
    config = _tiny_config()
    specs = _generate(config)
    save_batch(specs, tmp_path, config=config.to_dict())
    assert regenerate_batch(tmp_path) == []


def test_sr_f_11_6_regeneration_diff_names_graph_and_changed_fields(tmp_path):
    """SR-F-11.6: when the config no longer matches, the diff says what differs."""
    config = _tiny_config()
    specs = _generate(config)
    save_batch(specs, tmp_path, config=config.to_dict())

    manifest_path = tmp_path / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["hidden_sizes"] = [31]  # pretend the config drifted
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    lines = regenerate_batch(tmp_path)
    assert lines, "expected at least one difference line"
    assert any("fingerprint" in line for line in lines)
    assert any(
        "shape" in line and "saved" in line and "new" in line for line in lines
    ), lines


def test_sr_f_11_6_describe_diff_is_empty_for_equal_specs():
    """SR-F-11.6: equal structure means an empty diff, ignoring metadata."""
    assert describe_diff(_example(), _example().with_metadata(seed=9)) == []


def test_sr_f_11_6_describe_diff_shows_node_and_edge_changes():
    """SR-F-11.6: node field, node added and edge added differences each get a line."""
    moved = GraphSpec(
        (
            NodeSpec("input", "IdentityNode", (4,), "input"),
            NodeSpec("h0", "Linear", (16,), activation="relu"),
            NodeSpec("h1", "Linear", (16,), activation="tanh"),
            NodeSpec("output", "Linear", (3,), "output", activation="softmax"),
        ),
        (
            EdgeSpec("input", "h0"),
            EdgeSpec("h0", "output"),
            EdgeSpec("input", "h1"),
        ),
    )
    lines = describe_diff(_example(), moved)
    assert any("'h0' field 'shape'" in line for line in lines)
    assert any("'h1' is new" in line for line in lines)
    assert any("'input->h1:in' is new" in line for line in lines)


def test_sr_f_11_5_regenerate_graph_rebuilds_one_saved_graph(tmp_path):
    """SR-F-11.5: one graph can be regenerated alone from its metadata."""
    specs = _generate(_tiny_config(2))
    saved = load(save(specs[1], tmp_path / "g.json"))
    fresh = regenerate_graph(saved, exclude=[specs[0].fingerprint()])
    assert fresh.fingerprint() == specs[1].fingerprint()


def test_sr_f_11_5_regenerate_graph_needs_generation_metadata():
    """SR-F-11.5: a hand-made spec states what metadata regeneration is missing."""
    with pytest.raises(SerializationError) as exc:
        regenerate_graph(_example())
    assert "graph_index" in str(exc.value) or "config" in str(exc.value)


def test_sr_f_11_5_regenerate_batch_requires_saved_config(tmp_path):
    """SR-F-11.5: a manifest without a recorded config says how to fix it."""
    save_batch([_example()], tmp_path)
    with pytest.raises(SerializationError) as exc:
        regenerate_batch(tmp_path)
    assert MANIFEST_FILE in str(exc.value)
    assert "config" in str(exc.value)


def test_sr_f_11_5_check_regenerate_rejects_plain_file(tmp_path):
    """SR-F-11.5: the CLI refuses --regenerate outside a batch directory."""
    from fabricpc.fuzzer.cli import main

    path = save(_example(), tmp_path / "g.json")
    assert main(["check", "--regenerate", str(path)]) == 2


# ------------------------------------------------------------------ SR-F-11.8
def test_sr_f_11_8_reload_is_faithful_and_versioned(tmp_path):
    """SR-F-11.8: reload reproduces every field, and the envelope states its version."""
    spec = _example().with_metadata(master_seed=3, graph_index=1, graph_seed=4)
    path = save(spec, tmp_path / "g.json")
    again = load(path)
    assert again == spec
    assert again.metadata == spec.metadata
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == FORMAT_NAME
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["fingerprint"] == spec.fingerprint()


# ---------------------------------------------------------------- SR-NF-LE-02
def _scan_no_unsafe_load(path: Path):
    """Yield pickle/eval/exec style nodes an AST scan finds in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [
                alias.name
                for alias in node.names
                if alias.name.split(".")[0] in {"pickle", "marshal", "shelve"}
            ]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in {"pickle", "marshal", "shelve"}:
                bad.append(node.module)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in {"eval", "exec"}:
                bad.append(fn.id)
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "pickle"
            ):
                bad.append(f"pickle.{fn.attr}")
    return bad


@pytest.mark.parametrize("module", ["serialize.py", "spec.py"])
def test_sr_nf_le_02_load_path_uses_no_pickle_or_eval(module):
    """SR-NF-LE-02: JSON-only load cannot run code or unpickle arbitrary bytes."""
    assert _scan_no_unsafe_load(FUZZER_DIR / module) == []


# --------------------------------------------------------------------- helpers
def _generate(config: GeneratorConfig):
    """Generate with the real default stages (validation smoke-trains each graph)."""
    from fabricpc.fuzzer.pipeline import generate

    return generate(config)
