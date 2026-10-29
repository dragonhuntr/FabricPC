"""Save and reload generated graphs as JSON.

A single graph file looks like::

    {
      "format": "fabricpc-graph-spec",
      "schema_version": 1,
      "fingerprint": "3f1c…",
      "spec": {"nodes": [...], "edges": [...], "metadata": {...}}
    }

A batch is a directory of ``graph_0000.json``, ``graph_0001.json``, … plus a
``manifest.json`` listing each file with its fingerprint and the generator
config, so a whole run can be shared and reloaded in one step.

The stored fingerprint is checked on load, which catches hand edits that left
the file inconsistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from fabricpc.fuzzer.spec import GraphSpec, SpecError

FORMAT_NAME = "fabricpc-graph-spec"
MANIFEST_FORMAT_NAME = "fabricpc-graph-batch"
SCHEMA_VERSION = 1
MANIFEST_FILE = "manifest.json"

PathLike = Union[str, Path]


class SerializationError(ValueError):
    """A file is not a valid saved graph or batch."""


def to_document(spec: GraphSpec) -> Dict[str, Any]:
    """Wrap a spec in the versioned file envelope."""
    return {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "fingerprint": spec.fingerprint(),
        "spec": spec.to_dict(),
    }


def from_document(doc: Mapping[str, Any], verify: bool = True) -> GraphSpec:
    """Unwrap and check a file envelope produced by :func:`to_document`.

    Args:
        doc: Parsed JSON document.
        verify: Check the stored fingerprint and run ``GraphSpec.check()``.

    Raises:
        SerializationError: Wrong format, unsupported version, fingerprint
            mismatch, or a malformed spec.
    """
    if not isinstance(doc, Mapping) or doc.get("format") != FORMAT_NAME:
        raise SerializationError(f"not a {FORMAT_NAME!r} document")
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SerializationError(
            f"unsupported schema_version {version!r}; this fuzzer reads "
            f"version {SCHEMA_VERSION}"
        )
    try:
        spec = GraphSpec.from_dict(doc["spec"])
        if verify:
            spec.check()
    except (KeyError, SpecError, TypeError) as exc:
        raise SerializationError(f"malformed spec: {exc}") from None
    stored = doc.get("fingerprint")
    if verify and stored is not None and stored != spec.fingerprint():
        raise SerializationError(
            "fingerprint mismatch: the nodes or edges were changed after the file "
            "was written. Remove the 'fingerprint' field to accept the edit."
        )
    return spec


def dumps(spec: GraphSpec, indent: Optional[int] = 2) -> str:
    """Serialize a spec to a JSON string."""
    return json.dumps(to_document(spec), indent=indent, sort_keys=False)


def loads(text: str, verify: bool = True) -> GraphSpec:
    """Parse a spec from a JSON string."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"invalid JSON: {exc}") from None
    return from_document(doc, verify=verify)


def save(spec: GraphSpec, path: PathLike) -> Path:
    """Write one graph to ``path`` (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(spec) + "\n", encoding="utf-8")
    return path


def load(path: PathLike, verify: bool = True) -> GraphSpec:
    """Read one graph from ``path``."""
    path = Path(path)
    try:
        return loads(path.read_text(encoding="utf-8"), verify=verify)
    except SerializationError as exc:
        raise SerializationError(f"{path}: {exc}") from None


def save_batch(
    specs: Sequence[GraphSpec],
    directory: PathLike,
    config: Optional[Mapping[str, Any]] = None,
    overwrite: bool = False,
) -> List[Path]:
    """Write a batch of graphs plus ``manifest.json`` to ``directory``.

    Args:
        specs: Graphs to save, in order.
        directory: Output directory (created if needed).
        config: Generator config dict to record in the manifest.
        overwrite: Allow writing into a directory that already has a manifest.

    Returns:
        Paths of the written graph files.

    Raises:
        FileExistsError: ``directory`` already holds a batch and ``overwrite``
            is False.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / MANIFEST_FILE
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{directory} already contains a batch; pass overwrite=True "
            "(or --overwrite) to replace it"
        )
    paths = []
    entries = []
    for i, spec in enumerate(specs):
        path = save(spec, directory / f"graph_{i:04d}.json")
        paths.append(path)
        entries.append(
            {
                "file": path.name,
                "fingerprint": spec.fingerprint(),
                "num_nodes": len(spec.nodes),
                "num_edges": len(spec.edges),
                "cyclic": spec.is_cyclic,
                "graph_seed": spec.metadata.get("graph_seed"),
            }
        )
    manifest = {
        "format": MANIFEST_FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "config": dict(config) if config is not None else None,
        "graphs": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths


def load_batch(directory: PathLike, verify: bool = True) -> List[GraphSpec]:
    """Load every graph listed in ``directory/manifest.json``, in order.

    Raises:
        SerializationError: Missing or invalid manifest, or a listed graph whose
            fingerprint does not match the manifest.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    specs = []
    for entry in manifest["graphs"]:
        spec = load(directory / entry["file"], verify=verify)
        if verify and entry.get("fingerprint") not in (None, spec.fingerprint()):
            raise SerializationError(
                f"{entry['file']}: fingerprint does not match manifest.json"
            )
        specs.append(spec)
    return specs


def load_manifest(directory: PathLike) -> Dict[str, Any]:
    """Read and check ``directory/manifest.json``."""
    path = Path(directory) / MANIFEST_FILE
    if not path.exists():
        raise SerializationError(f"{path} not found; is this a saved batch?")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SerializationError(f"{path}: invalid JSON: {exc}") from None
    if manifest.get("format") != MANIFEST_FORMAT_NAME:
        raise SerializationError(f"{path}: not a {MANIFEST_FORMAT_NAME!r} manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SerializationError(
            f"{path}: unsupported schema_version {manifest.get('schema_version')!r}"
        )
    return manifest
