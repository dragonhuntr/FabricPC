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
        raise SerializationError(
            f"not a {FORMAT_NAME!r} document; expected a JSON object with "
            "'format', 'schema_version' and 'spec' keys"
        )
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


def describe_diff(before: GraphSpec, after: GraphSpec) -> List[str]:
    """Human-readable line diff of two specs' structure.

    Used by :func:`regenerate_batch` when a regenerated graph no longer matches
    the saved one: each line names the node or edge that differs and shows both
    values, so a reader can see which stage or setting drifted between runs.
    Metadata is ignored, like in :meth:`~fabricpc.fuzzer.spec.GraphSpec.fingerprint`.

    Args:
        before: The saved spec.
        after: The regenerated spec.

    Returns:
        One line per difference (removed/added/changed nodes, then
        removed/added edges). Empty when the structures are equal.
    """
    lines: List[str] = []
    before_nodes = {n.name: n.to_dict() for n in before.nodes}
    after_nodes = {n.name: n.to_dict() for n in after.nodes}
    for name in sorted(set(before_nodes) - set(after_nodes)):
        lines.append(f"node {name!r} is in the saved graph but not the new one")
    for name in sorted(set(after_nodes) - set(before_nodes)):
        lines.append(f"node {name!r} is new (fields: {sorted(after_nodes[name])})")
    for name in sorted(set(before_nodes) & set(after_nodes)):
        old, new = before_nodes[name], after_nodes[name]
        for field_name in sorted(set(old) | set(new)):
            if old.get(field_name) != new.get(field_name):
                lines.append(
                    f"node {name!r} field {field_name!r}: "
                    f"saved {old.get(field_name)!r}, new {new.get(field_name)!r}"
                )
    before_edges = {e.key: e for e in before.edges}
    after_edges = {e.key: e for e in after.edges}
    for key in sorted(set(before_edges) - set(after_edges)):
        lines.append(f"edge {key!r} is in the saved graph but not the new one")
    for key in sorted(set(after_edges) - set(before_edges)):
        lines.append(f"edge {key!r} is new")
    return lines


def regenerate_graph(spec: GraphSpec, exclude: Sequence[str] = ()) -> GraphSpec:
    """Regenerate one saved graph from the seed recorded in its metadata.

    A spec from :func:`fabricpc.fuzzer.generate` carries ``config`` and
    ``graph_index`` in its metadata, which together with the master seed fully
    determine it, so a saved graph can be rebuilt without the original run.

    Args:
        spec: A saved spec (e.g. from :func:`load`).
        exclude: Fingerprints to skip as duplicates, mirroring how a batch
            generation deduplicates earlier graphs in the batch.

    Returns:
        The freshly generated spec; compare with
        :meth:`~fabricpc.fuzzer.spec.GraphSpec.fingerprint` or
        :func:`describe_diff`.

    Raises:
        SerializationError: The spec carries no generation metadata, so it was
            not produced by the pipeline and cannot be regenerated.
    """
    from fabricpc.fuzzer.config import GeneratorConfig
    from fabricpc.fuzzer.pipeline import generate_one

    missing = [k for k in ("config", "graph_index") if k not in spec.metadata]
    if missing:
        raise SerializationError(
            f"cannot regenerate from seed: metadata is missing {missing}; "
            "only specs produced by generate() (or saved batches reloaded "
            "from them) record the config and graph index"
        )
    config = GeneratorConfig.from_dict(spec.metadata["config"])
    return generate_one(config, int(spec.metadata["graph_index"]), exclude=set(exclude))


def regenerate_batch(directory: PathLike) -> List[str]:
    """Re-run generation from a saved batch's seed and report every difference.

    Reads the generator config from ``manifest.json``, regenerates each graph
    from its recorded index (which fixes its per-graph seed), and compares
    fingerprints against the manifest. An empty result proves the batch is
    reproducible from its seed on this machine and FabricPC version.

    Args:
        directory: A batch directory written by :func:`save_batch`.

    Returns:
        Readable difference lines — empty means every graph regenerated
        identically. Each mismatched graph contributes one summary line plus
        the :func:`describe_diff` lines showing what changed.
    """
    from fabricpc.fuzzer.config import GeneratorConfig
    from fabricpc.fuzzer.pipeline import GenerationError, generate_one

    directory = Path(directory)
    manifest = load_manifest(directory)
    raw_config = manifest.get("config")
    if raw_config is None:
        raise SerializationError(
            f"{directory / MANIFEST_FILE}: no 'config' recorded, cannot "
            "regenerate from seed; save batches via the CLI or pass config= "
            "to save_batch"
        )
    config = GeneratorConfig.from_dict(raw_config).validate()
    lines: List[str] = []
    seen: set = set()
    for i, entry in enumerate(manifest["graphs"]):
        spec = load(directory / entry["file"])
        index = int(spec.metadata.get("graph_index", i))
        try:
            fresh = generate_one(config, index, exclude=seen)
        except GenerationError as exc:
            lines.append(f"{entry['file']} (graph {index}): regeneration failed: {exc}")
            continue
        seen.add(fresh.fingerprint())
        saved_fp = entry.get("fingerprint") or spec.fingerprint()
        if fresh.fingerprint() != saved_fp:
            lines.append(
                f"{entry['file']} (graph {index}): fingerprint {saved_fp[:12]} "
                f"does not match regeneration from seed {config.seed}: "
                f"{fresh.fingerprint()[:12]}"
            )
            lines.extend(f"  {d}" for d in describe_diff(spec, fresh))
    return lines


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
