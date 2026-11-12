"""Automated non-functional checks for the fuzzer package.

Each test encodes one Report 1.5 SR-NF requirement as a runnable check so the
Appendix T row can point at it. The requirements that are CI *steps* rather
than properties of the source — DV-01 (pre-commit), DV-02 (coverage floor),
DV-03 (mypy), OP-01..03 (runner matrix) and PE-02 (runtime budget) — live in
``.github/workflows/fuzzer-nf.yml`` and are recorded in ``docs/fuzzer/STATUS.md``.

Scans work on the source text via ``ast``: they never import FabricPC, never
execute fuzzed code, and therefore cannot be fooled or slowed by the runtime.
"""

import ast
import importlib
import inspect
import re
import pkgutil
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fabricpc.fuzzer
from fabricpc.fuzzer import registry
from fabricpc.fuzzer.build import to_fabricpc
from fabricpc.fuzzer.config import ClampedNode, GeneratorConfig
from fabricpc.nodes import NodeBase

FUZZER_ROOT = Path(fabricpc.fuzzer.__file__).parent


# ----------------------------------------------------------------- infrastructure
def _module_sources() -> Iterator[Tuple[Path, ast.Module]]:
    """Parse every module shipped in ``fabricpc/fuzzer``."""
    for path in sorted(FUZZER_ROOT.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _imported_roots(tree: ast.Module) -> List[str]:
    """Top-level package names imported by a module (``fabricpc.core`` -> ``fabricpc``)."""
    roots: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0:  # absolute only; relative stays in-package
                roots.append(node.module.split(".")[0])
    return roots


# ------------------------------------------------------------------- SR-NF-DV-04
def test_sr_nf_dv_04_public_api_has_docstrings():
    """SR-NF-DV-04: every public function, class and method is documented."""
    missing: List[str] = []
    package = fabricpc.fuzzer
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if info.name.endswith("__main__"):
            continue  # runs the CLI on import
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if obj.__module__ != info.name:
                continue  # imported, documented at its definition site
            if not obj.__doc__:
                missing.append(f"{info.name}.{name}")
            if inspect.isclass(obj):
                for m_name, member in inspect.getmembers(obj):
                    if m_name.startswith("_") or inspect.isclass(member):
                        continue
                    fn = member.fget if isinstance(member, property) else member
                    if isinstance(fn, (staticmethod, classmethod)):
                        fn = fn.__func__
                    if (
                        (inspect.isfunction(fn) or inspect.isroutine(fn))
                        and getattr(fn, "__module__", None) == info.name
                        and not fn.__doc__
                    ):
                        missing.append(f"{info.name}.{name}.{m_name}")
    assert not missing, "undocumented public API: " + ", ".join(missing)


# ------------------------------------------------------------------- SR-NF-US-01
@pytest.mark.parametrize("config_cls", [GeneratorConfig, ClampedNode])
def test_sr_nf_us_01_settings_document_meaning_type_and_default(config_cls):
    """SR-NF-US-01: each setting's doc line states meaning, type and default.

    The contract: inside the class docstring, every dataclass field has a line
    starting ``<name> (`` whose first segment names the accepted type and ends
    with ``default <value>)`` or ``required)``; text after ``):`` states the
    meaning.
    """
    import dataclasses

    doc = config_cls.__doc__ or ""
    doc_lines = [line.strip() for line in doc.splitlines()]
    header_re = re.compile(r"^\w+ \(")
    problems: List[str] = []
    for field in dataclasses.fields(config_cls):
        indexes = [
            i for i, line in enumerate(doc_lines) if line.startswith(f"{field.name} (")
        ]
        if not indexes:
            problems.append(f"{field.name}: no 'name (type, default ...):' line")
            continue
        header, _, rest = doc_lines[indexes[0]].partition("):")
        if "default" not in header and "required" not in header:
            problems.append(f"{field.name}: type/default clause lacks a default")
        if rest.strip():
            continue  # meaning stated on the same line
        # Google-style entries may wrap the meaning to following lines; the
        # block must continue with prose before the next field header.
        following = doc_lines[indexes[0] + 1 : indexes[0] + 3]
        wrapped = [t for t in following if t and not header_re.match(t)]
        if not any(len(t) > 8 for t in wrapped):
            problems.append(f"{field.name}: no meaning stated after the header")
    assert not problems, "; ".join(problems)


# ------------------------------------------------------------------- SR-NF-DE-03
def test_sr_nf_de_03_no_global_randomness_in_source():
    """SR-NF-DE-03 (UR-F-08): randomness only via the passed ``rng``.

    Forbids ``import random``/``from random import ...`` and any runtime use of
    ``np.random``/``numpy.random`` convenience functions (which draw from
    implicit global state). Two things are allowed: type annotations naming
    ``np.random.Generator``, and the seeded constructors in ``seeding.py``
    (``np.random.SeedSequence``, ``np.random.default_rng``) that *create* the
    one Generator stages are allowed to use. ``jax.random`` with a fixed
    literal key is also flagged (UR-F-08); validity derives its key from the
    graph fingerprint, not a literal.
    """
    seeded_constructors = {"default_rng", "SeedSequence"}
    offenders: List[str] = []
    for path, tree in _module_sources():
        # Ids of nodes inside annotations (``rng: np.random.Generator`` names
        # a type) and of the ``np.random`` base of a seeded constructor call.
        tolerated: set = set()
        for node in ast.walk(tree):
            for ann in (
                getattr(node, "annotation", None),
                getattr(node, "returns", None),
            ):
                if ann is not None:
                    tolerated.update(id(sub) for sub in ast.walk(ann))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in seeded_constructors
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "random"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id in {"np", "numpy"}
            ):
                tolerated.add(id(node.func.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders += [
                    f"{path.name}:{node.lineno} import random"
                    for alias in node.names
                    if alias.name == "random"
                ]
            elif isinstance(node, ast.ImportFrom) and (node.module or "") == "random":
                offenders.append(f"{path.name}:{node.lineno} from random import")
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "random"
                and isinstance(node.value, ast.Name)
                and node.value.id in {"np", "numpy"}
                and id(node) not in tolerated
            ):
                offenders.append(
                    f"{path.name}:{node.lineno} {node.value.id}.random use"
                )
            elif isinstance(node, ast.Call):
                fn = node.func
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr in {"PRNGKey", "fold_in"}
                    and isinstance(fn.value, ast.Attribute)
                    and fn.value.attr == "random"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, int)
                ):
                    offenders.append(
                        f"{path.name}:{node.lineno} jax.random key with a fixed "
                        f"literal ({node.args[0].value})"
                    )
    assert not offenders, "global randomness: " + "; ".join(offenders)


# ------------------------------------------------------------------- SR-NF-DC-02
def test_sr_nf_dc_02_no_torch_or_tensorflow_imports():
    """SR-NF-DC-02: the fuzzer only depends on the JAX stack."""
    banned = {"torch", "tensorflow", "tf_keras", "keras"}
    offenders = [
        f"{path.name}:{root}"
        for path, tree in _module_sources()
        for root in _imported_roots(tree)
        if root in banned
    ]
    assert not offenders, "heavy framework imports: " + ", ".join(offenders)


# ------------------------------------------------------------------- SR-NF-DC-04
def test_sr_nf_dc_04_no_gui_or_plotting_imports():
    """SR-NF-DC-04: generation is headless; plotting belongs to callers."""
    banned = {
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "wx",
        "pygame",
        "matplotlib",
    }
    offenders = [
        f"{path.name}:{root}"
        for path, tree in _module_sources()
        for root in _imported_roots(tree)
        if root in banned
    ]
    assert not offenders, "GUI imports: " + ", ".join(offenders)


# ------------------------------------------------------------------- SR-NF-LE-02
def test_sr_nf_le_02_no_pickle_eval_exec_anywhere():
    """SR-NF-LE-02: no saved-file format that can execute code, anywhere in the package.

    The targeted load-path scan also lives in ``test_fuzzer_serialize_checks.py``;
    this is the whole-package version so a new module cannot regress it.
    """
    banned_imports = {"pickle", "marshal", "shelve", "dbm"}
    offenders: List[str] = []
    for path, tree in _module_sources():
        for root in _imported_roots(tree):
            if root in banned_imports:
                offenders.append(f"{path.name} imports {root}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"eval", "exec"}
            ):
                offenders.append(f"{path.name}:{node.lineno} calls {node.func.id}()")
    assert not offenders, "; ".join(offenders)


# ------------------------------------------------------------------- SR-NF-IN-01
def test_sr_nf_in_01_only_public_fabricpc_apis():
    """SR-NF-IN-01: never import a private (leading-underscore) FabricPC module."""
    offenders: List[str] = []
    for path, tree in _module_sources():
        for node in ast.walk(tree):
            modules: List[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            for mod in modules:
                if not mod.startswith("fabricpc"):
                    continue
                if mod.startswith("fabricpc.fuzzer"):
                    continue  # our own package
                segments = mod.split(".")
                if any(
                    seg.startswith("_") and not seg.startswith("__") for seg in segments
                ):
                    offenders.append(f"{path.name}:{node.lineno} imports {mod}")
    assert not offenders, "private FabricPC imports: " + "; ".join(offenders)


# ------------------------------------------------------------------- SR-NF-DC-03
def test_sr_nf_dc_03_generated_nodes_map_to_fabricpc_classes():
    """SR-NF-DC-03: every NodeSpec's type resolves to a real FabricPC node class."""
    from fabricpc.fuzzer.pipeline import generate

    config = GeneratorConfig(
        num_graphs=4,
        seed=3,
        num_nodes=(2, 3),
        in_degree=(1, 2),
        input_shape=(8,),
        output_shape=(2,),
        hidden_sizes=(4, 8),
    ).validate()
    specs = generate(config)
    assert specs, "tiny config produced no graphs"
    for spec in specs:
        for node in spec.nodes:
            cls = registry.node_class(node.node_type)
            assert issubclass(cls, NodeBase)
            assert cls.__module__.split(".")[0] == "fabricpc"
            assert registry.NODE_TYPES[node.node_type] is cls
        to_fabricpc(spec)  # the whole batch assembles in FabricPC, not just the names


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    seed=st.integers(0, 10_000),
    n_lo=st.integers(1, 3),
    n_span=st.integers(0, 3),
)
def test_sr_nf_dc_03_every_generated_batch_maps_to_node_classes(seed, n_lo, n_span):
    """SR-NF-DC-03: the mapping holds for random valid configs, not just one."""
    from fabricpc.fuzzer.pipeline import generate

    config = GeneratorConfig(
        num_graphs=2,
        seed=seed,
        num_nodes=(n_lo, n_lo + n_span),
        input_shape=(8,),
        output_shape=(2,),
        hidden_sizes=(4, 8),
        max_attempts=50,
    )
    for spec in generate(config):
        for node in spec.nodes:
            assert node.node_type in registry.NODE_TYPES
            assert issubclass(registry.NODE_TYPES[node.node_type], NodeBase)
