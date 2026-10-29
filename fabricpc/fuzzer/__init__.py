"""Generative graph fuzzer for FabricPC.

Generates many structurally distinct, valid, non-sequential predictive coding
graphs from a few settings, and saves them so experiments can be shared and
rerun.

Quick start::

    from fabricpc.fuzzer import GeneratorConfig, generate, save_batch, to_fabricpc

    config = GeneratorConfig(num_graphs=10, seed=7, num_nodes=(3, 6))
    specs = generate(config)
    save_batch(specs, "runs/demo", config=config.to_dict())
    structure = to_fabricpc(specs[0])   # ready for initialize_params / train

Or from the shell::

    python -m fabricpc.fuzzer generate --config fuzz.toml --out runs/demo
"""

__version__ = "0.1.0"

from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec, SpecError  # noqa: E402
from fabricpc.fuzzer.config import ConfigError, GeneratorConfig  # noqa: E402
from fabricpc.fuzzer.serialize import (  # noqa: E402
    SerializationError,
    dumps,
    load,
    load_batch,
    loads,
    save,
    save_batch,
)
from fabricpc.fuzzer.build import BuildError, to_fabricpc  # noqa: E402
from fabricpc.fuzzer.pipeline import (  # noqa: E402
    GenerationError,
    GenerationReport,
    Skeleton,
    ValidationIssue,
    ValidationResult,
    generate,
    generate_one,
)

__all__ = [
    "__version__",
    "BuildError",
    "ConfigError",
    "EdgeSpec",
    "GenerationError",
    "GenerationReport",
    "GeneratorConfig",
    "GraphSpec",
    "NodeSpec",
    "SerializationError",
    "Skeleton",
    "SpecError",
    "ValidationIssue",
    "ValidationResult",
    "dumps",
    "generate",
    "generate_one",
    "load",
    "load_batch",
    "loads",
    "save",
    "save_batch",
    "to_fabricpc",
]
