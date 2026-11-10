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

    report = evaluate_batch(specs, data_factory)   # vs sequential baselines
    report.to_csv("runs/demo/results.csv")

Or from the shell::

    python -m fabricpc.fuzzer generate --config fuzz.toml --out runs/demo
"""

__version__ = "0.1.0"

from fabricpc.fuzzer.baselines import (
    describe_architecture,
    sequential_baseline,
)
from fabricpc.fuzzer.build import BuildError, to_fabricpc
from fabricpc.fuzzer.config import ConfigError, GeneratorConfig
from fabricpc.fuzzer.evaluate import (
    EvalSettings,
    EvaluationReport,
    ModelResult,
    evaluate_batch,
    evaluate_graph,
    read_csv,
)
from fabricpc.fuzzer.factory import ModelFactory, as_model_factory
from fabricpc.fuzzer.pipeline import (
    GenerationError,
    GenerationReport,
    Skeleton,
    ValidationIssue,
    ValidationResult,
    generate,
    generate_one,
)
from fabricpc.fuzzer.serialize import (
    SerializationError,
    dumps,
    load,
    load_batch,
    loads,
    save,
    save_batch,
)
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec, SpecError

__all__ = [
    "BuildError",
    "ConfigError",
    "EdgeSpec",
    "EvalSettings",
    "EvaluationReport",
    "GenerationError",
    "GenerationReport",
    "GeneratorConfig",
    "GraphSpec",
    "ModelFactory",
    "ModelResult",
    "NodeSpec",
    "SerializationError",
    "Skeleton",
    "SpecError",
    "ValidationIssue",
    "ValidationResult",
    "__version__",
    "as_model_factory",
    "describe_architecture",
    "dumps",
    "evaluate_batch",
    "evaluate_graph",
    "generate",
    "generate_one",
    "load",
    "load_batch",
    "loads",
    "read_csv",
    "save",
    "save_batch",
    "sequential_baseline",
    "to_fabricpc",
]
