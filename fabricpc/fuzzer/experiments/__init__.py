"""Evaluation experiments kept out of the fuzzer's default path.

The sponsor's Nov 3 decision moved "score generated graphs against sequential
baselines" out of the fuzzer's job: it is research, not a sync guarantee. The
code is kept here, importable and tested, but it is no longer re-exported from
:mod:`fabricpc.fuzzer`. See ``docs/fuzzer/STATUS.md`` (SR-F-12.x, "out of
scope, pending sponsor confirmation").
"""

from fabricpc.fuzzer.experiments.baselines import (
    describe_architecture,
    sequential_baseline,
)
from fabricpc.fuzzer.experiments.evaluate import (
    EvalSettings,
    EvaluationReport,
    ModelResult,
    evaluate_batch,
    evaluate_graph,
    read_csv,
)

__all__ = [
    "EvalSettings",
    "EvaluationReport",
    "ModelResult",
    "describe_architecture",
    "evaluate_batch",
    "evaluate_graph",
    "read_csv",
    "sequential_baseline",
]
