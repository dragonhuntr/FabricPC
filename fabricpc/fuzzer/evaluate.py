"""Train generated graphs and their sequential baselines, and report results.

Evaluation consumes the pipeline's output and never changes it::

    specs = generate(config)            # or load_batch("runs/demo")
    report = evaluate_batch(specs, data_factory, EvalSettings())
    report.to_csv("results.csv")

For every graph, :func:`evaluate_graph` builds the sequential baseline from
:mod:`fabricpc.fuzzer.baselines`, then trains both with FabricPC once per
training seed and records, per model:

- test classification accuracy, its mean and standard error over seeds,
- peak memory during training,
- wall-clock training time,
- layer sizes and connections.

Training seeds are separate from the generation seed. A spec is plain data
with no randomness left in it, and every training run draws its parameter
initialization, batch order and inference noise from ``PRNGKey(seed)`` and
``data_factory(seed)`` only. Changing ``EvalSettings.training_seeds``
therefore cannot change which graph is trained.

``data_factory(seed) -> (train_loader, test_loader)`` follows the same
convention as :mod:`fabricpc.experiments`: both models of a seed get freshly
built loaders from the same call, so they see the same batch stream.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import jax
import numpy as np
import optax

from fabricpc.core.inference import InferenceBase
from fabricpc.experiments.statistics import descriptive_stats
from fabricpc.fuzzer.baselines import (
    BASELINE_KIND,
    describe_architecture,
    sequential_baseline,
)
from fabricpc.fuzzer.build import to_fabricpc
from fabricpc.fuzzer.spec import GraphSpec
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import evaluate as fabricpc_evaluate
from fabricpc.training import train as fabricpc_train

GENERATED = "generated"
DataFactory = Callable[[int], Tuple[Any, Any]]


@dataclass(frozen=True)
class EvalSettings:
    """How every model is trained and scored.

    Attributes:
        training_seeds: One training run per seed, for every model (SR-F-12.4).
        num_epochs: Passed to ``fabricpc.training.train``.
        learning_rate: Adam learning rate, used when ``optimizer`` is ``None``.
        algorithm: ``"pc"`` or ``"backprop"``.
        inference: Inference algorithm for :func:`to_fabricpc`; ``None`` uses
            its default.
        optimizer: Optax optimizer; ``None`` means ``optax.adam(learning_rate)``.
        train_fn: Same signature as ``fabricpc.training.train``. Swap it to
            mock or shrink training in tests.
        eval_fn: Same signature as ``fabricpc.training.evaluate``; must return
            an ``"accuracy"`` entry.
        memory_poll_s: How often peak memory is sampled during training.
    """

    training_seeds: Tuple[int, ...] = (0, 1, 2)
    num_epochs: float = 1
    learning_rate: float = 1e-3
    algorithm: str = "pc"
    inference: Optional[InferenceBase] = None
    optimizer: Optional[optax.GradientTransformation] = None
    train_fn: Callable = fabricpc_train
    eval_fn: Callable = fabricpc_evaluate
    memory_poll_s: float = 0.01

    def __post_init__(self):
        seeds = tuple(int(s) for s in self.training_seeds)
        if not seeds:
            raise ValueError("training_seeds must not be empty")
        if any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"training_seeds must be distinct and non-negative, got {seeds}"
            )
        object.__setattr__(self, "training_seeds", seeds)

    def make_optimizer(self) -> optax.GradientTransformation:
        return self.optimizer or optax.adam(self.learning_rate)


def _device_bytes() -> Optional[int]:
    stats = jax.local_devices()[0].memory_stats()
    if stats and "bytes_in_use" in stats:
        return int(stats["bytes_in_use"])
    return None


def _process_rss() -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def _max_rss() -> int:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class PeakMemory:
    """Context manager that records the highest memory use seen inside it.

    Samples, in order of preference, the accelerator's ``bytes_in_use``, the
    process resident set size (needs ``psutil``), or the process's lifetime
    maximum RSS. ``source`` says which one was used; only the first two are
    specific to the code inside the block.
    """

    def __init__(self, poll_s: float = 0.01):
        self.poll_s = poll_s
        if _device_bytes() is not None:
            self.source, self._read = "device", _device_bytes
        elif _process_rss() is not None:
            self.source, self._read = "rss", _process_rss
        else:
            self.source, self._read = "max_rss", _max_rss
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, int(self._read() or 0))

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._sample()

    def __enter__(self) -> "PeakMemory":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._sample()
        self._stop.set()
        self._thread.join()


@dataclass(frozen=True)
class SeedRun:
    """One training run of one model."""

    seed: int
    accuracy: float
    train_time_s: float
    peak_memory_bytes: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResult:
    """Everything recorded for one model of one graph, over all seeds.

    Attributes:
        graph_id: Which generated graph this row belongs to.
        model: ``"generated"`` or ``"sequential"``.
        fingerprint: Structure fingerprint of the trained spec.
        graph_seed: Generation seed of the graph, when the spec records it.
        runs: One :class:`SeedRun` per training seed, in seed order.
        architecture: Layer sizes and connections from
            :func:`~fabricpc.fuzzer.baselines.describe_architecture`.
        num_params: Trainable parameter count.
        memory_source: How peak memory was measured (see :class:`PeakMemory`).
    """

    graph_id: int
    model: str
    fingerprint: str
    graph_seed: Optional[int]
    runs: Tuple[SeedRun, ...]
    architecture: Dict[str, Any]
    num_params: int
    is_cyclic: bool
    memory_source: str

    @property
    def training_seeds(self) -> List[int]:
        return [r.seed for r in self.runs]

    @property
    def accuracies(self) -> List[float]:
        return [r.accuracy for r in self.runs]

    @property
    def accuracy_mean(self) -> float:
        return float(descriptive_stats(np.asarray(self.accuracies)).mean)

    @property
    def accuracy_se(self) -> float:
        """Standard error of the mean over seeds; 0 for a single seed."""
        return float(descriptive_stats(np.asarray(self.accuracies)).se)

    @property
    def peak_memory_bytes(self) -> int:
        return max(r.peak_memory_bytes for r in self.runs)

    @property
    def train_time_mean_s(self) -> float:
        return float(np.mean([r.train_time_s for r in self.runs]))

    def to_dict(self) -> Dict[str, Any]:
        """Flat, JSON-safe row; also the CSV row (lists stay lists here)."""
        arch = self.architecture
        return {
            "graph_id": self.graph_id,
            "model": self.model,
            "fingerprint": self.fingerprint,
            "graph_seed": self.graph_seed,
            "training_seeds": self.training_seeds,
            "accuracies": self.accuracies,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_se": self.accuracy_se,
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_source": self.memory_source,
            "train_time_mean_s": self.train_time_mean_s,
            "train_times_s": [r.train_time_s for r in self.runs],
            "num_nodes": len(arch["layers"]),
            "num_edges": len(arch["connections"]),
            "num_params": self.num_params,
            "is_cyclic": self.is_cyclic,
            "layer_sizes": [math.prod(layer["shape"]) for layer in arch["layers"]],
            "connections": arch["connections"],
        }


def _count_params(params: Any) -> int:
    return int(sum(np.size(leaf) for leaf in jax.tree_util.tree_leaves(params)))


def train_seed(
    spec: GraphSpec, data_factory: DataFactory, seed: int, settings: EvalSettings
) -> Tuple[SeedRun, int, str]:
    """Train and test ``spec`` once with training seed ``seed``.

    Returns:
        ``(run, num_params, memory_source)``.
    """
    structure = to_fabricpc(spec, inference=settings.inference)
    graph_key, train_key, eval_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = initialize_params(structure, graph_key)
    train_loader, test_loader = data_factory(seed)
    config = {"num_epochs": settings.num_epochs}

    with PeakMemory(settings.memory_poll_s) as mem:
        start = time.perf_counter()
        result = settings.train_fn(
            params,
            structure,
            train_loader,
            settings.make_optimizer(),
            config,
            train_key,
            algorithm=settings.algorithm,
            verbose=False,
        )
        trained = jax.block_until_ready(result.params)
        elapsed = time.perf_counter() - start

    metrics = settings.eval_fn(
        trained, structure, test_loader, config, eval_key, algorithm=settings.algorithm
    )
    if "accuracy" not in metrics:
        raise KeyError(f"eval_fn returned no 'accuracy' (got {sorted(metrics)})")
    run = SeedRun(
        seed=seed,
        accuracy=float(metrics["accuracy"]),
        train_time_s=elapsed,
        peak_memory_bytes=mem.peak_bytes,
        metrics={k: float(v) for k, v in metrics.items()},
    )
    return run, _count_params(params), mem.source


def train_model(
    spec: GraphSpec,
    data_factory: DataFactory,
    settings: EvalSettings,
    graph_id: int,
    model: str,
) -> ModelResult:
    """Train ``spec`` once per training seed and collect a :class:`ModelResult`."""
    runs, num_params, source = [], 0, ""
    for seed in settings.training_seeds:
        run, num_params, source = train_seed(spec, data_factory, seed, settings)
        runs.append(run)
    graph_seed = spec.metadata.get("graph_seed")
    return ModelResult(
        graph_id=graph_id,
        model=model,
        fingerprint=spec.fingerprint(),
        graph_seed=None if graph_seed is None else int(graph_seed),
        runs=tuple(runs),
        architecture=describe_architecture(spec),
        num_params=num_params,
        is_cyclic=spec.is_cyclic,
        memory_source=source,
    )


def evaluate_graph(
    spec: GraphSpec,
    data_factory: DataFactory,
    settings: Optional[EvalSettings] = None,
    graph_id: Optional[int] = None,
    baseline: Optional[GraphSpec] = None,
) -> Tuple[ModelResult, ModelResult]:
    """Train ``spec`` and its sequential baseline over the same training seeds.

    Args:
        spec: A generated graph, from any stage or loaded from disk.
        data_factory: ``seed -> (train_loader, test_loader)``.
        settings: Training settings; defaults to :class:`EvalSettings()`.
        graph_id: Row id; defaults to ``spec.metadata["graph_index"]`` or 0.
        baseline: Use this baseline instead of :func:`sequential_baseline`.

    Returns:
        ``(generated_result, baseline_result)``.
    """
    settings = settings or EvalSettings()
    if graph_id is None:
        graph_id = int(spec.metadata.get("graph_index", 0))
    baseline = baseline if baseline is not None else sequential_baseline(spec)
    return (
        train_model(spec, data_factory, settings, graph_id, GENERATED),
        train_model(baseline, data_factory, settings, graph_id, BASELINE_KIND),
    )


@dataclass
class EvaluationReport:
    """Results for a batch: one :class:`ModelResult` per graph and model."""

    results: List[ModelResult] = field(default_factory=list)

    def __iter__(self) -> Iterator[ModelResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)


def evaluate_batch(
    specs: Sequence[GraphSpec],
    data_factory: DataFactory,
    settings: Optional[EvalSettings] = None,
) -> EvaluationReport:
    """:func:`evaluate_graph` for every spec. Row ids follow ``graph_index``
    metadata when every spec has it, else the position in ``specs``."""
    settings = settings or EvalSettings()
    use_meta = all("graph_index" in s.metadata for s in specs)
    report = EvaluationReport()
    for pos, spec in enumerate(specs):
        gid = int(spec.metadata["graph_index"]) if use_meta else pos
        report.results.extend(evaluate_graph(spec, data_factory, settings, gid))
    return report


class ArrayLoader:
    """Minimal loader over in-memory arrays; reshuffles every epoch from ``seed``."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.x, self.y, self.batch_size = x, y, batch_size
        self.shuffle, self.seed = shuffle, seed
        self._epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.x) / self.batch_size)

    def __iter__(self):
        idx = np.arange(len(self.x))
        if self.shuffle:
            np.random.default_rng((self.seed, self._epoch)).shuffle(idx)
            self._epoch += 1
        for i in range(0, len(idx), self.batch_size):
            j = idx[i : i + self.batch_size]
            yield self.x[j], self.y[j]


def synthetic_data_factory(
    input_shape: Sequence[int],
    num_classes: int,
    n_train: int = 256,
    n_test: int = 128,
    batch_size: int = 32,
    noise: float = 0.5,
    data_seed: int = 0,
) -> DataFactory:
    """A small, learnable classification task for tests and smoke runs.

    Each class is a Gaussian blob around a random centre. The data itself is
    fixed by ``data_seed``; the training seed only sets the batch order.
    """
    rng = np.random.default_rng(data_seed)
    shape = tuple(int(d) for d in input_shape)
    centres = rng.normal(size=(num_classes, *shape)).astype(np.float32)

    def draw(n: int) -> Tuple[np.ndarray, np.ndarray]:
        labels = rng.integers(num_classes, size=n)
        x = centres[labels] + noise * rng.normal(size=(n, *shape))
        return x.astype(np.float32), np.eye(num_classes, dtype=np.float32)[labels]

    train_xy, test_xy = draw(n_train), draw(n_test)

    def factory(seed: int) -> Tuple[ArrayLoader, ArrayLoader]:
        return (
            ArrayLoader(*train_xy, batch_size, shuffle=True, seed=seed),
            ArrayLoader(*test_xy, batch_size),
        )

    return factory
