"""Tests for fabricpc.fuzzer.evaluate: generated graphs vs sequential baselines.

Most tests swap in stub train/eval functions so they run in milliseconds; a
few train for real on a tiny synthetic task.
"""

import json
import time

import numpy as np
import pytest

from fabricpc.fuzzer import GeneratorConfig, generate
from fabricpc.fuzzer.baselines import sequential_baseline
from fabricpc.fuzzer.evaluate import (
    CSV_COLUMNS,
    EvalSettings,
    PeakMemory,
    evaluate_batch,
    evaluate_graph,
    read_csv,
    synthetic_data_factory,
)
from fabricpc.fuzzer.pipeline import chain_topology, linear_assign
from fabricpc.fuzzer.seeding import graph_rng
from fabricpc.fuzzer.spec import EdgeSpec, GraphSpec, NodeSpec
from fabricpc.training import TrainResult

IN, OUT = (6,), (3,)


def small_config(**kw):
    base = dict(
        num_graphs=3,
        seed=5,
        num_nodes=(1, 3),
        input_shape=IN,
        output_shape=OUT,
        hidden_sizes=(4, 8),
    )
    base.update(kw)
    return GeneratorConfig(**base)


def dag_spec():
    nodes = (
        NodeSpec("x", "IdentityNode", IN, "input"),
        NodeSpec("a", "Linear", (8,), activation="tanh"),
        NodeSpec("b", "Linear", (5,), activation="relu"),
        NodeSpec("y", "Linear", OUT, "output", "softmax", energy="cross_entropy"),
    )
    edges = (
        EdgeSpec("x", "a"),
        EdgeSpec("x", "b"),
        EdgeSpec("a", "b"),
        EdgeSpec("b", "y"),
        EdgeSpec("a", "y"),
    )
    return GraphSpec(nodes, edges)


class Recorder:
    """Stub train/eval pair. Accuracy depends on the seed and the graph size,
    so tests can predict it; every call is recorded."""

    def __init__(self, sleep_s=0.0):
        self.sleep_s = sleep_s
        self.calls = []

    def train(self, params, structure, loader, optimizer, config, key, **kw):
        time.sleep(self.sleep_s)
        self.calls.append(("train", kw))
        return TrainResult(params, None, 0, [], [])

    def evaluate(self, params, structure, loader, config, key, **kw):
        seed = loader.seed
        return {"accuracy": 0.1 * seed + 0.01 * len(structure.nodes), "energy": 1.0}

    def settings(self, **kw):
        return EvalSettings(train_fn=self.train, eval_fn=self.evaluate, **kw)


class SeededLoader:
    def __init__(self, seed):
        self.seed = seed

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


def seeded_factory(seed):
    return SeededLoader(seed), SeededLoader(seed)


# --------------------------------------------- SR-F-12.1, 12.2: coverage
def test_every_graph_gets_a_generated_and_baseline_row():
    """SR-F-12.1/12.2: each generated graph is trained alongside a baseline."""
    specs = generate(small_config())
    report = evaluate_batch(specs, seeded_factory, Recorder().settings())
    assert len(report) == 2 * len(specs)
    rows = [(r.graph_id, r.model) for r in report]
    assert rows == [
        (i, m) for i in range(len(specs)) for m in ("generated", "sequential")
    ]
    for spec, (gen, base) in zip(specs, zip(*[iter(report)] * 2)):
        assert gen.fingerprint == spec.fingerprint()
        assert base.fingerprint == sequential_baseline(spec).fingerprint()


def test_baseline_shares_input_and_output():
    gen, base = evaluate_graph(dag_spec(), seeded_factory, Recorder().settings())
    first, last = base.architecture["layers"][0], base.architecture["layers"][-1]
    assert first["name"] == "x" and first["shape"] == list(IN)
    assert last["name"] == "y" and last["shape"] == list(OUT)
    assert base.model == "sequential" and gen.model == "generated"


def test_custom_baseline_is_used():
    custom = sequential_baseline(dag_spec(), hidden_sizes=[2])
    _, base = evaluate_graph(
        dag_spec(), seeded_factory, Recorder().settings(), baseline=custom
    )
    assert base.fingerprint == custom.fingerprint()


# ----------------------------------------------- SR-F-12.3, 12.4, 12.5
def test_default_is_three_training_seeds():
    """SR-F-12.4: three training seeds by default, the same for both models."""
    rec = Recorder()
    gen, base = evaluate_graph(dag_spec(), seeded_factory, rec.settings())
    assert gen.training_seeds == base.training_seeds == [0, 1, 2]
    assert len([c for c in rec.calls if c[0] == "train"]) == 6


def test_accuracy_mean_and_standard_error():
    """SR-F-12.3/12.5: mean test accuracy and its standard error over seeds."""
    rec = Recorder()
    gen, _ = evaluate_graph(
        dag_spec(), seeded_factory, rec.settings(training_seeds=(1, 4, 6))
    )
    acc = np.array([0.1 * s + 0.04 for s in (1, 4, 6)])
    assert gen.accuracies == pytest.approx(acc.tolist())
    assert gen.accuracy_mean == pytest.approx(acc.mean())
    assert gen.accuracy_se == pytest.approx(acc.std(ddof=1) / np.sqrt(3))


def test_single_seed_has_zero_standard_error():
    gen, _ = evaluate_graph(
        dag_spec(), seeded_factory, Recorder().settings(training_seeds=(3,))
    )
    assert gen.accuracy_se == 0.0


def test_settings_reject_bad_seeds():
    for seeds in [(), (1, 1), (-1,)]:
        with pytest.raises(ValueError):
            EvalSettings(training_seeds=seeds)


def test_missing_accuracy_is_an_error():
    s = EvalSettings(
        training_seeds=(0,),
        train_fn=Recorder().train,
        eval_fn=lambda *a, **k: {"energy": 1.0},
    )
    with pytest.raises(KeyError, match="accuracy"):
        evaluate_graph(dag_spec(), seeded_factory, s)


# ----------------------------------------------------- SR-F-12.7, 12.8
def test_peak_memory_recorded():
    """SR-F-12.7: peak memory is recorded per model."""
    gen, base = evaluate_graph(dag_spec(), seeded_factory, Recorder().settings())
    for r in (gen, base):
        assert r.peak_memory_bytes > 0
        assert r.memory_source in ("device", "rss", "max_rss")


def test_peak_memory_sees_allocations():
    with PeakMemory() as idle:
        pass
    if idle.source == "max_rss":
        pytest.skip("only the lifetime maximum is available here")
    with PeakMemory(poll_s=0.001) as mem:
        block = np.ones(64 * 2**20, dtype=np.uint8)
        time.sleep(0.02)
        del block
    assert mem.peak_bytes >= idle.peak_bytes + 32 * 2**20


def test_training_time_recorded():
    """SR-F-12.8: wall-clock training time is recorded per model."""
    gen, _ = evaluate_graph(
        dag_spec(),
        seeded_factory,
        Recorder(sleep_s=0.03).settings(training_seeds=(0, 1)),
    )
    assert all(r.train_time_s >= 0.03 for r in gen.runs)
    assert gen.train_time_mean_s >= 0.03


# ------------------------------------------------------------ SR-F-12.9
def test_baseline_layers_and_connections_recorded():
    """SR-F-12.9: baseline layer sizes and connections are recorded."""
    _, base = evaluate_graph(dag_spec(), seeded_factory, Recorder().settings())
    row = base.to_dict()
    assert row["layer_sizes"] == [6, 8, 5, 3]
    assert row["connections"] == ["x->b0:in", "b0->b1:in", "b1->y:in"]
    assert row["num_nodes"] == 4 and row["num_edges"] == 3


# ----------------------------------------------------------- SR-F-12.10
def test_training_seeds_do_not_change_the_graph():
    """SR-F-12.10: changing training seeds never changes the graph."""
    config = small_config()
    specs = generate(config)
    before = [s.to_dict() for s in specs]
    rec = Recorder()
    a = evaluate_batch(specs, seeded_factory, rec.settings(training_seeds=(0, 1)))
    b = evaluate_batch(specs, seeded_factory, rec.settings(training_seeds=(7, 8)))
    assert [r.fingerprint for r in a] == [r.fingerprint for r in b]
    assert [s.to_dict() for s in specs] == before
    assert [s.to_dict() for s in generate(config)] == before
    assert [r.graph_seed for r in a] == [r.graph_seed for r in b]
    assert a.results[0].accuracies != b.results[0].accuracies


# ----------------------------------------------- independent of stage
def test_works_for_any_producing_stage():
    config = small_config()
    chain = linear_assign(
        chain_topology(config, graph_rng(0, 0)), config, graph_rng(0, 0)
    )
    for spec in (chain, dag_spec(), generate(config)[0]):
        gen, base = evaluate_graph(spec, seeded_factory, Recorder().settings())
        assert gen.fingerprint == spec.fingerprint()


def test_graph_id_defaults():
    specs = generate(small_config(num_graphs=2))
    stripped = [GraphSpec(s.nodes, s.edges) for s in specs]
    report = evaluate_batch(stripped, seeded_factory, Recorder().settings())
    assert [r.graph_id for r in report] == [0, 0, 1, 1]


# ------------------------------------------------- real, tiny training
def test_real_training_is_deterministic_per_seed():
    data = synthetic_data_factory(IN, 3, n_train=64, n_test=32, batch_size=16)
    s = EvalSettings(training_seeds=(0,), num_epochs=1, learning_rate=1e-2)
    a, _ = evaluate_graph(dag_spec(), data, s)
    b, _ = evaluate_graph(dag_spec(), data, s)
    assert a.accuracies == b.accuracies
    assert 0.0 <= a.accuracy_mean <= 1.0
    assert a.num_params > 0
    assert a.runs[0].train_time_s > 0


def test_synthetic_data_matches_spec_shapes():
    train, test = synthetic_data_factory(IN, 3, n_train=10, batch_size=4)(0)
    x, y = next(iter(train))
    assert x.shape == (4, *IN) and y.shape == (4, 3)
    assert len(train) == 3


# ------------------------------------------------------------ SR-F-12.6
def test_csv_has_one_row_per_graph_and_model(tmp_path):
    """SR-F-12.6: results are written as CSV, one row per graph and model."""
    specs = generate(small_config())
    report = evaluate_batch(specs, seeded_factory, Recorder().settings())
    path = report.to_csv(tmp_path / "out" / "results.csv")
    rows = read_csv(path)
    assert len(rows) == 2 * len(specs)
    assert [(r["graph_id"], r["model"]) for r in rows] == [
        (r.graph_id, r.model) for r in report
    ]
    assert list(rows[0]) == list(CSV_COLUMNS)


def test_csv_round_trips_to_plain_dicts(tmp_path):
    report = evaluate_batch([dag_spec()], seeded_factory, Recorder().settings())
    rows = read_csv(report.to_csv(tmp_path / "r.csv"))
    expected = report.to_dicts()
    for got, want in zip(rows, expected):
        assert got.keys() == want.keys()
        for key in want:
            if isinstance(want[key], float):
                assert got[key] == pytest.approx(want[key])
            else:
                assert got[key] == want[key], key
    assert rows[0]["graph_seed"] is None
    assert json.loads(json.dumps(rows)) == rows


def test_results_fit_in_generator_constraints(tmp_path):
    report = evaluate_batch([dag_spec()], seeded_factory, Recorder().settings())
    rows = read_csv(report.to_csv(tmp_path / "r.csv"))
    config = small_config(constraints={"evaluation": rows})
    config.validate()
    assert GeneratorConfig.from_dict(config.to_dict()).constraints["evaluation"] == rows
