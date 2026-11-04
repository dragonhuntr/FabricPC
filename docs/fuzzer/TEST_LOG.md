# Fuzzer test log

Append-only. Every row is a real run; failure rows stay after they are fixed.

| # | Date | Commit | Branch | Command | Result | Failures, cause and fix |
|---|---|---|---|---|---|---|
| 1 | 2026-11-10 | `88578ba` | `ws/status-audit` | `python -m pytest -q --cov=fabricpc --cov-report=term-missing` | **873 passed, 8 skipped, 1 warning** in 281.75 s; coverage **86%** | None. This is the Part 0 baseline. |
| 2 | 2026-11-10 | `88578ba` | `ws/status-audit` | `python -m pytest tests/test_fuzzer_*.py -q --cov=fabricpc/fuzzer --cov-report=term-missing` | **203 passed, 2 skipped** in 53.29 s; fuzzer coverage **86%** | None. `--cov=fabricpc.fuzzer` aborts coverage's source discovery (exit 134, no output) on this Python 3.12/macOS setup; the directory form works. |
| 3 | 2026-11-10 | `88578ba` | `ws/status-audit` | `python -m pytest tests/test_fuzzer_*.py -q` | **1 failed, 202 passed, 2 skipped** in 43.54 s | `test_property_valid_configs_give_valid_skeletons` failed. Cause: Hypothesis replayed `num_nodes=(0,1)`, `in_degree=(2,2)`, `allow_cycles=True`. `GeneratorConfig.validate()` accepts it, but no hidden count in range can supply two sources (n=0 has none, n=1 has one), so `random_dag_topology` correctly raises `TopologyError`. The test asserted `num_edges is not None` before allowing that error. Fix: relax the assertion in the Part 2 branch (an infeasible in-degree range can raise even with `num_edges=None`). |
