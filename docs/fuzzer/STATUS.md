# Graph fuzzer requirement status (Report 1.5)

Audited: **2026-11-10** · Commit: **`88578ba`** (`feature/fuzzer-foundation`)

## Headline numbers

| Measure | Result |
|---|---|
| Full test suite | **873 passed, 8 skipped, 1 warning** in 281.75 s |
| Full-suite command | `python -m pytest -q --cov=fabricpc --cov-report=term-missing` |
| Coverage, whole library | **86%** (6142 statements, 847 missed) |
| Fuzzer tests only | **203 passed, 2 skipped** in 53.29 s |
| Fuzzer command | `python -m pytest tests/test_fuzzer_*.py -q --cov=fabricpc/fuzzer --cov-report=term-missing` |
| Coverage, `fabricpc/fuzzer` | **86%** (1487 statements, 210 missed) |
| Status counts (assessed IDs) | **done 7 · partial 7 · not started 11 · out of scope 10** |
| Assessed / unassessed | **35 / 59** of the 94 SR-F IDs |

> **Scope of this table.** Report 1.5's Table 7 was not supplied with the task.
> The 35 IDs below are every requirement the repository already cites in code,
> tests or `docs/fuzzer/stage_contract.md`. The remaining 59 rows cannot be
> named without Table 7; they are counted as **not assessed** rather than
> guessed. Paste Table 7 and this table can be completed without changing the
> measured results above.

## Status key

- **done** — implemented and a test proves it.
- **partial** — implementation or test exists, but the requirement is not fully met.
- **not started** — no implementation.
- **out of scope** — sponsor has ruled it out or it is research, pending confirmation.
- **not assessed** — Table 7 row not available.

## Requirements

| ID | Status | Test that proves it | Notes |
|---|---|---|---|
| SR-F-03.3 | done | `tests/test_fuzzer_topology.py::test_hidden_count_inside_range_bounds_included` | Hidden count is inside `num_nodes`, both ends reachable. |
| SR-F-03.5 | done | `tests/test_fuzzer_topology.py::test_edge_count_inside_range_bounds_included` | Edge count is inside `num_edges`, both ends reachable. |
| SR-F-04.2 | done | `tests/test_fuzzer_topology.py::test_in_degree_limits` | Every hidden node and the output has in-degree inside `in_degree`. |
| SR-F-04.3 | not started | — | Out-degree **upper** bound missing; `GeneratorConfig` has no `out_degree`. Tracked by `docs/fuzzer/stage_contract.md`. |
| SR-F-04.4 | not started | — | Out-degree **lower** bound missing for the same reason. |
| SR-F-05.1 | not started | — | No user-supplied clamped input/output nodes. |
| SR-F-05.2 | not started | — | No guarantee that a whole batch shares one taskmap. |
| SR-F-05.3 | not started | — | No clamped-node in-degree / out-degree enforcement. |
| SR-F-06.2 | done | `tests/test_fuzzer_topology.py::test_same_seed_same_skeletons` | Same seed gives identical skeletons. |
| SR-F-07.2 | partial | `tests/test_fuzzer_assign_validate.py` (retry path) | Retries up to `max_attempts`, but the default is **100**, not the required **1000**, and a failure raises instead of returning a partial batch. |
| SR-F-07.5 | not started | — | No "made X of N" count in the CLI. |
| SR-F-07.6 | not started | — | No most-common-failing-check report. |
| SR-F-11 | partial | `tests/test_fuzzer_serialize.py` (whole file) | JSON `GraphSpec` save/load works and is fingerprint-checked, but the sponsor wants graphs produced from FabricPC itself rather than a JSON middle layer. Provisional; see `serialization_options.md`. |
| SR-F-12.1 | out of scope | `tests/test_fuzzer_evaluate.py::test_every_graph_gets_a_generated_and_baseline_row` | Evaluation implemented; sponsor says scoring against baselines is research, not the fuzzer's job. Pending confirmation. |
| SR-F-12.2 | out of scope | `tests/test_fuzzer_evaluate.py::test_every_graph_gets_a_generated_and_baseline_row` | Same. |
| SR-F-12.3 | out of scope | `tests/test_fuzzer_evaluate.py::test_accuracy_mean_and_standard_error` | Same. |
| SR-F-12.4 | out of scope | `tests/test_fuzzer_evaluate.py::test_default_is_three_training_seeds` | Same. |
| SR-F-12.5 | out of scope | `tests/test_fuzzer_evaluate.py::test_accuracy_mean_and_standard_error` | Same. |
| SR-F-12.6 | out of scope | `tests/test_fuzzer_evaluate.py::test_csv_has_one_row_per_graph_and_model` | Same. |
| SR-F-12.7 | out of scope | `tests/test_fuzzer_evaluate.py::test_peak_memory_recorded` | Same. |
| SR-F-12.8 | out of scope | `tests/test_fuzzer_evaluate.py::test_training_time_recorded` | Same. |
| SR-F-12.9 | out of scope | `tests/test_fuzzer_evaluate.py::test_baseline_layers_and_connections_recorded` | Same. |
| SR-F-12.10 | out of scope | `tests/test_fuzzer_evaluate.py::test_training_seeds_do_not_change_the_graph` | Same. |
| SR-F-14.1 | not started | — | No test suite that catches an upstream FabricPC change breaking the fuzzer. |
| SR-F-14.2 | not started | — | No fixed seed list for an upstream-sync suite. |
| SR-NF-IN-04 | done | `pyproject.toml` dependency list; `tests/test_fuzzer_*.py` | No new runtime dependency was added by the fuzzer. `pytest-cov` is a dev-only test tool. |
| SR-NF-LE-02 | partial | — | Fuzzer package coverage is 86%, but `cli.py` is 0% covered and `__main__.py` is never exercised. Needs CLI tests. |
| SR-NF-US-01 | partial | `tests/test_fuzzer_config.py::test_validation_names_the_parameter` | `ConfigError` names the offending field, but the new fields (`out_degree`, clamped nodes) do not exist yet. |
| SR-NF-US-02 | partial | `tests/test_fuzzer_config.py::test_unknown_key_suggests_fix` | Unknown keys get a "did you mean" hint; new settings need the same. |
| SR-NF-US-03 | partial | `tests/test_fuzzer_config.py::test_validation_names_the_parameter` | Combination checks name the field; clamped-node combinations are not checked. |
| SR-NF-PE-02 | not started | — | No upstream-sync CI job and no runtime budget for one. |
| UR-F-08 | done | `tests/test_fuzzer_assign_validate.py::test_ignores_unknown_constraints_and_global_rng`, `tests/test_fuzzer_topology.py::test_same_seed_same_skeletons` | Randomness flows only through the passed `numpy.random.Generator`. |
| UR-F-11 | done | `tests/test_fuzzer_assign_validate.py::test_rejects_edge_into_input` and the other localisation tests | Validation issues name the node or edge at fault. |
| UR-F-13 | partial | `tests/test_fuzzer_topology.py::test_is_the_default_topology_stage`, `tests/test_fuzzer_assign_validate.py::test_is_the_default_stage` | Tests run against `DEFAULT_STAGES`, but there is no shared parameterised harness across every stage. |
| UR-NF-DE-01 | not started | — | No golden fingerprints and no `--update-golden` flag. |

## Not assessed (59 IDs)

Table 7 has 94 SR-F rows. The repository cites 35 of them; the other 59 are
outside sections **SR-F-01**, **SR-F-02**, **SR-F-08**, **SR-F-09**,
**SR-F-10**, **SR-F-13** and the unreferenced sub-IDs of sections 03–07 and 14.
They are listed here as one group rather than invented one by one.

## How the numbers were produced

```bash
python -m pytest -q --cov=fabricpc --cov-report=term-missing
python -m pytest tests/test_fuzzer_*.py -q \
    --cov=fabricpc/fuzzer --cov-report=term-missing
```

`pytest-cov` is a dev dependency; `--cov=fabricpc.fuzzer` crashes coverage's
source discovery on this Python/macOS combination, so the directory form
`--cov=fabricpc/fuzzer` is used.
