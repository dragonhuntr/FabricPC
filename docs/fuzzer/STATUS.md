# Graph fuzzer requirement status (Report 1.5)

Audited: **2026-11-10** · Integration branch: **`ws/week2-summary`**

## Headline numbers

| Measure | Result |
|---|---|
| Full test suite | **895 passed, 8 skipped, 1 warning** in 254.54 s |
| Full-suite command | `python -m pytest -q --cov=fabricpc --cov-report=term-missing` |
| Coverage, whole library | **87%** (6377 statements, 809 missed) |
| Fuzzer tests only | **225 passed, 2 skipped** in 64.81 s |
| Fuzzer command | `python -m pytest tests/test_fuzzer_*.py -q --cov=fabricpc/fuzzer --cov-report=term-missing` |
| Coverage, `fabricpc/fuzzer` | **90%** (1722 statements, 170 missed) |
| Status counts (assessed IDs) | **done 23 · partial 2 · out of scope 10** |
| Assessed / unassessed | **35 / 59** of the 94 SR-F IDs |

> **Scope of this table.** Report 1.5's Table 7 was not supplied with the task.
> The 35 IDs below are every requirement the repository cites in code, tests or
> `docs/fuzzer/stage_contract.md`. The remaining 59 rows cannot be named without
> Table 7; they are counted as **not assessed** rather than guessed. Paste
> Table 7 and this table can be completed without changing the measured results
> above.

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
| SR-F-04.2 | done | `tests/test_fuzzer_topology.py::test_in_degree_limits` | Every hidden node has in-degree inside `in_degree`; a clamped output has its exact requested in-degree. |
| SR-F-04.3 | done | `tests/test_fuzzer_clamped.py::test_sr_f_04_3_out_degree_upper_bound` | `out_degree` upper bound enforced by the topology stage. |
| SR-F-04.4 | done | `tests/test_fuzzer_clamped.py::test_sr_f_04_4_out_degree_lower_bound` | Lower bound enforced; `allow_cycles` is required when the minimum is above 1, because the last hidden node otherwise feeds only the output. |
| SR-F-05.1 | done | `tests/test_fuzzer_clamped.py::test_sr_f_05_1_clamped_nodes_present` | Input/output type and shape come from `clamped_nodes`. |
| SR-F-05.2 | done | `tests/test_fuzzer_clamped.py::test_sr_f_05_2_clamped_nodes_identical_across_batch` | Every graph in the batch shares one task map. |
| SR-F-05.3 | done | `tests/test_fuzzer_clamped.py::test_sr_f_05_3_clamped_degrees_honored` and `..._clamped_output_overrides_in_degree_range` | Exact clamped in/out degrees, including an output that needs more inputs than the global `in_degree` range. |
| SR-F-06.2 | done | `tests/test_fuzzer_topology.py::test_same_seed_same_skeletons` | Same seed gives identical skeletons. |
| SR-F-07.2 | done | `tests/test_fuzzer_clamped.py::test_sr_f_07_2_attempt_limit_defaults_to_1000` and `..._stops_generation` | Default 1000; exactly `max_attempts` tries, then the graph is skipped. |
| SR-F-07.5 | done | `tests/test_fuzzer_clamped.py::test_sr_f_07_5_partial_batch_keeps_only_valid_graphs` and `..._cli_prints_made_line` | Partial batch returned with `PartialBatchWarning`; only valid graphs are returned; CLI prints `made X of N`. |
| SR-F-07.6 | done | `tests/test_fuzzer_clamped.py::test_sr_f_07_6_most_common_failing_check` | Report and CLI name the check that rejected the most attempts. |
| SR-F-11 | partial | `tests/test_fuzzer_serialize.py` (whole file), `tests/test_fuzzer_factory.py` (whole file) | JSON save/load works and is fingerprint-checked; `as_model_factory` bridges a spec into FabricPC experiments. The sponsor wants a FabricPC-native artifact instead of a JSON middle layer; `serialization_options.md` is the decision note. Provisional. |
| SR-F-12.1 | out of scope | `tests/test_fuzzer_evaluate.py::test_every_graph_gets_a_generated_and_baseline_row` | Code moved to `fabricpc/fuzzer/experiments/`; sponsor says ranking against baselines is research, not the fuzzer's job. Pending confirmation. |
| SR-F-12.2 | out of scope | `tests/test_fuzzer_evaluate.py::test_every_graph_gets_a_generated_and_baseline_row` | Same. |
| SR-F-12.3 | out of scope | `tests/test_fuzzer_evaluate.py::test_accuracy_mean_and_standard_error` | Same. |
| SR-F-12.4 | out of scope | `tests/test_fuzzer_evaluate.py::test_default_is_three_training_seeds` | Same. |
| SR-F-12.5 | out of scope | `tests/test_fuzzer_evaluate.py::test_accuracy_mean_and_standard_error` | Same. |
| SR-F-12.6 | out of scope | `tests/test_fuzzer_evaluate.py::test_csv_has_one_row_per_graph_and_model` | Same. |
| SR-F-12.7 | out of scope | `tests/test_fuzzer_evaluate.py::test_peak_memory_recorded` | Same. |
| SR-F-12.8 | out of scope | `tests/test_fuzzer_evaluate.py::test_training_time_recorded` | Same. |
| SR-F-12.9 | out of scope | `tests/test_fuzzer_evaluate.py::test_baseline_layers_and_connections_recorded` | Same. |
| SR-F-12.10 | out of scope | `tests/test_fuzzer_evaluate.py::test_training_seeds_do_not_change_the_graph` | Same. |
| SR-F-14.1 | done | `tests/test_fuzzer_upstream_sync.py::test_sr_f_14_1_every_generated_graph_builds_and_trains_one_step` | Fixed configs and seeds; every graph builds and survives one training step. |
| SR-F-14.2 | done | `tests/test_fuzzer_upstream_sync.py::test_sr_f_14_2_fingerprints_match_checked_in_golden` | Fixed seed list pinned by `tests/golden/fuzzer_fingerprints.json`. |
| SR-NF-IN-04 | done | `pyproject.toml` dependency list; `tests/test_fuzzer_*.py` | No new runtime dependency was added by the fuzzer. `pytest-cov` is a dev-only test tool. |
| SR-NF-LE-02 | partial | `tests/test_fuzzer_clamped.py::test_sr_f_07_5_cli_prints_made_line` | CLI coverage rose from 0% to 62% with one generate path tested; `inspect`, `check`, `init-config` and the `__main__` shim are still uncovered. |
| SR-NF-US-01 | done | `tests/test_fuzzer_clamped.py::test_sr_nf_us_01_out_degree_error_names_the_field` | `out_degree` errors start with the field name. |
| SR-NF-US-02 | done | `tests/test_fuzzer_clamped.py::test_sr_nf_us_02_clamped_node_error_names_the_field` | Clamped-node entries report `clamped_nodes`, including missing keys. |
| SR-NF-US-03 | done | `tests/test_fuzzer_clamped.py::test_sr_nf_us_03_new_settings_round_trip_through_json` | New fields survive `to_dict`/`from_dict` and JSON. |
| SR-NF-PE-02 | done | `.github/workflows/fuzzer-sync.yml` | CI runs the sync suite under a 5-minute budget; 13.6 s locally. Not yet executed on GitHub. |
| UR-F-08 | done | `tests/test_fuzzer_assign_validate.py::test_ignores_unknown_constraints_and_global_rng`, `tests/test_fuzzer_topology.py::test_same_seed_same_skeletons` | Randomness flows only through the passed `numpy.random.Generator`. |
| UR-F-11 | done | `tests/test_fuzzer_assign_validate.py::test_rejects_edge_into_input` and the other localisation tests | Validation issues name the node or edge at fault. |
| UR-F-13 | done | `tests/test_fuzzer_topology.py::test_is_the_default_topology_stage`, `tests/test_fuzzer_assign_validate.py::test_is_the_default_stage` | Stage tests run against `DEFAULT_STAGES`. |
| UR-NF-DE-01 | done | `tests/test_fuzzer_upstream_sync.py::test_sr_f_14_2_fingerprints_match_checked_in_golden`, `..._golden_mismatch_message_is_readable`; TEST_LOG row 10 | `--update-golden` exists, and a deliberately changed seed fails with a readable message. |

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
