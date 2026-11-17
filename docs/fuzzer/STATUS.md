# Graph fuzzer requirement status (Report 1.5)

Audited: **2026-11-16** · Commit: **`ebfbae2`** (`feature/fuzzer-foundation`)

## Headline numbers

| Measure | Result |
|---|---|
| Full test suite | **975 passed, 8 skipped, 1 warning** in 351.83 s |
| Full-suite command | `python -m pytest -q --cov=fabricpc --cov-report=term-missing` |
| Coverage, whole library | **87%** (6480 statements, 821 missed) |
| Fuzzer tests only | **305 passed, 2 skipped** in 162.84 s |
| Fuzzer command | `python -m pytest tests/test_fuzzer_*.py -q --cov=fabricpc/fuzzer --cov-report=term-missing` |
| Coverage, `fabricpc/fuzzer` | **90%** (1825 statements, 183 missed) |
| Status counts (assessed IDs) | **done 23 · partial 2 · not started 0 · out of scope 10** |
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
| SR-F-04.3 | done | `tests/test_fuzzer_clamped.py::test_sr_f_04_3_out_degree_upper_bound` | `out_degree` upper bound enforced in the topology draw and post-checked. |
| SR-F-04.4 | done | `tests/test_fuzzer_clamped.py::test_sr_f_04_4_out_degree_lower_bound` | Lower bound enforced (needs `allow_cycles` for minimum > 1; the config validator says so). |
| SR-F-05.1 | done | `tests/test_fuzzer_clamped.py::test_sr_f_05_1_clamped_nodes_present` | `clamped_nodes` pins input/output node types and shapes. |
| SR-F-05.2 | done | `tests/test_fuzzer_clamped.py::test_sr_f_05_2_clamped_nodes_identical_across_batch` | One task map for the whole batch; clamped nodes are identical per graph. |
| SR-F-05.3 | done | `tests/test_fuzzer_clamped.py::test_sr_f_05_3_clamped_degrees_honored` | Clamped in-/out-degrees are exact; the output can override `in_degree`. |
| SR-F-06.2 | done | `tests/test_fuzzer_topology.py::test_same_seed_same_skeletons` | Same seed gives identical skeletons. |
| SR-F-07.2 | done | `tests/test_fuzzer_clamped.py::test_sr_f_07_2_attempt_limit_defaults_to_1000` | `max_attempts` defaults to **1000** and generation stops after exactly that many tries. |
| SR-F-07.5 | done | `tests/test_fuzzer_clamped.py::test_sr_f_07_5_partial_batch_keeps_only_valid_graphs` | Short batches return the valid graphs plus a `PartialBatchWarning`; the CLI prints the made-line. |
| SR-F-07.6 | done | `tests/test_fuzzer_clamped.py::test_sr_f_07_6_most_common_failing_check` | The warning names the check that rejected the most attempts. |
| SR-F-11 | partial | `tests/test_fuzzer_serialize.py` + `tests/test_fuzzer_serialize_checks.py` | JSON `GraphSpec` save/load works, fingerprint-checked, with `check --train-step` / `check --regenerate`; still **provisional** until the sponsor picks A/B/C/D (`serialization_options.md`). |
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
| SR-F-14.1 | done | `tests/test_fuzzer_upstream_sync.py::test_sr_f_14_1_every_generated_graph_builds_and_trains_one_step` | Sync suite rebuilds fixed graphs on the current FabricPC and trains one step. |
| SR-F-14.2 | done | `tests/test_fuzzer_upstream_sync.py::test_sr_f_14_2_fingerprints_match_checked_in_golden` | Fixed seed list + checked-in golden fingerprints + `--update-golden`; runs in CI (`fuzzer-sync.yml`). |
| SR-NF-IN-04 | done | `pyproject.toml` dependency list; `tests/test_fuzzer_*.py` | No new runtime dependency was added by the fuzzer. `pytest-cov` is a dev-only test tool. |
| SR-NF-LE-02 | done | `tests/test_fuzzer_nonfunctional.py::test_sr_nf_le_02_no_pickle_eval_exec_anywhere` | No pickle/eval/exec anywhere in the package (AST scan); the load path stays pure-JSON with fingerprint checks; CLI and `__main__` are now exercised by tests. |
| SR-NF-US-01 | done | `tests/test_fuzzer_config.py::test_validation_names_the_parameter`, `tests/test_fuzzer_clamped.py::test_sr_nf_us_01_*` | Every setting including `out_degree` and clamped nodes has meaning + type + default in docs and a named-field error. |
| SR-NF-US-02 | done | `tests/test_fuzzer_error_messages.py` | Audit triggers every exception family and asserts the message names the setting/node/edge at fault. |
| SR-NF-US-03 | done | `tests/test_fuzzer_error_messages.py` | Messages state what to change; `GenerationError`/`TopologyError` carry explicit `Fix:` lines. |
| SR-NF-PE-02 | done | `.github/workflows/fuzzer-nf.yml` | Upstream-sync CI (`fuzzer-sync.yml`) plus a 300 s wall-clock budget asserted inside the fuzzer job. |
| UR-F-08 | done | `tests/test_fuzzer_assign_validate.py::test_ignores_unknown_constraints_and_global_rng`, `tests/test_fuzzer_topology.py::test_same_seed_same_skeletons` | Randomness flows only through the passed `numpy.random.Generator`. |
| UR-F-11 | done | `tests/test_fuzzer_assign_validate.py::test_rejects_edge_into_input` and the other localisation tests | Validation issues name the node or edge at fault. |
| UR-F-13 | partial | `tests/test_fuzzer_topology.py::test_is_the_default_topology_stage`, `tests/test_fuzzer_assign_validate.py::test_is_the_default_stage` | Tests run against `DEFAULT_STAGES`, but there is no shared parameterised harness across every stage. |
| UR-NF-DE-01 | done | `tests/test_fuzzer_upstream_sync.py::test_ur_nf_de_01_golden_mismatch_message_is_readable` | Golden fingerprints checked in; `--update-golden` re-records them; mismatch message is readable. |

## Non-functional (SR-NF) checks (added 2026-11-13)

Every SR-NF requirement the repository cites now has an automated check: a
named pytest test, or a CI step in `.github/workflows/fuzzer-nf.yml` whose
step name carries the ID. Measured values are from the local runs recorded in
`TEST_LOG.md` rows 15-22 and 26-30; the CI job adds the ubuntu matrix runs.

| ID | Check | Pass/Fail | Measured / test |
|---|---|---|---|
| SR-NF-DV-01 | `pre-commit run --all-files` (Ruff + Black) | pass | both hooks Passed, 15.1 s; CI step on py3.11 + 3.13 |
| SR-NF-DV-02 | `pytest tests/test_fuzzer_*.py --cov=fabricpc/fuzzer --cov-fail-under=80` | pass | **90%** coverage; 305 passed, 2 skipped |
| SR-NF-DV-03 | `mypy fabricpc/fuzzer` with 0 errors | pass | **0 errors, 17 files** (package fully checked; untyped upstream analyzed silently, no per-file ignores) |
| SR-NF-DV-04 | every public API documented | pass | `test_sr_nf_dv_04_public_api_has_docstrings` |
| SR-NF-US-01 | setting docs: meaning + type + default | pass | `test_sr_nf_us_01_settings_document_meaning_type_and_default` |
| SR-NF-US-02 | errors name the setting/node/edge | pass | error-message audit: `tests/test_fuzzer_error_messages.py` |
| SR-NF-US-03 | errors say what to change | pass | same file; `GenerationError`/`TopologyError` gained explicit `Fix:` lines |
| SR-NF-DE-03 | no global randomness (AST) | pass | `test_sr_nf_de_03_no_global_randomness_in_source`; seeded constructors allowlisted, fixed-key `jax.random` flagged |
| SR-NF-DC-02 | no torch/tensorflow imports | pass | `test_sr_nf_dc_02_no_torch_or_tensorflow_imports` |
| SR-NF-DC-03 | every NodeSpec maps to a FabricPC node class | pass | `test_sr_nf_dc_03_generated_nodes_map_to_fabricpc_classes` + Hypothesis variant |
| SR-NF-DC-04 | no GUI/plotting imports | pass | `test_sr_nf_dc_04_no_gui_or_plotting_imports` |
| SR-NF-DC-05 | peak memory < 4 GB, 100 graphs | **pass** | **672 MB peak** (+536 MB during run) on the 100-graph batch (macOS local; CI runner numbers pending) |
| SR-NF-IN-01 | no private FabricPC module imports | pass | `test_sr_nf_in_01_only_public_fabricpc_apis` |
| SR-NF-IN-04 | no new runtime dependencies | pass | unchanged since week 1 audit (dev-only: pytest-cov, hypothesis already listed) |
| SR-NF-LE-02 | no pickle/eval/exec, whole package | pass | `test_sr_nf_le_02_no_pickle_eval_exec_anywhere` (+ targeted load-path scan) |
| SR-NF-OP-01 | CI runs on ubuntu-latest | wired | `fuzzer-nf.yml` runs-on: ubuntu-latest |
| SR-NF-OP-02 | Python 3.11 and 3.13 matrix | wired | `fuzzer-nf.yml` strategy.matrix |
| SR-NF-OP-03 | CPU JAX | wired | base install ships `jax[cpu]`; no CUDA extra installed |
| SR-NF-PE-01 | 100 graphs (≤20 hidden nodes) < 2 min | **pass** | **55.8 s** local (`test_fuzzer_perf.py`, `-m slow`) |
| SR-NF-PE-02 | default CI run < 5 min | **pass** | 162.84 s with coverage locally; CI asserts `elapsed < 300` |
| SR-NF-CS-01 | docs usable by newcomers | **pass** | README "Graph Fuzzer" section + `examples/demo_presentation2.py --quick` (< 30 s) |

`tests/test_fuzzer_perf.py` is `@pytest.mark.slow`: deselected from the
5-minute budget, run on demand with `pytest -m slow`.

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
