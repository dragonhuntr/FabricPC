# Week 2 summary (Nov 10, 2026)

## What works now

- **Generation** produces distinct, validated, non-sequential graphs from
  `GeneratorConfig`, saves them as a batch, and reloads them.
- **Clamped nodes**: a batch can fix the input and output node type, shape and
  exact in/out degree, so every graph trains on the same task map.
- **Out-degree limits** for hidden nodes, with validation errors that name
  `out_degree`.
- **Attempt limit** defaults to 1000. A request that cannot be filled returns
  the valid graphs plus `made X of N, most common failure: <check>`; the CLI
  prints the same line and exits 1 only when nothing was made.
- **Model factory**: `as_model_factory(spec)` drops a fuzzed graph into the
  existing MNIST lateral A/B harness with no harness change. One epoch
  reproduced on cached MNIST: Fuzzed 93.35%, Lateral 92.75% (one trial; not a
  claim about architecture quality).
- **Upstream-sync suite**: fixed seeds 0, 1, 42 across affine, mixed and
  clamped configs. Every graph builds, survives one training step, and matches
  a checked-in fingerprint. `--update-golden` accepts an intentional change.
- **CI**: `.github/workflows/fuzzer-sync.yml` checks the rebase requirement and
  runs the sync suite under a 5-minute budget.

Final measured numbers are in `STATUS.md` and the last row of `TEST_LOG.md`.
The full suite was **873 passed, 8 skipped, 86% coverage** at the Part 0
baseline; this week adds 22 fuzzer tests (5 factory, 14 clamped/partial-batch,
3 upstream-sync).

## What moved out of scope, and why

- **Evaluation against sequential baselines (SR-F-12.x).** The sponsor said
  scoring graphs against baselines is research, not the fuzzer's job. The code
  moved to `fabricpc/fuzzer/experiments/` and is no longer re-exported from
  `fabricpc.fuzzer`. It is kept, tested and importable. The replacement is the
  upstream-sync suite, which answers the practical question: did a FabricPC
  change break generation?
- **JSON as the primary graph format (SR-F-11).** `serialize.py` still works
  and is the current sharing format, but it is marked provisional. The sponsor
  ruled out Python files and discouraged a second spec; the three options and a
  recommendation are in `serialization_options.md`. PR #38 is the closest match
  if it merges.

## Requirement -> test -> pass/fail (Appendix T)

| ID | Test | Result |
|---|---|---|
| SR-F-03.3 | `test_fuzzer_topology.py::test_hidden_count_inside_range_bounds_included` | PASS |
| SR-F-03.5 | `test_fuzzer_topology.py::test_edge_count_inside_range_bounds_included` | PASS |
| SR-F-04.2 | `test_fuzzer_topology.py::test_in_degree_limits` | PASS |
| SR-F-04.3 | `test_fuzzer_clamped.py::test_sr_f_04_3_out_degree_upper_bound` | PASS |
| SR-F-04.4 | `test_fuzzer_clamped.py::test_sr_f_04_4_out_degree_lower_bound` | PASS |
| SR-F-05.1 | `test_fuzzer_clamped.py::test_sr_f_05_1_clamped_nodes_present` | PASS |
| SR-F-05.2 | `test_fuzzer_clamped.py::test_sr_f_05_2_clamped_nodes_identical_across_batch` | PASS |
| SR-F-05.3 | `test_fuzzer_clamped.py::test_sr_f_05_3_clamped_degrees_honored` | PASS |
| SR-F-06.2 | `test_fuzzer_topology.py::test_same_seed_same_skeletons` | PASS |
| SR-F-07.2 | `test_fuzzer_clamped.py::test_sr_f_07_2_attempt_limit_defaults_to_1000` and `..._stops_generation` | PASS |
| SR-F-07.5 | `test_fuzzer_clamped.py::test_sr_f_07_5_partial_batch_keeps_only_valid_graphs` and `..._cli_prints_made_line` | PASS |
| SR-F-07.6 | `test_fuzzer_clamped.py::test_sr_f_07_6_most_common_failing_check` | PASS |
| SR-F-11 | `test_fuzzer_serialize.py` (whole file) and `test_fuzzer_factory.py` | PASS (format provisional) |
| SR-F-12.1–12.10 | `test_fuzzer_evaluate.py` (moved to `experiments`) | PASS, **out of scope** pending sponsor |
| SR-F-14.1 | `test_fuzzer_upstream_sync.py::test_sr_f_14_1_every_generated_graph_builds_and_trains_one_step` | PASS |
| SR-F-14.2 | `test_fuzzer_upstream_sync.py::test_sr_f_14_2_fingerprints_match_checked_in_golden` | PASS |
| SR-NF-IN-04 | dependency check in `pyproject.toml` | PASS |
| SR-NF-LE-02 | fuzzer coverage (see `STATUS.md`); CLI now has a test | PARTIAL |
| SR-NF-US-01 | `test_fuzzer_clamped.py::test_sr_nf_us_01_out_degree_error_names_the_field` | PASS |
| SR-NF-US-02 | `test_fuzzer_clamped.py::test_sr_nf_us_02_clamped_node_error_names_the_field` | PASS |
| SR-NF-US-03 | `test_fuzzer_clamped.py::test_sr_nf_us_03_new_settings_round_trip_through_json` | PASS |
| SR-NF-PE-02 | `.github/workflows/fuzzer-sync.yml`; sync suite 13.6 s locally | PASS (not yet run on GitHub) |
| UR-F-08 | `test_fuzzer_topology.py` and `test_fuzzer_assign_validate.py` RNG tests | PASS |
| UR-F-11 | validation localisation tests in `test_fuzzer_assign_validate.py` | PASS |
| UR-F-13 | default-stage tests in `test_fuzzer_topology.py` and `test_fuzzer_assign_validate.py` | PASS |
| UR-NF-DE-01 | `test_fuzzer_upstream_sync.py` positive test plus row 10's deliberate failure | PASS |

The 59 SR-F rows outside this table still need Table 7. They are counted as
"not assessed" in `STATUS.md` rather than guessed.

## Open questions for the sponsor

1. **Serialization artifact.** Is a run one artifact (structure + weights) or
   two (a shareable structure and a checkpoint)? That decides whether PR #38
   replaces `serialize.py` or sits beside it.
2. **Evaluation.** Confirm SR-F-12.x is out of scope. Keep the code under
   `experiments`, or remove it before the final report?
3. **Partial batches.** Should a short batch exit non-zero even when some
   graphs were made? Today it exits 0 if at least one graph exists.
4. **Clamped nodes.** Beyond type, shape and degrees, should the sponsor be able
   to clamp activation and energy too? Today those stay global config.
5. **Out-degree scope.** `out_degree` applies to hidden nodes. The input and
   output are governed by `clamped_nodes`. Confirm that is the intended reading
   of SR-F-04.3/04.4.
6. **CI meaning.** The fuzzer lives in the FabricPC repo, so the sync job tests
   the branch after the required rebase on main. Is that sufficient, or does the
   sponsor want a separate nightly run against upstream `main` before a merge?

## Weekly task list (for the faculty advisor)

| Task | Owner | Measurable output | Done? |
|---|---|---|---|
| Status audit against Report 1.5 | Wai Soon | `docs/fuzzer/STATUS.md` with counts at the top and a measured suite/coverage baseline | Done |
| Serialization spike and prototype | Wai Soon | `serialization_options.md`; fuzzed graph trains one epoch in the MNIST A/B harness | Done |
| Clamped nodes and taskmap | Adzfar | SR-F-05.1–05.3 tests pass | Done |
| Out-degree limits | Adzfar | SR-F-04.3/04.4 tests pass | Done |
| Attempt limit and partial batches | Adzfar | SR-F-07.2/07.5/07.6 tests pass; CLI prints the made line | Done |
| Move evaluation out of the default path | Evan | `fuzzer/experiments/` package; STATUS marks SR-F-12 out of scope | Done |
| Upstream-sync suite and golden file | Evan | `tests/test_fuzzer_upstream_sync.py` passes; golden committed; deliberate failure logged | Done |
| CI job under 5 minutes | Evan | `.github/workflows/fuzzer-sync.yml` added | Done locally, not yet run on GitHub |
| Week 2 summary and status update | Wai Soon | `WEEK2_SUMMARY.md`, updated `STATUS.md` | Done |
| Rebase and open PRs | each owner | PR linked with the AI-review findings answered | Not done: branches pushed, PRs still to open |
