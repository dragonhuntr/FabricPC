# Week 3 summary — Graph fuzzer (2026-11-10 to 2026-11-16)

Branch `feature/fuzzer-foundation`, pushed to `dragonhuntr/FabricPC`.
Commits this week: `1cd68e2` (save/load hardening) · `2c8749e` (non-functional
tests, mypy) · `5f2144f` (CI matrix, perf) · `202d4a3` (error-message audit) ·
`9846e7f` (README manual, demo script, topology fix) · this summary commit.
Johnny's mixed-node-type work (`35867a1`/`27a6e2a`/`0457a5b`, Nov 16 evening)
landed on the branch mid-week; our six commits sit on top of it and every
number below is from the merged tree.

## What shipped

1. **Save/load hardened for the provisional format (Wai Soon).** The sponsor
   has not chosen a serialization option, so option D of
   `serialization_options.md` applies: the JSON envelope stays and got harder.
   `fabricpc-fuzz check` gained `--train-step` (rebuild + one real training
   step per graph) and `--regenerate` (re-run the recorded seed and diff).
   `regenerate_graph` / `regenerate_batch` / `describe_diff` are public, and a
   regeneration diff prints human-readable line pairs, not hashes. The
   done-when demo passes: save a seed-7 batch, reload it in a **fresh
   process**, rebuild every graph, train one step — fingerprints identical.
2. **Every SR-NF requirement is now an automated check (Adzfar).** Ten
   requirement tests (AST scans for global randomness, torch/TF, GUI,
   pickle/eval, private imports; docstring scans for the public API and every
   setting), `mypy fabricpc/fuzzer` at **0 errors** with no blanket ignores,
   a CI workflow (`fuzzer-nf.yml`) whose step names carry the requirement IDs
   (ubuntu matrix, 3.11 + 3.13, CPU JAX, coverage floor 80%, 300 s budget),
   and a slow-marked 100-graph perf measurement (stdlib `resource`, no new
   deps).
3. **Error messages and docs (Evan).** An audit test triggers every exception
   family the fuzzer raises and asserts each message names the culprit and
   says what to change — it found two real message defects, now fixed
   (`Fix:` lines on `GenerationError`/`TopologyError`, clearer `SpecError`
   and `SerializationError` wording). The README gained a plain-English
   "Graph Fuzzer" manual (terms defined, every setting with defaults, clamped
   nodes, reload workflow, failure reports) and `examples/demo_presentation2.py`
   runs the whole story in ~11 s (< 30 s with `--quick`).
4. **A real topology bug, found by the demo.** Small clamped configs could
   run out of candidate parents mid-draw and die with numpy internals
   (`ValueError: high <= 0`). Dead ends now re-enter the stage's retry loop
   like any other failed draw; regression test over 6 seeds
   (TEST_LOG rows 26-28).

## Measured numbers (2026-11-16, TEST_LOG rows 31-32)

| Measure | Result | Target |
|---|---|---|
| Full test suite | 984 passed, 8 skipped, 352.39 s | — |
| Fuzzer suite | 314 passed, 2 skipped, 154.66 s | < 300 s (PE-02) ✓ |
| `fabricpc/fuzzer` coverage | 90% (1836 stmts) | ≥ 80% (DV-02) ✓ |
| Library coverage | 87% (6491 stmts) | — |
| mypy on `fabricpc/fuzzer` | 0 errors, 17 files | 0 (DV-03) ✓ |
| pre-commit (Ruff + Black) | pass, 15.1 s | pass (DV-01) ✓ |
| 100 graphs (2..20 hidden nodes) | 55.8 s | < 2 min (PE-01) ✓ |
| Peak RSS, 100-graph batch | 672 MB | < 4 GB (DC-05) ✓ |
| `demo --quick` | 4.9 s | < 30 s ✓ |

## Behavior changes since week 2 (for demo prep and anyone diffing)

* `fabricpc-fuzz check` flags `--train-step` and `--regenerate` (new);
  `check` without them is unchanged.
* New public API: `regenerate_graph`, `regenerate_batch`, `describe_diff`,
  `as_range` (config helper for `IntOrRange` fields).
* `as_model_factory` now returns a `_GraphModelFactory` **class instance**
  instead of a closure. Call behavior, `.spec` and the `factory_` `__name__`
  prefix are unchanged; code that stored attributes on the factory object
  itself would notice.
* Error wording changed: `GenerationError` and `TopologyError` end with an
  explicit `Fix:` line; `SpecError` duplicate-name restates the naming rule;
  `SerializationError` lists the required envelope keys. Tests/CI that match
  message text should key off the ID/setting names, not full sentences.
* `topology.random_dag_topology`: dead-end draws are now retried instead of
  raising numpy `ValueError`; a config that is genuinely impossible still
  raises `TopologyError` after `MAX_DRAWS` (message unchanged).
* STATUS.md was refreshed (its SR-F rows were a stale week-0 audit);
  `serialization_options.md` now leads with the A/B/C/D question for the
  sponsor.
* From Johnny's merged work: `validate()` now rejects clamped nodes whose
  node type `assign` cannot build, build errors no longer double the node
  name, and the shape table is guarded (TEST_LOG rows 31-32 cover the sync).

## Open issues and questions

1. **Serialization decision (blocking SR-F-11 closing out).** A: current
   JSON envelope; B: FabricPC-native artifact; C: both; D: status quo, keep
   hardening. We are executing D. Full question text is at the top of
   `docs/fuzzer/serialization_options.md`.
2. **Report 1.5 Table 7** still has not been supplied. 59 SR-F IDs remain
   unassessed; the SR-F-11.2-11.8 sub-IDs we implemented come from the task
   brief, with the mapping stated per test.
3. **SR-F-12.1-12.10 (evaluation/baselines)** stay **out of scope** pending
   written confirmation ("research, not the fuzzer's job").
4. **Partial batches on very tight configs**: the demo's StorkeyHopfield
   config returns `made 0 of 4` with a warning — correct per SR-F-07.5, but
   is silent constraint-relaxation (e.g. widening `hidden_sizes`) wanted, or
   is an empty batch + warning the final word?
5. **CI runner numbers**: DC-05/PE numbers so far are local macOS; first
   ubuntu runs land when the branch is pushed and `fuzzer-nf.yml` executes.

## Appendix T evidence map

| Appendix T row | What it shows | Where to demo it |
|---|---|---|
| Reproducibility | same seed → identical fingerprints, rerun and cross-process | `examples/demo_presentation2.py` step 2; `fabricpc-fuzz check --regenerate runs/demo` |
| Validity by construction | every graph builds in FabricPC and survives a training step | `fabricpc-fuzz check --train-step runs/demo`; `tests/test_fuzzer_upstream_sync.py` |
| One task map per batch | clamped input/output identical across graphs | demo bonus section; `test_sr_f_05_2_clamped_nodes_identical_across_batch` |
| Honest partial batches | made-line + most-common failure name | demo step 4; `test_sr_f_07_5_cli_prints_made_line` |
| Actionable errors | culprit named + what to change | demo step 3; `tests/test_fuzzer_error_messages.py` |
| Non-functional gates | mypy / coverage / deps / purity / runtime | STATUS.md SR-NF table; `fuzzer-nf.yml` step log lines |
| Performance at scale | 100 graphs: 55.8 s, 672 MB | `pytest -m slow tests/test_fuzzer_perf.py` |
| Safe reloading | pure-JSON load path, fingerprint verification | `test_sr_nf_le_02_no_pickle_eval_exec_anywhere`; `check` output |
| Newcomer onboarding | README manual + < 30 s guided demo | README "Graph Fuzzer"; `python examples/demo_presentation2.py --quick` |

## Next week (draft task list)

* All: sponsor answers on serialization and evaluation scope; ask for
  Table 7 so the remaining 59 IDs can be assessed.
* Wai Soon: if the decision is not D, port the hardening (regenerate,
  fingerprint diff, `--train-step`) onto the chosen format; cross-version
  regeneration check (FabricPC pin in manifest).
* Adzfar: read the first `fuzzer-nf.yml` ubuntu-run numbers and reconcile
  against the local 55.8 s / 672 MB; malformed-envelope fuzzing of the load
  path (random/corrupt JSON property tests); push fuzzer coverage 90% → 92%.
* Evan: presentation slides around `demo_presentation2.py` (the five steps
  map 1:1 to Appendix T rows); triage "why 0 of N" diagnostics — suggest the
  narrowest setting to widen when a batch comes back empty.
