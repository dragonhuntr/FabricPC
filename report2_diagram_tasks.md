# Report 2 diagram tasks

Each owner keeps their own section. A task counts as done only when the
diagram or figure exists in the shared Report 2 document, not just in a branch.

## Wai Soon

| Task | Diagram or figure | Source | Status |
|---|---|---|---|
| Requirement status at a glance | Pipeline stages with a done / partial / not started / out of scope badge per stage | `docs/fuzzer/STATUS.md` | draft |
| Serialization decision | Three-way comparison: model factory vs Orbax checkpoint vs PR #38, with a recommendation arrow | `docs/fuzzer/serialization_options.md` | draft |
| Evaluation out of scope | Before/after architecture: evaluation was a side path, now an experiments box outside the default pipeline | `docs/fuzzer/stage_contract.md` | draft |
| Upstream-sync loop | FabricPC main -> generate -> build -> one train step -> golden fingerprints -> accept or reject | `tests/test_fuzzer_upstream_sync.py` | draft |

## Adzfar

| Task | Diagram or figure | Source | Status |
|---|---|---|---|
| Config schema | `GeneratorConfig` fields grouped by stage, with the new `out_degree` and `clamped_nodes` | `fabricpc/fuzzer/config.py` | draft |
| Degree-constrained topology | Draw order with in-degree bounds, out-degree bounds, and the retry edge when a draw misses | `fabricpc/fuzzer/topology.py` | draft |
| Partial batch | Flow from `generate` to `GenerationReport` to the CLI line "made X of N, most common failure" | `fabricpc/fuzzer/pipeline.py`, `cli.py` | draft |

## Evan

| Task | Diagram or figure | Source | Status |
|---|---|---|---|
| Upstream-sync test matrix | Configs (affine, mixed, clamped) x fixed seeds x build / train / fingerprint checks | `tests/test_fuzzer_upstream_sync.py` | draft |
| Golden accept flow | Normal run vs `--update-golden`, showing the reviewed diff step | `tests/golden/README.md` | draft |
| Evaluation moved | Default fuzzer path vs `experiments` extra, with the sponsor question marked open | `fabricpc/fuzzer/experiments/` | draft |
