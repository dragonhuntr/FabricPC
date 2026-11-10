# Graph fuzzer: stage contract (PROPOSAL, open for review)

Status: **draft for team review**. Nothing here is frozen. Comment on the PR
if a signature doesn't fit your stage, and we change it before anyone builds
on top of it.

Owners: interface / spec / reproducibility, Adzfar · topology, Evan ·
typing and validity, Hai · evaluation and feedback, Wai Soon.

## The pipeline in one picture

```
GeneratorConfig ─topology─▶ Skeleton ─assign─▶ GraphSpec ─validate─▶ ValidationResult
   (Adzfar)        (Evan)                (Hai)                (Hai)
                                                  │
                                                  ├─▶ save / load (JSON)      (Adzfar)
                                                  └─▶ to_fabricpc → train     (Wai Soon)
```

`fabricpc/fuzzer/pipeline.py` runs the three stages for each graph, retries
failed attempts with fresh randomness up to `config.max_attempts` times (1000 by
default), skips duplicates, and stamps provenance metadata. A graph that
exhausts the limit is skipped: the valid graphs are returned with a
`PartialBatchWarning`, and the report names the most common failing check. The
default stages are `random_dag_topology`, `typed_assign` and `validate_spec`;
`chain_topology`, `linear_assign` and `check_builds` remain as stand-ins and as
the shape of a sequential baseline.

## Stage 1: topology (Evan)

```python
def my_topology(config: GeneratorConfig, rng: np.random.Generator) -> Skeleton
```

Returns a `Skeleton(hidden=("h0", "h1", ...), edges=(("input", "h0"), ...))`.

Expected of every skeleton:

- Hidden names are unique and are not `"input"` or `"output"`.
- The hidden node count is within `config.num_nodes`.
- No edge points into `input`; no self-edges; no duplicate `(source, target)` pairs.
- Every hidden node and `output` has at least one in-edge, and every
  hidden in-degree is within `config.in_degree`. A clamped output has exactly
  its requested in-degree instead.
- Every hidden node's out-degree is within `config.out_degree` when that
  setting is not `None`. A clamped input has exactly its requested out-degree.
- The edge count is within `config.num_edges` when that setting is not `None`.
- No directed cycle unless `config.allow_cycles` is true.
- Every hidden node lies on some path from `input` to `output` (no dead branches).

Slots are **not** chosen here. That is Hai's job.

Implemented by `random_dag_topology` in `fabricpc/fuzzer/topology.py`, now
the default. `skeleton_issues(skeleton, config)` lists every rule above that
a skeleton breaks.

## Stage 2: assign (Hai)

```python
def my_assign(skeleton: Skeleton, config: GeneratorConfig,
              rng: np.random.Generator) -> GraphSpec
```

For each node, choose a `node_type` from `config.node_types` (hidden nodes
only), a `shape`, an `activation` and, if needed, extra constructor kwargs in
`NodeSpec.config`. For each edge, choose a `slot`. The input node has role
`"input"`, shape `config.input_shape`, and is normally an `IdentityNode`. The
output node has role `"output"`, shape `config.output_shape`, and uses
`config.output_activation` and `config.output_energy`. When `clamped_nodes` is
set, its node type and shape replace the defaults for the input and output, so
every graph in the batch keeps one task map.

Assign **may** add or drop edges (for example, a `skip` edge that a
`SkipConnection` needs). If it does, the returned spec is the source of truth,
not the skeleton.

## Stage 3: validate (Hai)

```python
def my_validate(spec: GraphSpec) -> ValidationResult
```

Return `ValidationResult.passed()` or
`ValidationResult.failed(ValidationIssue(message, where="h3", check="shape"))`.
`where` is a node name or an edge key (`"src->dst:slot"`). That is what lets
UR-F-11 tell the researcher what is wrong. A failed graph is regenerated, not
returned, so a passing result has to mean "builds and trains".

## Evaluation (Wai Soon) — moved out of scope, pending sponsor confirmation

Evaluation is not a pipeline stage, and after the Nov 3 sponsor meeting it is
no longer part of the fuzzer's default path. The code is kept under
`fabricpc/fuzzer/experiments/` and is still tested; it is not re-exported from
`fabricpc.fuzzer`. The useful replacement is the upstream-sync suite, which
checks that FabricPC changes do not break generation.

Evaluation consumes the fuzzer's output:

```python
specs = generate(config)                  # or load_batch("runs/demo")
structure = to_fabricpc(spec)             # GraphStructure for initialize_params/train
```

Baselines are `GraphSpec`s too, so baselines and generated graphs go through
the same code path and serialize the same way.

Implemented in `fabricpc/fuzzer/experiments/baselines.py` and
`fabricpc/fuzzer/experiments/evaluate.py`:

```python
from fabricpc.fuzzer.experiments import EvalSettings, evaluate_batch, read_csv

report = evaluate_batch(specs, data_factory, EvalSettings(training_seeds=(0, 1, 2)))
report.to_csv("runs/demo/results.csv")   # one row per graph and model
rows = read_csv("runs/demo/results.csv")  # plain dicts
```

- `sequential_baseline(spec)` keeps the graph's input and output nodes and
  puts one `Linear` layer per hidden node (same width and activation) in a chain.
- Each model is trained once per training seed. Training seeds only seed
  parameter init, batch order and inference, never the graph.
- Per model: mean test accuracy, its standard error over seeds, peak memory,
  training time, parameter count, layer sizes and connections.
- `data_factory(seed) -> (train_loader, test_loader)`, as in
  `fabricpc.experiments`. `synthetic_data_factory` gives a small task for tests;
  `examples/fuzzer_baseline_mnist.py` runs the MNIST comparison.

Feedback (UR-F-12) goes in through `GeneratorConfig.constraints`, a free-form
JSON dict that every stage can read. `read_csv` rows fit there as-is; no
steering is built yet (out of scope per Report 1.5).

## Rules for every stage

1. **Randomness only from the `rng` argument.** Never `random`, `np.random.*`
   module functions, or `jax.random` with a fixed key. This is what makes
   "same seed → same graph" (UR-F-08) hold.
2. **Pure functions.** No reading files, no global state, no mutating `config`.
3. **Names are registry keys.** Node types, activations and energies are the
   strings in `fabricpc/fuzzer/registry.py`. A new FabricPC node type is
   registered there once.
4. **Unrecognized `constraints` keys are ignored, never an error.**

## How to plug in your stage

While developing, pass your function directly. Nothing else changes:

```python
from fabricpc.fuzzer import GeneratorConfig, generate
specs = generate(GeneratorConfig(num_graphs=20, seed=1), topology=my_topology)
```

When your stage is ready, change its entry in `DEFAULT_STAGES` in
`pipeline.py` in your PR. The shared test harness (UR-F-13, next on Adzfar's
list) will run against whatever is in `DEFAULT_STAGES`, so it checks your
stage automatically once it's the default.

## Open questions for the team

- Evan → Adzfar: the topology model is read from
  `constraints["topology_model"]` (`random_dag` default, `erdos_renyi`,
  `barabasi_albert`, `watts_strogatz`). Should it become a real
  `GeneratorConfig` field so it is validated up front?
- Evan → Adzfar: SR-F-04.3/04.4 need out-degree limits, but `GeneratorConfig`
  has no `out_degree` setting. Can it be added (same shape as `in_degree`)?
  Until then only in-degree limits are enforced and tested.
  **Resolved:** `out_degree` and `clamped_nodes` are now real settings, with
  validation errors that name the field, and the topology stage enforces both.
- Adzfar → sponsor: should a batch that cannot fill its request fail the
  command, or return the partial batch (current behaviour, exit 1 only when no
  graph at all was made)?
- Hai: does every node type's shape rule fit `NodeSpec` as-is, or do some types
  need extra fields (kernel size, number of heads)? Right now those would go in
  `NodeSpec.config`.
- Hai: should validate run a short training step (slower, catches NaNs), or
  should that be a separate opt-in check?
- Wai Soon: results are a CSV for now, linked to graphs by `fingerprint` and
  `graph_id`. Should they also go in an `evaluation` block next to `spec` in
  each graph's JSON file?
