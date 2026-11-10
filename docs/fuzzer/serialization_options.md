# Graph serialization options (PROVISIONAL)

Status: **provisional**. The sponsor has not chosen a format. `serialize.py`
still ships and still works; this note records the three options, the
prototype that was built, and a recommendation.

Requirements touched: **SR-F-11** (will be rewritten in Report 2),
**SR-NF-IN-04** (no new runtime dependency), **SR-NF-LE-02** (effort and
maintenance cost).

## Why the current JSON layer is provisional

`fabricpc/fuzzer/serialize.py` writes a `GraphSpec` as a versioned JSON
document (`format`, `schema_version`, `fingerprint`, `spec`) and the batch
manifest wraps a directory of them. It is reproducible and diff-friendly, but
it is a **second description of a graph** next to FabricPC's own
`GraphStructure`. A new FabricPC node option has to be added to `NodeSpec`,
the registry, `build.py` and the validator before it can be saved. The sponsor
ruled out Python files and discouraged that middle layer, so the options below
either remove it or make it the only copy.

## Option A — model factory pattern

**Where it already exists.** `fabricpc/experiments/ab_experiment.py` defines

```python
ModelFactory = Callable[[jax.Array], Tuple[Any, Any]]  # rng_key -> (params, structure)
```

and `examples/mnist_lateral_connections.py` implements it twice:
`create_lateral_model(rng_key)` and `create_mlp_model(rng_key)`. Each builds
nodes, calls `graph(...)`, then `initialize_params(structure, rng_key)` and
returns `(params, structure)`.

**How it works here.** `fabricpc/fuzzer/factory.py` adds

```python
as_model_factory(spec, inference=None, unroll=None) -> ModelFactory
```

It builds the `GraphStructure` once with the existing `to_fabricpc(spec)` and
returns a closure that calls `initialize_params(structure, rng_key)` on every
invocation. The signature is identical to the two example factories, so a
fuzzed graph can replace either arm of `ABExperiment` without touching the
harness. The prototype is `examples/fuzzer_model_factory_ab.py`.

**What it can store**

- The complete graph as FabricPC objects (nodes, slots, edges, task map,
  inference settings), plus a fresh initialisation from a seed.
- It can be trained immediately; no decode step.

**What it cannot store**

- The graph definition itself is still a `GraphSpec` in memory or JSON. A
  factory is code, and the sponsor ruled out shipping Python files. Sharing a
  result still needs a data file or a script.
- Trained weights are not stored. Parameters are regenerated from the key, so
  a checkpoint is still required to reproduce a finished run.

**Effort.** Low. The adapter is ~40 lines and reuses `to_fabricpc` and
`initialize_params`. No new dependency.

**Recommendation.** Use it for experiments and as the bridge into FabricPC's
A/B harness. It solves "how does a fuzzed graph train like a hand-written
model", not "what file do we ship". Pair it with a weight checkpoint for a
finished run.

## Option B — ORBAX checkpointing

**What the repository actually says.** `docs/user_guides/16_troubleshooting.md`
answers "How do I save and load trained parameters?" with Orbax:

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save("/path/to/checkpoint", params)
params = checkpointer.restore("/path/to/checkpoint")
```

That example stores `GraphParams` only. `docs/dev_plans_archive/model_checkpointing.md`
goes further and designs a **composite** checkpoint: `params` and `opt_state`
through Orbax array handlers, plus a JSON metadata item holding a best-effort
**structure snapshot** `(class_path, config)` per component, `node_order` and
muPC factors. That design is archived and is **not** implemented on this
branch: there is no `fabricpc/serialization.py` here.

**How it works.** Orbax saves a JAX pytree of arrays (and JSON metadata items).
A `GraphStructure` is not an array pytree: it contains node objects, slots,
task-map names and inference config. Orbax can store it only if the application
serialises it into a JSON sidecar first.

**What it can store**

- Weights, optimizer state, RNG/step metadata, and (through the archived
  design) a structure snapshot.
- Sharded arrays and atomic commits, which a hand-written format would have to
  reimplement.

**What it cannot store natively**

- The graph topology as a first-class object. Without the JSON sidecar you get
  weights whose meaning depends on rebuilding the same structure in code.
- A stable, reviewable diff: checkpoints are binary directories.

**Effort.** Medium to high. The archived plan is a sizable API
(`save_checkpoint`, `load_checkpoint`, `Checkpoint`, a training callback, a
constructor round-trip fix). Orbax is already a runtime dependency, but the
structure codec is new work.

**Recommendation.** Use Orbax for weights and resume, not as the
graph-definition format. If adopted, put `GraphSpec.to_dict()` in the
structure slot (it already has a version and a fingerprint) and reconstruct
with `GraphSpec.from_dict` -> `to_fabricpc`.

## Option C — Artem's checkpoint/loading PR

The open PR is reachable: **trueagi-io/FabricPC#38**, "Add model checkpointing:
save/load params, optimizer state, and structure"
(`feature/model-checkpointing`, open at the time of writing).

**Format.** A checkpoint is a directory:

- `metadata.json` — format magic, version, library versions, per-file sha256;
- `params.msgpack` — always, flax msgpack, self-describing;
- optional `structure.json` — a **versioned JSON snapshot** of `NodeInfo`,
  rebuilt from class import paths, with `node_order` and muPC factors;
- optional `opt_state.msgpack` and `state.msgpack`.

The PR also carries `docs/dev_plans_archive/COMPARISON_checkpointing.md`, whose
recommendation is to **decouple parameter decoding from structure
reconstruction**: params load without structure, and the structure snapshot
degrades gracefully if a class was renamed.

**How our format would line up**

| Our `GraphSpec` | PR #38 `structure.json` |
|---|---|
| `NodeSpec.node_type` is a registry key (`"Linear"`) | class import path |
| `NodeSpec.shape`, `role`, `activation`, `energy`, `config` | `NodeInfo` config fields |
| `EdgeSpec(source, target, slot)` | edges with slot names |
| `fingerprint` = SHA-256 of canonical nodes/edges | per-file sha256 checksums |
| envelope `format` + `schema_version` | metadata `format` + `format_version` |

A converter is straightforward for the node fields; the main semantic gap is
that our names are registry keys (stable in this repo) while the PR stores
import paths (stable across renames only if the snapshot degrades). Our
`GraphSpec` also deliberately excludes weights, while the PR's checkpoint
includes them.

**What it can store.** Both structure and weights, plus optimizer state and
inference state. That is the closest match to the sponsor's request.

**What it cannot store.** Nothing relevant, except that a structure snapshot
with stale class paths is best-effort by design; the params still load.

**Effort.** Medium. Either wait for the PR to merge and emit its
`structure.json` from `GraphSpec`, or keep our envelope and add a
`to_checkpoint_structure(spec)` / `from_checkpoint_structure(doc)` adapter now.
The adapter needs tests against the PR branch, which is not merged.

**Recommendation.** Prefer this shape if the PR lands. Keep our `GraphSpec`
JSON as the canonical structure payload and feed it into the PR's
`structure.json` slot rather than maintaining a second schema.

## Prototype result

`examples/fuzzer_model_factory_ab.py` generated graph
`8442411be394` (4 nodes, 4 edges, acyclic) with this shape:

```text
input (784) -> h0 (128), h1 (128); h0, h1 -> output (10)
```

It wrapped the spec with `as_model_factory` and ran it as arm A of the same
`ABExperiment` as `create_lateral_model` (arm B), one epoch, one trial. Both
arms used `InferenceSGD(eta_infer=0.20, infer_steps=20)` and
`optax.adamw(0.001, weight_decay=0.001)`:

```
Fuzzed:  accuracy=0.9335  (train: 1.8s)
Lateral: accuracy=0.9275  (train: 4.0s)
```

The point is not the accuracy (one epoch, one trial) but that the fuzzed graph
needed **no harness change** and no JSON decode step. Full command and output
are row 4 of `TEST_LOG.md`.

## Recommendation

1. **Now:** keep `serialize.py` (marked provisional) and use
   `as_model_factory` for training and A/B experiments. This is what the
   prototype demonstrates.
2. **If the sponsor wants one artifact per graph:** adopt PR #38's checkpoint
   directory once merged, with `GraphSpec.to_dict()` as the structure payload.
   That removes the "two specs in sync" problem because our spec is generated
   from the same `GraphSpec` that produced the FabricPC structure.
3. **If the sponsor wants no data layer at all:** ship a factory script per
   graph, which the sponsor has already ruled out. Do not choose this.

Open question for the sponsor: is a graph plus its trained weights one
artifact, or two (structure for sharing, checkpoint for resuming)? The answer
decides whether Option C replaces `serialize.py` or sits next to it.
