# FabricPC

**State-of-the-art predictive coding, made easy.**

FabricPC is an easy-to-use, high-performance open-source Python library for building and training predictive coding networks. It is designed to get researchers from idea to running experiment as fast as possible, eliminating algorithm boilerplate. A single directed edge between nodes is all that's needed to define a connection. Local derivatives are built in, following graph topology. The framework handles inference and learning dynamics automatically for whatever you write in a node's `forward()` method.

Built on JAX for GPU and multi-GPU acceleration with local (node-level) automatic differentiation.

## What It Does

FabricPC supports arbitrary graph topologies: feedforward, recurrent, skip connections, and cyclic architectures. Heterogeneous components such as linear, convolutional, and pooling nodes, transformer blocks, and Storkey-Hopfield associative memory coexist within the same energy-minimization graph. The same graph topology can be trained by predictive coding (`train(..., algorithm="pc")`) or by backpropagation (`train(..., algorithm="backprop")`), so controlled PC-vs-backprop comparisons reuse one model definition instead of two. See `examples/PC_backprop_compare.py`.

Internally, everything is organized around three abstractions: nodes (state and computation), edges (connections between nodes), and updates (inference and learning algorithms).

## Installation

Python 3.11–3.13. Install into a virtual environment, not the system Python. Create and activate the environment, then one command installs FabricPC, its optional dependencies, and a version-matched JAX backend — pick the line for your hardware:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U "fabricpc[all,cuda13]"   # GPU, CUDA 13 (NVIDIA driver ≥580)
pip install -U "fabricpc[all,cuda12]"   # GPU, CUDA 12
pip install -U "fabricpc[all]"          # CPU only
pip install fabricpc                    # core library only, CPU
```

`nvidia-smi` reports the CUDA version your driver supports.

**Platform:** GPU acceleration requires **Linux** (x86_64 or aarch64) — JAX publishes CUDA wheels for Linux only. On native Windows or macOS, install CPU-only; for GPU on Windows use WSL2 (JAX marks WSL2 GPU support experimental). The optional Aim experiment tracker in `[viz]`/`[all]` is Linux/macOS only and supports Python ≤3.12; on Windows or Python 3.13 it is skipped automatically and everything else installs normally.

See the [installation guide](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/01_installation.md) for details.

### From source (contributors)

```bash
git clone https://github.com/trueagi-io/FabricPC.git
cd FabricPC
python3 -m venv .venv && source .venv/bin/activate
pip install -U -e ".[all,dev]"    # add a backend extra for GPU: ".[all,dev,cuda12]"

# Install pre-commit hooks for code quality
pre-commit install

# Run an example
python examples/mnist_demo.py
```

## Build a Model

Define the graph. Initialize the parameters. Start experimenting.

```python
import jax
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD
from fabricpc import setup_jax

setup_jax()

layer1 = Linear(shape=(784,), name="input")
layer2 = Linear(shape=(256,), name="hidden")
layer3 = Linear(shape=(10,), name="output")

structure = graph(
    nodes=[layer1, layer2, layer3],
    edges=[Edge(source=layer1, target=layer2.slot("in")),
           Edge(source=layer2, target=layer3.slot("in"))],
    task_map=TaskMap(x=layer1, y=layer3),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

rng_key = jax.random.PRNGKey(0)
params = initialize_params(structure, rng_key)
```

## Demos

The [`examples`](https://github.com/trueagi-io/FabricPC/tree/main/examples) folder includes working demonstrations across image classification, sequence modeling, depth scaling (`examples/scaling/`), associative memory, and architectural probes. Start with [`mnist_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/mnist_demo.py) (over 98% accuracy on MNIST) and explore from there:

- [`mnist_conv_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/mnist_conv_demo.py) — convolutional MNIST classifier with `ConvNode` and `MaxPool`
- [`resnet18_cifar10_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/resnet18_cifar10_demo.py) — ResNet-18 as a PC graph, with global average pooling
- [`transformer_v2_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/transformer_v2_demo.py) — character- or BPE-level language modeling with text generation
- [`transformer_tuning.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/transformer_tuning.py) — two-phase hyperparameter search minimizing validation perplexity

## Graph Fuzzer

The graph fuzzer writes random predictive coding graphs for you — and guarantees every one is valid, so it can be built in FabricPC and trained without hand-checking. You describe what you want (how many hidden nodes, which node types, how connected), the fuzzer generates many structurally different graphs that match, and saves them so a whole run can be shared and reloaded.

Terms used below, in one line each: a **graph** is a set of **nodes** (small neural networks that make predictions) wired together by **edges** (one node's output feeding another node's input **slot**); **predictive coding (PC)** trains the graph by minimizing prediction errors locally instead of backpropagating one global loss; a **task map** says which node receives the data (**input node**) and which node is compared to the labels (**output node**); a **seed** is the number that makes a random run reproducible; a **fingerprint** is a short hash that identifies a graph's exact structure.

See it in 15 seconds:

```bash
python examples/demo_presentation2.py --quick
```

### First run

```bash
fabricpc-fuzz init-config fuzz.json          # write a starter settings file
fabricpc-fuzz generate --config fuzz.json --out runs/demo
fabricpc-fuzz inspect runs/demo              # print each saved graph
fabricpc-fuzz check runs/demo                # rebuild every graph in FabricPC
fabricpc-fuzz check --train-step runs/demo   # ...and run one real training step
fabricpc-fuzz check --regenerate runs/demo   # re-run the seed, prove graphs match
```

(`python -m fabricpc.fuzzer ...` works identically if you run from a source checkout.)

### Every setting, with an example

Put any of these in `fuzz.json` (or pass `--set key=value` on the command line). All are optional.

| Setting | Default | Meaning |
|---|---|---|
| `num_graphs` | `1` | How many graphs the run should return. |
| `seed` | `0` | Master random seed; same seed + settings = same graphs. |
| `num_nodes` | `[2, 6]` | Hidden-node count, as a number or a `[min, max]` range. |
| `num_edges` | `null` | Total edge count; `null` lets the fuzzer choose. |
| `in_degree` | `[1, 3]` | How many inputs a non-input node may have. |
| `out_degree` | `null` | How many nodes a hidden node may feed; `null` = no limit. A minimum > 1 needs `allow_cycles: true`. |
| `node_types` | `["Linear"]` | Node types allowed for hidden nodes (`Linear`, `LinearResidual`, `StorkeyHopfield`, `SkipConnection`, `IdentityNode`, `LinearExplicitGrad`). |
| `allow_cycles` | `false` | Allow feedback edges (recurrent graphs). |
| `unroll` | `2` | Cycle traversals used when building a cyclic graph. |
| `input_shape` | `[784]` | Shape of one data sample (784 = 28×28 MNIST pixels flattened). |
| `output_shape` | `[10]` | Shape of one target (10 = ten digit classes). |
| `hidden_sizes` | `[64, 128, 256]` | Widths to choose from for hidden nodes. |
| `hidden_activations` | `["relu", "tanh", "sigmoid"]` | Activations to choose from for hidden nodes. |
| `output_activation` | `"softmax"` | Output activation (`"cross_entropy"` energy requires `"softmax"`). |
| `output_energy` | `"cross_entropy"` | The prediction-error function the PC training minimizes at the output. |
| `max_attempts` | `1000` | Retries per graph before it is dropped from the batch. |
| `clamped_nodes` | `[]` | Pin the input/output nodes; see below. |
| `constraints` | `{}` | Free-form extras for the generation stages, e.g. `{"topology_model": "erdos_renyi"}`. |

A worked settings file for MNIST-shaped graphs:

```json
{
  "num_graphs": 20,
  "seed": 7,
  "num_nodes": [4, 8],
  "in_degree": [1, 3],
  "node_types": ["Linear", "LinearResidual"],
  "input_shape": [784],
  "output_shape": [10]
}
```

### Clamped nodes (one task map for the whole batch)

"Clamping" a node means pinning it so every graph in the batch shares it. Give exactly two entries, one `input` and one `output`, with their exact in- and out-degrees: the input can never receive edges (`in_degree: 0`) and must feed at least one node; the output is the mirror image. Their shapes also set `input_shape`/`output_shape`, so every graph trains and scores on the same data:

```json
"clamped_nodes": [
  {"role": "input",  "node_type": "Linear", "shape": [784], "in_degree": 0, "out_degree": 1},
  {"role": "output", "node_type": "Linear", "shape": [10],  "in_degree": 2, "out_degree": 0}
]
```

### Reloading a saved graph

A batch is a directory of `graph_0000.json`, … plus `manifest.json` (config + fingerprints):

```python
from fabricpc.fuzzer import load_batch, to_fabricpc, as_model_factory

specs = load_batch("runs/demo")              # fingerprints verified on load
structure = to_fabricpc(specs[0])            # a FabricPC graph, ready to train
factory = as_model_factory(specs[0])         # drop into an A/B experiment
```

Hand edits are welcome but caught: a changed graph no longer matches its stored fingerprint and the loader says so. `fabricpc-fuzz check --regenerate runs/demo` proves the whole batch reproduces from its recorded seed, on any machine running the same FabricPC. (The file format is provisional while the sponsor decides the canonical artifact — see `docs/fuzzer/serialization_options.md`.)

### Reading a failure report

* A whole settings problem is rejected before generation starts, and the error names the setting and says what to do:
  `num_edges: minimum 50 exceeds the 10 edges possible with num_nodes max 3, in_degree max 3 and no cycles`
* A partial batch warns instead of dying: `made 7 of 20, most common failure: shape`. `shape`/`slot`/`train` are the names of the fuzzer's checks; the manifest still holds the seven good graphs.
* Every per-graph rejection says where: `shape: [input->h0:in] Linear.in adds its inputs directly, so source shape (784,) must equal (64,)`.

### Running the upstream-sync tests

These regenerate fixed graphs on the current FabricPC and compare fingerprints, so a FabricPC change that alters generation is caught the day it lands:

```bash
pytest tests/test_fuzzer_upstream_sync.py -q        # the sync check
pytest tests/test_fuzzer_*.py -m "not slow" -q      # the fast CI suite (< 5 min)
pytest -m slow tests/test_fuzzer_perf.py            # the 100-graph perf measurement
```

If you change generation on purpose, re-record the golden fingerprints with `python -m pytest tests/test_fuzzer_upstream_sync.py -q --update-golden` and commit `tests/golden/`.

## Documentation

User guides, API reference, and tutorials live in [`docs/user_guides`](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/00_index.md). Development plans and technical design documents are in [`docs/dev_plans_archive`](https://github.com/trueagi-io/FabricPC/tree/main/docs/dev_plans_archive).

## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`. Implement the `get_slots()`, `initialize_params()`, and `forward()` methods. Nodes have a single output. Slots define incoming connections and are referenced in edges when building the graph.

See [`docs/user_guides/06_custom_nodes.md`](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/06_custom_nodes.md) for the node contract and a Conv2D teaching example (the production node is `fabricpc.nodes.ConvNode`).

## Contributing

Contributions are welcome! Please open issues or pull requests on the GitHub repository. See [CONTRIBUTING.md](https://github.com/trueagi-io/FabricPC/blob/main/CONTRIBUTING.md) for the development setup, the pull-request expectations, and the design-first workflow.

This is a research-first project.
- APIs may change frequently until the v1.0 release.
- Any breaking changes are documented in the changelog.

## Team

FabricPC is actively maintained by SingularityNET as part of the Artificial Superintelligence Alliance. Project lead: Dr. Matthew Behrend.

## License

This project is licensed under the [MIT License](https://github.com/trueagi-io/FabricPC/blob/main/LICENSE).
