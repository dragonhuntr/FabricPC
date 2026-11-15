"""Guided demo of the graph fuzzer for Presentation 2.

Walks through everything an end user needs (this is the script the README's
fuzzer section points at)::

    1. generate a small batch from a fixed seed, one line per graph
    2. re-run and show the output is identical   (same seed -> same graphs)
    3. an unsatisfiable config and its error message
    4. a partial batch and its warning line
    5. save a batch, reload it, and prove it is byte-identical

Run::

    python examples/demo_presentation2.py           # full (~2 minutes)
    python examples/demo_presentation2.py --quick   # < 30 s on a laptop

Everything trains on tiny CPU-friendly shapes; JAX imports once, at step 1.
"""

from __future__ import annotations

import sys
import tempfile
import time
import warnings
from pathlib import Path

from fabricpc.fuzzer import (
    ClampedNode,
    ConfigError,
    GeneratorConfig,
    PartialBatchWarning,
    generate,
    load_batch,
    regenerate_batch,
    save_batch,
)

FULL = "--quick" not in sys.argv


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def one_line(index: int, spec) -> str:
    return (
        f"  graph {index}: fingerprint={spec.fingerprint()[:12]} "
        f"nodes={len(spec.nodes)} edges={len(spec.edges)} "
        f"cyclic={spec.is_cyclic}"
    )


def main() -> int:
    start = time.perf_counter()
    n_graphs = 10 if FULL else 3
    hidden = (2, 5) if FULL else (2, 3)

    config = GeneratorConfig(
        num_graphs=n_graphs,
        seed=7,
        num_nodes=hidden,
        in_degree=(1, 3),
        input_shape=(64,),
        output_shape=(10,),
        hidden_sizes=(32, 64),
    ).validate()

    # --- 1. first batch ---------------------------------------------------
    rule(f"1. generate {config.num_graphs} graphs with seed {config.seed}")
    print("   (a 'graph' = one random predictive-coding network: input node,")
    print("    random hidden nodes, output node, random non-sequential edges)")
    specs = generate(config)
    for i, spec in enumerate(specs):
        print(one_line(i, spec))
    print("   every graph passed the fuzzer's checks: it builds in FabricPC")
    print("   and survives a short training run without NaNs")

    # --- 2. same seed, same graphs -----------------------------------------
    rule("2. re-run with the same seed")
    again = generate(config)
    same = [a.fingerprint() for a in specs] == [b.fingerprint() for b in again]
    for i, spec in enumerate(again[:2]):
        print(one_line(i, spec))
    print(f"   fingerprints all identical: {same}")
    if not same:
        raise SystemExit("reproducibility broken - demo must fail loudly")

    # --- 3. unsatisfiable config -------------------------------------------
    rule("3. impossible settings are rejected before generating")
    bad = GeneratorConfig(num_nodes=(2, 3), num_edges=(50, 60))
    try:
        bad.validate()
    except ConfigError as exc:
        print(f"   ConfigError: {exc}")

    # --- 4. partial batch ----------------------------------------------------
    rule("4. a config that fills only part of its request")
    partial = GeneratorConfig(
        num_graphs=4,
        seed=1,
        num_nodes=(3, 3),
        in_degree=(2, 2),
        allow_cycles=True,
        node_types=("StorkeyHopfield",),
        input_shape=(16,),
        output_shape=(2,),
        hidden_sizes=(8,),
        max_attempts=40,
    ).validate()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        made = generate(partial)
    for i, spec in enumerate(made):
        print(one_line(i, spec))
    lines = [str(w.message) for w in caught if w.category is PartialBatchWarning]
    if lines:
        print(f"   PartialBatchWarning: {lines[0]}")
    print(f"   {len(made)} valid graphs returned; the failed attempts are")
    print("   skipped, and the warning names the check that rejected the most")

    # --- 5. save, reload, prove identical ------------------------------------
    rule("5. save, reload and verify")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "demo_batch"
        save_batch(specs, out, config=config.to_dict())
        files = sorted(p.name for p in out.glob("graph_*.json"))
        print(f"   wrote {len(files)} graph files + manifest.json")
        reloaded = load_batch(out)
        identical = [a.fingerprint() for a in specs] == [
            b.fingerprint() for b in reloaded
        ]
        print(f"   reload in this process: fingerprints identical = {identical}")
        diff = regenerate_batch(out)
        print(f"   regenerate from the recorded seed: differences = {len(diff)}")
        if diff:
            for line in diff:
                print(f"     {line}")

    # --- clamped nodes teaser (details live in the README) -------------------
    rule("bonus: clamped nodes keep one task map across the batch")
    clamped = GeneratorConfig(
        num_graphs=2,
        seed=4,
        num_nodes=(1, 3),
        input_shape=(784,),
        output_shape=(10,),
        clamped_nodes=[
            ClampedNode("input", "Linear", (784,), in_degree=0, out_degree=1),
            ClampedNode("output", "Linear", (10,), in_degree=1, out_degree=0),
        ],
    ).validate()
    for i, spec in enumerate(generate(clamped)):
        print(one_line(i, spec))
    print("   input/output type, shape and degrees are fixed for every graph,")
    print("   so the whole batch can train and score on the same data")

    elapsed = time.perf_counter() - start
    print(f"\ndemo complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
