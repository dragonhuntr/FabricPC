"""
Generated graphs vs sequential baselines on MNIST
=================================================

Generates (or loads) a batch of fuzzer graphs, trains each one and its
sequential baseline with FabricPC over several training seeds, and writes one
CSV row per graph and model: mean test accuracy, its standard error, peak
memory, training time, and layer sizes and connections.

    python examples/fuzzer_baseline_mnist.py --num-graphs 5 --epochs 3
    python examples/fuzzer_baseline_mnist.py --batch runs/demo --out runs/demo/results.csv

Needs the ``tfds`` extra for MNIST. This is the long run; the test suite uses
a small synthetic task instead.
"""

import argparse

from fabricpc import setup_jax
from fabricpc.fuzzer import GeneratorConfig, generate, load_batch
from fabricpc.fuzzer.experiments import EvalSettings, evaluate_batch
from fabricpc.utils.data.dataloader import MnistLoader


def mnist_factory(batch_size):
    def factory(seed):
        return (
            MnistLoader(
                "train",
                batch_size=batch_size,
                tensor_format="flat",
                shuffle=True,
                seed=seed,
            ),
            MnistLoader(
                "test", batch_size=batch_size, tensor_format="flat", shuffle=False
            ),
        )

    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", help="load graphs from a saved batch directory")
    parser.add_argument("--num-graphs", type=int, default=5)
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out", default="fuzzer_mnist_results.csv")
    args = parser.parse_args()

    setup_jax()
    if args.batch:
        specs = load_batch(args.batch)
    else:
        config = GeneratorConfig(
            num_graphs=args.num_graphs,
            seed=args.generation_seed,
            num_nodes=(2, 4),
            input_shape=(784,),
            output_shape=(10,),
        )
        specs = generate(config)

    settings = EvalSettings(
        training_seeds=tuple(args.seeds), num_epochs=args.epochs, learning_rate=args.lr
    )
    report = evaluate_batch(specs, mnist_factory(args.batch_size), settings)
    path = report.to_csv(args.out)

    print(
        f"{'graph':>5} {'model':>10} {'accuracy':>16} {'time/seed':>10} {'peak MB':>8}"
    )
    for r in report:
        print(
            f"{r.graph_id:>5} {r.model:>10} "
            f"{r.accuracy_mean * 100:7.2f}% ± {r.accuracy_se * 100:5.2f} "
            f"{r.train_time_mean_s:9.1f}s {r.peak_memory_bytes / 2**20:8.0f}"
        )
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
