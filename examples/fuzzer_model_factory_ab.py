"""Run one fuzzed graph through the MNIST lateral-connections A/B harness.

This is the Part 1 prototype check: a generated ``GraphSpec`` is adapted with
:func:`fabricpc.fuzzer.as_model_factory` and dropped into the same
``ABExperiment`` that ``examples/mnist_lateral_connections.py`` uses. The other
arm is that example's hand-written lateral model, unchanged.

    python examples/fuzzer_model_factory_ab.py

One epoch, one trial. Set ``--epochs`` / ``--batch-size`` to change the cost.
Needs the ``[tfds]`` extra and a local or downloadable MNIST.
"""

from __future__ import annotations

import argparse

import jax
import optax

from fabricpc import setup_jax
from fabricpc.core.inference import InferenceSGD
from fabricpc.experiments import ABExperiment, ExperimentArm
from fabricpc.fuzzer import GeneratorConfig, as_model_factory, generate
from fabricpc.training import evaluate, train
from fabricpc.utils.data.dataloader import MnistLoader

try:  # running as ``python examples/...``
    from mnist_lateral_connections import create_lateral_model
except ImportError:  # running as a module from the repo root
    from examples.mnist_lateral_connections import create_lateral_model

# Match the hand-written arm so the comparison is about architecture, not
# inference hyperparameters.
INFERENCE = InferenceSGD(eta_infer=0.20, infer_steps=20)
OPTIMIZER = optax.adamw(0.001, weight_decay=0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--verbose", action="store_true", help="show per-epoch progress bars"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_jax()
    jax.config.update("jax_default_prng_impl", "threefry2x32")

    config = GeneratorConfig(
        num_graphs=1,
        seed=args.seed,
        num_nodes=(1, 2),
        in_degree=(1, 2),
        node_types=("Linear",),
        input_shape=(784,),
        output_shape=(10,),
        hidden_sizes=(args.hidden_size,),
        hidden_activations=("sigmoid",),
        output_activation="softmax",
        output_energy="cross_entropy",
    )
    spec = generate(config)[0]
    print("fuzzed graph:")
    print(spec.summary())
    print(f"fingerprint={spec.fingerprint()}")
    print()

    def loaders(seed: int):
        return (
            MnistLoader(
                "train",
                batch_size=args.batch_size,
                tensor_format="flat",
                shuffle=True,
                seed=seed,
            ),
            MnistLoader(
                "test",
                batch_size=args.batch_size,
                tensor_format="flat",
                shuffle=False,
            ),
        )

    train_config = {"num_epochs": args.epochs}
    arm_fuzzed = ExperimentArm(
        name="Fuzzed",
        model_factory=as_model_factory(spec, inference=INFERENCE),
        train_fn=train,
        eval_fn=evaluate,
        optimizer=OPTIMIZER,
        train_config=train_config,
    )
    arm_lateral = ExperimentArm(
        name="Lateral",
        model_factory=create_lateral_model,
        train_fn=train,
        eval_fn=evaluate,
        optimizer=OPTIMIZER,
        train_config=train_config,
    )

    experiment = ABExperiment(
        arm_a=arm_fuzzed,
        arm_b=arm_lateral,
        metric="accuracy",
        data_loader_factory=loaders,
        n_trials=1,
        verbose=args.verbose,
    )
    results = experiment.run()
    results.print_summary()


if __name__ == "__main__":
    main()
