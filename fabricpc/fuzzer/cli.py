"""Command-line interface: ``python -m fabricpc.fuzzer`` or ``fabricpc-fuzz``.

Subcommands::

    init-config  Write a starter config file to edit.
    generate     Generate a batch of graphs and save it to a directory.
    inspect      Print a saved graph (or every graph in a batch directory).
    check        Build saved graphs in FabricPC; exit code 1 if any fail.

Examples::

    fabricpc-fuzz init-config fuzz.json
    fabricpc-fuzz generate --config fuzz.json --out runs/demo
    fabricpc-fuzz generate -n 5 --seed 3 --set num_nodes=[3,6] --out runs/quick
    fabricpc-fuzz inspect runs/demo/graph_0000.json
    fabricpc-fuzz check runs/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from fabricpc.fuzzer import __version__
from fabricpc.fuzzer.config import ConfigError, GeneratorConfig, parse_override
from fabricpc.fuzzer.serialize import (
    MANIFEST_FILE,
    SerializationError,
    load,
    load_batch,
    save_batch,
)
from fabricpc.fuzzer.spec import GraphSpec

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fabricpc-fuzz",
        description="Generate valid, non-sequential predictive coding graphs for FabricPC.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-config", help="write a starter config file")
    p.add_argument("path", type=Path, help="where to write it (.json)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")

    p = sub.add_parser("generate", help="generate and save a batch of graphs")
    p.add_argument("--config", type=Path, help="settings file (.json or .toml)")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("-n", "--num-graphs", type=int, help="override num_graphs")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any setting, e.g. --set num_nodes=[3,6] (repeatable)",
    )
    p.add_argument("--overwrite", action="store_true", help="replace an existing batch")
    p.add_argument(
        "-q", "--quiet", action="store_true", help="print only the summary line"
    )

    p = sub.add_parser("inspect", help="print saved graphs")
    p.add_argument("path", type=Path, help="graph .json file or batch directory")

    p = sub.add_parser("check", help="build saved graphs in FabricPC")
    p.add_argument("path", type=Path, help="graph .json file or batch directory")
    return parser


def _load_any(path: Path) -> List[GraphSpec]:
    if path.is_dir() and (path / MANIFEST_FILE).exists():
        return load_batch(path)
    return [load(path)]


def _cmd_init_config(args: argparse.Namespace) -> int:
    if args.path.exists() and not args.force:
        print(f"error: {args.path} exists (use --force)", file=sys.stderr)
        return EXIT_USAGE
    GeneratorConfig().to_file(args.path)
    print(f"wrote {args.path}")
    return EXIT_OK


def _cmd_generate(args: argparse.Namespace) -> int:
    from fabricpc.fuzzer.pipeline import GenerationError, GenerationReport, generate

    config = (
        GeneratorConfig.from_file(args.config) if args.config else GeneratorConfig()
    )
    overrides = dict(parse_override(o) for o in args.overrides)
    if args.num_graphs is not None:
        overrides["num_graphs"] = args.num_graphs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        config = config.with_overrides(**overrides)
    config.validate()

    report = GenerationReport()
    try:
        specs = generate(config, report=report)
    except GenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    paths = save_batch(
        specs, args.out, config=config.to_dict(), overwrite=args.overwrite
    )

    if not args.quiet:
        for path, spec in zip(paths, specs):
            print(
                f"{path.name}  nodes={len(spec.nodes):<3} edges={len(spec.edges):<3} "
                f"cyclic={str(spec.is_cyclic):<5} {spec.fingerprint()[:12]}"
            )
    print(
        f"generated {len(specs)} graph(s) in {args.out} "
        f"(seed={config.seed}, attempts={sum(report.attempts)}, "
        f"rejected={len(report.rejected)}, duplicates={report.duplicates_skipped})"
    )
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    for spec in _load_any(args.path):
        meta = spec.metadata
        print(
            f"# graph_index={meta.get('graph_index')} graph_seed={meta.get('graph_seed')} "
            f"master_seed={meta.get('master_seed')}"
        )
        print(spec.summary())
        print()
    return EXIT_OK


def _cmd_check(args: argparse.Namespace) -> int:
    from fabricpc.fuzzer.build import BuildError, to_fabricpc

    failures = 0
    specs = _load_any(args.path)
    for i, spec in enumerate(specs):
        label = spec.metadata.get("graph_index", i)
        try:
            to_fabricpc(spec)
            print(f"graph {label}: ok")
        except BuildError as exc:
            failures += 1
            print(f"graph {label}: FAILED ({exc.stage}) {exc}")
    print(f"{len(specs) - failures}/{len(specs)} built successfully")
    return EXIT_FAILED if failures else EXIT_OK


COMMANDS = {
    "init-config": _cmd_init_config,
    "generate": _cmd_generate,
    "inspect": _cmd_inspect,
    "check": _cmd_check,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code instead of calling ``sys.exit``."""
    args = _build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (SerializationError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
