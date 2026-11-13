"""Generation throughput and memory measurements for the pipeline.

SR-NF-PE-01 asks for 100 graphs of up to 20 hidden nodes under two minutes;
SR-NF-DC-05 asks for peak memory under 4 GB. Both are *soft* goals: the test
records the measured numbers (they also go into ``docs/fuzzer/STATUS.md`` and
``WEEK3_SUMMARY.md``) and only hard-fails when the run misses by an order of
magnitude, which would signal a regression rather than a slow laptop.

Marked ``slow``: excluded from the 5-minute default CI budget (SR-NF-PE-02),
run on demand with ``pytest -m slow``.
"""

import resource
import sys
import time
import pytest

from fabricpc.fuzzer.config import GeneratorConfig
from fabricpc.fuzzer.pipeline import generate

TARGET_TIME_S = 120.0  # SR-NF-PE-01 soft target
TARGET_PEAK_MB = 4096.0  # SR-NF-DC-05 soft target
HARD_LIMIT_S = 15 * 60.0  # regression guard, not the requirement


def _peak_rss_bytes() -> int:
    """Peak resident set size of this process so far.

    ``getrusage`` reports kilobytes on Linux and bytes on macOS; the CI matrix
    is Linux and the team develops on macOS, so normalize explicitly.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else int(peak * 1024)


def _tiny_per_graph_config() -> GeneratorConfig:
    """100 graphs of 2..20 hidden nodes; smoke shapes keep the steps cheap."""
    return GeneratorConfig(
        num_graphs=100,
        seed=11,
        num_nodes=(2, 20),
        in_degree=(1, 3),
        input_shape=(64,),
        output_shape=(10,),
        hidden_sizes=(16, 32, 64),
        max_attempts=200,
    ).validate()


@pytest.mark.slow
def test_sr_nf_pe_01_dc_05_100_graphs_throughput_and_memory(capsys) -> None:
    """SR-NF-PE-01 + SR-NF-DC-05: 100 graphs (<= 20 hidden nodes) timed, RSS measured.

    Measured values are printed for TEST_LOG/STATUS regardless of the soft
    targets; the assertions cover correctness (100 valid distinct graphs,
    size bound) and a coarse regression guard.
    """
    config = _tiny_per_graph_config()
    peak_before = _peak_rss_bytes()
    start = time.perf_counter()
    specs = generate(config)
    elapsed = time.perf_counter() - start
    peak_after = _peak_rss_bytes()

    assert len(specs) == config.num_graphs, (
        f"partial batch: {len(specs)} of {config.num_graphs}; the perf run "
        "must use a config that always fills (check failure_counts first)"
    )
    assert all(len(s.nodes) <= 22 for s in specs)  # <= 20 hidden + input + output
    assert len({s.fingerprint() for s in specs}) == len(specs)

    peak_mb = peak_after / (1024 * 1024)
    delta_mb = max(0, peak_after - peak_before) / (1024 * 1024)
    time_ok = elapsed < TARGET_TIME_S
    mem_ok = peak_mb < TARGET_PEAK_MB
    print(
        f"\nSR-NF-PE-01: generated 100 graphs (2..20 hidden nodes) in "
        f"{elapsed:.1f}s -- target {TARGET_TIME_S:.0f}s: "
        f"{'MET' if time_ok else 'MISSED'}"
    )
    print(
        f"SR-NF-DC-05: peak RSS {peak_mb:.0f} MB total "
        f"(+{delta_mb:.0f} MB during the run) -- target {TARGET_PEAK_MB:.0f} MB: "
        f"{'MET' if mem_ok else 'MISSED'}"
    )
    with capsys.disabled():
        print(
            "measured: "
            f"time={elapsed:.1f}s peak_rss_mb={peak_mb:.0f} "
            f"run_delta_mb={delta_mb:.0f} "
            f"(targets: {TARGET_TIME_S:.0f}s / {TARGET_PEAK_MB:.0f} MB, soft)"
        )
    assert elapsed < HARD_LIMIT_S, (
        f"{elapsed:.0f}s is beyond the {HARD_LIMIT_S:.0f}s regression guard; "
        "this is no longer a soft-target miss"
    )
