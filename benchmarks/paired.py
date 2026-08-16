"""Paired before/after benchmark runner.

Timing differences worth chasing in this repository are in the `2-20%` range,
which is the same order as run-to-run noise on the development machine. A
baseline measured in one session and a candidate measured in another is not
evidence.

This module runs variants *interleaved* in the same session and compares them
using per-round paired differences. Interleaving means slow-moving drift
(thermal throttling, background load) affects both variants within a round and
cancels in the difference, which is what makes small effects resolvable at
modest repeat counts.

Statistics note: standard deviations here are *sample* standard deviations
(`ddof=1`), because the standard error of the paired difference requires them.
`common.time_call` reports population standard deviations instead; for the
repeat counts used in practice the two differ negligibly, but the field names
and the `std_ddof` entry in emitted payloads record which was used.

Typical use:

    from paired import compare_paired, format_report

    result = compare_paired(
        {"baseline": run_baseline, "candidate": run_candidate},
        repeat=7,
        warmup=1,
        check=max_abs_probability_difference,
        tolerance=1e-9,
    )
    print(format_report(result))

Self-test (validates the runner itself against known synthetic effects):

    .\\.venv\\Scripts\\python.exe benchmarks\\paired.py --self-test
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Callable, Mapping, Optional

from common import (
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)


# Two-sided 95% t critical values by degrees of freedom (n_rounds - 1).
# Using the normal approximation (1.96) at small n would overstate confidence.
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _t95(df: int) -> float:
    if df < 1:
        return math.inf
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.96
    # Between tabulated points, take the next lower tabulated df (conservative).
    return _T95[max(k for k in _T95 if k < df)]


@dataclass
class VariantTimings:
    """Per-round wall-clock times for one variant."""

    name: str
    times_s: list[float]

    @property
    def mean_s(self) -> float:
        return mean(self.times_s)

    @property
    def std_s(self) -> float:
        return stdev(self.times_s) if len(self.times_s) > 1 else 0.0

    @property
    def best_s(self) -> float:
        return min(self.times_s)

    def as_row(self) -> dict[str, Any]:
        return {
            "variant": self.name,
            "mean_s": self.mean_s,
            "std_s": self.std_s,
            "best_s": self.best_s,
            "rounds": len(self.times_s),
        }


@dataclass
class PairedComparison:
    """Comparison of one candidate against the baseline.

    `deltas_s` holds per-round `candidate - baseline` differences, so a negative
    mean means the candidate is faster.
    """

    baseline: str
    candidate: str
    deltas_s: list[float]
    baseline_mean_s: float
    candidate_mean_s: float
    baseline_std_s: float
    candidate_std_s: float
    max_abs_difference: Optional[float] = None
    tolerance: Optional[float] = None

    @property
    def delta_mean_s(self) -> float:
        return mean(self.deltas_s)

    @property
    def delta_std_s(self) -> float:
        return stdev(self.deltas_s) if len(self.deltas_s) > 1 else 0.0

    @property
    def delta_stderr_s(self) -> float:
        n = len(self.deltas_s)
        if n < 2:
            return math.inf
        return self.delta_std_s / math.sqrt(n)

    @property
    def ci95_s(self) -> tuple[float, float]:
        half = _t95(len(self.deltas_s) - 1) * self.delta_stderr_s
        return (self.delta_mean_s - half, self.delta_mean_s + half)

    @property
    def speedup(self) -> float:
        """Baseline mean divided by candidate mean. Above 1.0 means faster."""
        if self.candidate_mean_s == 0:
            return math.inf
        return self.baseline_mean_s / self.candidate_mean_s

    @property
    def relative_change(self) -> float:
        """Fractional runtime change. Negative means the candidate is faster."""
        if self.baseline_mean_s == 0:
            return math.nan
        return self.delta_mean_s / self.baseline_mean_s

    @property
    def separated_by_spread(self) -> bool:
        """Conservative unpaired check: means differ by more than summed spreads.

        This is the fallback criterion documented in `IMPROVEMENTS.md` for
        comparisons that are not paired. It is strictly weaker than the paired
        confidence interval and is reported only as a cross-check.
        """
        return abs(self.candidate_mean_s - self.baseline_mean_s) > (
            self.baseline_std_s + self.candidate_std_s
        )

    @property
    def verdict(self) -> str:
        """One of 'improvement', 'regression', or 'unproven'.

        Based on whether the 95% confidence interval of the paired difference
        excludes zero. With fewer than two rounds nothing can be concluded.
        """
        if len(self.deltas_s) < 2:
            return "unproven"
        low, high = self.ci95_s
        if high < 0:
            return "improvement"
        if low > 0:
            return "regression"
        return "unproven"

    @property
    def agrees(self) -> Optional[bool]:
        """Whether the candidate's output matched the baseline within tolerance."""
        if self.max_abs_difference is None or self.tolerance is None:
            return None
        return self.max_abs_difference <= self.tolerance

    def as_row(self) -> dict[str, Any]:
        low, high = self.ci95_s
        row = {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "baseline_mean_s": self.baseline_mean_s,
            "baseline_std_s": self.baseline_std_s,
            "candidate_mean_s": self.candidate_mean_s,
            "candidate_std_s": self.candidate_std_s,
            "delta_mean_s": self.delta_mean_s,
            "delta_std_s": self.delta_std_s,
            "delta_stderr_s": self.delta_stderr_s,
            "delta_ci95_low_s": low,
            "delta_ci95_high_s": high,
            "speedup": self.speedup,
            "relative_change": self.relative_change,
            "separated_by_spread": self.separated_by_spread,
            "verdict": self.verdict,
            "rounds": len(self.deltas_s),
        }
        if self.max_abs_difference is not None:
            row["max_abs_difference"] = self.max_abs_difference
            row["tolerance"] = self.tolerance
            row["agrees"] = self.agrees
        return row


@dataclass
class PairedResult:
    baseline: str
    timings: dict[str, VariantTimings]
    comparisons: list[PairedComparison]
    repeat: int
    warmup: int
    interleaved: bool
    results: dict[str, Any] = field(default_factory=dict, repr=False)

    def rows(self) -> list[dict[str, Any]]:
        rows = [
            result_row("timing", **timing.as_row()) for timing in self.timings.values()
        ]
        rows.extend(
            result_row("comparison", **comparison.as_row())
            for comparison in self.comparisons
        )
        return rows

    def config(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "variants": list(self.timings),
            "repeat": self.repeat,
            "warmup": self.warmup,
            "interleaved": self.interleaved,
            "std_ddof": 1,
        }


def compare_paired(
    variants: Mapping[str, Callable[[], Any]],
    *,
    repeat: int = 5,
    warmup: int = 1,
    baseline: Optional[str] = None,
    sync: Optional[Callable[[], None]] = None,
    check: Optional[Callable[[Any, Any], float]] = None,
    tolerance: Optional[float] = None,
    interleave: bool = True,
    progress: bool = False,
) -> PairedResult:
    """Time variants interleaved and compare each against the baseline.

    Parameters
    ----------
    variants:
        Mapping of name to a zero-argument callable performing the work. Each
        callable must be self-contained and repeatable; anything cached between
        calls will be measured as a speedup that does not exist.
    repeat:
        Number of rounds. Every variant runs once per round. Use at least `5`,
        and more when the expected effect is below `10%`.
    warmup:
        Untimed calls per variant before measurement begins. Covers import,
        allocation, JIT, and first-touch page faults.
    baseline:
        Name of the reference variant. Defaults to the first key.
    sync:
        Called after each variant invocation before the clock is stopped. Required
        for GPU work, where kernel launches are asynchronous; pass a CUDA
        synchronize (see `common.time_call`, which takes the same callback).
    check:
        Optional `check(baseline_result, candidate_result) -> float` returning a
        scalar discrepancy, evaluated on the final round's results. Use for any
        change that can alter numerical output.
    tolerance:
        Threshold the value from `check` must not exceed.
    interleave:
        Rotate variant order each round so no variant systematically occupies the
        same position. Disable only when a variant must run first for external
        reasons; doing so reintroduces drift bias.
    progress:
        Print round progress to stdout.

    Returns
    -------
    PairedResult with per-variant timings and per-candidate comparisons.
    """
    if not variants:
        raise ValueError("variants must not be empty")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if check is not None and tolerance is None:
        raise ValueError("tolerance is required when check is provided")

    names = list(variants)
    base_name = baseline if baseline is not None else names[0]
    if base_name not in variants:
        raise ValueError(f"baseline {base_name!r} is not one of {names}")

    for name in names:
        for _ in range(warmup):
            variants[name]()
            if sync is not None:
                sync()

    times: dict[str, list[float]] = {name: [] for name in names}
    last_results: dict[str, Any] = {}

    for round_idx in range(repeat):
        # Rotate the order so position within a round is not confounded with variant.
        if interleave:
            offset = round_idx % len(names)
            order = names[offset:] + names[:offset]
        else:
            order = names

        for name in order:
            start = time.perf_counter()
            last_results[name] = variants[name]()
            if sync is not None:
                sync()
            times[name].append(time.perf_counter() - start)

        if progress:
            summary = "  ".join(f"{name}={times[name][-1]:.4f}s" for name in names)
            print(f"round {round_idx + 1}/{repeat}: {summary}", flush=True)

    timings = {name: VariantTimings(name=name, times_s=times[name]) for name in names}

    comparisons: list[PairedComparison] = []
    for name in names:
        if name == base_name:
            continue

        max_abs_difference = None
        if check is not None:
            max_abs_difference = float(
                check(last_results[base_name], last_results[name])
            )

        comparisons.append(
            PairedComparison(
                baseline=base_name,
                candidate=name,
                deltas_s=[
                    cand - base
                    for cand, base in zip(times[name], times[base_name])
                ],
                baseline_mean_s=timings[base_name].mean_s,
                candidate_mean_s=timings[name].mean_s,
                baseline_std_s=timings[base_name].std_s,
                candidate_std_s=timings[name].std_s,
                max_abs_difference=max_abs_difference,
                tolerance=tolerance if check is not None else None,
            )
        )

    return PairedResult(
        baseline=base_name,
        timings=timings,
        comparisons=comparisons,
        repeat=repeat,
        warmup=warmup,
        interleaved=interleave,
        results=last_results,
    )


def format_report(result: PairedResult) -> str:
    """Render a paired result as a human-readable report."""
    lines: list[str] = []
    lines.append(
        f"paired benchmark: {result.repeat} rounds, {result.warmup} warmup, "
        f"{'interleaved' if result.interleaved else 'blocked'}, "
        f"baseline={result.baseline!r}"
    )
    lines.append("")
    lines.append(f"{'variant':<24}{'mean':>12}{'std':>12}{'best':>12}")
    for timing in result.timings.values():
        lines.append(
            f"{timing.name:<24}"
            f"{timing.mean_s:>11.4f}s"
            f"{timing.std_s:>11.4f}s"
            f"{timing.best_s:>11.4f}s"
        )

    for comparison in result.comparisons:
        low, high = comparison.ci95_s
        lines.append("")
        lines.append(f"{comparison.candidate} vs {comparison.baseline}")
        lines.append(
            f"  paired delta : {comparison.delta_mean_s:+.4f}s "
            f"+/- {comparison.delta_stderr_s:.4f}s (stderr)"
        )
        lines.append(f"  95% CI       : [{low:+.4f}s, {high:+.4f}s]")
        lines.append(
            f"  speedup      : {comparison.speedup:.3f}x "
            f"({comparison.relative_change * 100:+.1f}% runtime)"
        )
        lines.append(f"  verdict      : {comparison.verdict.upper()}")
        if comparison.verdict == "unproven":
            lines.append(
                "                 difference is within noise; increase --repeat "
                "or treat the change as having no measured effect"
            )
        if not comparison.separated_by_spread and comparison.verdict != "unproven":
            lines.append(
                "                 note: significant when paired, but means are "
                "within summed spreads; interleaving is doing the work here"
            )
        if comparison.agrees is not None:
            status = "OK" if comparison.agrees else "FAILED"
            lines.append(
                f"  agreement    : {status} "
                f"(max abs diff {comparison.max_abs_difference:.3e}, "
                f"tolerance {comparison.tolerance:.3e})"
            )

    return "\n".join(lines)


def emit(
    result: PairedResult,
    *,
    benchmark: str,
    output: Optional[str],
    csv_arg: Optional[str],
    extra_config: Optional[dict[str, Any]] = None,
    include_gpu_env: bool = True,
) -> None:
    """Write a paired result as JSON/CSV using the shared payload format."""
    config = result.config()
    if extra_config:
        config.update(extra_config)

    payload = make_payload(
        benchmark=benchmark,
        config=config,
        results=result.rows(),
        include_gpu_env=include_gpu_env,
    )
    write_results(
        payload,
        output=output,
        csv_path=resolve_csv_path(output, csv_arg, benchmark),
    )


def _self_test(repeat: int, workload_ms: float) -> int:
    """Validate the runner against synthetic effects of known sign and size.

    Checks that a deliberately slower variant is detected as a regression, and
    that a variant doing identical work is not reported as a change.
    """
    import numpy as np

    size = 220
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    matrix = matrix + matrix.conj().T

    def work(scale: float) -> Callable[[], Any]:
        iterations = max(1, int(workload_ms * scale))

        def run() -> Any:
            total = 0.0
            for _ in range(iterations):
                total += float(np.linalg.eigvalsh(matrix)[0])
            return total

        return run

    print("case 1: identical work, expect UNPROVEN")
    identical = compare_paired(
        {"baseline": work(1.0), "same": work(1.0)},
        repeat=repeat,
        warmup=1,
        progress=True,
    )
    print(format_report(identical))
    print()

    print("case 2: candidate doing 1.5x the work, expect REGRESSION")
    slower = compare_paired(
        {"baseline": work(1.0), "slower": work(1.5)},
        repeat=repeat,
        warmup=1,
        progress=True,
    )
    print(format_report(slower))
    print()

    failures = []
    if identical.comparisons[0].verdict != "unproven":
        failures.append(
            f"identical work reported {identical.comparisons[0].verdict!r}, "
            "expected 'unproven' (the runner is claiming an effect that is not there)"
        )
    if slower.comparisons[0].verdict != "regression":
        failures.append(
            f"1.5x work reported {slower.comparisons[0].verdict!r}, "
            "expected 'regression' (the runner is missing a real effect)"
        )

    if failures:
        print("SELF-TEST FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("SELF-TEST PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired before/after benchmark runner and its self-test."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the runner against synthetic effects of known sign.",
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument(
        "--workload",
        type=float,
        default=6,
        help="Self-test workload size (eigensolve iterations for the baseline).",
    )
    args = parser.parse_args()

    if not args.self_test:
        parser.print_help()
        print(
            "\nThis module is primarily a library. Import `compare_paired` from "
            "a benchmark script; run --self-test to verify the runner itself."
        )
        return 0

    return _self_test(repeat=args.repeat, workload_ms=args.workload)


if __name__ == "__main__":
    raise SystemExit(main())
