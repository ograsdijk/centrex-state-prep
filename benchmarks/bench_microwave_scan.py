from __future__ import annotations

import argparse
from typing import Any

from joblib import effective_n_jobs
import numpy as np

from common import (
    build_spa2_setup,
    configure_microwaves,
    make_detunings,
    make_payload,
    resolve_csv_path,
    result_row,
    time_call,
    write_results,
)


def run_shared_slow(
    setup: dict[str, Any],
    *,
    n_steps: int,
    detunings: np.ndarray,
    workers: int = 1,
    parallel_backend: str = "loky",
):
    from state_prep.simulator import Simulator

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )
    detunings_batched = np.column_stack([detunings, detunings])
    intensity_prefactors = np.ones((detunings.size, 2), dtype=float)
    return simulator.run_microwave_scan(
        detunings_hz=detunings_batched,
        intensity_prefactors=intensity_prefactors,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=workers,
        parallel_backend=parallel_backend,
    )


def run_naive(setup: dict[str, Any], *, n_steps: int, detunings: np.ndarray):
    from state_prep.simulator import Simulator

    probabilities = []
    main, bg = setup["microwave_fields"]
    for detuning in detunings:
        main.set_frequency(setup["frequency"] + float(detuning))
        bg.set_frequency(setup["frequency"] + float(detuning))
        simulator = Simulator(
            setup["trajectory"],
            setup["electric_field"],
            setup["magnetic_field"],
            setup["initial_states"],
            setup["hamiltonian"],
            setup["microwave_fields"],
        )
        result = simulator.run(
            N_steps=n_steps,
            store_psis=False,
            store_energies=False,
            store_probabilities=False,
            store_final_probabilities=True,
            monitor_states=setup["monitor_states"],
            store_final_monitor_probabilities=True,
            progress=False,
            eig_backend="zheevd",
        )
        probabilities.append(result.probabilities_final)

    configure_microwaves(
        setup["microwave_fields"],
        frequency=setup["frequency"],
        r0=setup["r0"],
    )
    return np.stack(probabilities, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU shared-slow microwave scans.")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare-naive", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--worker-sweep",
        type=int,
        nargs="+",
        help="Worker counts to benchmark. Overrides --workers. Use -1 for all available workers.",
    )
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    setup = build_spa2_setup()
    detunings = make_detunings(args.batch)
    results = []
    worker_sweep = args.worker_sweep if args.worker_sweep else [1, args.workers]
    worker_sweep = list(dict.fromkeys(worker_sweep))
    if 1 not in worker_sweep:
        worker_sweep.insert(0, 1)

    serial_timing, serial_result = time_call(
        lambda: run_shared_slow(
            setup,
            n_steps=args.n_steps,
            detunings=detunings,
            workers=1,
            parallel_backend="loky",
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    results.append(
        result_row(
            "shared_slow_scan_workers_1",
            **serial_timing,
            batch=args.batch,
            n=len(setup["hamiltonian"].QN),
            workers=1,
            effective_workers=1,
            parallel_backend="serial",
            chunk_count=1,
            speedup_vs_serial=1.0,
            max_abs_probability_diff_vs_serial=0.0,
        )
    )

    for workers in worker_sweep:
        if workers == 1:
            continue
        effective_workers = min(effective_n_jobs(workers), args.batch)
        parallel_timing, parallel_result = time_call(
            lambda workers=workers: run_shared_slow(
                setup,
                n_steps=args.n_steps,
                detunings=detunings,
                workers=workers,
                parallel_backend="loky",
            ),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        max_diff = float(
            np.max(
                np.abs(
                    parallel_result.probabilities_final
                    - serial_result.probabilities_final
                )
            )
        )
        speedup = serial_timing["time_best_s"] / parallel_timing["time_best_s"]
        results.append(
            result_row(
                f"shared_slow_scan_workers_{workers}",
                **parallel_timing,
                batch=args.batch,
                n=len(setup["hamiltonian"].QN),
                workers=workers,
                effective_workers=effective_workers,
                parallel_backend="loky",
                chunk_count=effective_workers,
                speedup_vs_serial=speedup,
                max_abs_probability_diff_vs_serial=max_diff,
            )
        )

    naive_timing = None
    if args.compare_naive:
        naive_timing, naive_probs = time_call(
            lambda: run_naive(setup, n_steps=args.n_steps, detunings=detunings),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        max_diff = float(np.max(np.abs(naive_probs - serial_result.probabilities_final)))
        speedup = naive_timing["time_best_s"] / serial_timing["time_best_s"]
        results.append(
            result_row(
                "naive_repeated_runs",
                **naive_timing,
                batch=args.batch,
                n=len(setup["hamiltonian"].QN),
                speedup_shared_slow=speedup,
                max_abs_probability_diff=max_diff,
            )
        )

    payload = make_payload(
        benchmark="microwave_scan",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "compare_naive": args.compare_naive,
            "worker_sweep": worker_sweep,
            "parallel_backend": "loky",
        },
        results=results,
        include_gpu_env=False,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "microwave_scan"),
    )


if __name__ == "__main__":
    main()
