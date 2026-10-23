from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from common import (
    build_spa2_setup,
    make_detunings,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)
from paired import compare_paired, format_report


def run_case(
    setup: dict[str, Any],
    *,
    path: str,
    n_steps: int,
    batch: int,
    workers: int,
    block_eigh: bool,
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
    if path == "run":
        return simulator.run(
            N_steps=n_steps,
            store_psis=False,
            store_energies=False,
            store_probabilities=False,
            store_final_probabilities=True,
            monitor_states=setup["monitor_states"],
            store_final_monitor_probabilities=True,
            progress=False,
            block_eigh=block_eigh,
        )

    detunings = make_detunings(batch)
    detunings_batched = np.column_stack((detunings, detunings))
    intensity_prefactors = np.ones_like(detunings_batched)
    return simulator.run_microwave_scan(
        detunings_hz=detunings_batched,
        intensity_prefactors=intensity_prefactors,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=workers,
        block_eigh=block_eigh,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense and automatic exact-block CPU SPA2 eigensolves."
    )
    parser.add_argument("--path", choices=("run", "scan"), default="scan")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    setup = build_spa2_setup()
    from state_prep.simulator import _find_exact_hamiltonian_blocks

    hamiltonians = [
        setup["hamiltonian"].get_H_t_func(),
        *[
            microwave.get_H_t_func(
                setup["trajectory"].R_t, setup["hamiltonian"].QN
            )
            for microwave in setup["microwave_fields"]
        ],
    ]
    t_array = np.linspace(0.0, setup["trajectory"].get_T(), args.n_steps)
    blocks = _find_exact_hamiltonian_blocks(hamiltonians, t_array, "mid")
    block_sizes = (
        [len(block) for block in blocks]
        if blocks is not None
        else [len(setup["hamiltonian"].QN)]
    )
    pair = compare_paired(
        {
            "dense": lambda: run_case(
                setup,
                path=args.path,
                n_steps=args.n_steps,
                batch=args.batch,
                workers=args.workers,
                block_eigh=False,
            ),
            "blocks": lambda: run_case(
                setup,
                path=args.path,
                n_steps=args.n_steps,
                batch=args.batch,
                workers=args.workers,
                block_eigh=True,
            ),
        },
        repeat=args.repeat,
        warmup=args.warmup,
        baseline="dense",
        check=lambda dense, blocks: float(
            np.max(np.abs(dense.probabilities_final - blocks.probabilities_final))
        ),
        tolerance=1e-5,
        progress=True,
    )
    print(format_report(pair), flush=True)

    comparison = pair.comparisons[0]
    results = []
    for name, timing in pair.timings.items():
        row = result_row(
            name,
            time_best_s=timing.best_s,
            time_mean_s=timing.mean_s,
            time_std_s=timing.std_s,
            repeat=args.repeat,
            warmup=args.warmup,
            path=args.path,
            n_steps=args.n_steps,
            batch=args.batch if args.path == "scan" else None,
            workers=args.workers if args.path == "scan" else None,
            eig_backend="zheevd",
        )
        if name == "blocks":
            row.update(
                max_abs_probability_difference=comparison.max_abs_difference,
                speedup_vs_dense=comparison.speedup,
                paired_delta_s=comparison.delta_mean_s,
                paired_ci95_s=list(comparison.ci95_s),
                verdict=comparison.verdict,
            )
        results.append(row)

    payload = make_payload(
        benchmark="cpu_block_eigh",
        config={
            "path": args.path,
            "n_steps": args.n_steps,
            "batch": args.batch if args.path == "scan" else None,
            "workers": args.workers if args.path == "scan" else None,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "eig_backend": "zheevd",
            "detected_block_sizes": block_sizes,
        },
        results=results,
        include_gpu_env=False,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "cpu_block_eigh"),
    )


if __name__ == "__main__":
    main()
