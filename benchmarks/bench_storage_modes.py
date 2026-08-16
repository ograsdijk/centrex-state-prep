from __future__ import annotations

import argparse
from typing import Any

from common import (
    build_fields_and_hamiltonian,
    make_payload,
    parse_js,
    resolve_csv_path,
    result_row,
    spa2_initial_states,
    spa2_monitor_states,
    stored_result_bytes,
    time_call,
    write_results,
)


def run_case(setup: dict[str, Any], *, n_steps: int, mode: dict[str, Any]):
    from state_prep.simulator import Simulator

    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        None,
    )
    return simulator.run(
        N_steps=n_steps,
        progress=False,
        eig_backend="zheevd",
        **mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU simulator storage modes.")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--js", default="0,1,2", help="Comma-separated J values, default: 0,1,2")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    js = parse_js(args.js)
    if 1 not in js or 2 not in js:
        raise ValueError("bench_storage_modes requires --js to include J=1 and J=2")

    trajectory, electric_field, magnetic_field, hamiltonian = build_fields_and_hamiltonian(js=js)
    setup = {
        "trajectory": trajectory,
        "electric_field": electric_field,
        "magnetic_field": magnetic_field,
        "hamiltonian": hamiltonian,
        "initial_states": spa2_initial_states(),
    }
    monitor_states = [spa2_monitor_states()[0]]

    modes = [
        (
            "full_default",
            {
                "store_psis": True,
                "store_energies": True,
                "store_probabilities": True,
                "store_final_probabilities": True,
                "store_every": 1,
            },
        ),
        (
            "downsample_20",
            {
                "store_psis": True,
                "store_energies": True,
                "store_probabilities": True,
                "store_final_probabilities": True,
                "store_every": 20,
            },
        ),
        (
            "final_only",
            {
                "store_psis": False,
                "store_energies": False,
                "store_probabilities": False,
                "store_final_probabilities": True,
                "store_every": 1,
            },
        ),
        (
            "monitor_final_only",
            {
                "store_psis": False,
                "store_energies": False,
                "store_probabilities": False,
                "store_final_probabilities": False,
                "monitor_states": monitor_states,
                "store_final_monitor_probabilities": True,
                "store_every": 1,
            },
        ),
    ]

    results = []
    for case, mode in modes:
        timing, result = time_call(
            lambda mode=mode: run_case(setup, n_steps=args.n_steps, mode=mode),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(
            result_row(
                case,
                **timing,
                stored_bytes=stored_result_bytes(result),
                saved_steps=len(result.t_array),
                n=len(hamiltonian.QN),
                n_initial_states=len(setup["initial_states"]),
            )
        )

    payload = make_payload(
        benchmark="storage_modes",
        config={
            "n_steps": args.n_steps,
            "js": js,
            "repeat": args.repeat,
            "warmup": args.warmup,
        },
        results=results,
        include_gpu_env=False,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "storage_modes"),
    )


if __name__ == "__main__":
    main()

