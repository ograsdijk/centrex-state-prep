from __future__ import annotations

import argparse
from typing import Any

from bench_gpu_spa2 import make_gpu_runner, make_scan_parameters
from common import (
    build_spa2_setup,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile SPA2 GPU propagation stages.")
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--dtype", choices=("complex64", "complex128"), default="complex128")
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    setup = build_spa2_setup()
    setup["initial_states"] = setup["initial_states"][: args.initial_states]
    detunings, powers, *_ = make_scan_parameters(args.batch, "combined")
    run, sync, _ = make_gpu_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
        complex_dtype=args.dtype,
    )
    timings: dict[str, Any] = {}
    output = run(profile_timings=timings)
    sync()
    print(f"final states: shape={output.psi_final.shape}, dtype={output.psi_final.dtype}")
    results = []
    for name, samples in timings.items():
        mean_per_sample = sum(samples) / max(1, len(samples))
        estimate = mean_per_sample * max(1, args.n_steps - 1)
        print(
            f"{name}: mean sample={mean_per_sample * 1e3:.3f} ms, "
            f"estimated total={estimate:.3f} s, samples={len(samples)}"
        )
        results.append(
            result_row(
                "gpu_spa2_stage_profile",
                stage=name,
                time_mean_s=mean_per_sample,
                estimated_total_s=estimate,
                samples=len(samples),
                batch=args.batch,
                initial_states=args.initial_states,
                n=64,
            )
        )
    payload = make_payload(
        benchmark="gpu_spa2_stage_profile",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "initial_states": args.initial_states,
            "dtype": args.dtype,
            "profile_step_stride": 25,
            "slow_eigh_sampling": "every tile, weighted per propagated step",
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_spa2_stage_profile"),
    )


if __name__ == "__main__":
    main()
