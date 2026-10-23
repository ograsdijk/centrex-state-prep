from __future__ import annotations

import argparse

import numpy as np

from bench_gpu_spa2 import make_gpu_runner, make_scan_parameters
from common import (
    build_spa2_setup,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)
from paired import compare_paired


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SPA2 GPU scan precision modes.")
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    setup = build_spa2_setup()
    setup["initial_states"] = setup["initial_states"][:4]
    detunings, powers, *_ = make_scan_parameters(args.batch, "combined")
    run128, sync, cp = make_gpu_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
        complex_dtype="complex128",
    )
    run64, _, _ = make_gpu_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
        complex_dtype="complex64",
    )
    paired = compare_paired(
        {"complex128": run128, "complex64": run64},
        repeat=args.repeat,
        warmup=args.warmup,
        baseline="complex128",
        sync=sync,
        progress=True,
    )
    ref = paired.results["complex128"]
    candidate = paired.results["complex64"]
    probabilities_ref = ref.probabilities_final.detach().cpu().numpy()
    probabilities_candidate = candidate.probabilities_final.detach().cpu().numpy()
    psi_ref = ref.psi_final.detach().cpu().numpy()
    psi_candidate = candidate.psi_final.detach().cpu().numpy()
    overlap = np.sum(np.conj(psi_ref) * psi_candidate, axis=-1)
    norm_product = np.sum(np.abs(psi_ref) ** 2, axis=-1) * np.sum(
        np.abs(psi_candidate) ** 2, axis=-1
    )
    fidelity = np.abs(overlap) ** 2 / norm_product
    print(f"complex128: {paired.timings['complex128'].mean_s:.4f} ± {paired.timings['complex128'].std_s:.4f}s")
    print(f"complex64:  {paired.timings['complex64'].mean_s:.4f} ± {paired.timings['complex64'].std_s:.4f}s")
    print(f"speedup: {paired.timings['complex128'].mean_s / paired.timings['complex64'].mean_s:.3f}x")
    print(f"max final probability difference: {np.max(np.abs(probabilities_ref - probabilities_candidate)):.3e}")
    print(f"max state infidelity: {np.max(np.maximum(0.0, 1.0 - fidelity)):.3e}")
    results = [
        result_row(
            f"gpu_{dtype}",
            time_mean_s=paired.timings[dtype].mean_s,
            time_std_s=paired.timings[dtype].std_s,
            batch=args.batch,
            n_steps=args.n_steps,
            initial_states=4,
            n=64,
            max_abs_probability_difference=(
                None if dtype == "complex128" else float(np.max(np.abs(probabilities_ref - probabilities_candidate)))
            ),
            max_state_infidelity=(
                None if dtype == "complex128" else float(np.max(np.maximum(0.0, 1.0 - fidelity)))
            ),
            speedup_vs_complex128=(
                None
                if dtype == "complex128"
                else paired.timings["complex128"].mean_s / paired.timings[dtype].mean_s
            ),
        )
        for dtype in ("complex128", "complex64")
    ]
    payload = make_payload(
        benchmark="gpu_precision",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "initial_states": 4,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_precision"),
    )


if __name__ == "__main__":
    main()
