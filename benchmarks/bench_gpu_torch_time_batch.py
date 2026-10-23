from __future__ import annotations

import argparse

import torch

from bench_gpu_spa2 import make_scan_parameters
from bench_gpu_torch_full_spa2 import make_torch_runner
from common import (
    build_spa2_setup,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)
from paired import compare_paired, format_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark time-chunked batched PyTorch SPA2 propagation."
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--scan-mode", choices=("detuning", "power", "combined"), default="detuning")
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    print("[setup] building real SPA2 setup", flush=True)
    setup = build_spa2_setup()
    if not 1 <= args.initial_states <= len(setup["initial_states"]):
        parser.error(f"--initial-states must be between 1 and {len(setup['initial_states'])}")
    setup["initial_states"] = setup["initial_states"][: args.initial_states]
    detunings, powers, n_detuning, n_power = make_scan_parameters(
        args.batch, args.scan_mode
    )

    variants = {
        f"chunk_{chunk}": make_torch_runner(
            setup,
            n_steps=args.n_steps,
            detunings=detunings,
            intensity_prefactors=powers,
            time_chunk_size=chunk,
        )
        for chunk in args.chunks
    }
    baseline = "chunk_1"
    if baseline not in variants:
        parser.error("--chunks must include 1 as the reference")

    def max_probability_difference(reference, candidate) -> float:
        return float(
            torch.max(
                torch.abs(
                    reference.probabilities_final - candidate.probabilities_final
                )
            ).item()
        )

    paired = compare_paired(
        variants,
        repeat=args.repeat,
        warmup=args.warmup,
        baseline=baseline,
        sync=torch.cuda.synchronize,
        check=max_probability_difference,
        tolerance=1e-5,
        progress=True,
    )
    print(format_report(paired), flush=True)

    results = []
    comparison_by_name = {item.candidate: item for item in paired.comparisons}
    for name, timing in paired.timings.items():
        comparison = comparison_by_name.get(name)
        chunk_size = int(name.removeprefix("chunk_"))
        row = result_row(
            name,
            time_best_s=timing.best_s,
            time_mean_s=timing.mean_s,
            time_std_s=timing.std_s,
            repeat=args.repeat,
            warmup=args.warmup,
            batch=args.batch,
            scan_mode=args.scan_mode,
            n_detuning=n_detuning,
            n_power=n_power,
            initial_states=args.initial_states,
            n=64,
            n_steps=args.n_steps,
            time_chunk_size=chunk_size,
        )
        if comparison is not None:
            row.update(
                max_probability_difference=comparison.max_abs_difference,
                speedup_vs_chunk_1=comparison.speedup,
                paired_delta_s=comparison.delta_mean_s,
                paired_ci95_s=list(comparison.ci95_s),
                verdict=comparison.verdict,
            )
        results.append(row)

    payload = make_payload(
        benchmark="gpu_torch_time_batch",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "scan_mode": args.scan_mode,
            "initial_states": args.initial_states,
            "chunks": args.chunks,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        results=results,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_torch_time_batch"),
    )


if __name__ == "__main__":
    main()
