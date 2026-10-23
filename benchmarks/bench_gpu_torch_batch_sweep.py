from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters, run_cpu
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
        description="Paired CPU versus full-PyTorch GPU scan batch sweep."
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batches", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--scan-mode", choices=("detuning", "power", "combined"), default="detuning")
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=10)
    parser.add_argument("--time-chunk-size", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    if any(batch < 1 for batch in args.batches):
        parser.error("batch sizes must be positive")
    print("[setup] building real SPA2 setup", flush=True)
    setup = build_spa2_setup()
    initial_states = setup["initial_states"]
    if not 1 <= args.initial_states <= len(initial_states):
        parser.error(f"--initial-states must be between 1 and {len(initial_states)}")
    setup["initial_states"] = initial_states[: args.initial_states]

    results = []
    for batch in args.batches:
        print(f"\n[batch={batch}] preparing both backends", flush=True)
        detunings, powers, n_detuning, n_power = make_scan_parameters(
            batch, args.scan_mode
        )
        cpu_run = lambda: run_cpu(
            setup,
            n_steps=args.n_steps,
            detunings=detunings,
            intensity_prefactors=powers,
            workers=args.cpu_workers,
        )
        torch_run = make_torch_runner(
            setup,
            n_steps=args.n_steps,
            detunings=detunings,
            intensity_prefactors=powers,
            time_chunk_size=args.time_chunk_size,
        )

        torch.cuda.reset_peak_memory_stats()
        torch_warmup = torch_run()
        torch.cuda.synchronize()
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        del torch_warmup
        torch.cuda.reset_peak_memory_stats()

        def probability_difference(reference, candidate) -> float:
            candidate_probabilities = candidate.probabilities_final.detach().cpu().numpy()
            return float(np.max(np.abs(reference.probabilities_final - candidate_probabilities)))

        paired = compare_paired(
            {"cpu": cpu_run, "torch_gpu": torch_run},
            repeat=args.repeat,
            warmup=args.warmup,
            baseline="cpu",
            sync=torch.cuda.synchronize,
            check=probability_difference,
            tolerance=1e-5,
            progress=True,
        )
        print(format_report(paired), flush=True)

        cpu_output = paired.results["cpu"]
        torch_output = paired.results["torch_gpu"]
        gpu_psi = torch_output.psi_final.detach().cpu().numpy()
        cpu_psi = np.asarray(cpu_output.psis_final)
        overlap = np.sum(np.conj(cpu_psi) * gpu_psi, axis=-1)
        norm_cpu = np.sum(np.abs(cpu_psi) ** 2, axis=-1)
        norm_gpu = np.sum(np.abs(gpu_psi) ** 2, axis=-1)
        fidelity = np.abs(overlap) ** 2 / (norm_cpu * norm_gpu)
        max_infidelity = float(np.max(np.maximum(0.0, 1.0 - fidelity)))
        max_probability_diff = probability_difference(cpu_output, torch_output)
        cpu_timing = paired.timings["cpu"]
        gpu_timing = paired.timings["torch_gpu"]
        print(
            f"max probability difference={max_probability_diff:.3e}; "
            f"max state infidelity={max_infidelity:.3e}; "
            f"PyTorch peak allocated={peak_memory_bytes / 2**30:.2f} GiB",
            flush=True,
        )

        for name, timing in paired.timings.items():
            results.append(
                result_row(
                    name,
                    time_best_s=timing.best_s,
                    time_mean_s=timing.mean_s,
                    time_std_s=timing.std_s,
                    repeat=args.repeat,
                    warmup=args.warmup,
                    batch=batch,
                    scan_mode=args.scan_mode,
                    n_detuning=n_detuning,
                    n_power=n_power,
                    initial_states=args.initial_states,
                    n=len(setup["hamiltonian"].QN),
                    n_steps=args.n_steps,
                    time_chunk_size=args.time_chunk_size if name == "torch_gpu" else None,
                    workers=args.cpu_workers if name == "cpu" else None,
                    points_per_second=batch / timing.mean_s,
                    max_probability_difference=max_probability_diff,
                    max_state_infidelity=max_infidelity,
                    torch_peak_memory_bytes=peak_memory_bytes if name == "torch_gpu" else None,
                )
            )
        results[-1]["speedup_vs_cpu_mean"] = cpu_timing.mean_s / gpu_timing.mean_s

    payload = make_payload(
        benchmark="gpu_torch_batch_sweep",
        config={
            "n_steps": args.n_steps,
            "batches": args.batches,
            "scan_mode": args.scan_mode,
            "initial_states": args.initial_states,
            "cpu_workers": args.cpu_workers,
            "time_chunk_size": args.time_chunk_size,
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
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_torch_batch_sweep"),
    )


if __name__ == "__main__":
    main()
