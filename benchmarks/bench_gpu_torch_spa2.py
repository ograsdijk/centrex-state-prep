from __future__ import annotations

import argparse

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters, make_torch_runner, run_cpu
from common import build_spa2_setup, make_payload, resolve_csv_path, result_row, write_results
from paired import compare_paired, format_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired CPU versus full-PyTorch SPA2 GPU benchmark."
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument(
        "--scan-mode", choices=("detuning", "power", "combined"), default="detuning"
    )
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is required")

    print("[setup] building real SPA2 setup", flush=True)
    setup = build_spa2_setup()
    all_initial_states = setup["initial_states"]
    if not 1 <= args.initial_states <= len(all_initial_states):
        parser.error(f"--initial-states must be between 1 and {len(all_initial_states)}")
    setup["initial_states"] = all_initial_states[: args.initial_states]
    detunings, powers, n_detuning, n_power = make_scan_parameters(
        args.batch, args.scan_mode
    )

    cpu_run = lambda: run_cpu(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
    )
    torch_run, sync = make_torch_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
    )

    def probability_discrepancy(reference, candidate) -> float:
        candidate_probs = candidate.probabilities_final.detach().cpu().numpy()
        return float(np.max(np.abs(reference.probabilities_final - candidate_probs)))

    paired = compare_paired(
        {"cpu": cpu_run, "pytorch": torch_run},
        repeat=args.repeat,
        warmup=args.warmup,
        baseline="cpu",
        sync=sync,
        check=probability_discrepancy,
        tolerance=1e-5,
        progress=True,
    )
    print(format_report(paired), flush=True)

    cpu_output = paired.results["cpu"]
    torch_output = paired.results["pytorch"]
    psi_cpu = np.asarray(cpu_output.psis_final)
    psi_torch = torch_output.psi_final.detach().cpu().numpy()
    overlap = np.sum(np.conj(psi_cpu) * psi_torch, axis=-1)
    norm_product = np.sum(np.abs(psi_cpu) ** 2, axis=-1) * np.sum(
        np.abs(psi_torch) ** 2, axis=-1
    )
    fidelity = np.abs(overlap) ** 2 / norm_product
    max_infidelity = float(np.max(np.maximum(0.0, 1.0 - fidelity)))
    max_probability_difference = probability_discrepancy(cpu_output, torch_output)
    print(
        f"max final probability difference={max_probability_difference:.3e}; "
        f"max state infidelity={max_infidelity:.3e}",
        flush=True,
    )

    rows = []
    for name, timing in paired.timings.items():
        rows.append(
            result_row(
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
                n=len(setup["hamiltonian"].QN),
                n_steps=args.n_steps,
                max_probability_difference=max_probability_difference,
                max_state_infidelity=max_infidelity,
            )
        )
    payload = make_payload(
        benchmark="gpu_torch_spa2",
        config={
            "n_steps": args.n_steps,
            "batch": args.batch,
            "scan_mode": args.scan_mode,
            "initial_states": args.initial_states,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        results=rows,
        include_gpu_env=True,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "gpu_torch_spa2"),
    )


if __name__ == "__main__":
    main()
