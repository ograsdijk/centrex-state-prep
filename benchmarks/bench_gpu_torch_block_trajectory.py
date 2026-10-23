from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters
from bench_gpu_torch_full_spa2 import make_torch_runner
from common import add_nvidia_dll_paths, build_spa2_setup
from state_prep.simulator import _find_exact_hamiltonian_blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dense and CPU-detector-compatible blockwise PyTorch CUDA "
            "eigensolves over a real SPA2 trajectory."
        )
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    add_nvidia_dll_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    setup = build_spa2_setup()
    setup["initial_states"] = setup["initial_states"][:4]
    trajectory = setup["trajectory"]
    hamiltonian = setup["hamiltonian"]
    microwave_fields = setup["microwave_fields"]
    t_array = np.linspace(0.0, float(trajectory.get_T()), args.n_steps)

    # Use the same exact support-union detector as the CPU simulator, over the
    # initial time and every midpoint used by this propagation.
    slow_hamiltonian = hamiltonian.get_H_t_func()
    microwave_hamiltonians = [
        mw.get_H_t_func(trajectory.R_t, hamiltonian.QN)
        for mw in microwave_fields
    ]
    detect_started = time.perf_counter()
    eigen_blocks = _find_exact_hamiltonian_blocks(
        [slow_hamiltonian, *microwave_hamiltonians], t_array, "mid"
    )
    detection_time_s = time.perf_counter() - detect_started
    if eigen_blocks is None:
        raise RuntimeError("The exact CPU-compatible detector found no useful blocks")

    detunings, powers, _, _ = make_scan_parameters(args.batch, "detuning")
    dense_runner = make_torch_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
    )
    block_runner = make_torch_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
        eigen_blocks=eigen_blocks,
    )

    timings = {"dense_torch": [], "blockwise_torch": []}
    if args.warmup:
        dense_runner()
        torch.cuda.synchronize()
        block_runner()
        torch.cuda.synchronize()

    outputs = {}
    for round_idx in range(args.repeat):
        variants = [
            ("dense_torch", dense_runner),
            ("blockwise_torch", block_runner),
        ]
        if round_idx % 2:
            variants.reverse()
        for name, run in variants:
            started = time.perf_counter()
            outputs[name] = run()
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - started)
        print(
            f"round {round_idx + 1}/{args.repeat}: "
            + "  ".join(f"{name}={timings[name][-1]:.3f}s" for name in timings),
            flush=True,
        )

    dense = outputs["dense_torch"]
    blocked = outputs["blockwise_torch"]
    max_prob_diff = torch.max(
        torch.abs(dense.probabilities_final - blocked.probabilities_final)
    ).item()
    max_state_diff = torch.max(torch.abs(dense.psi_final - blocked.psi_final)).item()
    block_sizes = [int(len(block)) for block in eigen_blocks]
    report = {
        "benchmark": "gpu_torch_block_trajectory_cpu_detector",
        "n_steps": args.n_steps,
        "batch": args.batch,
        "initial_states": 4,
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "block_sizes": block_sizes,
        "block_detection_time_s": detection_time_s,
        "timings_s": timings,
        "mean_speedup": float(
            np.mean(timings["dense_torch"]) / np.mean(timings["blockwise_torch"])
        ),
        "max_final_probability_difference": max_prob_diff,
        "max_final_state_vector_difference": max_state_diff,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
