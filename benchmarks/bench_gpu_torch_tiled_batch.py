from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters
from bench_gpu_torch_full_spa2 import make_torch_runner
from common import add_nvidia_dll_paths, build_spa2_setup
from paired import compare_paired, format_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark dense PyTorch time batching with bounded scan-point tiles."
        )
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--all-states", action="store_true")
    parser.add_argument("--time-chunks", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--scan-tiles", type=int, nargs="+", default=[250, 500])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    if len(args.time_chunks) != len(args.scan_tiles):
        parser.error("--time-chunks and --scan-tiles must have the same number of values")

    add_nvidia_dll_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")

    setup = build_spa2_setup()
    if not 1 <= args.initial_states <= len(setup["initial_states"]):
        parser.error(
            f"--initial-states must be between 1 and {len(setup['initial_states'])}"
        )
    setup["initial_states"] = setup["initial_states"][: args.initial_states]
    detunings, powers, n_detuning, n_power = make_scan_parameters(
        args.batch, "detuning"
    )

    print("[setup] building dense PyTorch reference and tiled candidates", flush=True)
    variants = {
        "time_chunk_1": make_torch_runner(
            setup,
            n_steps=args.n_steps,
            detunings=detunings,
            intensity_prefactors=powers,
            time_chunk_size=1,
            all_states=args.all_states,
        )
    }
    for time_chunk, scan_tile in zip(args.time_chunks, args.scan_tiles):
        name = f"time_chunk_{time_chunk}_scan_tile_{scan_tile}"
        variants[name] = make_torch_runner(
            setup,
            n_steps=args.n_steps,
            detunings=detunings,
            intensity_prefactors=powers,
            time_chunk_size=time_chunk,
            scan_chunk_size=scan_tile,
            all_states=args.all_states,
        )

    def max_probability_difference(reference, candidate):
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
        baseline="time_chunk_1",
        sync=torch.cuda.synchronize,
        check=max_probability_difference,
        tolerance=1e-5,
        progress=True,
    )
    print(format_report(paired), flush=True)

    baseline = paired.timings["time_chunk_1"]
    results = []
    max_difference_by_name = {
        comparison.candidate: comparison.max_abs_difference
        for comparison in paired.comparisons
    }
    for name, timing in paired.timings.items():
        results.append(
            {
                "variant": name,
                "mean_s": timing.mean_s,
                "std_s": timing.std_s,
                "speedup_vs_chunk_1": baseline.mean_s / timing.mean_s,
                "max_probability_difference": max_difference_by_name.get(name, 0.0),
            }
        )

    report = {
        "benchmark": "gpu_torch_tiled_batch",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "n_steps": args.n_steps,
        "batch": args.batch,
        "initial_states": args.initial_states,
        "all_states": args.all_states,
        "time_chunks": args.time_chunks,
        "scan_tiles": args.scan_tiles,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "scan_detuning_points": n_detuning,
        "scan_power_points": n_power,
        "results": results,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
