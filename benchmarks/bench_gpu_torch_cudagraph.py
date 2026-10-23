from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from bench_gpu_spa2 import make_scan_parameters
from bench_gpu_torch_full_spa2 import make_torch_runner
from common import add_nvidia_dll_paths, build_spa2_setup
from paired import compare_paired, format_report


def capture_runner(run):
    """Capture one fixed-shape full trajectory and return graph replay/output."""
    for _ in range(3):
        run()
        torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        graph.capture_begin()
        try:
            output = run()
        except torch.AcceleratorError as exc:
            message = str(exc)
            try:
                graph.capture_end()
            except torch.AcceleratorError:
                pass
            if "stream is capturing" not in message and "capture" not in message.lower():
                raise
            return None, time.perf_counter() - started, message
        graph.capture_end()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    capture_time_s = time.perf_counter() - started

    def replay():
        graph.replay()
        return output

    return replay, capture_time_s, None


def probe_basic_graph_capture() -> bool:
    left = torch.randn((64, 64), device="cuda")
    right = torch.randn((64, 64), device="cuda")
    for _ in range(2):
        left @ right
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = left @ right
    graph.replay()
    torch.cuda.synchronize()
    return bool(torch.isfinite(output).all().item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA Graph replay for the full PyTorch SPA2 trajectory."
    )
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--initial-states", type=int, default=4)
    parser.add_argument("--time-chunk-size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()

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
    eager_run = make_torch_runner(
        setup,
        n_steps=args.n_steps,
        detunings=detunings,
        intensity_prefactors=powers,
        time_chunk_size=args.time_chunk_size,
    )
    print("[capture] warming PyTorch and capturing the fixed trajectory", flush=True)
    basic_graph_capture_supported = probe_basic_graph_capture()
    graph_run, capture_time_s, capture_error = capture_runner(eager_run)
    if graph_run is None:
        report = {
            "benchmark": "gpu_torch_cudagraph",
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "n_steps": args.n_steps,
            "batch": args.batch,
            "initial_states": args.initial_states,
            "time_chunk_size": args.time_chunk_size,
            "capture_supported": False,
            "basic_graph_capture_supported": basic_graph_capture_supported,
            "capture_error": capture_error,
            "speedup": None,
        }
        print(json.dumps(report, indent=2), flush=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
        return
    print(f"[capture] completed in {capture_time_s:.3f}s", flush=True)

    def max_probability_difference(reference, candidate):
        return float(
            torch.max(
                torch.abs(
                    reference.probabilities_final - candidate.probabilities_final
                )
            ).item()
        )

    paired = compare_paired(
        {"eager": eager_run, "cuda_graph": graph_run},
        repeat=args.repeat,
        warmup=args.warmup,
        baseline="eager",
        sync=torch.cuda.synchronize,
        check=max_probability_difference,
        tolerance=1e-5,
        progress=True,
    )
    print(format_report(paired), flush=True)
    eager = paired.results["eager"]
    graphed = paired.results["cuda_graph"]
    probability_difference = max_probability_difference(eager, graphed)
    overlap = torch.sum(torch.conj(eager.psi_final) * graphed.psi_final, dim=-1)
    norm_eager = torch.sum(torch.abs(eager.psi_final) ** 2, dim=-1)
    norm_graph = torch.sum(torch.abs(graphed.psi_final) ** 2, dim=-1)
    fidelity = torch.abs(overlap) ** 2 / (norm_eager * norm_graph)
    max_infidelity = float(torch.max(torch.clamp(1.0 - fidelity, min=0.0)).item())

    report = {
        "benchmark": "gpu_torch_cudagraph",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "n_steps": args.n_steps,
        "batch": args.batch,
        "initial_states": args.initial_states,
        "time_chunk_size": args.time_chunk_size,
        "capture_supported": True,
        "basic_graph_capture_supported": basic_graph_capture_supported,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "capture_time_s": capture_time_s,
        "eager_mean_s": paired.timings["eager"].mean_s,
        "eager_std_s": paired.timings["eager"].std_s,
        "cuda_graph_mean_s": paired.timings["cuda_graph"].mean_s,
        "cuda_graph_std_s": paired.timings["cuda_graph"].std_s,
        "speedup": paired.timings["eager"].mean_s
        / paired.timings["cuda_graph"].mean_s,
        "max_probability_difference": probability_difference,
        "max_state_infidelity": max_infidelity,
        "scan_detuning_points": n_detuning,
        "scan_power_points": n_power,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
