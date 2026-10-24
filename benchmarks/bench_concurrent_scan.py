"""End-to-end paired throughput, including process startup and preparation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import dill
import numpy as np

from common import REPO_ROOT, build_spa2_setup, make_payload, write_results
from paired import compare_paired, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", default=["cpu:6", "cpu:8", "hybrid:6:0.6"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    setup_path = output / "setup.dill"
    if not setup_path.exists():
        with setup_path.open("wb") as stream:
            dill.dump(build_spa2_setup(), stream)
    variants = {}
    for label in args.variants:
        parts = label.split(":")
        mode, workers = parts[0], int(parts[1])
        fraction = 1.0 if mode == "cpu" else (0.0 if mode == "gpu" else float(parts[2]))
        if mode not in ("cpu", "gpu", "hybrid"):
            parser.error(f"unknown variant {label}")
        def make_runner(label, workers, fraction):
            counter = 0
            def run():
                nonlocal counter
                run_dir = output / label.replace(":", "_") / str(counter)
                counter += 1
                # Refuse cached runs: they are not a throughput measurement.
                if run_dir.exists():
                    raise ValueError(f"Benchmark run directory already exists: {run_dir}")
                command = [sys.executable, str(REPO_ROOT / "scripts" / "run_cpu_gpu_scan.py"),
                           "--setup-file", str(setup_path), "--steps", str(args.steps),
                           "--batch", str(args.batch), "--workers", str(workers),
                           "--cpu-fraction", str(fraction), "--output-dir", str(run_dir)]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                with np.load(run_dir / "merged.npz", allow_pickle=False) as saved:
                    return saved["probabilities_final"]
            return run
        variants[label] = make_runner(label, workers, fraction)
    pair = compare_paired(variants, baseline=args.variants[0], repeat=args.repeat,
                          warmup=args.warmup, progress=True,
                          check=lambda a, b: float(np.max(np.abs(a-b))), tolerance=1e-5)
    print(format_report(pair), flush=True)
    if not all(comparison.agrees for comparison in pair.comparisons):
        raise RuntimeError("Concurrent scan did not agree with CPU reference")
    payload = make_payload(benchmark="concurrent_scan", config={**vars(args), "output_dir": str(output)}, results=pair.rows())
    payload["rounds_s"] = {name: timing.times_s for name, timing in pair.timings.items()}
    write_results(payload, output=str(output / "comparison.json"), csv_path=None)


if __name__ == "__main__":
    main()
