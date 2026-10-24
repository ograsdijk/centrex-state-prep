"""Pilot two independent SPA2 jobs: sequential versus simultaneous CPU/GPU."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import dill
import numpy as np

from common import REPO_ROOT, align_lineshape, build_spa2_setup, make_payload, write_results
from paired import compare_paired, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=250)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--convergence-steps", nargs="+", type=int, default=[1000, 2000, 4000, 8000])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory must be new; cached runs cannot measure throughput")
    if args.batch < 2 or args.workers < 1 or args.steps < 2:
        parser.error("batch>=2, workers>=1, steps>=2 required")
    output.mkdir(parents=True)
    setup = build_spa2_setup()
    setup_path = output / "setup.dill"
    with setup_path.open("wb") as stream:
        dill.dump(setup, stream)
    detuning = np.linspace(-2e6, 1e6, args.batch)
    fields = len(setup["microwave_fields"])
    for name, power in (("cpu_job", 0.5), ("gpu_job", 1.0)):
        np.savez(output / f"{name}.npz",
                 detunings_hz=np.repeat(detuning[:, None], fields, axis=1),
                 intensity_prefactors=np.full((args.batch, fields), power))
    # All runs are projected into this fixed physical readout basis. Tracked
    # labels can differ across timesteps/backends and are not a comparison key.
    _, basis = np.linalg.eigh(setup["hamiltonian"].get_H_t_func()(setup["trajectory"].get_T()))
    j2 = np.array([state.J == 2 for state in setup["hamiltonian"].QN])
    np.save(output / "readout_basis.npy", basis)
    runs = []

    def run_job(name, backend, steps, directory):
        command = [sys.executable, str(REPO_ROOT / "scripts" / "run_cpu_gpu_scan.py"),
                   "--setup-file", str(setup_path), "--parameters", str(output / f"{name}.npz"),
                   "--steps", str(steps), "--workers", str(args.workers),
                   "--cpu-fraction", "1" if backend == "cpu" else "0",
                   "--output-dir", str(directory)]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "launcher.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name}/{backend} failed: {directory / 'launcher.log'}")
        with np.load(directory / "merged.npz", allow_pickle=False) as saved:
            psi = saved["psi_final"]
            initial = saved["initial_vectors"]
        norms = np.sum(np.abs(psi)**2, axis=-1)
        if not np.isfinite(psi).all() or np.max(np.abs(norms - 1)) > 1e-7:
            raise RuntimeError(f"Invalid final states in {directory}")
        with np.load(directory / f"{backend}.npz", allow_pickle=False) as saved:
            preparation = float(saved["preparation_s"])
            propagation = float(saved["propagation_s"])
        record = dict(job=name, backend=backend, steps=steps, wall_s=elapsed,
                      preparation_s=preparation, propagation_s=propagation,
                      max_norm_error=float(np.max(np.abs(norms - 1))), directory=str(directory))
        runs.append(record)
        print(f"{directory.relative_to(output)}: {elapsed:.2f}s", flush=True)
        return dict(psi=psi, initial=initial, probabilities=np.abs(psi @ basis.conj())**2,
                    j2=np.sum(np.abs(psi[..., j2])**2, axis=-1))

    convergence = {}
    for steps in sorted(set(args.convergence_steps)):
        convergence[steps] = {
            name: run_job(name, "gpu", steps, output / "convergence" / str(steps) / name)
            for name in ("cpu_job", "gpu_job")
        }
    convergence_rows = []
    levels = sorted(convergence)
    for coarse, fine in zip(levels, levels[1:]):
        for name in ("cpu_job", "gpu_job"):
            a, b = convergence[coarse][name], convergence[fine][name]
            alignment = [align_lineshape(a["j2"][:, i], b["j2"][:, i], detuning / 1e3,
                                         max_shift_khz=100, samples=2001)
                         for i in range(a["j2"].shape[1])]
            convergence_rows.append(dict(job=name, coarse_steps=coarse, fine_steps=fine,
                max_probability_difference=float(np.max(np.abs(a["probabilities"]-b["probabilities"]))),
                max_j2_difference=float(np.max(np.abs(a["j2"]-b["j2"]))),
                j2_alignment_shift_khz=[item[0] for item in alignment],
                j2_alignment_rms=[item[1] for item in alignment]))
    write_results(make_payload(benchmark="independent_job_convergence", config=vars(args),
                              results=convergence_rows),
                  output=str(output / "convergence.json"), csv_path=None)
    print(json.dumps(convergence_rows, indent=2), flush=True)

    # Independent cross-backend reference at the throughput timestep count.
    cpu_reference = run_job("gpu_job", "cpu", args.steps, output / "cpu_reference")
    counters = dict(sequential=0, concurrent=0)
    reference = {}

    def discrepancy(a, b):
        if not np.allclose(np.abs(np.sum(a["initial"].conj()*b["initial"], axis=-1)), 1, atol=1e-8, rtol=0):
            raise RuntimeError("Physical initial states differ")
        return float(np.max(np.abs(a["probabilities"]-b["probabilities"])))

    def run_pair(mode):
        count = counters[mode]
        counters[mode] += 1
        directory = output / mode / str(count)
        jobs = [("cpu_job", "cpu"), ("gpu_job", "gpu")]
        if mode == "concurrent":
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_job, name, backend, args.steps, directory / name)
                           for name, backend in jobs]
                results = [future.result() for future in futures]
        else:
            results = [run_job(name, backend, args.steps, directory / name) for name, backend in jobs]
        for (name, _), result in zip(jobs, results):
            if name not in reference:
                reference[name] = result
            error = discrepancy(reference[name], result)
            if name == "gpu_job":
                error = max(error, discrepancy(cpu_reference, result))
            if error > 1e-6:
                raise RuntimeError(f"{name} differs from reference: {error}")
        return np.concatenate([r["probabilities"] for r in results])

    paired = compare_paired({mode: lambda mode=mode: run_pair(mode)
                             for mode in ("sequential", "concurrent")},
                            warmup=args.warmup, repeat=args.repeat, progress=True,
                            check=lambda a, b: float(np.max(np.abs(a-b))), tolerance=1e-6)
    print(format_report(paired), flush=True)
    payload = make_payload(benchmark="independent_cpu_gpu_jobs", config=vars(args), results=paired.rows())
    payload["rounds_s"] = {name: timing.times_s for name, timing in paired.timings.items()}
    payload["job_runs"] = runs
    payload["cross_backend_max_probability_difference"] = discrepancy(cpu_reference, reference["gpu_job"])
    write_results(payload, output=str(output / "comparison.json"), csv_path=None)


if __name__ == "__main__":
    main()
