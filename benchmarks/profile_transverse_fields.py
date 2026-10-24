"""Profile aligned and synthetic non-collinear SPA2 fields without changing physics defaults."""
from __future__ import annotations

import argparse
import cProfile
import io
from pathlib import Path
import pstats
import time

import numpy as np

from bench_gpu_spa2 import make_torch_runner, run_cpu
from common import REPO_ROOT, build_spa2_setup, make_payload, time_call, write_results


def field_variant(case):
    setup = build_spa2_setup()
    if case == "transverse_b":
        original_b = setup["magnetic_field"].B_R
        # Relative to the benchmark's Bz, in the same existing field units.
        def b_r(position):
            b = original_b(position).copy()
            b[0] = 0.3*b[2]
            b[1] = 0.2*b[2]
            return b
        setup["magnetic_field"].B_R = b_r
    elif case == "varying_e":
        original_e = setup["electric_field"].E_R
        def e_r(position):
            e = original_e(position).copy()
            phase = 2*np.pi*(position[2]+0.08)/0.28
            e[0] = 0.02*e[2]*np.sin(phase)
            e[1] = 0.01*e[2]*np.cos(phase)
            return e
        setup["electric_field"].E_R = e_r
    elif case != "aligned":
        raise ValueError(case)
    # H_R closes over these field objects, so the slow Hamiltonian sees the
    # updated callables. Microwaves/carriers remain the benchmark reference.
    return setup


def profile_dense_cpu_native(run, batch, n):
    """Time LAPACK calls hidden inside the Python loop's cProfile entry.

    For the dense shared-slow path each step calls zheevd once for H_slow,
    followed by one call per scan point. Refuse a different call pattern.
    """
    import state_prep.simulator as module
    original = module.zheevd
    calls = 0
    totals = dict(slow_eigh_s=0., rotating_eigh_s=0.)
    def measured(matrix, *a, **kw):
        nonlocal calls
        if matrix.shape != (n, n):
            raise ValueError("Native dense profile encountered a block solve")
        section = "slow_eigh_s" if calls % (batch+1) == 0 else "rotating_eigh_s"
        started = time.perf_counter()
        result = original(matrix, *a, **kw)
        totals[section] += time.perf_counter()-started
        calls += 1
        return result
    module.zheevd = measured
    started = time.perf_counter()
    try:
        run()
    finally:
        module.zheevd = original
    elapsed = time.perf_counter()-started
    if calls % (batch+1):
        raise RuntimeError("Unexpected shared-slow LAPACK call pattern")
    return dict(wall_s=elapsed, calls=calls, steps=calls//(batch+1), **totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--cpu-batch", type=int, default=42)
    parser.add_argument("--gpu-batch", type=int, default=250)
    parser.add_argument("--cases", nargs="+", default=["aligned", "transverse_b", "varying_e"],
                        choices=["aligned", "transverse_b", "varying_e"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cpu_batch <= args.gpu_batch or args.steps < 2:
        parser.error("1 <= cpu-batch <= gpu-batch and steps >= 2 required")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("choose a new output directory")
    output.mkdir(parents=True)
    import torch
    import state_prep.simulator as cpu_module
    import state_prep_gpu._blocks as gpu_module
    torch.set_num_threads(1)
    gpu_det = np.repeat(np.linspace(-2e6, 1e6, args.gpu_batch)[:, None], 2, axis=1)
    subset = np.linspace(0, args.gpu_batch-1, args.cpu_batch, dtype=int)
    rows = []
    for case in args.cases:
        print(f"{case}: building setup", flush=True)
        setup = field_variant(case)
        cpu_blocks, gpu_blocks = [], []
        original_cpu, original_gpu = cpu_module._find_exact_hamiltonian_blocks, gpu_module.shared_slow_blocks
        def detect_cpu(*a, **kw):
            blocks = original_cpu(*a, **kw)
            cpu_blocks.append(None if blocks is None else [len(b) for b in blocks])
            return blocks
        def detect_gpu(*a, **kw):
            blocks = original_gpu(*a, **kw)
            gpu_blocks.append(None if blocks is None else [len(b) for b in blocks])
            return blocks
        cpu_module._find_exact_hamiltonian_blocks = detect_cpu
        gpu_module.shared_slow_blocks = detect_gpu
        try:
            def cpu():
                return run_cpu(setup, n_steps=args.steps, detunings=gpu_det[subset],
                               intensity_prefactors=np.ones_like(gpu_det[subset]), workers=1)
            timing, cpu_result = time_call(cpu, warmup=args.warmup, repeat=args.repeat)
            print(f"CPU {timing}, blocks={cpu_blocks[-1]}", flush=True)
            profiler = cProfile.Profile()
            profiler.enable(); cpu(); profiler.disable()
            profiler.dump_stats(str(output / f"{case}_cpu.prof"))
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
            (output / f"{case}_cpu.txt").write_text(stream.getvalue())
            native = (profile_dense_cpu_native(cpu, args.cpu_batch, len(setup["hamiltonian"].QN))
                      if cpu_blocks[-1] is None else None)
            rows.append(dict(case=case, device="cpu", batch=args.cpu_batch,
                             blocks=cpu_blocks[-1], native_profile=native, timing_std_ddof=0, **timing))

            start = time.perf_counter()
            run, sync = make_torch_runner(setup, n_steps=args.steps, detunings=gpu_det,
                                         intensity_prefactors=np.ones_like(gpu_det))
            sync()
            preparation = time.perf_counter()-start
            timing, gpu_result = time_call(run, warmup=args.warmup, repeat=args.repeat, sync=sync)
            print(f"GPU {timing}, blocks={gpu_blocks[-1]}", flush=True)
            stages = {}
            start = time.perf_counter()
            run(profile_timings=stages); sync()
            profile_wall = time.perf_counter()-start
            # Backend labels need not agree. Compare physical states, and
            # populations in one common basis at the physical readout time.
            psi_cpu = cpu_result.psis_final
            psi_gpu = gpu_result.psi_final.cpu().numpy()[subset]
            _, basis = np.linalg.eigh(setup["hamiltonian"].get_H_t_func()(setup["trajectory"].get_T()))
            error = float(np.max(np.abs(np.abs(psi_cpu @ basis.conj())**2 - np.abs(psi_gpu @ basis.conj())**2)))
            infidelity = float(np.max(np.abs(1-np.abs(np.sum(psi_cpu.conj()*psi_gpu, axis=-1))**2)))
            stage_report = {name: dict(mean_ms=float(np.mean(samples)*1000),
                                       estimated_total_s=float(np.mean(samples)*(args.steps-1)),
                                       samples=len(samples)) for name, samples in stages.items()}
            row = dict(case=case, device="gpu", batch=args.gpu_batch,
                       blocks=gpu_blocks[-1], preparation_s=preparation,
                       timing_std_ddof=0, profile_wall_s=profile_wall, stages=stage_report,
                       common_basis_max_probability_difference=error, max_infidelity=infidelity, **timing)
            rows.append(row)
            print(row, flush=True)
            if error > 1e-5 or infidelity > 1e-5:
                raise RuntimeError(f"CPU/GPU physical states disagree in {case}: {error}, {infidelity}")
        finally:
            cpu_module._find_exact_hamiltonian_blocks = original_cpu
            gpu_module.shared_slow_blocks = original_gpu
        payload = make_payload(benchmark="transverse_field_profile", config=vars(args), results=rows)
        write_results(payload, output=str(output / "profile.json"), csv_path=None)


if __name__ == "__main__":
    main()
