"""Investigate four remaining optimizations against production velocity scans."""
from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import time

import numpy as np

from bench_dense_investigation import patched
from bench_gpu_spa2 import make_torch_runner
from bench_velocity_scan import make_simulator
from common import make_payload, write_results
from dense_solver_prototypes import ReusableCusolver
from paired import compare_paired, format_report
from profile_transverse_fields import field_variant
from velocity_prototypes import cpu_variant, gpu_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'grouped', 'partition', 'gpu', 'combined'], default='all')
    parser.add_argument('--shapes', nargs='+', default=['150:5:8', '300:25:10'],
                        help='steps:microwave-points:velocities')
    parser.add_argument('--workers', nargs='+', type=int, default=[2, 4])
    parser.add_argument('--field-case', choices=['aligned', 'varying_e', 'transverse_b'], default='varying_e')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output file')
    setup = field_variant(args.field_case)
    _, basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
    rows = []

    def save():
        write_results(make_payload(benchmark='velocity_followup', config=vars(args), results=rows),
                      output=str(args.output), csv_path=None)

    for shape in args.shapes:
        steps, points, velocities = map(int, shape.split(':'))
        det = np.repeat(np.linspace(-3e5, 3e5, points)[:, None], 2, axis=1)
        det = np.tile(det, (velocities, 1))  # Deliberately velocity-major input order.
        speeds = np.repeat(np.linspace(140., 240., velocities), points)
        opts = dict(detunings_hz=det, intensity_prefactors=np.ones_like(det), forward_velocities=speeds,
                    N_steps=steps, block_eigh=False, progress=False)
        context = dict(steps=steps, parameter_points=points, velocities=velocities,
                       batch=len(det), field_case=args.field_case)

        def record(phase, variants, *, sync=None, as_numpy=np.asarray, extra=None):
            print(f'{phase}: {shape}', flush=True)
            def check(a, b):
                av, bv = as_numpy(a), as_numpy(b)
                error = float(np.max(np.abs(np.abs(av @ basis.conj())**2 - np.abs(bv @ basis.conj())**2)))
                infidelity = float(np.max(np.abs(1 - np.abs(np.sum(av.conj()*bv, axis=-1))**2)))
                if infidelity > 1e-5:
                    raise RuntimeError(f'Physical-state infidelity {infidelity}')
                return error
            pair = compare_paired(variants, warmup=args.warmup, repeat=args.repeat, sync=sync,
                                  check=check, tolerance=1e-5, progress=True)
            print(format_report(pair), flush=True)
            if not all(comparison.agrees for comparison in pair.comparisons):
                raise RuntimeError('Numerical disagreement')
            tags = dict(context, phase=phase, **(extra or {}))
            rows.extend(dict(row, **tags) for row in pair.rows())
            rows.append(dict(case='rounds', **tags,
                             times_s={name: timing.times_s for name, timing in pair.timings.items()}))
            reference = as_numpy(next(iter(pair.results.values())))
            for name, result in pair.results.items():
                values = as_numpy(result)
                rows.append(dict(case='validation', variant=name, **tags,
                    max_infidelity=float(np.max(np.abs(1-np.abs(np.sum(reference.conj()*values,axis=-1))**2))),
                    max_norm_error=float(np.max(np.abs(np.sum(np.abs(values)**2,axis=-1)-1)))))
            save()
            return pair

        if args.phase in ('all', 'grouped', 'partition'):
            from state_prep import Simulator
            import state_prep.simulator as cpu_module
            sim = make_simulator(setup, float(setup['trajectory'].Vini[2]))
            def current():
                return sim.run_microwave_scan(**opts).psis_final
            grouped = cpu_variant(grouped=True)
            def grouped_run():
                with patched(Simulator, '_time_evolve_mu_batched_shared_slow',
                             grouped._time_evolve_mu_batched_shared_slow):
                    return current()
            if args.phase in ('all', 'grouped'):
                pair = record('cpu_grouped', dict(current=current, grouped=grouped_run))
                native_s = []
                original = cpu_module.zheevd
                def timed(matrix, *a, **kw):
                    begin = time.perf_counter()
                    result = original(matrix, *a, **kw)
                    native_s.append(time.perf_counter()-begin)
                    return result
                profiler = cProfile.Profile()
                with patched(cpu_module, 'zheevd', timed):
                    begin = time.perf_counter()
                    profiler.runcall(current)
                    wall = time.perf_counter()-begin
                if len(native_s) != (steps-1)*(points+1):
                    raise RuntimeError(f'Unexpected solve count {len(native_s)}')
                times = np.asarray(native_s).reshape(steps-1, points+1)
                profile_path = args.output.with_name(args.output.stem+f'_cpu_{shape.replace(":", "_")}.prof')
                profiler.dump_stats(profile_path)
                rows.append(dict(case='cpu_profile', **context, wall_s=wall,
                                 slow_eigh_s=float(times[:, 0].sum()), rotating_eigh_s=float(times[:, 1:].sum()),
                                 native_eigh_fraction=float(times.sum()/wall), calls=len(native_s)))
                save()
            if args.phase in ('all', 'partition'):
                partitioned = cpu_variant(partitioned=True)
                serial = current()
                for workers in args.workers:
                    def contiguous():
                        return sim.run_microwave_scan(**opts, workers=workers).psis_final
                    def together():
                        with patched(Simulator, '_run_microwave_scan_parallel_loky',
                                     partitioned._run_microwave_scan_parallel_loky):
                            return contiguous()
                    pair = record('cpu_partition', dict(contiguous=contiguous, grouped_chunks=together),
                                  extra=dict(workers=workers))
                    for name, result in pair.results.items():
                        error = float(np.max(np.abs(np.abs(serial@basis.conj())**2 - np.abs(result@basis.conj())**2)))
                        if error > 1e-5:
                            raise RuntimeError('Parallel output ordering or state changed')
                        rows.append(dict(case='serial_agreement', variant=name, workers=workers,
                                         max_probability_error=error, **context))
                    save()

        if args.phase in ('all', 'grouped', 'gpu', 'combined'):
            import torch
            import state_prep_gpu.simulator as gpu_module
            torch.set_num_threads(1)
            def runner(function=None):
                if function is None:
                    return make_torch_runner(setup, n_steps=steps, detunings=det,
                        intensity_prefactors=np.ones_like(det), forward_velocities=speeds,
                        block_eigh=False, slow_eigh_chunk_size=128)
                with patched(gpu_module, 'simulate_batched_scalar_microwaves_shared_slow', function):
                    return runner()
            baseline, sync = runner()
            grouped_run, _ = runner(gpu_variant(grouped=True))
            variants = dict(current=lambda: baseline().psi_final, grouped=lambda: grouped_run().psi_final)
            native = []
            if args.phase in ('all', 'gpu', 'combined'):
                for chunk in (4, 8):
                    run, _ = runner(gpu_variant(full_chunk=chunk))
                    variants[f'full_time_{chunk}'] = lambda run=run: run().psi_final
                for deferred in (False, True):
                    solver = ReusableCusolver(deferred_info=deferred)
                    native.append(solver)
                    def native_run(solver=solver):
                        with patched(torch.linalg, 'eigh', solver):
                            result = baseline()
                        solver.flush()  # Every solver status is checked inside timing.
                        return result.psi_final
                    variants['native_deferred' if deferred else 'native_checked'] = native_run
            if args.phase == 'combined':
                full, _ = runner(gpu_variant(full_chunk=8))
                combined, _ = runner(gpu_variant(grouped=True, full_chunk=8))
                # Each closure retains its own checked solver object. Deferred
                # status checking remains inside the timed call.
                deferred_solver = native[-1]
                def tiled_native():
                    with patched(torch.linalg, 'eigh', deferred_solver):
                        result = full()
                    deferred_solver.flush()
                    return result.psi_final
                variants = dict(full_time_8=lambda: full().psi_final,
                                grouped_full_time_8=lambda: combined().psi_final,
                                native_full_time_8=tiled_native)
            try:
                pair = record('gpu', variants, sync=sync, as_numpy=lambda a: a.cpu().numpy())
                stage = {}
                baseline(profile_timings=stage)
                rows.append(dict(case='gpu_profile', **context,
                    stage_mean_s={name:float(np.mean(times)) for name,times in stage.items()}))
                for name, run in variants.items():
                    torch.cuda.reset_peak_memory_stats()
                    output = run()
                    sync()
                    rows.append(dict(case='gpu_memory', variant=name, **context,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    del output
                save()
            finally:
                for solver in native:
                    solver.close()


if __name__ == '__main__':
    main()
