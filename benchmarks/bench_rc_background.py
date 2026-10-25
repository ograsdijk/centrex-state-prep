"""RC J12 background: profile CPU multitone and compare beat-correct candidates."""
from __future__ import annotations

import argparse
import cProfile
from collections import Counter
from pathlib import Path
import pstats
import time

import numpy as np

from bench_convergence import build_multitone_setup, make_grid
from bench_dense_investigation import patched
from bench_velocity_scan import make_simulator
from common import make_payload, write_results
from paired import compare_paired, format_report
from rc_prototypes import cpu_variant, make_gpu_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'cpu', 'gpu', 'profile'], default='all')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--offsets', type=float, nargs='+', default=[0., -7.4])
    parser.add_argument('--detunings', type=float, nargs='+', default=[-2., -1., 0., 0.5, 1.5])
    parser.add_argument('--powers', type=float, nargs='+', default=[1., 2., 4., 8., 16.])
    parser.add_argument('--propagators', nargs='+', choices=['frozen', 'magnus'], default=['frozen', 'magnus'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production', action='store_true', help='Compare integrated code with the pre-RC baseline')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output path')
    rows = []
    def save():
        write_results(make_payload(benchmark='rc_background', config=vars(args), results=rows),
                      output=str(args.output), csv_path=None)
    for offset in args.offsets:
        setup = build_multitone_setup(bg_fraction=1/35, rc_bg_fraction=1/35, rc_offset_mhz=offset)
        det, power, _ = make_grid(args.detunings, args.powers, 3, True)
        sim = make_simulator(setup, float(setup['trajectory'].Vini[2]))
        from state_prep import Simulator
        import state_prep.simulator as module
        _, basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
        opts = dict(detunings_hz=det, intensity_prefactors=power, N_steps=args.steps,
            allow_multitone_same_manifold=True, progress=False,
            monitor_states=setup['monitor_states'])
        tags = dict(offset_mhz=offset, steps=args.steps, points=len(det),
                    rc_offset_from_spa2_mhz=setup['rc_info']['rc_offset_from_spa2_center_mhz'])
        def cpu(propagator='frozen', workers=1, variant=None):
            if variant is None:
                return sim.run_microwave_scan(**opts, propagator=propagator, workers=workers).psis_final
            with patched(Simulator, '_time_evolve_mu_batched_shared_slow_multitone',
                         variant._time_evolve_mu_batched_shared_slow_multitone):
                return cpu(propagator, workers)
        def as_numpy(value):
            return value.cpu().numpy() if hasattr(value, 'cpu') else value
        def compare(phase, variants, sync=None, **extra):
            print(f'{phase}: RC offset={offset:g} MHz, {len(det)} points, {args.steps} samples', flush=True)
            def error(a, b):
                a, b = as_numpy(a), as_numpy(b)
                infidelity = float(np.max(np.abs(1-np.abs(np.sum(a.conj()*b,axis=-1))**2)))
                if infidelity > 1e-5:
                    raise RuntimeError(f'State mismatch {infidelity}')
                return float(np.max(np.abs(np.abs(a@basis.conj())**2 - np.abs(b@basis.conj())**2)))
            pair = compare_paired(variants, warmup=args.warmup, repeat=args.repeat, sync=sync,
                                  check=error, tolerance=1e-5, progress=True)
            print(format_report(pair), flush=True)
            if not all(comparison.agrees for comparison in pair.comparisons):
                raise RuntimeError('RC background results disagree')
            rows.extend(dict(row, phase=phase, **tags, **extra) for row in pair.rows())
            rows.append(dict(case='rounds', phase=phase, **tags, **extra,
                             times_s={name: timing.times_s for name,timing in pair.timings.items()}))
            base = as_numpy(next(iter(pair.results.values())))
            for name, result in pair.results.items():
                value = as_numpy(result)
                rows.append(dict(case='validation', phase=phase, variant=name, **tags, **extra,
                    max_norm_error=float(np.max(np.abs(np.sum(np.abs(value)**2,axis=-1)-1))),
                    max_infidelity=float(np.max(np.abs(1-np.abs(np.sum(base.conj()*value,axis=-1))**2)))))
            save()

        if args.phase in ('all', 'cpu', 'profile'):
            component = cpu_variant(components=True)
            cached = cpu_variant(components=True, kernels=True)
            before = cpu_variant()
            for propagator in args.propagators:
                native = []
                original = module.zheevd
                def timed(matrix, *a, **kw):
                    start = time.perf_counter()
                    result = original(matrix, *a, **kw)
                    native.append((matrix.shape, time.perf_counter()-start))
                    return result
                profile = cProfile.Profile()
                with patched(module, 'zheevd', timed):
                    start = time.perf_counter()
                    profile.runcall(cpu, propagator)
                    wall = time.perf_counter()-start
                filename = args.output.with_name(args.output.stem+f'_{offset:g}_{propagator}.prof')
                profile.dump_stats(filename)
                print(f'Profile {propagator}, offset {offset}: wall={wall:.3f}s, eig={sum(t for _,t in native):.3f}s', flush=True)
                pstats.Stats(profile).sort_stats('cumtime').print_stats(12)
                rows.append(dict(case='cpu_profile', propagator=propagator, **tags, wall_s=wall,
                    eigh_s=sum(t for _,t in native), eig_shapes={str(k):v for k,v in Counter(shape for shape,_ in native).items()}))
                save()
                if args.phase != 'profile':
                    variants = dict(current=lambda: cpu(propagator, variant=before),
                                    components=lambda: cpu(propagator, variant=component))
                    if propagator == 'magnus':
                        variants['cached_integrals'] = lambda: cpu(propagator, variant=cached)
                    if args.production:
                        variants = dict(before=lambda: cpu(propagator,variant=before),
                                        production=lambda: cpu(propagator))
                    compare('cpu', variants, propagator=propagator)
        if args.phase in ('all', 'gpu'):
            import torch
            torch.set_num_threads(1)
            if args.production:
                from bench_gpu_spa2 import make_torch_runner
                step, sync = make_torch_runner(setup,n_steps=args.steps,detunings=det,
                    intensity_prefactors=power,rotating_eigh_chunk_size=1)
                tiled, _ = make_torch_runner(setup,n_steps=args.steps,detunings=det,
                    intensity_prefactors=power,rotating_eigh_chunk_size=8)
            else:
                step, sync = make_gpu_runner(setup, det, power, steps=args.steps)
                tiled, _ = make_gpu_runner(setup, det, power, steps=args.steps, full_chunk=8)
            compare('gpu_frozen', dict(cpu_workers=lambda: cpu(workers=args.workers),
                gpu_step=lambda: step().psi_final, gpu_time8=lambda: tiled().psi_final),
                sync=sync, cpu_workers=args.workers)


if __name__ == '__main__':
    main()
