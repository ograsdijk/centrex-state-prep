"""Isolated propagation timing for SPA referee S19; never touches its queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path
import sys

import dill
import numpy as np

from common import make_payload, write_results
from paired import compare_paired, format_report


def initialize_cpu(payload, paper_scripts):
    global _cpu
    sys.path.insert(0, paper_scripts)
    with open(payload, 'rb') as stream:
        _cpu = dill.load(stream)


def cpu_unit(index):
    from state_prep import Simulator
    import r2_s19_gpu_io as io
    case, vectors, det, powers, steps = _cpu
    sim = Simulator(case['traj'], case['efield'], case['bfield'],
                    [io.core.VecState(v) for v in vectors], case['ham'],
                    [f['field'] for f in case['fields']])
    sl = slice(index*34, (index+1)*34)
    return sim.run_microwave_scan(
        detunings_hz=det[sl], intensity_prefactors=powers[sl], N_steps=steps,
        store_final_probabilities=False, store_final_monitor_probabilities=False,
        progress=False, workers=1, blas_threads=1, time_sampling='mid').psis_final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-scripts', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--workers', type=int, nargs='+', default=[8, 10, 12])
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--profile-cpu', action='store_true', help='Profile one unit instead of running timing comparisons')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output path')
    sys.path.insert(0, str(args.paper_scripts))
    import r2_s19_gpu_io as io
    # This adapter otherwise creates a snapshot if missing. Never permit that here.
    if not io.SOLVER.is_file():
        raise RuntimeError('Existing referee solver snapshot required; no external writes allowed')
    points, _, _ = io.s18.base.data_points(all_off=True)
    molecules, _ = io.s18.base.samples(0)
    configs = list(itertools.product([.45, .55, .65], [.001, .003], [.03, .06]))
    task = dict(points=points.tolist(), mol=molecules[1].tolist(), phase=0.,
                jmax=4, k=16, configs=configs)
    case, vectors, _ = io.case_vectors(task)
    det, powers = io.scan_arrays(case, points, configs)
    arrays = io.build_inputs(task, n_steps_override=args.steps)
    validation = io.check_inputs(task, arrays)
    if not validation['passed']:
        raise RuntimeError(validation)
    _, basis = np.linalg.eigh(case['ham'].get_H_t_func()(case['traj'].get_T()))
    payload = args.output.with_suffix('.cpu.dill')
    payload.parent.mkdir(parents=True, exist_ok=True)
    with payload.open('wb') as stream:
        dill.dump((case, vectors, det, powers, args.steps), stream)
    if args.profile_cpu:
        import cProfile
        import pstats
        import time
        import state_prep.simulator as simulator_module
        initialize_cpu(str(payload), str(args.paper_scripts))
        profile = cProfile.Profile()
        native_times = {}
        original_eigh = simulator_module.zheevd
        def timed_eigh(*values, **options):
            line = sys._getframe(1).f_lineno
            start = time.perf_counter()
            try:
                return original_eigh(*values, **options)
            finally:
                entry = native_times.setdefault(line, [0, 0.])
                entry[0] += 1
                entry[1] += time.perf_counter()-start
        simulator_module.zheevd = timed_eigh
        try:
            profile.runcall(cpu_unit, 0)
        finally:
            simulator_module.zheevd = original_eigh
        profile.dump_stats(str(args.output.with_suffix('.prof')))
        stats = pstats.Stats(profile)
        stats.sort_stats('cumulative').print_stats(25)
        rows = [dict(filename=filename, line=line, function=function,
                     primitive_calls=data[0], calls=data[1], self_seconds=data[2], cumulative_seconds=data[3])
                for (filename,line,function),data in stats.stats.items()]
        rows.sort(key=lambda row: row['cumulative_seconds'], reverse=True)
        source_lines = Path(simulator_module.__file__).read_text().splitlines()
        for line, (calls, seconds) in native_times.items():
            rows.append(dict(case='native_eigh', line=line, calls=calls, seconds=seconds,
                             source=source_lines[line-1].strip()))
        write_results(make_payload(benchmark='referee_s19_cpu_profile',
            config=dict(steps=args.steps, task=task, units=1, frequencies=len(points),
                        workers=1, blas_threads=1, profile_total_seconds=stats.total_tt,
                        caveat='Profiling overhead included; attribution only, not a speed benchmark'),
            results=rows), output=str(args.output), csv_path=None)
        payload.unlink()
        return
    import r2_s19_gpu_worker as worker
    import torch
    rows = []
    def gpu():
        return worker.propagate(arrays)[0]
    def check(a, b):
        err = float(np.max(np.abs(np.abs(a@basis.conj())**2-np.abs(b@basis.conj())**2)))
        if err > 1e-5:
            raise RuntimeError(f'CPU/GPU probability disagreement: {err}')
        return err
    config = dict(steps=args.steps, production_steps=io.n_steps(task['mol']),
                  basis_states=100, initial_states=16, frequencies=34, units=12,
                  workers=args.workers, repeat=args.repeat, warmup=1,
                  task=task, timing_scope='propagation with prepared inputs; GPU transfers included',
                  caveat='Reduced full-trajectory timestep count: timing pilot, not converged physics')
    for count in args.workers:
        print(f'Preparing persistent {count}-worker pool', flush=True)
        with ProcessPoolExecutor(max_workers=count, initializer=initialize_cpu,
                                 initargs=(str(payload), str(args.paper_scripts))) as pool:
            def cpu():
                return np.concatenate(list(pool.map(cpu_unit, range(12))))
            result = compare_paired({'cpu': cpu, 'gpu': gpu}, warmup=1,
                repeat=args.repeat, check=check, tolerance=1e-5, progress=True,
                sync=torch.cuda.synchronize)
            print(f'workers={count}\n{format_report(result)}', flush=True)
            rows.extend(dict(row, workers=count) for row in result.rows())
            rows.append(dict(case='rounds', workers=count,
                times_s={name: timing.times_s for name, timing in result.timings.items()},
                validation=validation))
            write_results(make_payload(benchmark='referee_s19_cpu_gpu', config=config,
                results=rows), output=str(args.output), csv_path=None)
    payload.unlink()


if __name__ == '__main__':
    main()
