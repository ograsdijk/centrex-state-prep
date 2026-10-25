"""Short S19 GPU tiling, profiling, and concurrent-device pilots, outside its queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import itertools
from pathlib import Path
import sys

import dill
import numpy as np

from bench_referee_s19 import initialize_cpu, cpu_unit
from common import make_payload, write_results
from paired import compare_paired, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-scripts', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--phase', choices=['gpu', 'concurrent', 'all'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.steps <= 3000 or args.repeat < 5:
        parser.error('Use a fresh output, 2–3000 pilot samples and at least five repeats')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.paper_scripts))
    import r2_s19_gpu_io as io
    import r2_s19_gpu_worker as worker
    import torch
    from state_prep_gpu import simulate_batched_multitone_shared_slow
    torch.set_num_threads(1)
    if not io.SOLVER.is_file():
        raise RuntimeError('Existing snapshot required; external writes prohibited')
    points, _, _ = io.s18.base.data_points(all_off=True)
    if len(points) != 34:
        raise RuntimeError('Referee frequency definition changed')
    molecules, _ = io.s18.base.samples(0)
    configs = list(itertools.product([.45, .55, .65], [.001, .003], [.03, .06]))
    task = dict(points=points.tolist(), mol=molecules[1].tolist(), phase=0.,
                jmax=4, k=16, configs=configs)
    case, vectors, _ = io.case_vectors(task)
    z = io.build_inputs(task, n_steps_override=args.steps)
    validation = io.check_inputs(task, z)
    if not validation['passed']:
        raise RuntimeError(validation)
    # The S19 factorization contains paired Cartesian upper/lower pieces.
    # Sum amp*base is Hermitian. Half that sum plus its adjoint reconstructs
    # exactly the same microwave Hamiltonian, without introducing beat phases.
    h = np.einsum('tm,mij->tij', z['amps'], z['base'])
    validation['half_plus_adjoint_max_error'] = float(np.max(np.abs(h-(h+h.conj().transpose(0,2,1))*.5)))
    if not np.allclose(h, h.conj().transpose(0,2,1), rtol=1e-12, atol=1e-10):
        raise RuntimeError('Microwave factorization is not Hermitian')
    del h
    _, basis = np.linalg.eigh(case['ham'].get_H_t_func()(case['traj'].get_T()))
    rows = []
    config = dict(steps=args.steps, production_steps=io.n_steps(task['mol']),
        warmup=1, repeat=args.repeat, task=task, workers=12,
        timing_scope='prepared physical inputs; GPU transfers and state retrieval included',
        caveat='Coarse timing pilot, not converged production or production ETA')
    def save():
        write_results(make_payload(benchmark='referee_followup', config=config,
            results=rows), output=str(args.output), csv_path=None)
    def check(a, b):
        err = float(np.max(np.abs(np.abs(a@basis.conj())**2-np.abs(b@basis.conj())**2)))
        norm = max(float(np.max(np.abs(np.sum(np.abs(x)**2,axis=-1)-1))) for x in (a,b))
        infidelity = float(np.max(np.abs(1-np.abs(np.sum(a.conj()*b,axis=-1))**2)))
        if not np.isfinite(err+norm+infidelity) or max(err, norm, infidelity) > 1e-5:
            raise RuntimeError(dict(probability_error=err, norm_error=norm, infidelity=infidelity))
        return err
    def baseline():
        return worker.propagate(z)[0]
    def tiled(width=8, cap=512):
        # Keep all conversions inside the timer, like the existing worker.
        tensor = lambda a, dtype=torch.complex128: torch.as_tensor(a, dtype=dtype, device='cuda')
        terms = worker.BatchedSlowHamiltonianTerms(**{k:tensor(z[k]) for k in worker.TERMS})
        e = np.vstack([z['E_t'], z['E_t'][-1:]])
        count = len(z['scales'])
        dt = z['dt']
        result = simulate_batched_multitone_shared_slow(terms=terms,
            E_t=tensor(e, torch.float64), B_t=tensor(np.broadcast_to(z['B'],e.shape).copy(),torch.float64),
            E0=tensor(z['E0'],torch.float64), B0=tensor(z['B'],torch.float64),
            psi0=tensor(z['psi0'])[None].expand(count,-1,-1).contiguous(), dt=tensor(dt,torch.float64),
            H_mu_upper_fields=tensor(z['base']*.5), amp_t=tensor(z['amps']),
            coupling_scales=tensor(z['scales'],torch.float64), D_mu_diag_batch=tensor(z['dmu'],torch.float64),
            beat_angular_frequencies=tensor(np.zeros_like(z['scales']),torch.float64),
            sample_times=tensor(np.cumsum(dt)-.5*dt,torch.float64),
            return_final_probabilities=False, rotating_eigh_chunk_size=width, max_eigh_matrices=cap)
        return result.psi_final.cpu().numpy()
    def record(phase, variants):
        pair = compare_paired(variants, warmup=1, repeat=args.repeat, progress=True,
            sync=torch.cuda.synchronize, check=check, tolerance=1e-5)
        print(f'{phase}\n{format_report(pair)}',flush=True)
        rows.extend(dict(row, phase=phase) for row in pair.rows())
        rows.append(dict(case='rounds', phase=phase,
            times_s={name:timing.times_s for name,timing in pair.timings.items()}, validation=validation))
        save()
    if args.phase in ('gpu','all'):
        variants = {'scalar':baseline, 'tile8_cap512':tiled,
                    'tile8_cap1024':lambda:tiled(cap=1024)}
        record('gpu_tiling_12_units',variants)
        for name, run in variants.items():
            torch.cuda.reset_peak_memory_stats()
            value=run()
            rows.append(dict(case='memory',variant=name,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                norm_error=float(np.max(np.abs(np.sum(abs(value)**2,axis=-1)-1)))))
        original = worker.simulate_batched_scalar_microwaves_shared_slow
        profile = {}
        def profiled(**kwargs):
            return original(**kwargs, profile_timings=profile, profile_step_stride=16)
        worker.simulate_batched_scalar_microwaves_shared_slow = profiled
        try:
            baseline()
        finally:
            worker.simulate_batched_scalar_microwaves_shared_slow = original
        rows.append(dict(case='gpu_profile', mean_cuda_seconds_per_step={
            name:float(np.mean(values)) for name,values in profile.items() if len(values)}))
        save()
    if args.phase in ('concurrent','all'):
        # Three repetitions of twelve configurations: 36 independent units.
        # CPU-only has three complete waves; mixed uses two CPU waves plus
        # twelve GPU units. Both variants compute identical ordered outputs.
        det, powers = io.scan_arrays(case, points, configs*3)
        payload = args.output.with_suffix('.cpu.dill')
        with payload.open('wb') as stream:
            dill.dump((case,vectors,det,powers,args.steps),stream)
        try:
            with ProcessPoolExecutor(max_workers=12, initializer=initialize_cpu,
                    initargs=(str(payload),str(args.paper_scripts))) as pool, ThreadPoolExecutor(max_workers=1) as thread:
                def cpu_only():
                    return np.concatenate(list(pool.map(cpu_unit,range(36))))
                def mixed():
                    future = thread.submit(baseline)
                    cpu = np.concatenate(list(pool.map(cpu_unit,range(24))))
                    return np.concatenate([cpu,future.result()])
                record('concurrent_36_units',{'cpu_only':cpu_only, 'cpu24_gpu12':mixed})
        finally:
            payload.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
