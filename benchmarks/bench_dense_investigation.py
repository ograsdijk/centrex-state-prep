"""Investigate dense solvers, layout, dispatch and cross-call slow caches.

All adapters and memoization live in this benchmark. Production APIs and the
frozen propagation method are unchanged. Output directories must be fresh.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from scipy.linalg import lapack

from bench_gpu_spa2 import make_torch_runner, run_cpu
from common import make_payload, write_results, REPO_ROOT
from dense_solver_prototypes import ReusableCusolver, ReusableLapack
from paired import compare_paired, format_report
from profile_transverse_fields import field_variant


@contextmanager
def patched(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def parameters(batch):
    det = np.repeat(np.linspace(-2e6, 1e6, batch)[:, None], 2, axis=1)
    return det, np.ones_like(det)


def cpu_call(setup, steps, batch, solver=None, simulator=None):
    import state_prep.simulator as module
    det, power = parameters(batch)
    if simulator is not None:
        sim = simulator(setup['trajectory'], setup['electric_field'], setup['magnetic_field'],
                        setup['initial_states'], setup['hamiltonian'], setup['microwave_fields'])
        return sim.run_microwave_scan(detunings_hz=det, intensity_prefactors=power,
                                      N_steps=steps, progress=False, workers=1, block_eigh=False)
    if solver is None:
        return run_cpu(setup, n_steps=steps, detunings=det, intensity_prefactors=power)
    with patched(module, 'zheevd', solver):
        return run_cpu(setup, n_steps=steps, detunings=det, intensity_prefactors=power)


def capture(case, output):
    import state_prep.simulator as module
    setup = field_variant(case)
    original, matrices, slow, count = module.zheevd, [], [], 0
    steps, batch = 300, 25
    selected = set(np.linspace(0, steps-2, 12, dtype=int))
    def collect(a, *args, **kw):
        nonlocal count
        step, point = divmod(count, batch+1)
        if step in selected:
            (slow if point == 0 else matrices).append(a.copy())
        count += 1
        return original(a, *args, **kw)
    cpu_call(setup, steps, batch, collect)
    assert count == (steps-1)*(batch+1)
    np.savez(output, rotating=np.stack(matrices), slow=np.stack(slow))
    return np.stack(matrices)


def cpu_solvers():
    native = ReusableLapack()
    n = 64
    lw, liw, lrw, _ = lapack.zheevd_lwork(n)
    def qr(a):
        return lapack.zheev(a)
    def evr(a):
        w, v, m, _, info = lapack.zheevr(a)
        if m != len(a):
            raise RuntimeError('Incomplete eigensystem')
        return w, v, info
    return {
        'zheevd': lapack.zheevd,
        'numpy': lambda a: (*np.linalg.eigh(a, UPLO='U'), 0),
        'zheev': qr, 'zheevr': evr,
        'optimal_sizes': lambda a: lapack.zheevd(a, lwork=int(lw.real), liwork=int(liw), lrwork=int(lrw)),
        'fortran_overwrite_copy': lambda a: lapack.zheevd(np.array(a, order='F', copy=True), overwrite_a=True),
        'reusable_workspace': native,
    }


def record(pair, rows, **context):
    print(format_report(pair), flush=True)
    if any(comparison.agrees is False for comparison in pair.comparisons):
        raise RuntimeError('Solver output failed agreement check')
    rows.extend(dict(row, **context) for row in pair.rows())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['cpu_screen', 'gpu_screen', 'cpu_propagate', 'gpu_propagate'], required=True)
    parser.add_argument('--cases', nargs='+', default=['transverse_b', 'varying_e'])
    parser.add_argument('--steps', nargs='+', type=int, default=[150, 300])
    parser.add_argument('--batches', nargs='+', type=int)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    rows, objects = [], []
    def save():
        write_results(make_payload(benchmark='dense_investigation', config=vars(args), results=rows),
                      output=str(args.output), csv_path=None)
    for case in args.cases:
        path = args.capture_dir / f'{case}.npz'
        matrices = np.load(path)['rotating'] if path.exists() else capture(case, path)
        print(f'{args.phase}: {case}, captured {len(matrices)} matrices', flush=True)
        if args.phase == 'cpu_screen':
            from state_prep import limit_blas_threads
            solvers = cpu_solvers()
            def runner(solver):
                def run():
                    results = [solver(matrix) for matrix in matrices]
                    if any(r[2] for r in results):
                        raise RuntimeError('LAPACK failed')
                    return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])
                return run
            with limit_blas_threads(1):
                pair = compare_paired({name: runner(solver) for name, solver in solvers.items()},
                    warmup=args.warmup, repeat=args.repeat, progress=True,
                    check=lambda a,b: float(np.max(np.abs(a[0]-b[0])) / np.max(np.abs(a[0]))), tolerance=1e-12)
            record(pair, rows, field_case=case)
            for name, (w,v) in pair.results.items():
                residual = float(np.max(np.abs(matrices@v-v*w[:,None,:]))/np.max(np.abs(matrices)))
                orthogonal = float(np.max(np.abs(v.conj().swapaxes(-1,-2)@v-np.eye(64))))
                if residual > 1e-12 or orthogonal > 1e-10:
                    raise RuntimeError(f'{name}: {residual}, {orthogonal}')
                rows.append(dict(case='validation', variant=name, field_case=case, residual=residual, orthogonality_error=orthogonal))
            # The f2py sizes are scratch lengths, not reusable workspace buffers.
            rows.append(dict(case='workspace_api', field_case=case,
                scipy_wrapper_accepts_work_arrays=False, default_lwork=4224,
                queried_lwork=int(lapack.zheevd_lwork(64)[0].real)))
        elif args.phase == 'gpu_screen':
            import torch
            torch.set_num_threads(1)
            original = torch.backends.cuda.preferred_linalg_library()
            checked, deferred = ReusableCusolver(), ReusableCusolver(True)
            jacobi = ReusableCusolver(True, driver='jacobi')
            objects += [checked, deferred, jacobi]
            for batch in args.batches or [25,250]:
                a = torch.tensor(matrices[:batch], device='cuda')
                def builtin(backend):
                    def run():
                        torch.backends.cuda.preferred_linalg_library(backend)
                        return torch.linalg.eigh(a)
                    return run
                def native(solver):
                    def run():
                        result = solver(a)
                        solver.flush()
                        return result
                    return run
                variants = dict(default=builtin('default'), cusolver=builtin('cusolver'), magma=builtin('magma'),
                                native_checked=native(checked), native_deferred=native(deferred), jacobi=native(jacobi))
                # Screen unsupported backend/library combinations explicitly.
                for name in list(variants):
                    try:
                        variants[name](); torch.cuda.synchronize()
                    except Exception as exc:
                        rows.append(dict(case='unsupported', variant=name, message=str(exc)))
                        del variants[name]
                pair = compare_paired(variants, warmup=args.warmup, repeat=args.repeat,
                    sync=torch.cuda.synchronize, progress=True,
                    check=lambda x,y: float((x[0]-y[0]).abs().max().item()/x[0].abs().max().item()), tolerance=1e-12)
                record(pair, rows, field_case=case, batch=batch)
                for name,(w,v) in pair.results.items():
                    residual = float((a@v-v*w[:,None,:]).abs().max().item()/a.abs().max().item())
                    orthogonal = float((v.mH@v-torch.eye(64,device='cuda')).abs().max().item())
                    if residual > 1e-12 or orthogonal > 1e-10:
                        raise RuntimeError(f'{name}: {residual}, {orthogonal}')
                    rows.append(dict(case='validation', variant=name, field_case=case, batch=batch,
                                     residual=residual, orthogonality_error=orthogonal))
            torch.backends.cuda.preferred_linalg_library(original)
        elif args.phase == 'cpu_propagate':
            setup = field_variant(case)
            solvers = cpu_solvers()
            # Isolate layout/copy changes in a temporary module, not production.
            import state_prep.simulator as module
            source = Path(module.__file__).read_text()
            source = source.replace('H_rot = (coupling_scales[b, 0] * H_mu_rot[0]).copy()',
                'H_rot = (coupling_scales[b, 0] * H_mu_rot[0]).copy(order="F")')
            source = source.replace('H_rot = np.zeros((n, n), dtype=np.complex128)',
                'H_rot = np.zeros((n, n), dtype=np.complex128, order="F")')
            source = source.replace('zheevd(H_rot)', 'zheevd(H_rot, overwrite_a=True)')
            layout = types.ModuleType('state_prep._dense_layout_experiment')
            sys.modules[layout.__name__] = layout
            exec(compile(source, 'layout_experiment.py', 'exec'), layout.__dict__)
            for steps in args.steps:
                for batch in args.batches or [8,25]:
                    def make(solver):
                        return lambda: cpu_call(setup, steps, batch, solver)
                    # QR/MRRR lost decisively on captured matrices; retain
                    # only the current driver and allocation/layout candidates.
                    variants = {name: make(solvers[name]) for name in ['zheevd','reusable_workspace']}
                    variants['fortran_assembly_overwrite'] = lambda: cpu_call(setup, steps, batch, simulator=layout.Simulator)
                    pair = compare_paired(variants, warmup=args.warmup, repeat=args.repeat, progress=True,
                        check=lambda a,b: float(np.max(np.abs(a.probabilities_final-b.probabilities_final))), tolerance=1e-5)
                    record(pair, rows, field_case=case, steps=steps, batch=batch)
                    for name, result in pair.results.items():
                        infidelity = float(np.max(np.abs(1-np.abs(np.sum(pair.results['zheevd'].psis_final.conj()*result.psis_final,axis=-1))**2)))
                        if infidelity > 1e-5:
                            raise RuntimeError(f'{name} infidelity {infidelity}')
                        rows.append(dict(case='state_validation', variant=name, field_case=case, steps=steps, batch=batch, max_infidelity=infidelity))
                    save()
        elif args.phase == 'gpu_propagate':
            import torch
            torch.set_num_threads(1)
            original_eigh = torch.linalg.eigh
            original_backend = torch.backends.cuda.preferred_linalg_library()
            checked, deferred = ReusableCusolver(), ReusableCusolver(True)
            objects += [checked, deferred]
            setup = field_variant(case)
            for steps in args.steps:
                for batch in args.batches or [25,250]:
                    det,power = parameters(batch)
                    run,sync = make_torch_runner(setup,n_steps=steps,detunings=det,intensity_prefactors=power,
                                                block_eigh=False,slow_eigh_chunk_size=1)
                    def backend(name, solver=None):
                        def call():
                            torch.backends.cuda.preferred_linalg_library(name)
                            if solver is None:
                                return run()
                            with patched(torch.linalg,'eigh',solver):
                                result = run()
                            solver.flush()
                            return result
                        return call
                    # Installed PyTorch reports that MAGMA selection falls back
                    # to cuSOLVER. Screen native_checked separately; retain the
                    # most aggressive dispatch/workspace prototype end to end.
                    variants = dict(default=backend('default'), native_deferred=backend('default',deferred))
                    pair = compare_paired(variants,warmup=args.warmup,repeat=args.repeat,sync=sync,progress=True,
                        check=lambda a,b: float((a.probabilities_final-b.probabilities_final).abs().max().item()),tolerance=1e-5)
                    record(pair,rows,field_case=case,steps=steps,batch=batch)
                    for name,result in pair.results.items():
                        infidelity = float((1-(pair.results['default'].psi_final.conj()*result.psi_final).sum(-1).abs()**2).abs().max().item())
                        if infidelity > 1e-5:
                            raise RuntimeError(f'{name} infidelity {infidelity}')
                        rows.append(dict(case='state_validation',variant=name,field_case=case,steps=steps,batch=batch,max_infidelity=infidelity))
                    save()
            torch.backends.cuda.preferred_linalg_library(original_backend)
            assert torch.linalg.eigh == original_eigh
        save()
    for obj in objects:
        obj.close()


if __name__ == '__main__':
    main()
