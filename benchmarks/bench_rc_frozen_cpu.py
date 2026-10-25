"""Compare installed CPU solvers and frozen RC propagation without changing integrators."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg.lapack import zheevd
from threadpoolctl import threadpool_limits, threadpool_info

from bench_convergence import build_multitone_setup, make_grid
from bench_dense_investigation import patched
from bench_gpu_spa2 import run_cpu
from common import make_payload, write_results
from multitone_setup import prepare_multitone_inputs
from paired import compare_paired, format_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--offset',type=float,default=-7.4)
    parser.add_argument('--phase',choices=['solvers','propagate','all'],default='all')
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output file')
    import torch
    import state_prep.simulator as module
    from state_prep_gpu import simulate_batched_multitone_shared_slow
    torch.set_num_threads(1)
    setup = build_multitone_setup(bg_fraction=1/35,rc_bg_fraction=1/35,rc_offset_mhz=args.offset)
    det,power,_ = make_grid([-2.,-1.,0.,0.5,1.5],[1.,2.,4.,8.,16.],3,True)
    _,basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
    rows = []
    def save():
        payload = make_payload(benchmark='rc_frozen_cpu',config=vars(args),results=rows)
        payload['cpu_libraries'] = threadpool_info()
        payload['torch_build'] = torch.__config__.show()
        write_results(payload,output=str(args.output),csv_path=None)
    def current():
        return run_cpu(setup,n_steps=args.steps,detunings=det,intensity_prefactors=power).psis_final
    with threadpool_limits(limits=1):
        if args.phase in ('all','solvers'):
            matrices = []
            count = 0
            original = module.zheevd
            selected_steps = {args.steps//4,args.steps//2,3*args.steps//4}
            def capture(matrix,*a,**kw):
                nonlocal count
                step,point = divmod(count,len(det)+1)
                if step in selected_steps and point:
                    matrices.append(matrix.copy())
                count += 1
                return original(matrix,*a,**kw)
            with patched(module,'zheevd',capture):
                current()
            if len(matrices) != len(selected_steps)*len(det):
                raise RuntimeError('Unexpected RC eigensolve pattern')
            corpus = np.stack(matrices)
            tensors = torch.from_numpy(corpus)
            def scipy_solve():
                results = [zheevd(matrix) for matrix in corpus]
                if any(info for _,_,info in results):
                    raise RuntimeError('LAPACK failed')
                return np.stack([values for values,_,_ in results]),np.stack([vectors for _,vectors,_ in results])
            def torch_loop():
                results = [torch.linalg.eigh(matrix,UPLO='U') for matrix in tensors]
                return torch.stack([d for d,_ in results]).numpy(),torch.stack([v for _,v in results]).numpy()
            def torch_batch():
                values,vectors = torch.linalg.eigh(tensors,UPLO='U')
                return values.numpy(),vectors.numpy()
            reference_matrix = np.triu(corpus)+np.triu(corpus,1).conj().swapaxes(-1,-2)
            for name,solve in [('scipy',scipy_solve),('torch_loop',torch_loop),('torch_batch',torch_batch)]:
                values,vectors = solve()
                residual = float(np.linalg.norm(reference_matrix@vectors-vectors*values[:,None,:]) / np.linalg.norm(reference_matrix))
                orthogonality = float(np.max(np.abs(vectors.conj().swapaxes(-1,-2)@vectors-np.eye(corpus.shape[-1]))))
                if max(residual,orthogonality)>1e-10:
                    raise RuntimeError(f'Invalid {name} eigensystems: {residual}, {orthogonality}')
                rows.append(dict(case='eigensystem_validation',variant=name,residual=residual,orthogonality_error=orthogonality))
            pair = compare_paired(dict(scipy=scipy_solve,torch_loop=torch_loop,torch_batch=torch_batch),
                warmup=args.warmup,repeat=args.repeat,progress=True,
                check=lambda a,b:float(np.max(np.abs(a[0]-b[0]))/np.max(np.abs(a[0]))),tolerance=1e-12)
            print(format_report(pair),flush=True)
            if not all(comparison.agrees for comparison in pair.comparisons):
                raise RuntimeError('Eigensolver eigenvalues disagree')
            rows.extend(dict(row,phase='solvers',matrices=len(corpus)) for row in pair.rows())
            rows.append(dict(case='rounds',phase='solvers',times_s={name:timing.times_s for name,timing in pair.timings.items()}))
            save()
        if args.phase in ('all','propagate'):
            inputs = prepare_multitone_inputs(setup,det,power,steps=args.steps,device='cpu')
            def run(chunk):
                return simulate_batched_multitone_shared_slow(**inputs,slow_eigh_chunk_size=128,
                    rotating_eigh_chunk_size=chunk).psi_final.numpy()
            def error(a,b):
                probability = float(np.max(np.abs(np.abs(a@basis.conj())**2-np.abs(b@basis.conj())**2)))
                infidelity = float(np.max(np.abs(1-np.abs(np.sum(a.conj()*b,axis=-1))**2)))
                if infidelity>1e-5:
                    raise RuntimeError('Physical states disagree')
                return probability
            pair = compare_paired(dict(scipy_cpu=current,torch_cpu_step=lambda:run(1),torch_cpu_time8=lambda:run(8)),
                warmup=args.warmup,repeat=args.repeat,progress=True,check=error,tolerance=1e-5)
            print(format_report(pair),flush=True)
            if not all(comparison.agrees for comparison in pair.comparisons):
                raise RuntimeError('Frozen RC propagation disagrees')
            rows.extend(dict(row,phase='propagate') for row in pair.rows())
            rows.append(dict(case='rounds',phase='propagate',times_s={name:timing.times_s for name,timing in pair.timings.items()}))
            save()


if __name__=='__main__':
    main()
