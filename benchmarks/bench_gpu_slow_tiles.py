"""Production slow-eigensolve tiles versus the pre-tile Git implementation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bench_block_overhead import load_baseline
from bench_dense_investigation import patched, parameters
from bench_gpu_spa2 import make_torch_runner, run_cpu
from common import make_payload, write_results
from paired import compare_paired, format_report
from profile_transverse_fields import field_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='b02b546')
    parser.add_argument('--cases', nargs='+', default=['aligned','transverse_b','varying_e'])
    parser.add_argument('--shapes', nargs='+', default=['150:25','300:250'])
    parser.add_argument('--chunk-size',type=int,default=128)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output file')
    import torch
    import state_prep_gpu.simulator as module
    torch.set_num_threads(1)
    baseline = load_baseline('state_prep_gpu._slow_tile_baseline','src/state_prep_gpu/simulator.py',args.baseline_ref)
    original_baseline = baseline.simulate_batched_scalar_microwaves_shared_slow
    def legacy(**kw):
        kw.pop('slow_eigh_chunk_size',None)
        return original_baseline(**kw)
    rows = []
    for case in args.cases:
        setup = field_variant(case)
        _,basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
        for shape in args.shapes:
            steps,batch = map(int,shape.split(':'))
            det,power = parameters(batch)
            with patched(module,'simulate_batched_scalar_microwaves_shared_slow',legacy):
                old,sync = make_torch_runner(setup,n_steps=steps,detunings=det,intensity_prefactors=power)
            tiled,_ = make_torch_runner(setup,n_steps=steps,detunings=det,intensity_prefactors=power,
                                       slow_eigh_chunk_size=args.chunk_size)
            print(f'{case}: steps={steps}, batch={batch}',flush=True)
            pair = compare_paired(dict(untiled=old,tiled=tiled),repeat=args.repeat,warmup=args.warmup,
                sync=sync,progress=True,
                check=lambda a,b:float((a.probabilities_final-b.probabilities_final).abs().max().item()),tolerance=1e-5)
            print(format_report(pair),flush=True)
            if not all(c.agrees for c in pair.comparisons):
                raise RuntimeError('Tracked output probabilities changed')
            a,b = [pair.results[key].psi_final.cpu().numpy() for key in ('untiled','tiled')]
            infidelity = float(np.max(np.abs(1-np.abs(np.sum(a.conj()*b,axis=-1))**2)))
            common_error = float(np.max(np.abs(np.abs(a@basis.conj())**2-np.abs(b@basis.conj())**2)))
            subset = np.linspace(0,batch-1,min(batch,7),dtype=int)
            cpu = run_cpu(setup,n_steps=steps,detunings=det[subset],intensity_prefactors=power[subset])
            cpu_error = float(np.max(np.abs(np.abs(cpu.psis_final@basis.conj())**2-np.abs(b[subset]@basis.conj())**2)))
            cpu_infidelity = float(np.max(np.abs(1-np.abs(np.sum(cpu.psis_final.conj()*b[subset],axis=-1))**2)))
            if max(infidelity,common_error,cpu_error,cpu_infidelity) > 1e-5:
                raise RuntimeError(f'Physical states disagree: {infidelity}, {common_error}, {cpu_error}, {cpu_infidelity}')
            rows.extend(dict(row,field_case=case,steps=steps,batch=batch) for row in pair.rows())
            rows.append(dict(case='physical_validation',field_case=case,steps=steps,batch=batch,
                max_infidelity=infidelity,common_basis_error=common_error,cpu_common_basis_error=cpu_error,
                cpu_max_infidelity=cpu_infidelity,cpu_points=len(subset)))
            rows.append(dict(case='rounds',field_case=case,steps=steps,batch=batch,
                             times_s={name:timing.times_s for name,timing in pair.timings.items()}))
            write_results(make_payload(benchmark='gpu_slow_tiles',config=vars(args),results=rows),
                          output=str(args.output),csv_path=None)


if __name__ == '__main__':
    main()
