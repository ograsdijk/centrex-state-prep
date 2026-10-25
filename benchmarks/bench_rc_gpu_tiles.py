"""Paired tuning of RC frozen GPU time tiles and eigensolver workspace caps."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bench_convergence import build_multitone_setup,make_grid
from common import make_payload,write_results
from multitone_setup import prepare_multitone_inputs
from paired import compare_paired,format_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--offset',type=float,default=-7.4)
    parser.add_argument('--tiles',nargs='+',default=['8:256','16:512','32:1024'],help='time tile:maximum matrices per solve')
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output path')
    import torch
    from state_prep_gpu import simulate_batched_multitone_shared_slow
    torch.set_num_threads(1)
    setup=build_multitone_setup(bg_fraction=1/35,rc_bg_fraction=1/35,rc_offset_mhz=args.offset)
    det,power,_=make_grid([-2.,-1.,0.,0.5,1.5],[1.,2.,4.,8.,16.],3,True)
    inputs=prepare_multitone_inputs(setup,det,power,steps=args.steps)
    _,basis=np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
    variants={}
    for tile in args.tiles:
        width,cap=map(int,tile.split(':'))
        def run(width=width,cap=cap):
            return simulate_batched_multitone_shared_slow(**inputs,
                rotating_eigh_chunk_size=width,max_eigh_matrices=cap).psi_final
        variants[tile]=run
    def error(a,b):
        a,b=a.cpu().numpy(),b.cpu().numpy()
        fidelity=float(np.max(np.abs(1-np.abs(np.sum(a.conj()*b,axis=-1))**2)))
        if fidelity>1e-5:
            raise RuntimeError(f'Physical-state mismatch: {fidelity}')
        return float(np.max(np.abs(np.abs(a@basis.conj())**2-np.abs(b@basis.conj())**2)))
    pair=compare_paired(variants,warmup=args.warmup,repeat=args.repeat,progress=True,
        sync=torch.cuda.synchronize,check=error,tolerance=1e-5)
    print(format_report(pair),flush=True)
    rows=pair.rows()
    rows.append(dict(case='rounds',times_s={name:timing.times_s for name,timing in pair.timings.items()}))
    if not all(comparison.agrees for comparison in pair.comparisons):
        raise RuntimeError('RC tile configurations disagree')
    for name,run in variants.items():
        torch.cuda.reset_peak_memory_stats()
        initial=torch.cuda.memory_allocated()
        result=run()
        torch.cuda.synchronize()
        rows.append(dict(case='memory',variant=name,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            incremental_peak_bytes=torch.cuda.max_memory_allocated()-initial,
            max_norm_error=float((result.abs().square().sum(-1)-1).abs().max().item())))
        del result
    write_results(make_payload(benchmark='rc_gpu_tiles',config=vars(args),results=rows),
                  output=str(args.output),csv_path=None)


if __name__=='__main__':
    main()
