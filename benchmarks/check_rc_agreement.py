"""Longer RC numerical checks; deliberately makes no single-run speed claim."""
from pathlib import Path
import argparse

import numpy as np

from bench_convergence import build_multitone_setup, make_grid
from bench_dense_investigation import patched
from bench_gpu_spa2 import make_torch_runner, run_cpu
from bench_velocity_scan import make_simulator
from common import make_payload, write_results
from rc_prototypes import cpu_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--offsets',type=float,nargs='+',default=[0.,-7.4])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output file')
    import torch
    from state_prep import Simulator
    torch.set_num_threads(1)
    rows = []
    for offset in args.offsets:
        setup = build_multitone_setup(bg_fraction=1/35,rc_bg_fraction=1/35,rc_offset_mhz=offset)
        det,power,_ = make_grid([-1.,0.,0.5],[1.,4.,16.],3,True)
        det,power = np.tile(det,(3,1)),np.tile(power,(3,1))
        speeds = np.repeat([150.,184.,230.],9)
        _,basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
        runner,sync = make_torch_runner(setup,n_steps=args.steps,detunings=det,
            intensity_prefactors=power,forward_velocities=speeds,max_eigh_matrices=64)
        profile = {}
        gpu = runner(store_every=333,profile_timings=profile)
        sync()
        vectors = gpu.psi_final.cpu().numpy()
        subset = np.linspace(0,len(det)-1,7,dtype=int)
        cpu = run_cpu(setup,n_steps=args.steps,detunings=det[subset],
            intensity_prefactors=power[subset],forward_velocities=speeds[subset])
        error = float(np.max(np.abs(np.abs(cpu.psis_final@basis.conj())**2-np.abs(vectors[subset]@basis.conj())**2)))
        fidelity = float(np.max(np.abs(1-np.abs(np.sum(cpu.psis_final.conj()*vectors[subset],axis=-1))**2)))
        norm = float(np.max(np.abs(np.sum(np.abs(vectors)**2,axis=-1)-1)))
        if max(error,fidelity,norm) > 1e-5:
            raise RuntimeError(f'RC frozen CPU/GPU mismatch: {error}, {fidelity}, {norm}')
        row = dict(case='gpu_cpu_frozen',offset_mhz=offset,points=len(det),steps=args.steps,
                   max_probability_error=error,max_infidelity=fidelity,max_norm_error=norm,
                   profile_stage_means_s={name:float(np.mean(values)) for name,values in profile.items()})
        rows.append(row)
        print(row,flush=True)
        sim = make_simulator(setup,184.)
        opts = dict(detunings_hz=det,intensity_prefactors=power,forward_velocities=speeds,
            N_steps=args.steps,progress=False,allow_multitone_same_manifold=True,propagator='magnus')
        before = cpu_variant()
        with patched(Simulator,'_time_evolve_mu_batched_shared_slow_multitone',before._time_evolve_mu_batched_shared_slow_multitone):
            old = sim.run_microwave_scan(**opts)
        current = sim.run_microwave_scan(**opts)
        error = float(np.max(np.abs(np.abs(old.psis_final@basis.conj())**2-np.abs(current.psis_final@basis.conj())**2)))
        fidelity = float(np.max(np.abs(1-np.abs(np.sum(old.psis_final.conj()*current.psis_final,axis=-1))**2)))
        if max(error,fidelity) > 1e-5:
            raise RuntimeError('CPU RC cache changed physical states')
        row = dict(case='cpu_magnus_cache',offset_mhz=offset,steps=args.steps,points=len(det),
                   max_probability_error=error,max_infidelity=fidelity)
        rows.append(row)
        print(row,flush=True)
        write_results(make_payload(benchmark='rc_long_agreement',config=vars(args),results=rows),
                      output=str(args.output),csv_path=None)


if __name__ == '__main__':
    main()
