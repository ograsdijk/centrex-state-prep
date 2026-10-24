"""Interleaved independent-velocity runs versus a shared spatial-grid ensemble."""
from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import numpy as np

from bench_gpu_spa2 import make_torch_runner
from common import make_payload, write_results
from paired import compare_paired, format_report
from profile_transverse_fields import field_variant
from state_prep import Simulator, Trajectory


def make_simulator(setup, speed):
    trajectory = setup['trajectory']
    path = Trajectory(trajectory.Rini.copy(), trajectory.Vini * (speed / trajectory.Vini[2]), trajectory.zfin)
    hamiltonian = copy(setup['hamiltonian'])
    hamiltonian.trajectory = path
    return Simulator(path, setup['electric_field'], setup['magnetic_field'],
                     setup['initial_states'], hamiltonian, setup['microwave_fields'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-steps', type=int, default=300)
    parser.add_argument('--velocities', type=int, default=8)
    parser.add_argument('--parameter-points', type=int, default=5)
    parser.add_argument('--field-case', choices=['aligned', 'varying_e', 'transverse_b'], default='varying_e')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a fresh output file')
    if min(args.n_steps, args.velocities, args.parameter_points) < 2:
        parser.error('steps, velocities and parameter-points must each be >= 2')
    setup = field_variant(args.field_case)
    speeds = np.linspace(140., 240., args.velocities)
    reference_speed = float(setup['trajectory'].Vini[2])
    det = np.repeat(np.linspace(-3e5, 3e5, args.parameter_points)[:, None], 2, axis=1)
    power = np.ones_like(det)
    det_batch = np.tile(det, (len(speeds), 1))
    power_batch = np.tile(power, (len(speeds), 1))
    speed_batch = np.repeat(speeds, len(det))
    _, basis = np.linalg.eigh(setup['hamiltonian'].get_H_t_func()(setup['trajectory'].get_T()))
    rows, outputs = [], {}
    for backend in ('cpu', 'gpu'):
        if (args.cpu or args.gpu) and not getattr(args, backend):
            continue
        if backend == 'cpu':
            simulators = [make_simulator(setup, speed) for speed in speeds]
            shared = make_simulator(setup, reference_speed)
            opts = dict(N_steps=args.n_steps, progress=False, block_eigh=False)
            def independent():
                return np.concatenate([sim.run_microwave_scan(detunings_hz=det,
                    intensity_prefactors=power, **opts).psis_final for sim in simulators])
            def ensemble():
                return shared.run_microwave_scan(detunings_hz=det_batch,
                    intensity_prefactors=power_batch, forward_velocities=speed_batch, **opts).psis_final
            sync = None
            as_numpy = np.asarray
        else:
            import torch
            torch.set_num_threads(1)
            runners = [make_torch_runner(setup, n_steps=args.n_steps, detunings=det,
                intensity_prefactors=power, forward_velocities=np.full(len(det), speed),
                block_eigh=False)[0] for speed in speeds]
            runner, sync = make_torch_runner(setup, n_steps=args.n_steps, detunings=det_batch,
                intensity_prefactors=power_batch, forward_velocities=speed_batch, block_eigh=False)
            def independent():
                return torch.cat([run().psi_final for run in runners])
            def ensemble():
                return runner().psi_final
            as_numpy = lambda tensor: tensor.cpu().numpy()
        def error(a, b):
            return float(np.max(np.abs(np.abs(as_numpy(a) @ basis.conj())**2
                                       - np.abs(as_numpy(b) @ basis.conj())**2)))
        print(f'{backend}: {args.parameter_points} microwave rows x {args.velocities} speeds', flush=True)
        pair = compare_paired(dict(independent=independent, ensemble=ensemble),
            warmup=args.warmup, repeat=args.repeat, sync=sync, check=error, tolerance=1e-5, progress=True)
        print(format_report(pair), flush=True)
        if not all(comparison.agrees for comparison in pair.comparisons):
            raise RuntimeError('Independent velocity results disagree')
        outputs[backend] = as_numpy(pair.results['ensemble'])
        rows.extend(dict(row, backend=backend) for row in pair.rows())
        rows.append(dict(case='rounds', backend=backend,
                         times_s={name: timing.times_s for name, timing in pair.timings.items()}))
        write_results(make_payload(benchmark='velocity_scan', config=vars(args), results=rows),
                      output=str(args.output), csv_path=None)
    if len(outputs) == 2:
        discrepancy = float(np.max(np.abs(np.abs(outputs['cpu'] @ basis.conj())**2
                                          - np.abs(outputs['gpu'] @ basis.conj())**2)))
        rows.append(dict(case='cpu_gpu_agreement', max_probability_error=discrepancy))
        if discrepancy > 1e-5:
            raise RuntimeError(f'CPU/GPU disagreement: {discrepancy}')
        print(f'CPU/GPU common-basis probability error: {discrepancy:.3g}', flush=True)
        write_results(make_payload(benchmark='velocity_scan', config=vars(args), results=rows),
                      output=str(args.output), csv_path=None)


if __name__ == '__main__':
    main()
