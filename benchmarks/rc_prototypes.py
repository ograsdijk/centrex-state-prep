"""Benchmark-only multitone optimizations; preserve the reference beat phases."""
from pathlib import Path
import sys
import subprocess

from velocity_prototypes import clone, replace, gpu_variant


def cpu_variant(*, components=False, kernels=False):
    import state_prep.simulator as module
    source = subprocess.check_output(['git', 'show', '0d6e981:src/state_prep/simulator.py'],
                                     cwd=Path(__file__).resolve().parents[1], text=True)
    start = source.index('    def _time_evolve_mu_batched_shared_slow_multitone(')
    end = source.index('\n    def ', start+5)
    method = source[start:end]
    if components:
        method = replace(method, '                lower_rot.append(Vh @ lower @ V)', '''                lower_rot.append(upper_rot[-1].conj().T if np.array_equal(lower, upper.conj().T)
                                 else Vh @ lower @ V)''')
        method = replace(method, '            beat_phases = [', '''            static_rot = {field: upper_rot[field] + lower_rot[field]
                          for field in static_field_indices}
            beat_phases = [''')
        method = method.replace('(upper_rot[field_idx] + lower_rot[field_idx])', 'static_rot[field_idx]')
        method = replace(method, '''                        upper_rot[field_idx] + lower_rot[field_idx]
''', '''                        static_rot[field_idx]
''')
    if kernels:
        method = replace(method, '        last_evecs = V_ref', '''        keys = [D_mu_diag_batch, velocity_scales[:, None]]
        keys.extend(frequency[:, None] for _, frequency in beat_fields)
        _, kernel_groups = np.unique(np.concatenate(keys, axis=1), axis=0, return_inverse=True)
        last_evecs = V_ref''')
        method = replace(method, '            for b in range(batch):', '''            integrated_fields = {}
            for b in range(batch):''')
        a = method.index('                    delta = D + D_mu_diag_batch[b]')
        b = method.index('                    psis_slow[b] = apply_magnus_taylor', a)
        method = method[:a] + '''                    group = int(kernel_groups[b])
                    if group not in integrated_fields:
                        delta = D + D_mu_diag_batch[b]
                        e_delta = np.exp(1j * delta * dt_point)
                        outer = np.outer(e_delta, e_delta.conj())
                        gaps = delta[:, None] - delta[None, :]
                        fields = []
                        if static_field_indices:
                            static_kernel = _magnus_integral_from_outer(outer, gaps, dt_point)
                            for field_idx in static_field_indices:
                                fields.append((field_idx, (upper_rot[field_idx] + lower_rot[field_idx]) * static_kernel))
                        for field_idx, delta_omega_col in beat_fields:
                            w = float(delta_omega_col[b])
                            beat = np.exp(-1j * w * t0_step * velocity_scales[b])
                            fields.append((field_idx,
                                beat * upper_rot[field_idx] * _magnus_integral_from_outer(outer, gaps, dt_point, -w)
                                + beat.conjugate() * lower_rot[field_idx] * _magnus_integral_from_outer(outer, gaps, dt_point, w)))
                        integrated_fields[group] = delta, fields
                    delta, fields = integrated_fields[group]
                    A = np.zeros((n, n), dtype=np.complex128)
                    for field_idx, field in fields:
                        A += coupling_scales[b, field_idx] * field
''' + method[b:]
    source = source[:start] + method + source[end:]
    return clone(module, f'rc_cpu_{components}_{kernels}', source).Simulator


def gpu_multitone(*, full_chunk=0):
    """Use Hermitian X/Y quadratures, with weights generated in bounded tiles.

    Bases are [X_0,...X_M,Y_0,...Y_M], where X=upper+lower and
    Y=-i(upper-lower). cos(phi)*X+sin(phi)*Y is exactly the CPU
    exp(-i*phi)*upper+exp(+i*phi)*lower Hamiltonian.
    """
    import state_prep_gpu.simulator as module
    function = gpu_variant(full_chunk=full_chunk)
    source = sys.modules[function.__module__].__source__
    start = source.index('def simulate_batched_scalar_microwaves_shared_slow(')
    end = source.index('\ndef _as_torch_complex(', start)
    method = source[start:end]
    method = replace(method, '    reference_forward_velocity: Optional[float] = None,', '''    reference_forward_velocity: Optional[float] = None,
    beat_angular_frequencies: Any = None,
    sample_times: Any = None,''')
    method = replace(method, '    solve_indices = solve_inverse = None', '''    frequencies = torch.as_tensor(beat_angular_frequencies, dtype=torch.float64, device=psi.device)
    times = torch.as_tensor(sample_times, dtype=torch.float64, device=psi.device)
    if M % 2 or frequencies.shape != (batch, M // 2) or times.shape != (T - 1,):
        raise ValueError("Quadrature bases/frequencies/sample times have incompatible shapes")
    rates = frequencies * (scales[:, None] if forward_velocities is not None else 1)
    solve_indices = solve_inverse = None''')
    method = replace(method, '        features = [coupling_scales, shift_features]',
                     '        features = [coupling_scales, shift_features, rates]')
    if full_chunk:
        method = replace(method, '            shifts = D_mu_diag_batch', '''            phases = times[i:i + tile_end-offset, None, None] * rates[None]
            weights = weights * torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
            shifts = D_mu_diag_batch''')
    else:
        method = replace(method, '        weights = coupling_scales * amp_i', '''        phases = times[i] * rates
        weights = coupling_scales * amp_i * torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)''')
    source = source[:start] + method + source[end:]
    return clone(module, f'rc_gpu_{full_chunk}', source).simulate_batched_scalar_microwaves_shared_slow


def make_gpu_runner(setup, detunings, powers, *, steps, full_chunk=0, velocities=None):
    import centrex_tlf
    import numpy as np
    import torch
    from common import compute_eb_t, microwave_amp_t_np
    from state_prep import Simulator
    from state_prep.microwaves import build_rotating_frame_shift
    from state_prep_gpu.slow_hamiltonian import terms_from_centrex_tlf
    function = gpu_multitone(full_chunk=full_chunk)
    path = setup['trajectory']
    fields = setup['microwave_fields']
    qn = setup['hamiltonian'].QN
    batch, n = len(detunings), len(qn)
    grid = np.linspace(0, path.get_T(), steps)
    mids = (grid[1:] + grid[:-1]) / 2
    sampled = np.r_[mids, grid[-1]]
    e, b = compute_eb_t(trajectory=path, electric_field=setup['electric_field'],
                        magnetic_field=setup['magnetic_field'], t_array=sampled)
    e0, b0 = compute_eb_t(trajectory=path, electric_field=setup['electric_field'],
                          magnetic_field=setup['magnetic_field'], t_array=np.array([0.]))
    amplitude = microwave_amp_t_np(fields, trajectory=path, t_array=sampled).real
    upper = []
    lower = []
    from state_prep.microwaves import XConstants
    for j, field in enumerate(fields):
        component = field.get_H_t_components_func(path.R_t, qn)
        pol = np.asarray(field.p_t(float(mids[0])), dtype=complex)
        u = sum(pol[axis]*np.triu(field.H_list[axis]) for axis in range(3))
        l = sum(pol[axis].conj()*np.tril(field.H_list[axis]) for axis in range(3))
        if not np.allclose(l, u.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError('Nonadjoint microwave components')
        # Verify factorization at every propagated spatial sample, including
        # boundaries where a background envelope switches on or off.
        for i, t in enumerate(mids):
            actual_u, actual_l = component(float(t))
            if not (np.allclose(actual_u, amplitude[i, j]*u, atol=1e-10, rtol=1e-12)
                    and np.allclose(actual_l, amplitude[i, j]*l, atol=1e-10, rtol=1e-12)):
                raise ValueError('Spatially changing polarization cannot use scalar quadratures')
        upper.append(u)
        lower.append(l)
    upper, lower = np.asarray(upper), np.asarray(lower)
    bases = np.concatenate((upper+lower, -1j*(upper-lower)))
    shifts = build_rotating_frame_shift(fields, qn, detunings, skip_repeated_manifolds=True)[0]
    frequencies = np.zeros_like(detunings)
    reference_for_manifold = {}
    for j, field in enumerate(fields):
        ref = reference_for_manifold.setdefault(field.Je, j)
        if np.isclose(field.muW_freq, fields[ref].muW_freq):
            if not np.allclose(detunings[:, j], detunings[:, ref]):
                raise ValueError('Same carrier fields require equal detunings')
        else:
            frequencies[:, j] = 2*np.pi*((field.muW_freq+detunings[:, j])
                                        -(fields[ref].muW_freq+detunings[:, ref]))
    sim = Simulator(path, setup['electric_field'], setup['magnetic_field'],
                    setup['initial_states'], setup['hamiltonian'], fields)
    sim.init_state_vecs(setup['hamiltonian'].get_H_t_func()(0.))
    terms = terms_from_centrex_tlf(centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(qn),
                                  dtype=torch.complex128, device='cuda')
    def tensor(array, dtype=torch.complex128):
        return torch.as_tensor(array, dtype=dtype, device='cuda')
    options = dict(terms=terms, E_t=tensor(e, torch.float64), B_t=tensor(b, torch.float64),
        E0=tensor(e0[0], torch.float64), B0=tensor(b0[0], torch.float64),
        psi0=tensor(np.repeat(sim.psis[None], batch, axis=0)), dt=grid[1]-grid[0],
        H_mu_base_fields=tensor(bases), amp_t=tensor(np.tile(amplitude, (1, 2))),
        coupling_scales=tensor(np.tile(np.sqrt(powers), (1, 2)), torch.float64),
        D_mu_diag_batch=tensor(shifts), sample_times=tensor(mids, torch.float64),
        beat_angular_frequencies=tensor(frequencies, torch.float64), block_eigh=False,
        slow_eigh_chunk_size=128,
        monitor_state_vectors=tensor(np.stack([s.state_vector(qn) for s in setup['monitor_states']])))
    if velocities is not None:
        options.update(forward_velocities=velocities, reference_forward_velocity=float(path.Vini[2]))
    return lambda: function(**options), lambda: torch.cuda.synchronize()
