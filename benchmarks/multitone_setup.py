"""CPU-object bridge for the flat GPU multitone API, shared by scan runners."""
import numpy as np


def uses_multitone(fields):
    carriers = {}
    for field in fields:
        carriers.setdefault(field.Je, set()).add(round(float(field.muW_freq), 6))
    return any(len(values) > 1 for values in carriers.values())


def prepare_multitone_inputs(setup, detunings, powers, *, steps,
                             velocities=None, complex_dtype='complex128', device='cuda'):
    import centrex_tlf
    import torch
    from common import compute_eb_t, microwave_amp_t_np
    from state_prep import Simulator
    from state_prep.microwaves import build_rotating_frame_shift
    from state_prep_gpu.slow_hamiltonian import terms_from_centrex_tlf
    path, fields, qn = setup['trajectory'], setup['microwave_fields'], setup['hamiltonian'].QN
    detunings, powers = np.asarray(detunings), np.asarray(powers)
    if (steps < 2 or detunings.ndim != 2 or detunings.shape != powers.shape
            or detunings.shape[1] != len(fields) or not len(detunings)
            or not np.isfinite(detunings).all() or not np.isfinite(powers).all() or (powers < 0).any()):
        raise ValueError('Require steps >= 2 and matching finite (B,M) parameters with nonnegative powers')
    batch, n = len(detunings), len(qn)
    grid = np.linspace(0, path.get_T(), steps)
    mids = (grid[1:]+grid[:-1])/2
    e, b = compute_eb_t(trajectory=path, electric_field=setup['electric_field'],
                        magnetic_field=setup['magnetic_field'], t_array=np.r_[mids,grid[-1]])
    e0, b0 = compute_eb_t(trajectory=path, electric_field=setup['electric_field'],
                          magnetic_field=setup['magnetic_field'], t_array=np.array([0.]))
    amplitude = microwave_amp_t_np(fields, trajectory=path, t_array=np.r_[mids,grid[-1]]).real
    upper = []
    for j, field in enumerate(fields):
        component = field.get_H_t_components_func(path.R_t, qn)
        pol = np.asarray(field.p_t(float(mids[0])), dtype=complex)
        u = sum(pol[axis]*np.triu(field.H_list[axis]) for axis in range(3))
        l = sum(pol[axis].conj()*np.tril(field.H_list[axis]) for axis in range(3))
        if not np.allclose(l, u.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError('GPU multitone requires adjoint upper/lower components')
        for i, t in enumerate(mids):
            actual_u, actual_l = component(float(t))
            if not (np.allclose(actual_u, amplitude[i,j]*u, atol=1e-10, rtol=1e-12)
                    and np.allclose(actual_l, amplitude[i,j]*l, atol=1e-10, rtol=1e-12)):
                raise ValueError('GPU multitone scalar envelopes do not reproduce spatially changing polarization')
        upper.append(u)
    shifts = build_rotating_frame_shift(fields, qn, detunings, skip_repeated_manifolds=True)[0]
    frequencies = np.zeros_like(detunings, dtype=float)
    references = {}
    for j, field in enumerate(fields):
        ref = references.setdefault(field.Je, j)
        if np.isclose(field.muW_freq, fields[ref].muW_freq):
            if not np.allclose(detunings[:,j], detunings[:,ref]):
                raise ValueError('Same carrier fields require equal detunings')
        else:
            frequencies[:,j] = 2*np.pi*((float(field.muW_freq)+detunings[:,j])
                                        -(float(fields[ref].muW_freq)+detunings[:,ref]))
    sim = Simulator(path, setup['electric_field'], setup['magnetic_field'],
                    setup['initial_states'], setup['hamiltonian'], fields)
    sim.init_state_vecs(setup['hamiltonian'].get_H_t_func()(0.))
    dtype = getattr(torch, complex_dtype)
    terms = terms_from_centrex_tlf(centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(qn),
                                  dtype=dtype, device=device)
    def tensor(array, kind=dtype):
        return torch.as_tensor(np.asarray(array), dtype=kind, device=device)
    inputs = dict(terms=terms, E_t=tensor(e,torch.float64), B_t=tensor(b,torch.float64),
        E0=tensor(e0[0],torch.float64), B0=tensor(b0[0],torch.float64),
        psi0=tensor(np.repeat(sim.psis[None],batch,axis=0)), dt=grid[1]-grid[0],
        H_mu_upper_fields=tensor(upper), amp_t=tensor(amplitude),
        coupling_scales=tensor(np.sqrt(powers),torch.float64), D_mu_diag_batch=tensor(shifts),
        sample_times=tensor(mids,torch.float64), beat_angular_frequencies=tensor(frequencies,torch.float64))
    monitors = setup.get('monitor_states')
    if monitors:
        inputs['monitor_state_vectors'] = tensor(np.stack([s.state_vector(qn) for s in monitors]))
    if velocities is not None:
        inputs.update(forward_velocities=velocities, reference_forward_velocity=float(path.Vini[2]))
    return inputs


def make_multitone_torch_runner(setup, *, n_steps, detunings, intensity_prefactors,
        forward_velocities=None, complex_dtype='complex128', slow_eigh_chunk_size=None,
        rotating_eigh_chunk_size=8, max_eigh_matrices=256):
    import torch
    from state_prep_gpu import simulate_batched_multitone_shared_slow
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch is unavailable')
    inputs = prepare_multitone_inputs(setup,detunings,intensity_prefactors,steps=n_steps,
        velocities=forward_velocities,complex_dtype=complex_dtype)
    def run(store_every=None, profile_timings=None):
        return simulate_batched_multitone_shared_slow(**inputs, store_every=store_every,
            profile_timings=profile_timings,
            slow_eigh_chunk_size=128 if slow_eigh_chunk_size is None else slow_eigh_chunk_size,
            rotating_eigh_chunk_size=rotating_eigh_chunk_size, max_eigh_matrices=max_eigh_matrices)
    return run, lambda: torch.cuda.synchronize()
