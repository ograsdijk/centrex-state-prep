"""Batched frozen propagation with explicit same-manifold microwave beats."""
from __future__ import annotations

from operator import index
from typing import Any, Optional

from ._torch import torch as get_torch
from ._reorder import reorder_evecs_batched
from .slow_hamiltonian import BatchedSlowHamiltonianTerms, slow_hamiltonian_batch
from .simulator import (BatchedSimulationOutput, _as_torch_complex, _normalize_dt,
                        _dt_for_step, _validate_store_every, _new_history, _store_history)


def simulate_batched_multitone_shared_slow(
    *, terms: BatchedSlowHamiltonianTerms, E_t: Any, B_t: Any,
    psi0: Any, dt: Any, H_mu_upper_fields: Any, amp_t: Any,
    coupling_scales: Any, D_mu_diag_batch: Any,
    beat_angular_frequencies: Any, sample_times: Any,
    E0: Optional[Any] = None, B0: Optional[Any] = None,
    forward_velocities: Optional[Any] = None,
    reference_forward_velocity: Optional[float] = None,
    monitor_state_vectors: Optional[Any] = None,
    return_final_probabilities: bool = True,
    store_every: Optional[int] = None,
    slow_eigh_chunk_size: int = 128,
    rotating_eigh_chunk_size: int = 8,
    max_eigh_matrices: int = 256,
    profile_timings: Optional[dict[str, list[float]]] = None,
    profile_step_stride: int = 25,
) -> BatchedSimulationOutput:
    """Evolve multitone points sharing one spatial path, with dense eigensolves.

    E_t/B_t have shape (T,3), psi0 (B,S,n), H_mu_upper_fields (M,n,n),
    coupling_scales (B,M), D_mu_diag_batch (B,n), and amp_t (T-1,M) or
    (T-1,B,M). Each lower component is the upper component's adjoint.
    Beat angular frequencies (B,M) are in rad/s relative to the selected
    rotating-frame carrier; zero marks a static field. sample_times are
    seconds, shape (T-1,) or (T-1,B), at the actual Hamiltonian samples.
    Their origin must match the beat phase convention (normally entrance t=0).

    Optional forward velocities scale reference dt and sample_times by
    reference_forward_velocity / speed, preserving the supplied spatial path.
    The caller supplies the correct frame shifts; the function adds no Doppler
    correction. Different velocities retain distinct full eigensolves.

    Slow eigensystems and full microwave eigensystems are built in bounded
    time tiles. Each solver call contains at most max_eigh_matrices matrices
    (scan tiles are used when needed), while state propagation and label
    tracking remain sequential. Only the current tiles are retained. No
    cross-call cache, block approximation, or Magnus approximation is used.
    The final E_t/B_t sample is not propagated; E0/B0 specify true initial fields.

    Profiling requires CUDA. Tile build/solve intervals are amortized over
    covered steps; tracking/update intervals use profile_step_stride.
    """
    torch = get_torch()
    device, dtype = terms.Hff.device, terms.Hff.dtype
    for name, value in (('slow_eigh_chunk_size', slow_eigh_chunk_size),
                        ('rotating_eigh_chunk_size', rotating_eigh_chunk_size),
                        ('max_eigh_matrices', max_eigh_matrices),
                        ('profile_step_stride', profile_step_stride)):
        try:
            valid = not isinstance(value, bool) and index(value) > 0
        except TypeError:
            valid = False
        if not valid:
            raise ValueError(f'{name} must be a positive integer')
    slow_eigh_chunk_size = min(index(slow_eigh_chunk_size), index(max_eigh_matrices))
    rotating_eigh_chunk_size = min(index(rotating_eigh_chunk_size), index(max_eigh_matrices))
    max_eigh_matrices = index(max_eigh_matrices)
    store_every = _validate_store_every(store_every)
    e, b = torch.as_tensor(E_t, device=device), torch.as_tensor(B_t, device=device)
    psi = _as_torch_complex(psi0, dtype=dtype, device=device)
    if e.ndim != 2 or e.shape[-1] != 3 or e.shape != b.shape or len(e) < 2:
        raise ValueError('E_t/B_t must have matching shape (T,3), T >= 2')
    if psi.ndim != 3 or psi.shape[-1] != terms.n or psi.shape[0] < 1:
        raise ValueError('psi0 must have nonempty shape (B,S,n)')
    T, batch, n = len(e), psi.shape[0], terms.n
    upper = _as_torch_complex(H_mu_upper_fields, dtype=dtype, device=device)
    if upper.ndim != 3 or upper.shape[1:] != (n,n) or len(upper) < 1:
        raise ValueError('H_mu_upper_fields must have shape (M,n,n), M >= 1')
    M = len(upper)
    amp = _as_torch_complex(amp_t, dtype=dtype, device=device)
    scales = torch.as_tensor(coupling_scales, dtype=torch.float64, device=device)
    shifts = _as_torch_complex(D_mu_diag_batch, dtype=dtype, device=device)
    frequencies = torch.as_tensor(beat_angular_frequencies, dtype=torch.float64, device=device)
    times = torch.as_tensor(sample_times, dtype=torch.float64, device=device)
    if amp.shape not in ((T-1,M), (T-1,batch,M)):
        raise ValueError('amp_t must have shape (T-1,M) or (T-1,B,M)')
    if scales.shape != (batch,M) or frequencies.shape != (batch,M) or shifts.shape != (batch,n):
        raise ValueError('coupling_scales/beat frequencies must be (B,M), shifts (B,n)')
    if times.shape not in ((T-1,), (T-1,batch)) or not bool(torch.isfinite(times).all()):
        raise ValueError('sample_times must be finite with shape (T-1,) or (T-1,B)')
    if not bool(torch.isfinite(frequencies).all()):
        raise ValueError('beat_angular_frequencies must be finite')
    if not bool((shifts.imag == 0).all()):
        raise ValueError('D_mu_diag_batch must be real for a Hermitian Hamiltonian')
    dt_arr = _normalize_dt(torch, dt, T, batch, device)
    if not bool((torch.isfinite(dt_arr) & (dt_arr > 0)).all()):
        raise ValueError('dt must be finite and positive')
    traversal = torch.ones(batch, dtype=torch.float64, device=device)
    if forward_velocities is not None:
        velocities = torch.as_tensor(forward_velocities, dtype=torch.float64, device=device)
        reference = torch.as_tensor(reference_forward_velocity if reference_forward_velocity is not None else float('nan'),
                                    dtype=torch.float64, device=device)
        if (velocities.ndim > 1 or velocities.numel() not in (1,batch)
                or not bool((torch.isfinite(velocities) & (velocities > 0)).all())):
            raise ValueError('forward_velocities must be finite positive scalar or (B,)')
        if reference.ndim != 0 or not bool(torch.isfinite(reference) & (reference > 0)):
            raise ValueError('reference_forward_velocity must be finite and positive')
        traversal = reference / velocities.reshape(-1).expand(batch)
        dt_arr = (dt_arr[:,None] if dt_arr.ndim == 1 else dt_arr) * traversal[None]
    elif reference_forward_velocity is not None:
        raise ValueError('reference_forward_velocity requires forward_velocities')
    initial_e = e[0] if E0 is None else torch.as_tensor(E0, device=device)
    initial_b = b[0] if B0 is None else torch.as_tensor(B0, device=device)
    if initial_e.shape != (3,) or initial_b.shape != (3,):
        raise ValueError('E0/B0 must have shape (3,)')
    _, reference_vectors = torch.linalg.eigh(slow_hamiltonian_batch(terms, initial_e, initial_b))
    monitor_indices = None
    if monitor_state_vectors is not None:
        monitors = _as_torch_complex(monitor_state_vectors, dtype=dtype, device=device)
        if monitors.ndim != 2 or monitors.shape[-1] != n:
            raise ValueError('monitor_state_vectors must have shape (K,n)')
        monitor_indices = torch.abs(reference_vectors.mH @ monitors.T).argmax(dim=0)
    if profile_timings is not None and device.type != 'cuda':
        raise ValueError('profile_timings requires CUDA tensors')
    events = []
    def begin():
        if profile_timings is None:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device))
        return event
    def finish(name, start, width=1):
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(device))
            events.append((name,start,end,width))
    history, history_indices, cursor = _new_history(torch, psi, T, store_every)
    diagonal = torch.arange(n, device=device)
    for slow_start in range(0, T-1, slow_eigh_chunk_size):
        slow_end = min(slow_start+slow_eigh_chunk_size, T-1)
        stamp = begin()
        slow_values, slow_vectors = torch.linalg.eigh(
            slow_hamiltonian_batch(terms, e[slow_start:slow_end], b[slow_start:slow_end]))
        rotated_upper = slow_vectors.mH[:,None] @ (upper[None] @ slow_vectors[:,None])
        finish('slow_eigh_and_microwave_rotate', stamp, slow_end-slow_start)
        for tile_start in range(slow_start, slow_end, rotating_eigh_chunk_size):
            tile_end = min(tile_start+rotating_eigh_chunk_size, slow_end)
            width = tile_end-tile_start
            local = tile_start-slow_start
            stamp = begin()
            rot_values = torch.empty((width,batch,n), dtype=psi.real.dtype, device=device)
            rot_vectors = torch.empty((width,batch,n,n), dtype=dtype, device=device)
            # Cap workspace independently of total scan size; retain just the
            # eigensystems of this bounded time tile for sequential evolution.
            scan_width = max(1, max_eigh_matrices // width)
            for point_start in range(0, batch, scan_width):
                point_end = min(point_start+scan_width, batch)
                physical_times = times[tile_start:tile_end]
                physical_times = (physical_times[:,None] if times.ndim == 1
                                  else physical_times[:,point_start:point_end])
                phase = physical_times[:,:,None] * frequencies[None,point_start:point_end]
                phase = phase * traversal[None,point_start:point_end,None]
                envelope = amp[tile_start:tile_end]
                envelope = envelope[:,None] if amp.ndim == 2 else envelope[:,point_start:point_end]
                weights = scales[None,point_start:point_end] * envelope * torch.exp(-1j*phase)
                matrix = torch.einsum('tbm,tmij->tbij', weights.to(dtype),
                                      rotated_upper[local:local+width])
                matrix = matrix + matrix.mH
                matrix[:,:,diagonal,diagonal] += slow_values[local:local+width,None]
                matrix[:,:,diagonal,diagonal] += shifts[None,point_start:point_end]
                values, vectors = torch.linalg.eigh(matrix.reshape(-1,n,n))
                rot_values[:,point_start:point_end] = values.reshape(width,-1,n)
                rot_vectors[:,point_start:point_end] = vectors.reshape(width,-1,n,n)
                del matrix, values, vectors, weights
            finish('microwave_build_and_eigh', stamp, width)
            for offset in range(width):
                step = tile_start+offset
                k = local+offset
                v, d = slow_vectors[k], slow_values[k]
                stamp = begin() if step % profile_step_stride == 0 else None
                _, tracked = reorder_evecs_batched(v[None], d[None], reference_vectors[None])
                reference_vectors = tracked[0]
                state = psi @ v.conj()
                state = state @ rot_vectors[offset].conj()
                phases = torch.exp(-1j*rot_values[offset]*_dt_for_step(dt_arr,step))
                state *= phases[:,None]
                psi = (state @ rot_vectors[offset].mT) @ v.T
                cursor = _store_history(history, history_indices, cursor, step+1, psi)
                finish('tracking_and_state_update', stamp)
            # Release previous-tile views before constructing the next tile.
            del rot_values, rot_vectors, phases, state, v, d
        del slow_values, slow_vectors, rotated_upper
    if profile_timings is not None:
        torch.cuda.synchronize(device)
        profile_timings.clear()
        for name,start,end,width in events:
            profile_timings.setdefault(name, []).extend([start.elapsed_time(end)/(1000*width)]*width)
    probabilities = monitors_final = None
    if return_final_probabilities or monitor_indices is not None:
        amplitudes = psi @ reference_vectors.conj()
        if return_final_probabilities:
            probabilities = amplitudes.abs()**2
        if monitor_indices is not None:
            monitors_final = amplitudes[:,:,monitor_indices].abs()**2
    return BatchedSimulationOutput(psi_final=psi, probabilities_final=probabilities,
        monitor_probabilities_final=monitors_final, last_evecs=reference_vectors,
        psi_history=history, history_indices=None if history_indices is None else
        torch.tensor(history_indices,dtype=torch.int64,device=device))
