"""Microwave and field inputs for the GPU solvers, built from `state_prep` objects (numpy only).

The GPU entry points take plain arrays: per-field coupling matrices with time envelopes
(`simulate_batched`, `simulate_batched_multitone_shared_slow`) and sampled E/B fields. These
functions build them from a `state_prep` trajectory, fields and `MicrowaveField`s, exactly at the
propagated sample times. Importing this module does not require PyTorch.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "compute_eb_t",
    "microwave_base_matrices_np",
    "microwave_amp_t_np",
    "microwave_factors_np",
    "microwave_upper_factors_np",
]


def compute_eb_t(*, trajectory: Any, electric_field: Any, magnetic_field: Any, t_array: np.ndarray):
    e_t = np.empty((t_array.size, 3), dtype=float)
    b_t = np.empty((t_array.size, 3), dtype=float)
    for idx, t in enumerate(t_array):
        position = trajectory.R_t(float(t))
        e_t[idx] = electric_field.E_R(position)
        b_t[idx] = magnetic_field.B_R(position)
    return e_t, b_t


def microwave_base_matrices_np(
    microwave_fields: Sequence[Any],
    *,
    trajectory: Any,
    t0: float,
    n: int,
) -> np.ndarray:
    h_base = np.empty((len(microwave_fields), n, n), dtype=np.complex128)
    for field_idx, field in enumerate(microwave_fields):
        field.get_H_t_func(trajectory.R_t, field.QN)
        polarization = np.asarray(field.p_t(float(t0)), dtype=np.complex128)
        h_sum = np.zeros((n, n), dtype=np.complex128)
        for axis in range(3):
            h_axis = np.asarray(field.H_list[axis], dtype=np.complex128)
            h_sum += polarization[axis] * np.triu(h_axis)
            h_sum += np.conj(polarization[axis]) * np.tril(h_axis)
        h_base[field_idx] = h_sum
    return h_base


def microwave_amp_t_np(
    microwave_fields: Sequence[Any],
    *,
    trajectory: Any,
    t_array: np.ndarray,
) -> np.ndarray:
    from state_prep.microwaves import XConstants

    for field in microwave_fields:
        field.get_H_t_func(trajectory.R_t, field.QN)

    amp_t = np.empty((t_array.size - 1, len(microwave_fields)), dtype=np.complex128)
    for time_idx, t in enumerate(t_array[:-1]):
        for field_idx, field in enumerate(microwave_fields):
            e_field = float(field.E_t(float(t)))
            amp_t[time_idx, field_idx] = 2 * np.pi * XConstants.D_TlF * e_field / 2.0
    return amp_t


def microwave_factors_np(
    microwave_fields: Sequence[Any],
    *,
    trajectory: Any,
    t_array: np.ndarray,
    n: int,
    rtol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar-envelope factorization of each field, exact at every propagated sample.

    `MicrowaveField.get_H_t_func` gives
    `H(t) = pref E(t) sum_k [p_k(t) triu(H_k) + conj(p_k(t)) tril(H_k)]`.
    A field whose polarization is constant over `t_array[:-1]` keeps a single base
    matrix with envelope `pref E(t)`, identical to `microwave_base_matrices_np` and
    `microwave_amp_t_np`. A field whose polarization varies (switching, position
    dependence) is split into up to six fixed parts, `triu(H_k)` and `tril(H_k)`,
    with complex envelopes `pref E(t) p_k(t)` and its conjugate. Parts whose matrix or
    envelope is identically zero are dropped.

    Returns `(base, amp, field_index)`: base `(K, n, n)`, amp `(T-1, K)` and the
    source field of each part `(K,)`, so per-field coupling scales expand as
    `scales[:, field_index]`.
    """
    from state_prep.microwaves import XConstants

    samples = np.asarray(t_array, dtype=float)[:-1]
    bases, amps, owners = [], [], []
    for field_idx, field in enumerate(microwave_fields):
        field.get_H_t_func(trajectory.R_t, field.QN)
        pref = np.array(
            [2 * np.pi * XConstants.D_TlF * float(field.E_t(float(t))) / 2.0 for t in samples]
        )
        pol = np.array([np.asarray(field.p_t(float(t)), dtype=np.complex128) for t in samples])
        h_axes = [np.asarray(h, dtype=np.complex128) for h in field.H_list]
        if np.abs(pol - pol[0]).max() <= rtol * max(np.abs(pol[0]).max(), 1e-300):
            h_sum = np.zeros((n, n), dtype=np.complex128)
            for axis in range(3):
                h_sum += pol[0, axis] * np.triu(h_axes[axis])
                h_sum += np.conj(pol[0, axis]) * np.tril(h_axes[axis])
            bases.append(h_sum)
            amps.append(pref.astype(np.complex128))
            owners.append(field_idx)
            continue
        for axis in range(3):
            envelope = pref * pol[:, axis]
            for part, env in ((np.triu(h_axes[axis]), envelope), (np.tril(h_axes[axis]), envelope.conj())):
                if not part.any() or not env.any():
                    continue
                bases.append(part)
                amps.append(env)
                owners.append(field_idx)
    return np.stack(bases), np.stack(amps, axis=1), np.asarray(owners, dtype=np.int64)


def microwave_upper_factors_np(
    microwave_fields: Sequence[Any],
    *,
    trajectory: Any,
    t_array: np.ndarray,
    n: int,
    rtol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle factorization for the GPU multitone API.

    `simulate_batched_multitone_shared_slow` takes upper components and adds
    their adjoints, `w U + conj(w) U^H`. Since `tril(H_k) = triu(H_k)^H` for the
    Hermitian coupling matrices, a field is reproduced exactly by:

    - constant polarization: one part `sum_k p_k triu(H_k)` with envelope `pref E(t)`;
    - varying polarization: up to three parts `triu(H_k)` with complex envelopes
      `pref E(t) p_k(t)`.

    Returns `(upper, amp, field_index)`: upper `(K, n, n)`, amp `(T-1, K)` complex
    and the source field of each part, so per-field coupling scales and beat
    frequencies expand as `x[:, field_index]`.
    """
    from state_prep.microwaves import XConstants

    samples = np.asarray(t_array, dtype=float)[:-1]
    uppers, amps, owners = [], [], []
    for field_idx, field in enumerate(microwave_fields):
        field.get_H_t_func(trajectory.R_t, field.QN)
        pref = np.array(
            [2 * np.pi * XConstants.D_TlF * float(field.E_t(float(t))) / 2.0 for t in samples]
        )
        pol = np.array([np.asarray(field.p_t(float(t)), dtype=np.complex128) for t in samples])
        h_axes = [np.asarray(h, dtype=np.complex128) for h in field.H_list]
        if not all(np.allclose(np.tril(h), np.triu(h).conj().T, atol=0, rtol=rtol) for h in h_axes):
            raise ValueError("coupling matrices must be Hermitian for the upper factorization")
        if np.abs(pol - pol[0]).max() <= rtol * max(np.abs(pol[0]).max(), 1e-300):
            uppers.append(sum(pol[0, axis] * np.triu(h_axes[axis]) for axis in range(3)))
            amps.append(pref.astype(np.complex128))
            owners.append(field_idx)
            continue
        for axis in range(3):
            envelope = pref * pol[:, axis]
            if not h_axes[axis].any() or not envelope.any():
                continue
            uppers.append(np.triu(h_axes[axis]))
            amps.append(envelope)
            owners.append(field_idx)
    return np.stack(uppers), np.stack(amps, axis=1), np.asarray(owners, dtype=np.int64)
