from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


def _build_fields_and_hamiltonian(*, v_forward: float):
    from state_prep.electric_fields import ElectricField, Ez_from_csv
    from state_prep.hamiltonians import SlowHamiltonian
    from state_prep.magnetic_fields import MagneticField
    from state_prep.trajectory import Trajectory

    trajectory = Trajectory(
        Rini=np.array((0, 0, -80e-3)), Vini=np.array((0, 0, float(v_forward))), zfin=200e-3
    )

    Ez = Ez_from_csv()

    def E_R(R):
        return np.array([0.0, 0.0, float(Ez(R[2]))])

    electric_field = ElectricField(E_R, trajectory.R_t)

    B0 = np.array((0.0, 0.0, 1e-3))

    def B_R(R):
        if len(np.shape(R)) == 1:
            return B0
        return B0.reshape((3, 1)) * np.ones(np.shape(R))

    magnetic_field = MagneticField(B_R, R_t=trajectory.R_t)

    Js = [0, 1, 2, 3]
    # SlowHamiltonian's type hints are looser than its runtime expectations
    # (it expects objects with .E_R/.B_R). Keep the benchmark script pragmatic.
    hamiltonian = SlowHamiltonian(
        Js,
        trajectory,
        cast(Any, electric_field),
        cast(Any, magnetic_field),
    )
    return trajectory, electric_field, magnetic_field, hamiltonian


def _build_microwaves(hamiltonian, trajectory):
    from state_prep.approximate_states import J1_triplet_0, J2_triplet_0
    from state_prep.intensity_profiles import BackgroundField, GaussianBeam
    from state_prep.microwaves import MicrowaveField, Polarization
    from state_prep.utils import calculate_transition_frequency

    state1 = J1_triplet_0
    state2 = J2_triplet_0

    R0 = np.array((0.00, 0.0, 0.0254 * 1.125))

    p_z = np.array([0.0, 0.0, 1.0])

    def P_R(R):
        return p_z / np.sqrt(np.sum(p_z**2))

    k = np.array((1.0, 0.0, 0.0))
    pol = Polarization(P_R, k, f_long=1)

    muW_freq_2 = calculate_transition_frequency(state1, state2, hamiltonian.H_R(R0), hamiltonian.QN)

    intensity = GaussianBeam(
        power=0.5e-3,
        sigma=25.4e-3 / (2 * np.sqrt(2 * np.log(2))),
        R0=R0,
        k=k,
        freq=muW_freq_2,
    )

    mf12 = MicrowaveField(1, 2, intensity, pol, muW_freq_2, QN=hamiltonian.QN)

    # Background field: start with some nominal intensity; we set the final intensity later
    lims = [
        (-1.0, 1.0),
        (-1.0, 1.0),
        (+25.4e-3 / (2 * np.sqrt(2 * np.log(2))), 1.0),
    ]
    intensity_bg = BackgroundField(cast(Any, lims), intensity=cast(Any, mf12.intensity).I_R(R0) / 20)
    pol_bg = Polarization(P_R)
    mf12_bg = MicrowaveField(1, 2, intensity_bg, pol_bg, muW_freq_2, hamiltonian.QN, background_field=True)

    return mf12, mf12_bg, muW_freq_2, R0


def _compute_EB_t(*, trajectory, electric_field, magnetic_field, t_array: np.ndarray):
    E_single = np.empty((t_array.size, 3), dtype=float)
    B_single = np.empty((t_array.size, 3), dtype=float)
    for i, t in enumerate(t_array):
        R = trajectory.R_t(float(t))
        E_single[i] = electric_field.E_R(R)
        B_single[i] = magnetic_field.B_R(R)

    return E_single, B_single


def _make_D_mu_diag_batch(*, microwave_fields, QN, detunings_hz_batched: np.ndarray) -> np.ndarray:
    """Rotating-frame diagonal shift, delegating to the CPU package implementation.

    This was a second copy of the construction in `Simulator.run_microwave_scan`.
    Sharing it is what makes the CPU/GPU comparison meaningful: if the two used
    different rotating-frame conventions, any disagreement would be attributed to
    the GPU kernels rather than to the setup.
    """
    from state_prep.microwaves import build_rotating_frame_shift

    d_mu_diag_batch, _, _ = build_rotating_frame_shift(
        list(microwave_fields), list(QN), detunings_hz_batched
    )
    return d_mu_diag_batch


def _microwave_coupling_matrices_np(microwave_fields, *, n: int):
    """Return Hu/Hl arrays with shape (M,3,n,n) as complex128."""

    Hu = np.empty((len(microwave_fields), 3, n, n), dtype=np.complex128)
    Hl = np.empty((len(microwave_fields), 3, n, n), dtype=np.complex128)

    for j, mw in enumerate(microwave_fields):
        for k in range(3):
            Hk = np.asarray(mw.H_list[k], dtype=np.complex128)
            Hu[j, k] = np.triu(Hk)
            Hl[j, k] = np.tril(Hk)

    return Hu, Hl


def _microwave_base_matrices_np(microwave_fields, *, trajectory, t0: float, n: int) -> np.ndarray:
    """Build per-field fixed microwave matrices assuming fixed polarization.

    Returns H_base_fields with shape (M,n,n) such that:
        H_mu_field(t) = amp(t) * H_base_fields[field]

    where amp(t) is the scalar envelope (proportional to E-field magnitude).
    """

    H_base = np.empty((len(microwave_fields), n, n), dtype=np.complex128)

    for j, mw in enumerate(microwave_fields):
        # Ensure internal callables exist
        mw.get_H_t_func(trajectory.R_t, mw.QN)

        p = np.asarray(mw.p_t(float(t0)), dtype=np.complex128)

        H_sum = np.zeros((n, n), dtype=np.complex128)
        for k in range(3):
            Hk = np.asarray(mw.H_list[k], dtype=np.complex128)
            H_sum += p[k] * np.triu(Hk) + np.conj(p[k]) * np.tril(Hk)

        H_base[j] = H_sum

    return H_base


def _microwave_amp_t_np(microwave_fields, *, trajectory, t_array: np.ndarray) -> np.ndarray:
    """Return amp_t with shape (T-1,M), assuming fixed polarization per field."""

    from state_prep.microwaves import XConstants

    for mw in microwave_fields:
        mw.get_H_t_func(trajectory.R_t, mw.QN)

    Tm1 = int(t_array.size - 1)
    M = len(microwave_fields)
    amp_t = np.empty((Tm1, M), dtype=np.complex128)

    for i, t in enumerate(t_array[:-1]):
        for j, mw in enumerate(microwave_fields):
            E = float(mw.E_t(float(t)))
            amp_t[i, j] = (2 * np.pi * XConstants.D_TlF * E / 2.0)

    return amp_t


@dataclass(frozen=True)
class BenchConfig:
    n_steps: int = 6000
    v_forward: float = 184.0
    detuning_lo_mhz: float = -2.0
    detuning_hi_mhz: float = 1.5
    n_detunings: int = 25
    power_spa2_w: float = 5e-5
    bg_fraction: float = 1 / 35


def main() -> None:
    from centrex_tlf import states
    from state_prep.approximate_states import (
        J1_singlet,
        J1_triplet_m,
        J1_triplet_0,
        J1_triplet_p,
        J2_singlet,
        J2_triplet_m,
        J2_triplet_p,
    )
    from state_prep.utils import find_max_overlap_idx
    from state_prep.simulator import Simulator

    from state_prep_gpu._cupy import cupy
    from state_prep_gpu.slow_hamiltonian import terms_from_centrex_tlf
    from state_prep_gpu.simulator import simulate_batched_scalar_microwaves_shared_slow

    cfg = BenchConfig()

    trajectory, electric_field, magnetic_field, hamiltonian = _build_fields_and_hamiltonian(
        v_forward=cfg.v_forward
    )
    mf12, mf12_bg, muW_freq_2, R0 = _build_microwaves(hamiltonian, trajectory)

    # Match notebook-style setup: set base power and base background intensity
    mf12.set_frequency(muW_freq_2)
    mf12_bg.set_frequency(muW_freq_2)
    mf12.set_power(cfg.power_spa2_w)
    cast(Any, mf12_bg.intensity).intensity = cast(Any, mf12.intensity).I_R(R0) * float(cfg.bg_fraction)

    microwave_fields = [mf12, mf12_bg]

    initial_states_approx = [J1_singlet, J1_triplet_m, J1_triplet_0, J1_triplet_p]

    monitor_states = [
        J2_singlet,
        states.CoupledBasisState(
            J=2,
            F1=3 / 2,
            F=1,
            mF=0,
            electronic_state=states.ElectronicState.X,
            I1=1 / 2,
            I2=1 / 2,
            Omega=0,
            P=1,
        ).transform_to_uncoupled(),
        J2_triplet_m,
        J2_triplet_p,
    ]

    simulator = cast(Any, Simulator)(
        trajectory,
        electric_field,
        magnetic_field,
        initial_states_approx,
        hamiltonian,
        microwave_fields,
    )

    detunings = np.linspace(cfg.detuning_lo_mhz, cfg.detuning_hi_mhz, cfg.n_detunings) * 1e6
    batch = int(detunings.size)

    # Two fields at same frequency => detuning duplicated
    detunings_hz_batched = np.column_stack([detunings, detunings])
    intensity_prefactors = np.ones((batch, 2), dtype=float)

    # CPU run (batched scan)
    t0 = time.perf_counter()
    cpu_res = simulator.run_microwave_scan(
        detunings_hz=detunings_hz_batched,
        intensity_prefactors=intensity_prefactors,
        N_steps=cfg.n_steps,
        monitor_states=monitor_states,
        progress=False,
    )
    t1 = time.perf_counter()

    assert cpu_res.V_ini is not None
    assert cpu_res.probabilities_final is not None

    # A representative observable (same as notebook uses): initial_state_idx=2 (J1_triplet_0)
    initial_state_idx = 2
    init_idx = find_max_overlap_idx(
        initial_states_approx[initial_state_idx].state_vector(simulator.hamiltonian.QN),
        cpu_res.V_ini,
    )
    cpu_prob_ini = cpu_res.probabilities_final[:, initial_state_idx, init_idx]

    # GPU run (same physics; compact microwave construction)
    cp = cupy()

    # Slow Hamiltonian term matrices
    import centrex_tlf

    H_uncoupled_x = centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(cast(Any, hamiltonian.QN))
    terms = terms_from_centrex_tlf(H_uncoupled_x, dtype=cp.complex128)

    # Time grid
    T_total = trajectory.get_T()
    t_array = np.linspace(0.0, float(T_total), int(cfg.n_steps))
    dt = float(t_array[1] - t_array[0])

    # Field arrays (shared across batch)
    E_t, B_t = _compute_EB_t(
        trajectory=trajectory,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        t_array=t_array,
    )

    # Initial state vectors in QN basis (same mapping as CPU)
    H0 = hamiltonian.get_H_t_func()(0.0)
    simulator.init_state_vecs(H0)
    psi0 = np.repeat(simulator.psis[None, :, :], batch, axis=0)

    # Monitor vectors in QN basis
    monitor_vecs = np.stack([s.state_vector(cast(Any, hamiltonian.QN)) for s in monitor_states], axis=0)

    coupling_scales = np.sqrt(intensity_prefactors)
    D_mu_diag_batch = _make_D_mu_diag_batch(
        microwave_fields=microwave_fields,
        QN=hamiltonian.QN,
        detunings_hz_batched=detunings_hz_batched,
    )

    n = len(hamiltonian.QN)
    H_mu_base_fields = _microwave_base_matrices_np(
        microwave_fields,
        trajectory=trajectory,
        t0=float(t_array[0]),
        n=n,
    )
    amp_t = _microwave_amp_t_np(microwave_fields, trajectory=trajectory, t_array=t_array)

    # Warmup + timed run
    out_gpu = simulate_batched_scalar_microwaves_shared_slow(
        terms=terms,
        E_t=E_t,
        B_t=B_t,
        psi0=psi0,
        dt=dt,
        H_mu_base_fields=H_mu_base_fields,
        amp_t=amp_t,
        coupling_scales=coupling_scales,
        D_mu_diag_batch=D_mu_diag_batch,
        monitor_state_vectors=monitor_vecs,
        return_final_probabilities=True,
    )
    cp.cuda.Stream.null.synchronize()

    t2 = time.perf_counter()
    out_gpu = simulate_batched_scalar_microwaves_shared_slow(
        terms=terms,
        E_t=E_t,
        B_t=B_t,
        psi0=psi0,
        dt=dt,
        H_mu_base_fields=H_mu_base_fields,
        amp_t=amp_t,
        coupling_scales=coupling_scales,
        D_mu_diag_batch=D_mu_diag_batch,
        monitor_state_vectors=monitor_vecs,
        return_final_probabilities=True,
    )
    cp.cuda.Stream.null.synchronize()
    t3 = time.perf_counter()

    probs_gpu = cp.asnumpy(out_gpu.probabilities_final)

    # Compare final probabilities (same tracked slow-eigenbasis semantics)
    max_diff = float(np.max(np.abs(cpu_res.probabilities_final - probs_gpu)))

    print("=== SPA2-like real simulation benchmark ===")
    print(f"Config: N_steps={cfg.n_steps}  batch(detunings)={batch}  n={n}")
    print(f"muW_freq_2: {muW_freq_2/1e9:.6f} GHz")
    print(f"CPU run_microwave_scan: {t1 - t0:.3f} s")
    print(f"GPU simulate_batched_scalar_microwaves_shared_slow: {t3 - t2:.3f} s")
    if (t3 - t2) > 0:
        print(f"Speedup: {(t1 - t0) / (t3 - t2):.2f}x")
    print(f"Max |Δprobabilities_final|: {max_diff:.3e}")
    print(f"CPU example depletion (1 - prob_ini) range: {(1 - cpu_prob_ini).min():.3f}..{(1 - cpu_prob_ini).max():.3f}")


if __name__ == "__main__":
    main()
