"""Multitone beat sign: the same drive described in two rotating frames.

Frame A puts the carrier on the tone, so the field is static. Frame B puts the
carrier `w` lower: the tone then beats at `+w` against it and the higher level's
frame shift grows by `+w`. Populations are frame independent, so both must agree.

This is the test the CPU/GPU agreement suite could not be: both backends shared
the opposite sign (`exp(-i*w*t)` on `<lower|H|higher>`), which in frame B moves
the drive `2w` off resonance. Each case also checks that the old sign fails, so
the comparison has teeth.
"""
import numpy as np
import pytest

import analytic_models as am

# Higher level at E, tone at E - DETUNING; W is the frame-B carrier offset.
# Frozen is second order in the frame-B diagonal (~DETUNING + W) against the
# rotating coupling: 8e-4 at 40k steps, 3e-3 at 20k; Magnus is exact here.
E, DETUNING, RABI, W = 3.0e3, 1.0e2, 4.0e2, 5.0e2


def _run_cpu(propagator, beat, frame_offset, n_steps=40_000):
    from state_prep.simulator import Simulator

    model = am.rotating_coupling()  # Only for the stub: n=2, T=1, psi0=|lower>.
    upper = 0.5 * RABI * am.SIGMA_PLUS
    lower = upper.conj().T
    shift = -(E - DETUNING) + frame_offset
    psis, *_ = Simulator._time_evolve_mu_batched_shared_slow_multitone(
        am.stub_simulator(model),
        H_slow_t=lambda t: np.diag([0.0, E]).astype(complex),
        component_hams=[lambda t: (upper, lower)],
        coupling_scales=np.ones((1, 1)),
        D_mu_diag_batch=np.array([[0.0, shift]]),
        t_array=np.linspace(0.0, model.T, n_steps),
        static_field_indices=[] if beat else [0],
        beat_fields=[(0, np.array([beat]))] if beat else [],
        monitor_states=None,
        store_final_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend="zheevd",
        time_sampling="mid",
        propagator=propagator,
    )
    return np.abs(psis[0]) ** 2


@pytest.mark.parametrize("propagator", ["frozen", "magnus"])
def test_cpu_multitone_beat_is_frame_invariant(propagator):
    static = _run_cpu(propagator, 0.0, 0.0)
    beating = _run_cpu(propagator, W, W)
    old_sign = _run_cpu(propagator, -W, W)
    assert static[0, 1] > 0.1, "drive must move population"
    assert np.abs(beating - static).max() < 2e-3
    assert np.abs(old_sign - static).max() > 0.1


def _gpu_options(device, beat, frame_offset):
    torch = pytest.importorskip("torch")
    from state_prep_gpu.slow_hamiltonian import BatchedSlowHamiltonianTerms

    rng = np.random.default_rng(17)
    n, low, batch, T = 5, 2, 2, 401
    def tensor(a, kind=torch.complex128):
        return torch.as_tensor(a, dtype=kind, device=device)
    # Lower manifold {0,1} near 0 Hz, higher {2,3,4} near 3 Hz; fields couple only
    # lower -> higher, as one rotational transition does.
    hff = np.diag([0.0, 0.05, 3.0, 3.04, 3.1]).astype(complex)
    hsz = np.diag([0.0, 0.02, 0.0, -0.03, 0.03]).astype(complex)
    zero = np.zeros_like(hff)
    terms = BatchedSlowHamiltonianTerms(*map(tensor, (hff, zero, zero, hsz, zero, zero, zero)))
    upper = np.zeros((2, n, n), complex)
    upper[:, :low, low:] = 3.0 * (rng.normal(size=(2, low, n - low))
                                  + 1j * rng.normal(size=(2, low, n - low)))
    higher = np.r_[np.zeros(low), np.ones(n - low)]
    shifts = -2 * np.pi * 2.9 * higher + frame_offset * higher
    dt = np.full(T - 1, 1e-3)
    grid = np.r_[0.0, np.cumsum(dt)]
    e = np.zeros((T, 3))
    e[:, 2] = np.linspace(-0.5, 0.5, T)
    psi = np.zeros((batch, 1, n), complex)
    psi[:, 0, 0] = 1.0
    return dict(terms=terms, E_t=tensor(e, torch.float64), B_t=tensor(np.zeros_like(e), torch.float64),
        E0=tensor([0.0, 0.0, -0.5], torch.float64), B0=tensor(np.zeros(3), torch.float64),
        psi0=tensor(psi), dt=tensor(dt, torch.float64), H_mu_upper_fields=tensor(upper),
        amp_t=tensor(np.ones((T - 1, 2))), coupling_scales=tensor([[1.0, 0.5], [0.7, 1.0]], torch.float64),
        D_mu_diag_batch=tensor(np.tile(shifts, (batch, 1))),
        sample_times=tensor((grid[:-1] + grid[1:]) / 2, torch.float64),
        beat_angular_frequencies=tensor(np.full((batch, 2), beat), torch.float64),
        forward_velocities=[170.0, 200.0], reference_forward_velocity=184.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gpu_multitone_beat_is_frame_invariant(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from state_prep_gpu import simulate_batched_multitone_shared_slow as simulate

    def populations(beat, frame_offset):
        out = simulate(**_gpu_options(device, beat, frame_offset), return_final_probabilities=False)
        return out.psi_final.abs().square().cpu().numpy()

    w = 2 * np.pi * 5.0
    static, beating, old_sign = populations(0.0, 0.0), populations(w, w), populations(-w, w)
    assert static[..., 2:].sum(-1).min() > 0.1, "drive must move population"
    assert np.abs(beating - static).max() < 2e-3
    assert np.abs(old_sign - static).max() > 0.1
