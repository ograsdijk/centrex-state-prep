"""`coupling_tolerance` shrinks the per-point solve without changing the answer."""
import numpy as np
import pytest

import state_prep.simulator as module
from state_prep import Simulator
from state_prep.simulator import (
    _apply_coupled_solution,
    _coupled_eigen_blocks,
    _shift_difference_bounds,
    _solve_coupled_blocks,
)

# Measured: 1e-8 moves final probabilities by below 1e-8 on SPA2 and the cascade.
TOLERANCE = 1e-8
AGREEMENT = 1e-7


def make_simulator(setup):
    return Simulator(setup['trajectory'], setup['electric_field'], setup['magnetic_field'],
                     setup['initial_states'], setup['hamiltonian'], setup['microwave_fields'])


def scan_options(n_fields, n_steps=120, **extra):
    detunings = np.repeat(np.linspace(-1.0e6, 0.5e6, 4)[:, None], n_fields, axis=1)
    return dict(detunings_hz=detunings, intensity_prefactors=np.ones_like(detunings),
                N_steps=n_steps, progress=False, **extra)


def largest_solves(monkeypatch, run):
    """Run `run()` and return the largest matrix handed to the eigensolver."""
    sizes = []
    original = module.zheevd

    def recording(matrix, *args, **kwargs):
        sizes.append(matrix.shape[0])
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(module, 'zheevd', recording)
    result = run()
    monkeypatch.setattr(module, 'zheevd', original)
    return result, sizes


@pytest.mark.parametrize('block_eigh', [False, True])
def test_spa2_matches_full_solve(spa2_setup, block_eigh):
    sim = make_simulator(spa2_setup)
    opts = scan_options(2, block_eigh=block_eigh, monitor_states=spa2_setup['monitor_states'])
    full = sim.run_microwave_scan(**opts)
    reduced = sim.run_microwave_scan(**opts, coupling_tolerance=TOLERANCE)
    np.testing.assert_allclose(reduced.probabilities_final, full.probabilities_final, atol=AGREEMENT)
    np.testing.assert_allclose(reduced.monitor_probabilities_final,
                               full.monitor_probabilities_final, atol=AGREEMENT)
    np.testing.assert_allclose(np.linalg.norm(reduced.psis_final, axis=-1), 1.0, atol=1e-10)


def test_cascade_matches_full_solve(cascade_setup):
    sim = make_simulator(cascade_setup)
    opts = scan_options(2, block_eigh=False)
    full = sim.run_microwave_scan(**opts)
    reduced = sim.run_microwave_scan(**opts, coupling_tolerance=TOLERANCE)
    np.testing.assert_allclose(reduced.probabilities_final, full.probabilities_final, atol=AGREEMENT)


def test_dense_fields_solve_only_the_driven_manifolds(spa2_setup, monkeypatch):
    """With non-collinear DC fields there are no exact blocks, yet J=1,2 still separate."""
    electric_field = spa2_setup['electric_field']
    original_e = electric_field.E_R

    def e_r(position):
        e = original_e(position).copy()
        phase = 2 * np.pi * (position[2] + 0.08) / 0.28
        e[0] = 0.02 * e[2] * np.sin(phase)
        e[1] = 0.01 * e[2] * np.cos(phase)
        return e

    monkeypatch.setattr(electric_field, 'E_R', e_r)
    sim = make_simulator(spa2_setup)
    opts = scan_options(2)
    full = sim.run_microwave_scan(**opts)
    n = full.psis_final.shape[-1]
    reduced, sizes = largest_solves(
        monkeypatch, lambda: sim.run_microwave_scan(**opts, coupling_tolerance=TOLERANCE))
    np.testing.assert_allclose(reduced.probabilities_final, full.probabilities_final, atol=AGREEMENT)
    # Only the shared slow solves are full size; per-point solves stay near J=1,2.
    per_point = [size for size in sizes if size != n]
    assert sizes.count(n) == opts['N_steps'] - 1
    assert per_point and max(per_point) <= 40


def test_velocity_groups_match_full_solve(spa2_setup):
    sim = make_simulator(spa2_setup)
    opts = dict(detunings_hz=np.zeros((1, 2)), intensity_prefactors=np.ones((1, 2)),
                N_steps=120, progress=False, forward_velocities=[150., 184., 230.])
    full = sim.run_microwave_scan(**opts)
    reduced = sim.run_microwave_scan(**opts, coupling_tolerance=TOLERANCE)
    np.testing.assert_allclose(reduced.probabilities_final, full.probabilities_final, atol=AGREEMENT)


def test_workers_agree_with_serial(spa2_setup):
    sim = make_simulator(spa2_setup)
    opts = scan_options(2, n_steps=40, coupling_tolerance=TOLERANCE)
    serial = sim.run_microwave_scan(**opts)
    parallel = sim.run_microwave_scan(**opts, workers=2)
    np.testing.assert_allclose(parallel.probabilities_final, serial.probabilities_final, atol=AGREEMENT)


@pytest.mark.parametrize('bad', [0.0, -1e-6, 1.0, np.nan])
def test_rejects_invalid_tolerance(simulator, bad):
    with pytest.raises(ValueError, match='coupling_tolerance'):
        simulator.run_microwave_scan(**scan_options(2), coupling_tolerance=bad)


def test_rejects_unsupported_paths(spa2_setup, multitone_fields):
    sim = make_simulator(spa2_setup)
    with pytest.raises(ValueError, match='frozen'):
        sim.run_microwave_scan(**scan_options(2), propagator='magnus', coupling_tolerance=TOLERANCE)
    multitone = Simulator(spa2_setup['trajectory'], spa2_setup['electric_field'],
                          spa2_setup['magnetic_field'], spa2_setup['initial_states'],
                          spa2_setup['hamiltonian'], multitone_fields)
    with pytest.raises(ValueError, match='multitone'):
        multitone.run_microwave_scan(**scan_options(3), allow_multitone_same_manifold=True,
                                     coupling_tolerance=TOLERANCE)


def test_block_detection_keeps_near_resonant_couplings():
    # States 0-1 resonantly coupled, 2 far detuned from both, 3 exactly decoupled
    # and degenerate with 0.
    energies = np.array([0.0, 1.0, 1e9, 0.0])
    coupling = np.zeros((4, 4))
    coupling[0, 1] = coupling[1, 0] = 0.5
    coupling[1, 2] = coupling[2, 1] = 0.5
    zero = np.zeros((4, 4))
    isolated, blocks = _coupled_eigen_blocks(coupling, energies, zero, zero, 1e-6)
    np.testing.assert_array_equal(isolated, [2, 3])
    assert [block.tolist() for block in blocks] == [[0, 1]]
    # A shift range that can bring 2 into resonance with 1 must keep it.
    shifts = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -2e9, 0.0]])
    lo, hi = _shift_difference_bounds(shifts)
    isolated, blocks = _coupled_eigen_blocks(coupling, energies, lo, hi, 1e-6)
    np.testing.assert_array_equal(isolated, [3])
    assert [block.tolist() for block in blocks] == [[0, 1, 2]]


def test_block_propagation_matches_dense_exponential():
    rng = np.random.default_rng(3)
    n, dt = 7, 0.37
    matrix = np.diag(rng.normal(size=n)).astype(complex)
    blocks = [np.array([0, 3, 5]), np.array([1, 6])]
    isolated = np.array([2, 4])
    for indices in blocks:
        sub = rng.normal(size=(len(indices),) * 2) + 1j * rng.normal(size=(len(indices),) * 2)
        sub = sub + sub.conj().T
        matrix[indices[:, None], indices] = sub
    psis = rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))
    values, vectors = np.linalg.eigh(matrix)
    expected = psis @ (vectors * np.exp(-1j * values * dt) @ vectors.conj().T).T
    for backend in ('zheevd', 'numpy'):
        actual = psis.copy()
        solution = _solve_coupled_blocks(matrix, isolated, blocks, backend)
        _apply_coupled_solution(actual, isolated, solution, dt)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
