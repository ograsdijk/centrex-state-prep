"""Velocity ensembles retain a common path and use physical per-point time."""
from copy import copy

import numpy as np
import pytest

from state_prep import Simulator, Trajectory


def velocity_simulator(setup, velocity, fields=None):
    trajectory = setup['trajectory']
    path = Trajectory(trajectory.Rini.copy(),
                      trajectory.Vini * (velocity / trajectory.Vini[2]), trajectory.zfin)
    hamiltonian = copy(setup['hamiltonian'])
    hamiltonian.trajectory = path
    return Simulator(path, setup['electric_field'], setup['magnetic_field'],
                     setup['initial_states'], hamiltonian,
                     setup['microwave_fields'] if fields is None else fields)


@pytest.mark.parametrize('block_eigh', [False, True])
@pytest.mark.parametrize('propagator', ['frozen', 'magnus'])
def test_velocity_scan_matches_separate_trajectories(spa2_setup, block_eigh, propagator):
    velocities = np.array([150., 184., 230.])
    sim = velocity_simulator(spa2_setup, 184.)
    opts = dict(detunings_hz=np.zeros((1, 2)), intensity_prefactors=np.ones((1, 2)),
                N_steps=160, progress=False, block_eigh=block_eigh, propagator=propagator,
                monitor_states=spa2_setup['monitor_states'])
    actual = sim.run_microwave_scan(**opts, forward_velocities=velocities)
    for b, velocity in enumerate(velocities):
        reference = velocity_simulator(spa2_setup, velocity).run_microwave_scan(**opts)
        # Compare in a common physical basis, independent of tracked labels.
        basis = actual.V_fin
        np.testing.assert_allclose(np.abs(actual.psis_final[b] @ basis.conj())**2,
                                   np.abs(reference.psis_final[0] @ basis.conj())**2,
                                   atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(actual.monitor_probabilities_final[b],
                                   reference.monitor_probabilities_final[0], atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(actual.times_per_point[b], reference.t_array, atol=1e-18)
    np.testing.assert_array_equal(actual.forward_velocities, velocities)


@pytest.mark.parametrize('propagator', ['frozen', 'magnus'])
def test_multitone_velocity_scan_uses_own_beat_time(spa2_setup, multitone_fields, propagator):
    velocities = [155., 215.]
    opts = dict(detunings_hz=np.zeros((1, 3)), intensity_prefactors=np.ones((1, 3)),
                N_steps=180, progress=False, block_eigh=False, propagator=propagator,
                allow_multitone_same_manifold=True)
    sim = velocity_simulator(spa2_setup, 184., multitone_fields)
    actual = sim.run_microwave_scan(**opts, forward_velocities=velocities)
    for b, velocity in enumerate(velocities):
        reference = velocity_simulator(spa2_setup, velocity, multitone_fields).run_microwave_scan(**opts)
        np.testing.assert_allclose(np.abs(actual.psis_final[b] @ actual.V_fin.conj())**2,
                                   np.abs(reference.psis_final[0] @ actual.V_fin.conj())**2,
                                   atol=1e-7, rtol=1e-7)


def test_velocity_scan_shares_only_identical_microwave_solves(spa2_setup, monkeypatch):
    import state_prep.simulator as module
    original = module.zheevd
    calls = []
    def record(matrix, *args, **kwargs):
        calls.append(matrix.shape)
        return original(matrix, *args, **kwargs)
    monkeypatch.setattr(module, 'zheevd', record)
    sim = velocity_simulator(spa2_setup, 184.)
    det = np.array([[0., 0.], [1e5, 1e5], [0., 0.]])
    result = sim.run_microwave_scan(detunings_hz=det, intensity_prefactors=np.ones_like(det),
        forward_velocities=[150., 184., 230.], N_steps=8, block_eigh=False, progress=False)
    assert len(calls) == 7 * (1 + 2)  # One slow and two distinct microwave solves.
    assert result.psis_final.shape[0] == 3


def test_velocity_scan_parallel_preserves_order_and_times(spa2_setup):
    sim = velocity_simulator(spa2_setup, 184.)
    opts = dict(detunings_hz=np.zeros((1, 2)), intensity_prefactors=np.ones((1, 2)),
                forward_velocities=[150., 184., 230.], N_steps=30, progress=False)
    serial = sim.run_microwave_scan(**opts)
    parallel = sim.run_microwave_scan(**opts, workers=2)
    np.testing.assert_array_equal(parallel.psis_final, serial.psis_final)
    np.testing.assert_array_equal(parallel.times_per_point, serial.times_per_point)


def test_velocity_scan_preserves_transverse_slope_and_left_sampling(spa2_setup):
    setup = dict(spa2_setup)
    original = setup['trajectory']
    setup['trajectory'] = Trajectory(original.Rini.copy(), np.array([0.12, -0.08, 184.]), original.zfin)
    velocities = [150., 230.]
    opts = dict(detunings_hz=np.array([[0., 0.], [1e5, 1e5]]),
                intensity_prefactors=np.array([[1., 1.], [0.7, 0.7]]),
                N_steps=80, progress=False, time_sampling='left', block_eigh=False)
    actual = velocity_simulator(setup, 184.).run_microwave_scan(**opts, forward_velocities=velocities)
    for b, velocity in enumerate(velocities):
        reference = velocity_simulator(setup, velocity).run_microwave_scan(
            **dict(opts, detunings_hz=opts['detunings_hz'][b:b+1],
                   intensity_prefactors=opts['intensity_prefactors'][b:b+1]))
        np.testing.assert_allclose(np.abs(actual.psis_final[b] @ actual.V_fin.conj())**2,
                                   np.abs(reference.psis_final[0] @ actual.V_fin.conj())**2,
                                   atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('velocities', [[], [0.], [-1.], [np.nan], [np.inf], [[184.]], [1., 2.]])
def test_velocity_scan_rejects_bad_speeds(simulator, velocities):
    with pytest.raises(ValueError, match='forward_velocities'):
        simulator.run_microwave_scan(detunings_hz=np.zeros((3, 2)),
            intensity_prefactors=np.ones((3, 2)), forward_velocities=velocities, progress=False)
