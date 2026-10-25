"""Multitone integral reuse must retain physical beat and velocity timing."""
import numpy as np
import pytest

from bench_dense_investigation import patched
from bench_velocity_scan import make_simulator
from rc_prototypes import cpu_variant
from state_prep import Simulator


@pytest.mark.parametrize('velocity_scan', [False, True])
@pytest.mark.parametrize('blocks', [False, True])
def test_integral_cache_matches_before_for_powers_beats_and_velocities(spa2_setup, multitone_fields, velocity_scan, blocks):
    setup = dict(spa2_setup, microwave_fields=multitone_fields)
    sim = make_simulator(setup,184.)
    det = np.zeros((6,3))
    det[3:,2] = 2e5  # Same reference shift but different RC beat frequency.
    powers = np.tile([[1.,1.,1.],[1.,2.,2.],[1.,4.,4.]],(2,1))
    opts = dict(detunings_hz=det,intensity_prefactors=powers,N_steps=80,progress=False,
        propagator='magnus',allow_multitone_same_manifold=True,block_eigh=blocks,
        monitor_states=setup['monitor_states'])
    if velocity_scan:
        opts['forward_velocities'] = [150.,150.,150.,220.,220.,220.]
    old = cpu_variant()
    with patched(Simulator,'_time_evolve_mu_batched_shared_slow_multitone',old._time_evolve_mu_batched_shared_slow_multitone):
        reference = sim.run_microwave_scan(**opts)
    actual = sim.run_microwave_scan(**opts)
    np.testing.assert_allclose(actual.psis_final,reference.psis_final,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(actual.monitor_probabilities_final,reference.monitor_probabilities_final,atol=1e-10,rtol=1e-10)


def test_integrals_are_built_per_detuning_and_velocity_not_power(spa2_setup,multitone_fields,monkeypatch):
    import state_prep.simulator as module
    setup = dict(spa2_setup,microwave_fields=multitone_fields)
    sim = make_simulator(setup,184.)
    original = module._magnus_integral_from_outer
    calls = []
    def record(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'_magnus_integral_from_outer',record)
    sim.run_microwave_scan(detunings_hz=np.zeros((6,3)),intensity_prefactors=np.ones((6,3)),
        N_steps=10,progress=False,propagator='magnus',allow_multitone_same_manifold=True,
        forward_velocities=[150.,150.,150.,220.,220.,220.])
    assert len(calls) == 9*2*3  # Static and two beat senses per velocity group.
