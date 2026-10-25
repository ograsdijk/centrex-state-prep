"""Multitone integral reuse must retain physical beat and velocity timing."""
import numpy as np
import pytest

from bench_velocity_scan import make_simulator


@pytest.mark.parametrize('velocity_scan', [False, True])
@pytest.mark.parametrize('blocks', [False, True])
def test_integral_cache_matches_individual_scans_for_powers_beats_and_velocities(spa2_setup, multitone_fields, velocity_scan, blocks):
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
    actual = sim.run_microwave_scan(**opts)
    # Single-point scans cannot reuse integrals across power points. They also
    # provide a reference in source archives and shallow CI checkouts, without
    # loading an older simulator from Git history.
    references = []
    for point in range(len(det)):
        point_opts = dict(opts, detunings_hz=det[point:point+1],
                          intensity_prefactors=powers[point:point+1])
        if velocity_scan:
            point_opts['forward_velocities'] = opts['forward_velocities'][point:point+1]
        references.append(sim.run_microwave_scan(**point_opts))
    for field in ('psis_final', 'monitor_probabilities_final'):
        reference = np.concatenate([getattr(result, field) for result in references])
        np.testing.assert_allclose(getattr(actual, field),reference,atol=1e-10,rtol=1e-10)


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
