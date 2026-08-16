"""The rotating-frame shift, and the multitone path built on top of it.

`build_rotating_frame_shift` is shared by the simulator, the GPU benchmark and
`benchmarks/common.py`. It previously existed as three independent copies, so
these tests exist to keep them from diverging again.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import CROSS_BACKEND_TOL


def test_shift_is_diagonal_only_and_shifts_excited_manifold(spa2_setup):
    from state_prep.microwaves import build_rotating_frame_shift

    fields = spa2_setup["microwave_fields"]
    qn = spa2_setup["hamiltonian"].QN
    det = np.zeros((3, len(fields)))

    shift, omegas, groups = build_rotating_frame_shift(fields, qn, det)

    assert shift.shape == (3, len(qn))
    # The two SPA2 fields share a carrier, so they form one frequency group.
    assert len(omegas) == 1
    assert groups == [[0, 1]]

    # Only states in the addressed excited manifold are shifted.
    shifted = np.abs(shift[0]) > 0
    excited_J = fields[0].Je
    assert all(qn[i].J == excited_J for i in np.flatnonzero(shifted))
    assert shifted.any()


def test_detuning_enters_linearly(spa2_setup):
    from state_prep.microwaves import build_rotating_frame_shift

    fields = spa2_setup["microwave_fields"]
    qn = spa2_setup["hamiltonian"].QN

    zero = build_rotating_frame_shift(fields, qn, np.zeros((1, len(fields))))[0][0]
    one = build_rotating_frame_shift(fields, qn, np.ones((1, len(fields))) * 1e6)[0][0]

    delta = zero - one
    mask = np.abs(zero) > 0
    # A detuning of +1 MHz lowers the shifted energies by 2*pi*1e6 rad/s.
    assert np.allclose(delta[mask], 2 * np.pi * 1e6)
    assert np.allclose(delta[~mask], 0.0)


def test_same_carrier_fields_must_share_detuning(spa2_setup):
    """Otherwise the rotating frame would be ambiguous."""
    from state_prep.microwaves import build_rotating_frame_shift

    fields = spa2_setup["microwave_fields"]
    qn = spa2_setup["hamiltonian"].QN
    det = np.column_stack([np.zeros(2), np.ones(2) * 1e5])

    with pytest.raises(ValueError, match="share the same detuning"):
        build_rotating_frame_shift(fields, qn, det)


def test_all_call_sites_agree(spa2_setup):
    """The simulator, the benchmarks and the GPU benchmark must build one shift."""
    from common import make_d_mu_diag_batch
    from state_prep.microwaves import build_rotating_frame_shift
    from state_prep_gpu import benchmark_spa2_real as gpu_bench

    fields = spa2_setup["microwave_fields"]
    qn = spa2_setup["hamiltonian"].QN
    d = np.linspace(-1e6, 1e6, 4)
    det = np.column_stack([d, d])

    reference, _, _ = build_rotating_frame_shift(fields, qn, det)
    from_common = make_d_mu_diag_batch(
        microwave_fields=fields, qn=qn, detunings_hz_batched=det
    )
    from_gpu = gpu_bench._make_D_mu_diag_batch(
        microwave_fields=fields, QN=qn, detunings_hz_batched=det
    )

    assert np.array_equal(reference, from_common)
    assert np.array_equal(reference, from_gpu)


def test_multitone_rejected_by_default(simulator, spa2_setup, multitone_fields, detunings):
    """Two carriers on one manifold is ambiguous unless explicitly allowed."""
    from state_prep import Simulator, scan_grid

    sim = Simulator(
        spa2_setup["trajectory"],
        spa2_setup["electric_field"],
        spa2_setup["magnetic_field"],
        spa2_setup["initial_states"],
        spa2_setup["hamiltonian"],
        multitone_fields,
    )
    det, pref = scan_grid(n_fields=3, detunings_hz=detunings, detuning_fields=[0, 1])

    with pytest.raises(ValueError, match="allow_multitone_same_manifold"):
        sim.run_microwave_scan(
            detunings_hz=det,
            intensity_prefactors=pref,
            N_steps=100,
            store_final_probabilities=True,
            progress=False,
        )


def test_multitone_runs_and_normalises(spa2_setup, multitone_fields, detunings):
    from state_prep import Simulator, scan_grid

    sim = Simulator(
        spa2_setup["trajectory"],
        spa2_setup["electric_field"],
        spa2_setup["magnetic_field"],
        spa2_setup["initial_states"],
        spa2_setup["hamiltonian"],
        multitone_fields,
    )
    det, pref = scan_grid(n_fields=3, detunings_hz=detunings, detuning_fields=[0, 1])

    result = sim.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=200,
        monitor_states=spa2_setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        allow_multitone_same_manifold=True,
    )

    assert result.probabilities_final.shape[0] == len(detunings)
    totals = result.probabilities_final.sum(axis=-1)
    assert np.allclose(totals, 1.0, atol=1e-9)
