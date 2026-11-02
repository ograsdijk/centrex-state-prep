"""The GPU scalar-envelope factorization must reproduce every field, including varying polarization."""
import numpy as np

from state_prep_gpu.microwave_factors import (microwave_amp_t_np, microwave_base_matrices_np, microwave_factors_np,
                                               microwave_upper_factors_np)


def _samples(setup, steps=400):
    times = np.linspace(0, setup["trajectory"].get_T(), steps)
    return np.r_[(times[:-1] + times[1:]) / 2, times[-1]]


def _reconstruct_error(fields, trajectory, qn, t_array, base, amp, owner):
    worst = 0.0
    for j, field in enumerate(fields):
        ham = field.get_H_t_func(trajectory.R_t, qn)
        parts = owner == j
        for i, t in enumerate(t_array[:-1]):
            actual = ham(float(t))
            scale = max(np.abs(actual).max(), 1e-300)
            worst = max(worst, np.abs(np.tensordot(amp[i, parts], base[parts], 1) - actual).max() / scale)
    return worst


def test_static_polarization_matches_single_base_helpers(spa2_setup):
    fields, trajectory = spa2_setup["microwave_fields"], spa2_setup["trajectory"]
    n = len(spa2_setup["hamiltonian"].QN)
    t_array = _samples(spa2_setup)
    base, amp, owner = microwave_factors_np(fields, trajectory=trajectory, t_array=t_array, n=n)
    np.testing.assert_array_equal(owner, np.arange(len(fields)))
    np.testing.assert_array_equal(base, microwave_base_matrices_np(fields, trajectory=trajectory, t0=float(t_array[0]), n=n))
    np.testing.assert_array_equal(amp, microwave_amp_t_np(fields, trajectory=trajectory, t_array=t_array))


def test_switching_polarization_is_reproduced_by_parts(spa2_setup):
    from state_prep.microwaves import MicrowaveField, Polarization

    main = spa2_setup["microwave_fields"][0]
    trajectory, qn = spa2_setup["trajectory"], spa2_setup["hamiltonian"].QN

    def switching(position):
        # Elliptical polarization that rotates with z and turns off in a window.
        angle = 40.0 * position[2]
        on = 0.0 if 0.02 < position[2] < 0.05 else 1.0
        p = np.array([np.cos(angle), 1j * np.sin(angle), 0.5])
        return on * p / np.linalg.norm(p)

    field = MicrowaveField(main.Jg, main.Je, main.intensity, Polarization(switching), main.muW_freq, QN=qn)
    fields = [main, field]
    t_array = _samples(spa2_setup)
    base, amp, owner = microwave_factors_np(fields, trajectory=trajectory, t_array=t_array, n=len(qn))
    assert (owner == 0).sum() == 1
    assert 1 < (owner == 1).sum() <= 6
    # A single base matrix cannot represent this field; the parts must.
    single = microwave_base_matrices_np([field], trajectory=trajectory, t0=float(t_array[0]), n=len(qn))
    single_amp = microwave_amp_t_np([field], trajectory=trajectory, t_array=t_array)
    assert _reconstruct_error([field], trajectory, qn, t_array, single, single_amp, np.array([0])) > 1e-3
    assert _reconstruct_error(fields, trajectory, qn, t_array, base, amp, owner) < 1e-12


def test_upper_factors_reproduce_static_and_switching_fields(spa2_setup):
    """Multitone GPU input: w U + conj(w) U^H must equal the field at every sample."""
    from state_prep.microwaves import MicrowaveField, Polarization

    main = spa2_setup["microwave_fields"][0]
    trajectory, qn = spa2_setup["trajectory"], spa2_setup["hamiltonian"].QN

    def switching(position):
        angle = 40.0 * position[2]
        on = 0.0 if 0.02 < position[2] < 0.05 else 1.0
        p = np.array([np.cos(angle), 1j * np.sin(angle), 0.5])
        return on * p / np.linalg.norm(p)

    field = MicrowaveField(main.Jg, main.Je, main.intensity, Polarization(switching), main.muW_freq, QN=qn)
    fields = [main, field]
    t_array = _samples(spa2_setup)
    upper, amp, owner = microwave_upper_factors_np(fields, trajectory=trajectory, t_array=t_array, n=len(qn))
    assert (owner == 0).sum() == 1
    assert 1 < (owner == 1).sum() <= 3
    worst = 0.0
    for j, f in enumerate(fields):
        ham = f.get_H_t_func(trajectory.R_t, qn)
        parts = owner == j
        for i, t in enumerate(t_array[:-1]):
            u = np.tensordot(amp[i, parts], upper[parts], 1)
            actual = ham(float(t))
            worst = max(worst, np.abs(u + u.conj().T - actual).max() / max(np.abs(actual).max(), 1e-300))
    assert worst < 1e-12
