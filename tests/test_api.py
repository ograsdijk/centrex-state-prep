"""Public API, result shapes, and the scan-construction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import EXACT_TOL


def test_package_exports_are_importable():
    import state_prep as sp

    assert sp.__version__
    for name in sp.__all__:
        assert hasattr(sp, name), f"{name} is in __all__ but not importable"


def test_scan_grid_1d():
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, 5))
    assert det.shape == pref.shape == (5, 2)
    assert np.all(pref == 1.0)
    assert np.all(det[:, 0] == det[:, 1])


def test_scan_grid_outer_product_detuning_slowest():
    from state_prep import scan_grid

    det, pref = scan_grid(
        n_fields=1, detunings_hz=[0.0, 1.0, 2.0], intensity_prefactors=[1.0, 4.0]
    )
    assert det.shape == (6, 1)
    assert list(det[:, 0]) == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    assert list(pref[:, 0]) == [1.0, 4.0, 1.0, 4.0, 1.0, 4.0]


def test_scan_grid_fixed_source_keeps_zero_detuning():
    """A physically fixed source must not follow the scan."""
    from state_prep import scan_grid

    det, _ = scan_grid(
        n_fields=3, detunings_hz=[1e6, 2e6], detuning_fields=[0, 1]
    )
    assert np.all(det[:, 2] == 0.0)
    assert np.all(det[:, 0] == det[:, 1])


def test_scan_grid_rejects_bad_input():
    from state_prep import scan_grid

    with pytest.raises(ValueError):
        scan_grid(n_fields=2, intensity_prefactors=[-1.0])
    with pytest.raises(ValueError):
        scan_grid(n_fields=2, detuning_fields=[0, 0])
    with pytest.raises(ValueError):
        scan_grid(n_fields=2, detuning_fields=[5])


def test_scan_storage_defaults_are_accepted_by_run(simulator, spa2_setup):
    from state_prep import SCAN_STORAGE_DEFAULTS

    result = simulator.run(
        N_steps=100,
        monitor_states=spa2_setup["monitor_states"],
        progress=False,
        **SCAN_STORAGE_DEFAULTS,
    )
    assert result.probabilities_final is not None
    assert result.psis is None or result.psis.size == 0
    assert result.energies is None


def test_scan_result_accessors(simulator, spa2_setup, detunings):
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    result = simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=200,
        monitor_states=spa2_setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
    )

    assert result.batch_size == len(detunings)

    monitored = spa2_setup["monitor_states"][0]
    initial = spa2_setup["initial_states"][0]

    by_state = result.get_state_probability(monitored, initial)
    by_monitor = result.get_monitor_probability(monitored, initial)
    assert by_state.shape == (len(detunings),)
    # Both resolve the same tracked eigenstate, so they must agree.
    assert np.max(np.abs(by_state - by_monitor)) < EXACT_TOL

    frame = result.to_polars(detuning_hz=det[:, 0])
    assert frame.height == len(detunings)
    assert "detuning_hz" in frame.columns


def test_scan_result_roundtrips_through_pickle(simulator, spa2_setup, detunings, tmp_path):
    from state_prep import scan_grid
    import dill

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    result = simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=100,
        store_final_probabilities=True,
        progress=False,
    )
    path = tmp_path / "scan.pkl"
    result.save_to_pickle(path)

    with open(path, "rb") as f:
        loaded = dill.load(f)
    assert np.array_equal(loaded.probabilities_final, result.probabilities_final)


def test_units_and_shapes(simulator, spa2_setup):
    """Energies are angular frequencies; z follows from t and forward velocity."""
    result = simulator.run(
        N_steps=100,
        store_psis=False,
        store_energies=True,
        store_probabilities=False,
        store_final_probabilities=True,
        progress=False,
    )
    n = len(spa2_setup["hamiltonian"].QN)
    assert result.energies.shape[-1] == n
    assert result.t_array[0] == 0.0
    expected_z = (
        result.t_array * spa2_setup["trajectory"].Vini[2]
        + spa2_setup["trajectory"].Rini[2]
    )
    assert np.allclose(result.z_array, expected_z)
