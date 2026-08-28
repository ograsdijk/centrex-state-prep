"""Non-uniform integration grids.

The evolution loops already read `dt = t_array[i+1] - t_array[i]` per step, so
grading the grid is a change to how `t_array` is built and to nothing else.
These tests pin the two things that matter: the default is untouched, and a
graded run still agrees with the unbatched path it is supposed to reproduce.

What grading is worth is a benchmark question, not a test one -- see
`benchmarks/bench_step_grid.py` and Priority E in `IMPROVEMENTS.md`. The ceiling
on the SPA2 setup is about `1.58x`, and it is not uniform across a scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SAME_BACKEND_TOL


N_STEPS = 400


def test_default_grid_is_linspace_exactly():
    """The default path must stay bit-identical, not merely close."""
    from state_prep import build_time_grid

    T = 1.5217e-3
    for n in (2, 3, 101, 10000):
        assert np.array_equal(build_time_grid(T, n), np.linspace(0, T, n))


def test_graded_grid_invariants():
    from state_prep import build_time_grid

    T = 1.5217e-3
    peak = lambda ts: 1.0 + 1e3 * np.exp(-(((ts - 0.3 * T) / (0.05 * T)) ** 2))

    grid = build_time_grid(T, 1000, density=peak)
    assert len(grid) == 1000
    assert grid[0] == 0.0
    assert grid[-1] == T
    assert np.all(np.diff(grid) > 0), "grid must be strictly increasing"
    # Steps must actually be redistributed, or grading is a no-op.
    assert np.diff(grid).max() / np.diff(grid).min() > 5


def test_graded_grid_concentrates_where_the_density_is_high():
    """Steps go where the density is, which is the whole point."""
    from state_prep import build_time_grid

    T = 1.0
    grid = build_time_grid(
        T, 2000, density=lambda ts: 1.0 + 1e4 * np.exp(-(((ts - 0.25) / 0.02) ** 2))
    )
    near_peak = np.mean((grid >= 0.20) & (grid <= 0.30))
    assert near_peak > 0.10, f"only {near_peak:.1%} of steps near the peak (10% of T)"


def test_uniform_density_reproduces_a_uniform_grid():
    from state_prep import build_time_grid

    T = 2.5e-3
    grid = build_time_grid(T, 500, density=lambda ts: np.full_like(ts, 7.0))
    assert np.allclose(grid, np.linspace(0, T, 500), rtol=0, atol=1e-10 * T)


def test_max_ratio_caps_the_longest_step():
    """The cap is a ceiling on the longest step, and it binds exactly.

    Getting this right needs a fixed point: flooring the weight also raises the
    mean weight, which lengthens every step, so a floor derived from the
    unfloored mean overshoots. Where the cap does not bind it must leave the
    grid alone rather than distort it.
    """
    from state_prep import build_time_grid

    T, n = 1.0, 1000
    uniform_dt = T / (n - 1)
    peak = lambda ts: 1.0 + 1e6 * np.exp(-(((ts - 0.5) / 0.01) ** 2))

    uncapped = np.diff(build_time_grid(T, n, density=peak, max_ratio=1e9)).max()

    for cap in (2.0, 8.0, 64.0):
        dt_max = np.diff(build_time_grid(T, n, density=peak, max_ratio=cap)).max()
        target = cap * uniform_dt
        assert dt_max <= target * 1.02, f"cap {cap} exceeded"
        if uncapped > target:
            assert dt_max >= target * 0.98, f"cap {cap} binds but was not reached"
        else:
            assert np.isclose(dt_max, uncapped, rtol=1e-9), (
                f"cap {cap} does not bind but changed the grid"
            )


def test_degenerate_density_falls_back_to_uniform():
    from state_prep import build_time_grid

    T = 1.0
    grid = build_time_grid(T, 100, density=lambda ts: np.zeros_like(ts))
    assert np.array_equal(grid, np.linspace(0, T, 100))


def test_bad_density_is_rejected():
    from state_prep import build_time_grid

    T = 1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_time_grid(T, 100, density=lambda ts: -np.ones_like(ts))
    with pytest.raises(ValueError, match="one value per probe"):
        build_time_grid(T, 100, density=lambda ts: np.ones(3))


def test_bad_step_density_argument_is_rejected(simulator):
    with pytest.raises(ValueError, match="None, 'auto', or a callable"):
        simulator.run(N_steps=10, progress=False, step_density="uniform")
    with pytest.raises(ValueError, match="None, 'auto', or a callable"):
        simulator.run(N_steps=10, progress=False, step_density=3.0)


def test_step_density_none_matches_the_previous_default(simulator, detunings):
    """Opting out must be bitwise identical to not having the option."""
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    kwargs = dict(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
    )
    plain = simulator.run_microwave_scan(**kwargs)
    explicit = simulator.run_microwave_scan(**kwargs, step_density=None)
    assert np.array_equal(plain.probabilities_final, explicit.probabilities_final)
    assert np.array_equal(plain.t_array, explicit.t_array)
    assert plain.time_grid == "uniform"


def test_graded_scan_uses_a_nonuniform_grid(spa2_setup, simulator, detunings):
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    result = simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
        step_density="auto",
    )
    dt = np.diff(result.t_array)
    assert result.time_grid == "graded"
    assert len(result.t_array) == N_STEPS
    assert np.isclose(result.t_array[-1], spa2_setup["trajectory"].get_T())
    assert dt.max() / dt.min() > 2, "grid is not actually graded"
    assert np.allclose(result.probabilities_final.sum(axis=-1), 1.0, atol=1e-10)


def test_graded_shared_slow_matches_repeated_runs(spa2_setup, simulator, detunings):
    """The batched graded path must reproduce looped `run()` on the same grid.

    This is the graded twin of `test_shared_slow_matches_repeated_runs`: both
    paths build their grid independently, so it also pins that the two agree.
    """
    from state_prep import Simulator, scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    scan = simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
        step_density="auto",
    )

    for index, (detuning_row, prefactor_row) in enumerate(zip(det, pref)):
        single = Simulator(
            spa2_setup["trajectory"],
            spa2_setup["electric_field"],
            spa2_setup["magnetic_field"],
            spa2_setup["initial_states"],
            spa2_setup["hamiltonian"],
            spa2_setup["microwave_fields"],
        ).run_microwave_scan(
            detunings_hz=detuning_row[None, :],
            intensity_prefactors=prefactor_row[None, :],
            N_steps=N_STEPS,
            store_final_probabilities=True,
            progress=False,
            step_density="auto",
        )
        assert np.allclose(
            scan.probabilities_final[index],
            single.probabilities_final[0],
            rtol=0,
            atol=SAME_BACKEND_TOL,
        )


def test_graded_grid_is_batch_independent(simulator):
    """One grid for the whole scan, or the shared slow eigensolve is lost."""
    from state_prep import scan_grid

    grids = []
    for detunings in ([-1.0e6, 0.5e6], [-2.0e6, 1.5e6]):
        det, pref = scan_grid(n_fields=2, detunings_hz=np.array(detunings))
        grids.append(
            simulator.run_microwave_scan(
                detunings_hz=det,
                intensity_prefactors=pref * 4.0,
                N_steps=N_STEPS,
                store_final_probabilities=True,
                progress=False,
                step_density="auto",
            ).t_array
        )
    assert np.array_equal(grids[0], grids[1])


# ---------------------------------------------------------------------------
# Timestep sampling
# ---------------------------------------------------------------------------


def test_midpoint_is_the_default(simulator):
    result = simulator.run(
        N_steps=100, store_probabilities=False, store_final_probabilities=True,
        progress=False,
    )
    assert result.time_sampling == "mid"


def test_left_sampling_is_available_and_differs(simulator, detunings):
    """`left` must stay reachable: it reproduces pre-`time_sampling` results."""
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    kwargs = dict(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
    )
    mid = simulator.run_microwave_scan(**kwargs)
    left = simulator.run_microwave_scan(**kwargs, time_sampling="left")

    assert mid.time_sampling == "mid" and left.time_sampling == "left"
    assert not np.array_equal(mid.probabilities_final, left.probabilities_final)
    for result in (mid, left):
        assert np.allclose(result.probabilities_final.sum(axis=-1), 1.0, atol=1e-10)


def test_bad_time_sampling_is_rejected(simulator):
    with pytest.raises(ValueError, match="must be 'mid' or 'left'"):
        simulator.run(N_steps=10, progress=False, time_sampling="middle")


@pytest.mark.parametrize("time_sampling", ["mid", "left"])
def test_sampling_is_threaded_through_loky(simulator, detunings, time_sampling):
    """A missed hand-off to the workers would silently change only the parallel path."""
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    kwargs = dict(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
        time_sampling=time_sampling,
    )
    serial = simulator.run_microwave_scan(**kwargs, workers=1)
    parallel = simulator.run_microwave_scan(**kwargs, workers=2)
    assert np.array_equal(serial.probabilities_final, parallel.probabilities_final)
    assert parallel.time_sampling == time_sampling


def test_sampling_and_grading_compose(simulator, detunings):
    """Both options apply together, and each is visible in the result."""
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    result = simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
        step_density="auto",
        time_sampling="mid",
    )
    assert (result.time_grid, result.time_sampling) == ("graded", "mid")
    assert np.diff(result.t_array).max() / np.diff(result.t_array).min() > 2
    assert np.allclose(result.probabilities_final.sum(axis=-1), 1.0, atol=1e-10)
