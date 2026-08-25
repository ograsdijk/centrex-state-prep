"""Identifying eigenstates by quantum numbers rather than by a tracked index.

`reorder_evecs` carries a label from timestep to timestep by overlap, which is
the adiabatic convention. It is not always the physical one: where two levels
cross, the population stays on its diabatic branch while the label follows the
other. In this system that is not rare -- the Stark ramp-down folds the
hyperfine structure together and most labels cross something.

The observable consequence was a `depletion` of exactly 1.000000 instead of
about 0.90 at `N_steps=160000`. These tests pin the fix and the bug it replaces.
No microwaves are needed to reproduce it, which keeps them fast.
"""

from __future__ import annotations

import numpy as np
import pytest


# (J, F1, F, mF) of the state the SPA2 initial state becomes once the Stark
# field has ramped off. Determined by propagating the DC fields with the
# microwaves switched off; stable to 0.99999 across N_steps 2000..160000.
READOUT_IDENTITY = {"J": 1, "F1": 1.5, "F": 2, "mF": 0}
INITIAL_IDX = 2


@pytest.fixture(scope="module")
def fields_only(spa2_setup):
    """Simulator with the microwaves off: the DC ramp alone."""
    from state_prep import Simulator

    return lambda: Simulator(
        spa2_setup["trajectory"],
        spa2_setup["electric_field"],
        spa2_setup["magnetic_field"],
        spa2_setup["initial_states"],
        spa2_setup["hamiltonian"],
        None,
    )


@pytest.fixture(scope="module")
def runs(fields_only):
    """Field-only runs at a coarse and a fine step count."""
    return {
        n: fields_only().run(
            N_steps=n,
            store_probabilities=False,
            store_final_probabilities=True,
            progress=False,
        )
        for n in (10000, 160000)
    }


def test_quantum_numbers_are_good_at_readout(runs):
    """With the field off, J, F1, F and mF are all sharp for the initial state."""
    from state_prep import select_eigenstate

    table = runs[10000].final_quantum_numbers()
    index = select_eigenstate(table, READOUT_IDENTITY)
    for key in ("J", "F1", "F", "mF"):
        assert table[f"{key}_spread"][index] < 1e-3, f"{key} is not a good quantum number"


def test_population_is_stable_across_step_count(runs):
    """The physics does not depend on N_steps; only the label does."""
    values = [
        result.population(**READOUT_IDENTITY)[INITIAL_IDX] for result in runs.values()
    ]
    assert all(value > 0.999 for value in values), values
    assert abs(values[0] - values[1]) < 1e-4


def test_tracked_index_swaps_but_quantum_numbers_do_not(runs):
    """The exact failure that produced `depletion == 1.000000`.

    At 160000 steps the tracked index moves off the state it named, so a
    `V_ini`-derived index reads ~0 population where the physical state holds
    ~1. Selection by quantum numbers is unaffected.
    """
    from state_prep import select_eigenstate
    from state_prep.utils import find_max_overlap_idx

    tracked, selected = {}, {}
    for n, result in runs.items():
        tracked[n] = find_max_overlap_idx(
            result.initial_states[INITIAL_IDX].state_vector(result.hamiltonian.QN),
            result.V_ini,
        )
        selected[n] = select_eigenstate(result.final_quantum_numbers(), READOUT_IDENTITY)

    # The bug: the tracked index is the same at both step counts, but at the
    # finer one it no longer names the state that holds the population.
    assert tracked[10000] == tracked[160000]
    assert selected[160000] != tracked[160000], (
        "expected the label swap at N_steps=160000; if this fails the crossing "
        "may have moved rather than the fix having regressed"
    )

    by_index = runs[160000].probabilities_final[INITIAL_IDX, tracked[160000]]
    by_qn = runs[160000].population(**READOUT_IDENTITY)[INITIAL_IDX]
    assert by_index < 1e-6, "tracked index should read empty here"
    assert by_qn > 0.999, "quantum-number selection should find the population"


def test_ambiguous_quantum_numbers_are_rejected(runs):
    """(J, F, mF) is not a complete label: J=1 has two F=1 levels."""
    from state_prep import select_eigenstate

    table = runs[10000].final_quantum_numbers()
    with pytest.raises(ValueError, match="found 2"):
        select_eigenstate(table, {"J": 1, "F": 1, "mF": 0})


def test_crossings_are_recorded(runs):
    """The diagnostic sees the ramp-down crossings and reports where they are."""
    result = runs[10000]
    assert result.label_gaps is not None
    flagged = result.unreliable_labels()
    assert flagged, "expected the Stark ramp-down to produce crossings"

    transit = result.trajectory.get_T()
    times = np.array([detail["at_time_s"] for detail in flagged.values()]) / transit
    assert np.median(times) > 0.5, "crossings should cluster in the ramp-down"


def test_tracked_index_accessor_warns_once(runs):
    """Using a tracked index warns; using quantum numbers does not."""
    result = runs[10000]
    initial = result.initial_states[INITIAL_IDX]

    object.__setattr__(result, "_crossing_warning_issued", False)
    with pytest.warns(UserWarning, match="crossing|conserved"):
        result.get_state_probability(initial, initial)

    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result.population(**READOUT_IDENTITY)
    assert not caught, "quantum-number selection should not warn"


def test_mF_mismatch_is_reported_as_definite(runs):
    """mF is exactly conserved, so a label on the wrong mF is definitely wrong."""
    result = runs[10000]
    vector = result.initial_states[INITIAL_IDX].state_vector(result.hamiltonian.QN)
    table = result.final_quantum_numbers()

    wrong = int(np.flatnonzero(np.abs(table["mF"] - 1.0) < 1e-6)[0])
    object.__setattr__(result, "_crossing_warning_issued", False)
    with pytest.warns(UserWarning, match="exactly conserved"):
        result._check_tracked_index(wrong, "test", vector)


def test_label_gap_tracker_handles_degenerate_and_tiny_inputs():
    """Edge cases the trajectory loops can hit."""
    from state_prep import LabelGapTracker

    tracker = LabelGapTracker(1)
    tracker.update(np.array([0.0]), 0.0)  # must not raise
    assert tracker.summary(1.0)["crossings"].size == 1

    tracker = LabelGapTracker(3)
    for t, energies in enumerate([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]):
        tracker.update(np.array(energies), float(t))
    summary = tracker.summary(2.0)
    assert summary["crossings"].tolist() == [0, 0, 0]
    assert summary["fourier_width_hz"] == pytest.approx(0.5)
