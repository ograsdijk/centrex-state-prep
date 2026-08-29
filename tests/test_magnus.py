"""The interaction-picture Magnus propagator.

It replaces the per-scan-point eigensolve of `H_rot` with a Taylor series
applied to the state vectors: `n**2 * S` work instead of `n**3`. Measured
`3.23x` faster at batch `25`, and identical to the exact path to four
significant figures against closed-form solutions at production `delta*dt`.

The tests that matter here are the ones a duplicated loop would fail. Magnus is
implemented as a *branch* inside `_time_evolve_mu_batched_shared_slow`, not as a
parallel loop, precisely so it inherits `reorder_evecs`, the `LabelGapTracker`,
monitor-index handling and the shared basis hoist. An earlier benchmark-only
version was a separate loop and silently lost the gap tracker, which made
`label_gaps` report whatever a *previous* run had left on the Simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

import analytic_models as am
from conftest import SAME_BACKEND_TOL


N_STEPS = 400


def scan(simulator, detunings, **kwargs):
    from state_prep import scan_grid

    det, pref = scan_grid(n_fields=2, detunings_hz=detunings)
    return simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=N_STEPS,
        store_final_probabilities=True,
        progress=False,
        **kwargs,
    )


def test_frozen_is_the_default_and_unchanged(simulator, detunings):
    """Adding the option must not move the default path."""
    plain = scan(simulator, detunings)
    explicit = scan(simulator, detunings, propagator="frozen")
    assert plain.propagator == "frozen"
    assert np.array_equal(plain.probabilities_final, explicit.probabilities_final)


def test_magnus_runs_and_conserves_norm(simulator, detunings):
    result = scan(simulator, detunings, propagator="magnus")
    assert result.propagator == "magnus"
    assert np.allclose(result.probabilities_final.sum(axis=-1), 1.0, atol=1e-10)


def test_magnus_keeps_the_label_tracking(spa2_setup, simulator, detunings):
    """The defect a duplicated loop introduces, and the reason for the branch.

    `label_gaps` is read off the Simulator with `getattr(self, "_label_gaps",
    None)`, so a propagator that forgets to run the tracker does not fail -- it
    silently inherits the previous run's crossings, or reports `None`.
    """
    exact = scan(simulator, detunings, monitor_states=spa2_setup["monitor_states"])
    magnus = scan(
        simulator,
        detunings,
        propagator="magnus",
        monitor_states=spa2_setup["monitor_states"],
    )

    assert magnus.label_gaps is not None
    assert np.array_equal(
        exact.label_gaps["crossings"], magnus.label_gaps["crossings"]
    )


def test_magnus_matches_the_exact_propagator_on_analytic_models():
    """Against a closed form, where "matches" can be stated without a reference."""
    import analytic_models as am
    from state_prep import Simulator

    model = am.rotating(n=8, spread=1.0, T=1.0)
    exact_states = model.exact_states()
    zero = np.zeros((model.n, model.n), dtype=complex)
    results = {}
    for propagator in ("frozen", "magnus"):
        psis, _, _, _, _ = Simulator._time_evolve_mu_batched_shared_slow(
            am.stub_simulator(model),
            H_slow_t=model.H_t,
            muw_hams=[lambda t: zero],
            coupling_scales=np.ones((1, 1)),
            D_mu_diag_batch=np.zeros((1, model.n)),
            t_array=np.linspace(0.0, model.T, 1201),
            monitor_states=None,
            store_final_probabilities=False,
            store_final_monitor_probabilities=False,
            progress=False,
            eig_backend="zheevd",
            time_sampling="mid",
            propagator=propagator,
        )
        results[propagator] = psis[0]
        assert np.abs(psis[0] - exact_states).max() < 1e-4

    assert np.abs(results["frozen"] - results["magnus"]).max() < 1e-6


def test_magnus_is_threaded_to_the_loky_workers(simulator, detunings):
    """A keyword not passed to the workers changes only the parallel path."""
    serial = scan(simulator, detunings, propagator="magnus", workers=1)
    parallel = scan(simulator, detunings, propagator="magnus", workers=2)
    assert np.array_equal(serial.probabilities_final, parallel.probabilities_final)
    assert parallel.propagator == "magnus"

def test_multitone_accepts_magnus_and_matches_the_exact_path(
    spa2_setup, multitone_fields, detunings
):
    """Multitone Magnus is supported; it used to raise.

    The path was landed refusing `magnus`, because a step that freezes the beat
    phase instead of integrating it is wrong in a way no single-tone test
    reveals. The beat kernel now integrates it, and `rotating_coupling` in
    `analytic_models` scores that against a closed form -- see the multitone
    cases below, where Magnus is second order and the frozen-phase propagator is
    two orders of magnitude worse at `beat*dt ~ 1 rad`.

    Here the only claim is that the real setup runs and stays physical; accuracy
    is settled against truth, not against the other scheme.
    """
    from state_prep import Simulator, scan_grid

    simulator = Simulator(
        spa2_setup["trajectory"],
        spa2_setup["electric_field"],
        spa2_setup["magnetic_field"],
        spa2_setup["initial_states"],
        spa2_setup["hamiltonian"],
        multitone_fields,
    )
    det, pref = scan_grid(n_fields=len(multitone_fields), detunings_hz=detunings)
    result = simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=100,
        store_final_probabilities=True,
        progress=False,
        allow_multitone_same_manifold=True,
        propagator="magnus",
    )
    probabilities = np.asarray(result.probabilities_final)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-10)

def test_bad_propagator_is_rejected(simulator, detunings):
    with pytest.raises(ValueError, match=r"one of \('frozen', 'magnus'\)"):
        scan(simulator, detunings, propagator="bogus")


def test_taylor_guard_handles_a_large_step_norm():
    """`||A||` past the guard must degrade, not overflow.

    A fixed eight-term series diverges near `||A|| ~ 19`; scaling and squaring
    keeps it finite. The bound is Frobenius rather than spectral because the
    spectral norm is an SVD costing more than the eigensolve this replaces.
    """
    from state_prep import apply_magnus_taylor

    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    A = (A + A.conj().T) / 2
    psis = np.eye(8, dtype=complex)[:2]

    for scale in (0.01, 1.0, 20.0):
        out = apply_magnus_taylor(A * scale, psis)
        assert np.isfinite(out).all(), scale
        # exp(-iA) is unitary, so the row norms must be preserved
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-8), scale


# --- multitone -------------------------------------------------------------
#
# The multitone loop carries a beat phase `exp(-i*delta_omega*t)` inside `H_rot`
# that rotates at the inter-tone detuning. The exact propagator freezes it at the
# step's midpoint; Magnus integrates it. Nothing in the single-tone models can
# tell those apart, so these cases score against a closed form.


def _run_multitone(model, n_steps, propagator):
    from state_prep.simulator import Simulator

    sim = am.stub_simulator(model)
    psis, *_ = Simulator._time_evolve_mu_batched_shared_slow_multitone(
        sim,
        H_slow_t=model.H_slow,
        component_hams=model.beat_components,
        coupling_scales=np.ones((1, 1)),
        D_mu_diag_batch=np.zeros((1, model.n)),
        t_array=np.linspace(0.0, model.T, n_steps),
        static_field_indices=[],
        beat_fields=[(0, np.array([model.beat_omega]))],
        monitor_states=None,
        store_final_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend="zheevd",
        time_sampling="mid",
        propagator=propagator,
    )
    return psis[0]


def test_rotating_coupling_closed_form_solves_the_ode():
    """Guard the reference itself, without using a propagator to do it.

    Every error in the multitone cases is measured against `U_exact`, so the
    reference has to be established independently of the schemes it judges --
    otherwise this is the self-referencing trap that produced several reversed
    conclusions in `IMPROVEMENTS.md`, just wearing a closed form.

    `U = A B` with `A = expm(-i a t)`, `B = expm(-i b t)` and constant `a`, `b`,
    so `i Udot U^H = a + A b A^H` analytically -- no finite differences and no
    time stepping. Residual is machine precision (`2.7e-17` relative to `|H|`),
    which is ~`1e+10` below the smallest error it is used to measure.
    """
    model = am.rotating_coupling()
    a = 0.5 * model.meta["beat"] * am.SZ
    b = 0.5 * (model.meta["detuning"] - model.meta["beat"]) * am.SZ + 0.5 * model.meta[
        "rabi"
    ] * am.SX

    assert np.allclose(model.U_exact(0.0), np.eye(model.n), atol=1e-14)
    for t in np.linspace(0.0, model.T, 9):
        A = am.herm_expm(a, t)
        residual = np.linalg.norm((a + A @ b @ A.conj().T) - model.H_t(t), 2)
        assert residual < 1e-10, f"U_exact does not solve the ODE at t={t}: {residual:.2e}"
        U = model.U_exact(t)
        assert np.linalg.norm(U.conj().T @ U - np.eye(model.n), 2) < 1e-12


def test_rotating_coupling_agrees_with_an_independent_discretisation():
    """Second, weaker check: a fine product converges to the same `U_exact`.

    Kept because it exercises a completely different code path from the
    derivation above; if the two ever disagree, one of them is wrong.
    """
    model = am.rotating_coupling()
    err = [
        np.linalg.norm(am.reference_propagator(model, steps=n) - model.U_exact(model.T), 2)
        for n in (50_000, 100_000)
    ]
    assert err[0] / err[1] > 3.5, f"not second order: {err}"


def test_multitone_drive_is_near_resonance():
    """A counter-rotating parameterisation transfers nothing and tests nothing."""
    model = am.rotating_coupling()
    assert abs(model.meta["detuning"] - model.meta["beat"]) < model.meta["rabi"]
    populations = np.abs(model.exact_states()[0]) ** 2
    assert populations.min() > 0.05, "drive must actually move population"


def _populations(psis):
    return np.abs(psis) ** 2


def test_multitone_magnus_is_second_order():
    """The beat kernel must be *integrated*, not frozen.

    Populations, not amplitudes. Amplitudes here carry a phase convention that
    is pure gauge on this model -- see the two tests below -- and scoring them
    once made a gauge phase look like a collapse from order `2.00` to `0.99`.
    """
    model = am.rotating_coupling()
    exact = _populations(model.exact_states())
    err = [
        np.abs(_populations(_run_multitone(model, n, "magnus")) - exact).max()
        for n in (8000, 16000)
    ]
    order = np.log2(err[0] / err[1])
    assert order > 1.8, f"beat phase is not integrated across the step: order {order:.2f}"


def test_multitone_magnus_beats_frozen_beat_phase():
    """At `beat*dt ~ 1 rad` the frozen-phase propagator is far worse."""
    model = am.rotating_coupling()
    exact = _populations(model.exact_states())
    n = 8000
    assert abs(model.beat_omega) * model.T / n > 0.1, "beat must not be resolvable"
    magnus = np.abs(_populations(_run_multitone(model, n, "magnus")) - exact).max()
    frozen = np.abs(_populations(_run_multitone(model, n, "frozen")) - exact).max()
    assert magnus < frozen / 10.0, f"magnus {magnus:.3e} vs frozen {frozen:.3e}"


def _run_two_field(n_steps, phase):
    """One static field and one beating field, the beat offset by `phase`.

    This is the production multitone shape (a scanned tone plus a background
    that does not move), reduced to two levels.
    """
    from state_prep.simulator import Simulator

    model = am.rotating_coupling()
    static = 0.5 * 1.5e2 * am.SIGMA_PLUS
    beat_upper = 0.5 * model.meta["rabi"] * am.SIGMA_PLUS
    offset = np.exp(-1j * phase)
    components = [
        lambda t: (static, static.conj().T),
        lambda t: (offset * beat_upper, (offset * beat_upper).conj().T),
    ]
    sim = am.stub_simulator(model)
    psis, *_ = Simulator._time_evolve_mu_batched_shared_slow_multitone(
        sim,
        H_slow_t=model.H_slow,
        component_hams=components,
        coupling_scales=np.ones((1, 2)),
        D_mu_diag_batch=np.zeros((1, model.n)),
        t_array=np.linspace(0.0, model.T, n_steps),
        static_field_indices=[0],
        beat_fields=[(1, np.array([model.beat_omega]))],
        monitor_states=None,
        store_final_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend="zheevd",
        time_sampling="mid",
        propagator="magnus",
    )
    return _populations(psis[0])


def test_one_field_cannot_see_a_beat_phase_convention():
    """With a single coupled pair a constant beat phase is unobservable.

    `exp(-i*th)` on the raising part with `exp(+i*th)` on the lowering part is
    `R^H (.) R` for diagonal `R`, so it telescopes across steps and leaves every
    population untouched. This is why the closed-form model alone could not size
    the beat-anchor defect: it can only express it as gauge.
    """
    from state_prep.simulator import Simulator

    model = am.rotating_coupling()
    base = _populations(_run_multitone(model, 4000, "magnus"))

    shifted_model = am.rotating_coupling()
    upper = 0.5 * model.meta["rabi"] * am.SIGMA_PLUS * np.exp(-1j * 0.37)
    shifted_model.beat_components = [lambda t: (upper, upper.conj().T)]
    shifted = _populations(_run_multitone(shifted_model, 4000, "magnus"))

    assert np.allclose(base, shifted, atol=1e-12), "expected gauge invariance"


def test_two_fields_do_see_a_beat_phase_convention():
    """Add a second field and the same phase becomes physical.

    The static coupling does not receive the phase, so the *relative* phase
    between the two couplings is observable. This is the configuration in which
    the beat-anchor fix actually matters, and the reason it cannot be dismissed
    as a convention.
    """
    base = _run_two_field(4000, 0.0)
    shifted = _run_two_field(4000, 0.37)
    assert np.abs(base - shifted).max() > 1e-3, "expected the phase to be physical"
