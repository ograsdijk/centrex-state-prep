"""Propagators against exact solutions.

Every other test in this suite checks internal consistency -- one code path
against another, or an invariant. `AGENTS.md:58-61` names the gap that leaves:
"a change to the underlying `centrex_tlf` Hamiltonian would move every result
together and leave all tests passing." These are the first checks against an
answer that is true independently of the code.

Nothing here imports `centrex_tlf` or uses the session-scoped `spa2_setup`
fixture, so the whole module runs in seconds. The models live in
`benchmarks/analytic_models.py`, which `conftest.py` already puts on `sys.path`.

All of this runs in the *clean* regime, `delta*dt << 1`. That is deliberate: it
is the only regime where convergence order is a meaningful quantity, so it is
the only regime where a wrong order is evidence of a bug. At the production
`delta*dt ~ 7.65e+04` the error has not reached its asymptotic power law and no
order can be asserted -- see Priority E in `IMPROVEMENTS.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import analytic_models as am
from state_prep import Simulator


N_STEPS = 1201


def run_batched(model, n_steps, time_sampling="mid", impl=None):
    """Drive the shipped batched loop on a synthetic model; return final states."""
    sim = am.stub_simulator(model)
    loop = impl or Simulator._time_evolve_mu_batched_shared_slow
    zero = np.zeros((model.n, model.n), dtype=complex)
    psis, _, _, _, _ = loop(
        sim,
        H_slow_t=model.H_t,
        muw_hams=[lambda t: zero],
        coupling_scales=np.ones((1, 1)),
        D_mu_diag_batch=np.zeros((1, model.n)),
        t_array=np.linspace(0.0, model.T, n_steps),
        monitor_states=None,
        store_final_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend="zheevd",
        time_sampling=time_sampling,
    )
    return psis[0]


def run_field_only(model, n_steps, time_sampling="mid"):
    """Drive the shipped field-only loop on a synthetic model."""
    sim = am.stub_simulator(model)
    Simulator._time_evolve(
        sim,
        H_slow=model.H_t,
        t_array=np.linspace(0.0, model.T, n_steps),
        save_idx=np.array([0, n_steps - 1]),
        store_psis=False,
        store_energies=False,
        store_probabilities=False,
        store_final_probabilities=False,
        monitor_states=None,
        store_monitor_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend="zheevd",
        time_sampling=time_sampling,
    )
    return sim.psis


# ---------------------------------------------------------------------------
# The models are oracles only if their closed forms are right
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [am.rotating, am.scalar])
def test_closed_form_matches_a_fine_product(factory):
    """The n-level closed forms, against brute force. If these drift, nothing below means anything."""
    model = factory(n=4, spread=1.0, T=1.0)
    U = am.reference_propagator(model, steps=20000)
    assert np.linalg.norm(U - model.U_exact(model.T), 2) < 1e-8


@pytest.mark.parametrize(
    "factory,tol",
    [
        (am.rosen_zener, 1e-7),
        (am.allen_eberly, 1e-7),
        # Landau-Zener's formula is a t -> +/-inf asymptote, not an oracle: the
        # sweep never switches off, so a finite window leaves the population
        # still oscillating toward the limit. Checked at the level it is good for.
        (am.landau_zener, 5e-3),
    ],
)
def test_two_level_transfer_matches_analytic(factory, tol):
    model = factory()
    U = am.reference_propagator(model, steps=100_000)
    transfer = abs((U @ model.psi0[0])[1]) ** 2
    assert abs(transfer - model.analytic_transfer) < tol


def test_reference_propagator_is_converged_for_two_level():
    """Brute force is the oracle where no closed form exists -- but only if converged.

    Checked on the transfer probability, which is what the oracle is used for.
    The propagator *matrix* converges more slowly in norm because it also carries
    an accumulated global phase over a long window, and that phase is irrelevant
    to every observable here.
    """
    model = am.spa2_replica()
    probs = [
        abs((am.reference_propagator(model, steps=steps) @ model.psi0[0])[1]) ** 2
        for steps in (25_000, 50_000, 100_000)
    ]
    assert abs(probs[1] - probs[0]) < 1e-9, probs
    assert abs(probs[2] - probs[1]) < 1e-9, probs


# ---------------------------------------------------------------------------
# The shipped loops, against truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("time_sampling", ["mid", "left"])
@pytest.mark.parametrize("factory", [am.rotating, am.scalar])
def test_batched_loop_reproduces_the_exact_solution(factory, time_sampling):
    model = factory(n=4, spread=1.0, T=1.0)
    psis = run_batched(model, N_STEPS, time_sampling)
    assert np.abs(psis - model.exact_states()).max() < 1e-4
    assert np.allclose(np.linalg.norm(psis, axis=-1), 1.0, atol=1e-12)


@pytest.mark.parametrize("factory", [am.rotating, am.scalar])
def test_field_only_and_batched_agree_on_synthetic_models(factory):
    """Two loops, one answer -- now anchored to truth rather than to each other."""
    model = factory(n=4, spread=1.0, T=1.0)
    batched = run_batched(model, N_STEPS)
    field_only = run_field_only(model, N_STEPS)
    assert np.abs(batched - field_only).max() < 1e-10
    assert np.abs(field_only - model.exact_states()).max() < 1e-4


def test_midpoint_is_second_order_and_left_is_first():
    """The order claim, where order is a meaningful quantity.

    `scalar` has all `H(t)` commuting, so the time-ordering error vanishes and
    the only error is quadrature of `F`. Midpoint quadrature is second order and
    left-endpoint first, exactly. This is the sharpest available check that the
    loops are implemented correctly -- and it is why the erratic orders measured
    on the real problem are a statement about `delta*dt`, not about the code.
    """
    model = am.scalar(n=4, spread=1.0, T=1.0)
    exact = model.exact_states()

    orders = {}
    for sampling in ("mid", "left"):
        errors = [
            np.abs(run_batched(model, n + 1, sampling) - exact).max()
            for n in (500, 1000, 2000)
        ]
        # Two independent ratios; both must agree with the claimed order.
        orders[sampling] = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]

    assert all(1.8 < p < 2.2 for p in orders["mid"]), orders
    assert all(0.8 < p < 1.2 for p in orders["left"]), orders


def test_order_collapse_is_a_time_ordering_effect():
    """Pins the mechanism behind the whole convergence story.

    Midpoint is second order at `delta*dt << 1`. Push the spectral scale up and
    the observed order leaves that value entirely -- but **only when `H(t)` fails
    to commute with itself at different times**:

        model      commuting   delta*dt   observed order
        scalar          True       2e+02             2.00
        rotating       False       2e+01            -0.45
        rotating       False       2e+02             0.18

    For commuting `H` the error is pure quadrature of the scalar factor, which is
    second order at any scale. The collapse is therefore a *time-ordering*
    effect: the neglected terms carry powers of `delta*dt`, and once that exceeds
    `1` no asymptotic power law is available to measure.

    SPA2 runs at `delta*dt ~ 7.65e+04` with a non-commuting `H`, which is why no
    order could ever be measured there, why midpoint never showed its formal
    second order, and why Richardson extrapolation is unavailable.
    """
    def order(model):
        exact = model.exact_states()
        errors = [
            np.abs(run_batched(model, n + 1) - exact).max() for n in (500, 1000)
        ]
        return np.log2(errors[0] / errors[1])

    # Non-commuting: order 2 when resolved, gone when not.
    assert 1.8 < order(am.rotating(n=4, spread=1.0, T=1.0)) < 2.2
    rough = am.rotating(n=4, spread=2.0e4, T=1.0)
    assert rough.delta_dt(1000) > 1.0
    assert not (1.8 < order(rough) < 2.2), "expected the power law to be gone"

    # Commuting: order 2 survives the same scale, isolating the cause.
    commuting = am.scalar(n=4, spread=2.0e4, T=1.0)
    assert commuting.delta_dt(1000) > 1.0
    assert 1.8 < order(commuting) < 2.2, (
        "commuting H has no time-ordering error, so order 2 must survive"
    )


def test_spa2_replica_is_an_unsaturated_partial_transfer():
    """The replica must reproduce what makes the real `det=-1.0` cell hard.

    Saturated cells (transfer at 0 or 1) are insensitive and converge cleanly.
    The hard cell is partial transfer at a weakly coupled crossing, which is
    what this asserts -- otherwise the model tests the easy case and proves
    nothing about the regime that actually blocks convergence.
    """
    transfers = []
    for detuning in (-2.0, 0.0, 2.0):
        model = am.spa2_replica(detuning=detuning)
        U = am.reference_propagator(model, steps=20_000)
        transfers.append(abs((U @ model.psi0[0])[1]) ** 2)

    assert all(0.05 < p < 0.95 for p in transfers), transfers
    assert max(transfers) - min(transfers) > 0.05, (
        f"lineshape is flat, so the model is not detuning-sensitive: {transfers}"
    )


# ---------------------------------------------------------------------------
# Graded time grids
# ---------------------------------------------------------------------------


def graded_grid_for(model, n_steps, order):
    """Build the grid the way the shipped code does, from the model's own split."""
    from state_prep import build_time_grid, field_variation_density

    return build_time_grid(
        model.T,
        n_steps,
        density=field_variation_density(model.H_slow, model.muw_hams),
        order=order,
    )


def run_two_level(model, n_steps, grid=None):
    """Drive the batched loop on a two-level transfer model with its real split."""
    sim = am.stub_simulator(model)
    t_array = np.linspace(0.0, model.T, n_steps) if grid is None else grid
    psis, _, _, _, _ = Simulator._time_evolve_mu_batched_shared_slow(
        sim,
        H_slow_t=model.H_slow,
        muw_hams=model.muw_hams,
        coupling_scales=np.ones((1, len(model.muw_hams))),
        D_mu_diag_batch=np.zeros((1, model.n)),
        t_array=t_array,
        monitor_states=None,
        store_final_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend="zheevd",
        time_sampling="mid",
    )
    return psis[0]


def test_rosen_zener_density_comes_only_from_the_microwave():
    """The case that proves `muw_hams` reaches the step-density monitor.

    Rosen-Zener has a *constant* detuning, so `||dH_slow/dt||` is identically
    zero and every bit of structure lives in the pulse. If the shipped
    `field_variation_density` ever stopped summing `muw_hams`, the density here
    would collapse to zero and the graded grid would silently become uniform.
    """
    from state_prep import field_variation_density

    model = am.rosen_zener()
    probes = np.linspace(0.0, model.T, 400)

    slow_only = field_variation_density(model.H_slow)(probes)
    with_microwaves = field_variation_density(model.H_slow, model.muw_hams)(probes)

    assert slow_only.max() == 0.0, "H_slow should be constant for Rosen-Zener"
    assert with_microwaves.max() > 0.1, with_microwaves.max()


@pytest.mark.parametrize("factory,floor", [(am.rosen_zener, 20.0), (am.allen_eberly, 20.0)])
def test_grading_beats_uniform_on_a_localised_pulse(factory, floor):
    """Grading against exact solutions, on a pulse that is mostly quiet time.

    `exp(-i H dt)` is exact for constant `H`, so a stretch where the fields
    barely move costs nothing to cross in one long step. These models are quiet
    for most of their window, which is where that pays: measured gains are
    `100x`-`245x`, so a floor of `20x` is loose enough to be stable and tight
    enough to catch the grid silently degenerating to uniform.

    The gain is a strong function of how much of the trajectory is quiet -- on
    the real SPA2 profile, which is `61%` quiet rather than `97%`, it is
    `1.1x`-`7.7x`. Do not read this number as applying to production runs.
    """
    model = factory()
    truth = model.psi0 @ am.reference_propagator(model, steps=50_000).T

    n_steps = 1000
    uniform = np.abs(run_two_level(model, n_steps) - truth).max()
    graded = min(
        np.abs(run_two_level(model, n_steps, graded_grid_for(model, n_steps, order)) - truth).max()
        for order in (1, 2)
    )
    assert uniform / graded > floor, f"uniform {uniform:.3e}, best graded {graded:.3e}"


def test_grading_needs_a_non_flat_density_to_help():
    """A flat density means there is nothing to grade, and grading then costs.

    Recorded as a test because it produced a confident wrong answer during
    development: a model whose `||dH/dt||` range was `1.01x` showed grading
    losing, and that was reported as evidence against the feature. It was
    evidence about the model. Always check the dynamic range before reading a
    grading result -- below about `10x` the comparison is meaningless.
    """
    from state_prep import field_variation_density

    flat = am.rotating(n=4, spread=1.0, rotation=0.3, T=1.0)
    probes = np.linspace(0.0, flat.T, 200)
    density = field_variation_density(flat.H_t)(probes)

    assert density.max() / density.min() < 2.0, (
        "the rotating model is supposed to have an essentially flat density; "
        f"got {density.max() / density.min():.2f}x"
    )


def test_engineered_model_has_the_properties_the_others_lack():
    """The three models, and why a third was needed.

    Testing a graded grid or the Magnus propagator needs a closed form, a
    non-flat density, *and* a non-commuting localised coupling, all at once.
    `rotating` has a flat density by construction; `scalar` must have its
    coupling proportional to `H0` to stay solvable. Several measurements during
    development were invalidated by not checking that before reading a result.
    """
    from state_prep import field_variation_density

    engineered = am.engineered(n=6, spread=1.0e4, T=1.0)
    probes = np.linspace(0.0, engineered.T, 300)

    # closed form, and the split preserves it
    U = am.reference_propagator(engineered, steps=40_000)
    assert np.linalg.norm(U - engineered.U_exact(engineered.T), 2) < 1e-7
    assert np.allclose(
        engineered.H_slow(0.3) + engineered.muw_hams[0](0.3), engineered.H_t(0.3)
    )

    # non-flat density, unlike `rotating`
    density = field_variation_density(engineered.H_slow, engineered.muw_hams)(probes)
    assert (density < 0.1 * density.max()).mean() > 0.5, "should be mostly quiet"
    flat = field_variation_density(am.rotating(n=6, spread=1.0e4).H_t)(probes)
    assert flat.max() / flat.min() < 2.0, "rotating is the flat-density control"

    # non-commuting, unlike `scalar`
    a, b = engineered.H_t(0.2), engineered.H_t(0.6)
    assert np.linalg.norm(a @ b - b @ a, 2) > 0.0


def test_magnus_step_norm_tracks_the_grid():
    """`||A||` is a property of the grid, and grading changes it.

    Guards the interaction between two options that are fine alone: a graded
    grid stretches `dt`, and Magnus's accuracy and cost both scale with
    `||A|| = ||H_mu|| * dt`. Which way it moves is geometry -- here the density
    peaks where the coupling is, so grading shortens steps there and `||A||`
    falls. On SPA2 the density peaks at the Stark ramp instead and `max ||A||`
    rises. Either way it is checkable without propagating anything.
    """
    from common import magnus_step_norms
    from state_prep import build_time_grid, field_variation_density

    model = am.engineered(n=6, spread=1.0e4, T=1.0, angle=200.0)
    n_steps = 2000
    uniform = np.linspace(0.0, model.T, n_steps)
    graded = build_time_grid(
        model.T,
        n_steps,
        density=field_variation_density(model.H_slow, model.muw_hams),
        order=1,
    )

    a_uniform = magnus_step_norms(model.muw_hams, uniform).max()
    a_graded = magnus_step_norms(model.muw_hams, graded).max()

    assert a_uniform > 0.0 and a_graded > 0.0
    assert np.diff(graded).max() / np.diff(uniform).max() > 5.0, "grid should be graded"
    # Here the coupling sits on the density peak, so grading *reduces* ||A||.
    assert a_graded < a_uniform, (a_uniform, a_graded)


def test_spa_like_reproduces_the_real_geometry():
    """`spa_like` is the model meant to transfer to production, so pin its shape.

    Three features, all of which drove real conclusions:

    - the density peaks at a Stark-like ramp near the start, not at the beam;
    - the beam sits well away from that peak, which is why a graded grid puts
      *long* steps where the coupling lives and `max ||A||` rises rather than
      falls;
    - a level pair crosses twice, giving partial transfer at a weakly coupled
      crossing -- the structure behind the `det=-1.0` conditioning.
    """
    from state_prep import field_variation_density

    model = am.spa_like(n=12, spread=1.0e4, T=1.0)
    probes = np.linspace(0.0, model.T, 400)

    U = am.reference_propagator(model, steps=60_000)
    assert np.linalg.norm(U - model.U_exact(model.T), 2) < 1e-6

    density = field_variation_density(model.H_slow, model.muw_hams)(probes)
    peak_at = probes[int(np.argmax(density))]
    assert abs(peak_at - model.meta["ramp_centre"]) < 0.1, peak_at
    assert (density < 0.1 * density.max()).mean() > 0.5

    coupling = np.array(
        [np.linalg.norm(model.muw_hams[0](float(t)), 2) for t in probes]
    )
    beam_at = probes[int(np.argmax(coupling))]
    assert abs(beam_at - model.meta["beam_centre"]) < 0.05, beam_at
    # The point of the model: these must be separated, as they are on SPA2.
    assert abs(beam_at - peak_at) > 0.3, (beam_at, peak_at)

    gaps = np.array(
        [np.diff(np.linalg.eigvalsh(model.H_slow(float(t))))[0] for t in probes]
    )
    near = probes[gaps < 0.02 * gaps.max()]
    assert near.size >= 2, "expected a level pair to approach degeneracy twice"
    assert near.max() - near.min() > 0.3, "the two crossings should be separated"


def test_spa_like_grading_raises_the_magnus_step_norm():
    """The interaction that only shows up when beam and density peak differ.

    Which way it goes depends on how visible the beam is in the density. Once
    the beam is a few percent of the density peak -- SPA2's is about `1/3300` --
    the graded grid stops shortening steps there, the long steps land near the
    coupling, and `||A||` **rises**. That is the direction that can walk into the
    Taylor guard, so pin it in the regime that matches.

    With a beam large enough to shape the density the sign flips: measured
    `2.814e-07 -> 1.771e-07` at `beam_share=0.5`, against
    `5.627e-08 -> 6.307e-08` at `0.1`. Testing the wrong regime asserts the wrong
    sign, which is how this test first failed.
    """
    from common import magnus_step_norms
    from state_prep import build_time_grid, field_variation_density

    model = am.spa_like(n=12, spread=1.0e6, T=1.0, beam_share=0.05)
    n_steps = 2000
    uniform = np.linspace(0.0, model.T, n_steps)
    graded = build_time_grid(
        model.T,
        n_steps,
        density=field_variation_density(model.H_slow, model.muw_hams),
        order=1,
    )
    assert np.diff(graded).max() / np.diff(uniform).max() > 5.0

    assert magnus_step_norms(model.muw_hams, graded).max() > magnus_step_norms(
        model.muw_hams, uniform
    ).max()
