"""Time-dependent Hamiltonians with known exact solutions.

Every convergence claim in this repository has so far been scored against a
finer run of some scheme, and each time the reference's own error turned out to
be comparable to the quantity being measured -- `N_steps=128000` on a uniform
grid sat `2.5e-04` from converged and inverted a conclusion about graded grids.
These models remove the reference: the answer is known in closed form, or (for
two-level systems) computable to the floating-point floor.

They also close a gap `AGENTS.md:58-61` names explicitly -- the suite "checks
internal consistency, so a change to the underlying `centrex_tlf` Hamiltonian
would move every result together and leave all tests passing". Nothing here
imports `centrex_tlf`, so these are the first checks against an answer that is
true independently of the code.

**Why the spectral scale is the parameter that matters.** The observed order of
the shipped midpoint scheme collapses once `delta*dt` exceeds about `1`:

    scale   delta*dt @N=2000   err N=2000   err N=4000   observed order
    1e+00           2.95e-03    8.031e-08    2.008e-08             2.00
    1e+02           2.95e-01    6.818e-06    1.704e-06             2.00
    1e+04           2.95e+01    2.689e-03    2.747e-04             3.29
    1e+05           2.95e+02    5.162e-04    2.436e-04             1.08
    1e+06           2.95e+03    1.265e-03    3.533e-04             1.84

The scheme *is* second order; below `delta*dt ~ 1` it shows exactly that. Above,
the error has not entered its asymptotic power law and "order" stops being a
meaningful quantity. SPA2 runs at `delta*dt ~ 7.65e+04`, and reaching `1` would
need `N ~ 7.6e+08` steps. Every model here therefore takes a `spread` so both
regimes can be produced: `clean` to verify a propagator is implemented correctly
and attains its formal order, `realistic` to rank propagators where it counts.

Models, all verified numerically before being committed:

- `rotating` -- `H = e^{-iGt} H0 e^{iGt}`, exact `U = e^{-iGt} e^{-i(H0-G)t}`.
  Non-commuting, fixed spectrum, slowly rotating eigenbasis. The structure of the
  real problem.
- `scalar` -- `H = f(t) H0`, exact `U = exp(-i F(t) H0)`. All `H(t)` commute, so
  the only error is quadrature of `F` and midpoint is provably second order. The
  sharp diagnostic: failure here in the clean regime is a bug, not physics.
- `rosen_zener` -- sech pulse, constant detuning. Closed-form transfer, verified
  to `5e-10`.
- `allen_eberly` -- sech pulse with tanh chirp: envelope *and* sweep, which is
  SPA2's actual structure. Closed-form transfer, verified to `2e-11`.
- `landau_zener` -- linear sweep, constant coupling. Its `exp(-pi Om^2/2a)` is a
  `t -> +/-inf` asymptote and is **not** an oracle: over a finite window it is
  off by `2e-03` to `6e-03`. Physics sanity check only.
- `spa2_replica` -- Gaussian envelope with a U-shaped sweep crossing resonance
  twice, as SPA2 does. No closed form and none needed: a fine two-level product
  converges to `~1e-11`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from scipy.special import erf

import numpy as np

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def hermitian(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random Hermitian matrix. Same construction as the GPU benchmarks."""
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a + a.conj().T) / 2


def herm_expm(H: np.ndarray, dt: float) -> np.ndarray:
    """`exp(-i H dt)` for Hermitian `H`, by eigendecomposition."""
    w, W = np.linalg.eigh(H)
    return (W * np.exp(-1j * w * dt)[None, :]) @ W.conj().T


def _rescale_spread(H: np.ndarray, spread: float) -> np.ndarray:
    """Scale `H` so its eigenvalue spread is exactly `spread` rad/s."""
    w = np.linalg.eigvalsh(H)
    current = float(w.max() - w.min())
    if current <= 0:
        raise ValueError("degenerate Hamiltonian has no spread to rescale")
    return H * (spread / current)


@dataclass
class Model:
    """A time-dependent problem with a known answer.

    `U_exact` is `None` for models with no closed form; use `reference_propagator`
    there, which is machine-exact for `n = 2`.
    """

    name: str
    H_t: Callable[[float], np.ndarray]
    T: float
    n: int
    psi0: np.ndarray
    U_exact: Optional[Callable[[float], np.ndarray]] = None
    # Probability of ending in state |2>, i.e. of *transferring*. Uniform across
    # every two-level model here. Landau-Zener's textbook `exp(-pi Om^2/2a)` is
    # the probability of *staying*, so it is stored as its complement -- mixing
    # the two conventions silently turns a `2e-03` agreement into `9e-02`.
    analytic_transfer: Optional[float] = None
    # Physical split for the transfer models: detuning in `H_slow`, pulse in
    # `muw_hams`. This is what makes them usable for testing GRADED grids -- the
    # shipped `field_variation_density` takes both, and a model that hides the
    # pulse inside `H_slow` cannot exercise the microwave half of the monitor.
    H_slow: Optional[Callable[[float], np.ndarray]] = None
    muw_hams: Optional[list] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def delta_dt(self, n_steps: int) -> float:
        """`spectral spread * dt`, the number that decides whether order exists."""
        return float(self.meta["spread"]) * self.T / (n_steps - 1)

    def exact_states(self) -> np.ndarray:
        """Final states, `(S, n)` row vectors, from the closed form."""
        if self.U_exact is None:
            raise ValueError(f"model {self.name!r} has no closed form")
        return self.psi0 @ self.U_exact(self.T).T


# ---------------------------------------------------------------------------
# n-level models with closed forms
# ---------------------------------------------------------------------------


def rotating(
    *, n: int = 6, spread: float = 1.0, rotation: float = 0.3, T: float = 1.0, seed: int = 0
) -> Model:
    """`H(t) = e^{-iGt} H0 e^{+iGt}`; `U(t) = e^{-iGt} e^{-i(H0-G)t}`.

    Derivation: substituting `psi = e^{-iGt} phi` into `i psi' = H psi` gives
    `i phi' = (H0 - G) phi`, which is time-independent.

    Fixed spectrum with a rotating eigenbasis, which is the real problem's
    structure: `H0` carries the large energies and `G` sets how fast the
    eigenvectors turn. Non-commuting, so the frozen-`H` step makes its genuine
    time-ordering error here.
    """
    rng = np.random.default_rng(seed)
    H0 = _rescale_spread(hermitian(rng, n), spread)
    G = hermitian(rng, n) * rotation
    psi0 = np.eye(n, dtype=complex)[:2]

    def H_t(t: float) -> np.ndarray:
        R = herm_expm(G, float(t))
        return R @ H0 @ R.conj().T

    def U_exact(t: float) -> np.ndarray:
        return herm_expm(G, float(t)) @ herm_expm(H0 - G, float(t))

    return Model(
        name="rotating",
        H_t=H_t,
        T=T,
        n=n,
        psi0=psi0,
        U_exact=U_exact,
        meta={"spread": spread, "rotation": rotation, "seed": seed, "commuting": False},
    )


def scalar(
    *, n: int = 6, spread: float = 1.0, T: float = 1.0, seed: int = 0, rate: float = 1.3,
    profile: str = "smooth", peak_amp: float = 400.0, peak_sigma: float = 0.03,
) -> Model:
    """`H(t) = f(t) H0`; `U(t) = exp(-i F(t) H0)` with `F = integral of f`.

    All `H(t)` commute, so the time-ordering error vanishes identically and the
    only error left is the quadrature of `F`. Midpoint quadrature is second order
    and left-endpoint is first, exactly and provably -- which makes this the
    sharpest available check that a loop is implemented correctly. A propagator
    that misses those orders in the clean regime has a bug.

    `profile` selects the time dependence, and it decides whether the model can
    say anything about **graded grids**:

    - `"smooth"`: `f = 1 + 0.7 sin(rate*t)`. `||dH/dt||` varies by about `2x`
      across the trajectory, so there is nothing for a graded grid to exploit and
      grading can only lose. Use it for order and correctness checks.
    - `"peaked"`: `f = 1 + A exp(-(t-T/2)^2 / 2 s^2)`. With the defaults
      `||dH/dt||` spans about three orders of magnitude, which is the situation
      the real trajectory presents (`63774x`, concentrated at the Stark ramp).
      **Use this to test grading.** Its antiderivative is an error function, so
      the closed form survives.

    Testing a graded grid on a flat density measures nothing; that is why the
    two profiles are separate rather than one compromise.
    """
    rng = np.random.default_rng(seed)
    H0 = _rescale_spread(hermitian(rng, n), spread)
    psi0 = np.eye(n, dtype=complex)[:2]
    centre, width = 0.5 * T, peak_sigma * T

    if profile == "smooth":
        def f(t: float) -> float:
            return 1.0 + 0.7 * np.sin(rate * float(t))

        def F(t: float) -> float:
            return float(t) - 0.7 / rate * (np.cos(rate * float(t)) - 1.0)
    elif profile == "peaked":
        scale = peak_amp * width * np.sqrt(np.pi / 2.0)

        def f(t: float) -> float:
            return 1.0 + peak_amp * float(
                np.exp(-((float(t) - centre) ** 2) / (2.0 * width**2))
            )

        def F(t: float) -> float:
            return float(t) + scale * float(
                erf((float(t) - centre) / (width * np.sqrt(2.0)))
                - erf((0.0 - centre) / (width * np.sqrt(2.0)))
            )
    else:
        raise ValueError(f"profile must be 'smooth' or 'peaked'; got {profile!r}")

    return Model(
        name=f"scalar-{profile}",
        H_t=lambda t: f(t) * H0,
        T=T,
        n=n,
        psi0=psi0,
        U_exact=lambda t: herm_expm(H0, F(t)),
        meta={"spread": spread, "seed": seed, "rate": rate, "commuting": True,
              "profile": profile},
    )


def engineered(
    *, n: int = 6, spread: float = 1.0, T: float = 1.0, seed: int = 0,
    centre: float = 0.45, width: float = 0.05, angle: float = 3.0,
) -> Model:
    """A closed form built by choosing `U(t)` first, then reading off `H(t)`.

    The other models here are the two textbook cases -- commuting `H` (`scalar`)
    and a frame transformation that makes `H` constant (`rotating`) -- and
    between them they cannot produce what testing graded grids and Magnus
    actually needs: a **closed form at any spectral scale, with a non-flat
    density and a non-commuting localised coupling**. `rotating` has a flat
    density by construction; `scalar` needs its coupling proportional to `H0`.

    Inverting the problem removes that limit. Any smooth unitary path `U(t)`
    defines an exactly solvable `H(t) = i U'(t) U(t)^H`. Writing
    `U = V(t) exp(-i Phi(t))` gives

        H(t) = V diag(Phi') V^H + i V' V^H

    and choosing `V = exp(-i G theta(t))` makes the second term exactly
    `theta'(t) G`. So:

        H(t) = exp(-i G theta) diag(w) exp(i G theta) + theta'(t) G
        U(t) = exp(-i G theta(t)) diag(exp(-i w t))

    Every knob is then independent: `w` sets the spectral spread and therefore
    `delta*dt`; `theta'` localised (a Gaussian bump of width `width` at `centre`)
    makes the density non-flat and supplies the coupling; `G` does not commute
    with `diag(w)`, so the time-ordering error is genuine. With the defaults
    about `78%` of the trajectory is quiet, comparable to SPA2's `63%`.

    `H_slow` and `muw_hams` split it the physical way -- rotating spectrum in the
    slow part, localised `theta'(t) G` as the microwave term -- and the two still
    sum to `H_t`, so the closed form survives the split.
    """
    rng = np.random.default_rng(seed)
    G = hermitian(rng, n)
    G = G / np.linalg.norm(G, 2)
    w = np.linspace(-0.5, 0.5, n) * spread

    root = width * np.sqrt(2.0)
    offset = erf((0.0 - centre) / root)

    def theta(t: float) -> float:
        return angle * width * np.sqrt(np.pi / 2.0) * float(
            erf((float(t) - centre) / root) - offset
        )

    def theta_dot(t: float) -> float:
        return angle * float(np.exp(-((float(t) - centre) ** 2) / (2.0 * width**2)))

    def H_slow(t: float) -> np.ndarray:
        R = herm_expm(G, theta(t))
        return R @ np.diag(w.astype(complex)) @ R.conj().T

    def H_mu(t: float) -> np.ndarray:
        return theta_dot(t) * G

    return Model(
        name="engineered",
        H_t=lambda t: H_slow(t) + H_mu(t),
        T=T,
        n=n,
        psi0=np.eye(n, dtype=complex)[:2],
        U_exact=lambda t: herm_expm(G, theta(t)) @ np.diag(np.exp(-1j * w * float(t))),
        meta={"spread": spread, "seed": seed, "commuting": False,
              "centre": centre, "width": width, "angle": angle},
        H_slow=H_slow,
        muw_hams=[H_mu],
    )


def spa_like(
    *, n: int = 16, spread: float = 1.0, T: float = 1.0, seed: int = 0,
    ramp_centre: float = 0.12, ramp_width: float = 0.04, ramp_depth: float = 4.5e-5,
    sweep_depth: float = 4.5e-5,
    beam_centre: float = 0.65, beam_width: float = 0.05, beam_share: float = 0.1,
) -> Model:
    """SPA2's geometry, with a closed form.

    The other models each miss something the real problem has. `rotating` has a
    flat density (`1.01x`); `scalar` is commuting, so it collapses to a scalar
    quadrature; `engineered` is generic. This one reproduces the three features
    that actually drive SPA2's numerics:

    1. **Density peaked at the Stark ramp.** `omega_k(t)` carries a `tanh` ramp
       at `ramp_centre`, so `||dH/dt||` peaks near the start of the flight, as
       the measured profile does (peak at `z = -80 mm`, `63%` of the flight
       below `10%` of peak).
    2. **The beam somewhere else.** `theta'(t)` is a Gaussian bump at
       `beam_centre`, well away from the density peak -- which is why on SPA2 a
       graded grid puts *long* steps where the coupling lives, and why
       `max ||A||` rises under grading there rather than falling.
    3. **A U-shaped sweep, so resonance is crossed twice.** The quadratic term in
       `omega_k(t)` dips and returns, as the Stark-shifted transition frequency
       does, which is what produces partial Landau-Zener transfer at a weakly
       coupled crossing.

    Construction is the inverse method: pick `U(t)`, read off `H = i U' U^H`.
    With `U = exp(-i G theta(t)) diag(exp(-i Integral omega_k))`,

        H(t) = V diag(omega(t)) V^H + theta'(t) G,   V = exp(-i G theta(t))

    Both `omega` terms integrate in closed form (`Integral tanh = log cosh`, and
    the quadratic trivially), so `U_exact` is exact at any spectral scale --
    including production `delta*dt`, which no brute-force oracle reaches.

    **The depth parameters are ratios of the spectral spread, and their SPA2
    values are tiny.** The Stark shift sweeps the transition frequency by about
    `3.6 MHz` against a `5.0e+11 rad/s` rotational spread, so `ramp_depth` and
    `sweep_depth` default to `4.5e-05`. Setting them to order-unity values -- as
    an earlier version did -- makes the eigenvalues traverse most of the spectrum
    within the flight, which is enormously harder than the real problem and
    drives every propagator to a maximal `||psi - exact|| ~ 2` at every usable
    step count. The *shape* matters, but so do the magnitudes.

    **A structural limit worth knowing: this model cannot have both a
    SPA-realistic coupling strength and a ramp-dominated density.** The coupling
    `theta'(t) G` is the generator that rotates the eigenbasis, so it contributes
    `~theta' * spread` to `||dH/dt||` while the ramp contributes
    `~ramp_depth * spread / ramp_width`. Requiring the ramp to dominate forces
    `theta' < ramp_depth / ramp_width`, i.e. `||H_mu|| / spread` around `1e-11`
    against SPA2's `5e-07`. The two are tied together by the construction.

    The default resolves it in favour of the **density profile**, since that is
    what this model exists to get right: `beam_share` sets the beam's share of
    the density (default `0.1`, so the ramp still peaks) and the coupling is
    correspondingly weaker than SPA2's. **Use it for grading geometry; use
    `engineered` or `rotating` with `split_model` when what matters is a
    realistic `||A|| = ||H_mu|| * dt` for Magnus.**

    **Do not use this to evaluate splitting methods with the natural split.**
    Like `engineered`, it is built as a product of two exponentials, which is the
    Lie-split form; Lie reproduces that structure and looks `73000x` better than
    it is. Midpoint and Magnus are not favoured -- they measure identically on
    it -- but any integrator sharing the `V`/`Phi` factorisation will be.
    """
    rng = np.random.default_rng(seed)
    G = hermitian(rng, n)
    G = G / np.linalg.norm(G, 2)

    base = np.linspace(-0.5, 0.5, n) * spread
    amp_ramp = rng.uniform(0.5, 1.0, n) * ramp_depth * spread
    amp_sweep = rng.uniform(0.5, 1.0, n) * sweep_depth * spread
    # `beam_share` is the beam's share of the density, NOT a coupling strength.
    # See the docstring: in this construction the coupling generates the basis
    # rotation, so its density contribution is `theta' * spread` while the ramp's
    # is `ramp_depth * spread / ramp_width`. Fixing the ratio of the two is the
    # only way to control the density profile.
    beam_angle = beam_share * ramp_depth / ramp_width

    # Force levels 0 and 1 to cross TWICE, as SPA2's transition frequency does.
    # Their difference is then `S * (u^2 - 1/12)` with `u = t/T - 1/2`, which
    # vanishes at `u = +-1/sqrt(12)`, i.e. `t/T = 0.211` and `0.789` -- both
    # inside the window. Equal `base` and equal ramp amplitude keep the ramp out
    # of the gap, so the quadratic alone sets where the crossings fall.
    base[1] = base[0]
    amp_ramp[1] = amp_ramp[0]
    amp_sweep[1] = amp_sweep[0] + sweep_depth * spread

    def omega(t: float) -> np.ndarray:
        x = (float(t) - ramp_centre) / ramp_width
        u = float(t) / T - 0.5
        return base + amp_ramp * np.tanh(x) + amp_sweep * (u**2 - 1.0 / 12.0)

    def omega_integral(t: float) -> np.ndarray:
        t = float(t)
        x, x0 = (t - ramp_centre) / ramp_width, (0.0 - ramp_centre) / ramp_width
        ramp = ramp_width * (np.logaddexp(x, -x) - np.logaddexp(x0, -x0))
        u, u0 = t / T - 0.5, -0.5
        quad = T * (u**3 - u0**3) / 3.0 - t / 12.0
        return base * t + amp_ramp * ramp + amp_sweep * quad

    root = beam_width * np.sqrt(2.0)
    off = erf((0.0 - beam_centre) / root)

    def theta(t: float) -> float:
        return beam_angle * beam_width * np.sqrt(np.pi / 2.0) * float(
            erf((float(t) - beam_centre) / root) - off
        )

    def theta_dot(t: float) -> float:
        return beam_angle * float(
            np.exp(-((float(t) - beam_centre) ** 2) / (2.0 * beam_width**2))
        )

    def H_slow(t: float) -> np.ndarray:
        V = herm_expm(G, theta(t))
        return V @ np.diag(omega(t).astype(complex)) @ V.conj().T

    def H_mu(t: float) -> np.ndarray:
        return theta_dot(t) * G

    return Model(
        name="spa_like",
        H_t=lambda t: H_slow(t) + H_mu(t),
        T=T,
        n=n,
        psi0=np.eye(n, dtype=complex)[:2],
        U_exact=lambda t: herm_expm(G, theta(t)) @ np.diag(
            np.exp(-1j * omega_integral(t))
        ),
        meta={"spread": spread, "seed": seed, "commuting": False,
              "ramp_centre": ramp_centre, "beam_centre": beam_centre,
              "ramp_depth": ramp_depth, "beam_share": beam_share,
              "crossings_at": (0.5 - 12**-0.5, 0.5 + 12**-0.5),
              "splitting_biased": True},
        H_slow=H_slow,
        muw_hams=[H_mu],
    )


# ---------------------------------------------------------------------------
# Two-level transfer models -- the microwave analogues
# ---------------------------------------------------------------------------


def _two_level(name, H_t, T, analytic, meta, H_slow=None, muw_hams=None) -> Model:
    return Model(
        name=name,
        H_t=H_t,
        T=T,
        n=2,
        psi0=np.array([[1.0, 0.0]], dtype=complex),
        U_exact=None,
        analytic_transfer=analytic,
        meta=meta,
        H_slow=H_slow,
        muw_hams=muw_hams,
    )


def rosen_zener(*, rabi: float = 1.0, tau: float = 1.0, detuning: float = 0.5,
                windows: float = 40.0) -> Model:
    """Sech pulse at constant detuning.

    `P = sin^2(pi*Om0*tau/2) * sech^2(pi*Delta*tau/2)`, verified to `5e-10`
    against a converged propagator: the sech envelope decays exponentially, so
    truncating the window costs nothing measurable.

    A pulse with no sweep -- the transfer analogue with one of SPA2's two
    ingredients removed.
    """
    T = 2.0 * windows * tau
    t0 = -windows * tau

    def H_slow(t: float) -> np.ndarray:
        return 0.5 * detuning * SZ                      # constant: ||dH_slow/dt|| == 0

    def H_mu(t: float) -> np.ndarray:
        return 0.5 * rabi / np.cosh((float(t) + t0) / tau) * SX

    def H_t(t: float) -> np.ndarray:
        return H_slow(t) + H_mu(t)

    P = np.sin(np.pi * rabi * tau / 2) ** 2 / np.cosh(np.pi * detuning * tau / 2) ** 2
    return _two_level(
        "rosen_zener", H_t, T, float(P),
        {"spread": max(abs(detuning), rabi), "rabi": rabi, "tau": tau,
         "detuning": detuning, "swept": False},
        H_slow=H_slow, muw_hams=[H_mu],
    )


def allen_eberly(*, rabi: float = 1.0, tau: float = 1.0, sweep: float = 0.3,
                 windows: float = 40.0) -> Model:
    """Sech pulse with a tanh chirp -- envelope *and* sweep, as SPA2 has.

    `P = 1 - cos^2(pi/2 sqrt((Om0 tau)^2 - (D0 tau)^2)) / cosh^2(pi D0 tau/2)`,
    verified to `2e-11`.

    This is the closest closed-form analogue of the real microwave transfer: the
    molecule crosses resonance while the coupling is switched on and off by the
    beam profile.
    """
    T = 2.0 * windows * tau
    t0 = -windows * tau

    def H_slow(t: float) -> np.ndarray:
        return 0.5 * sweep * np.tanh((float(t) + t0) / tau) * SZ

    def H_mu(t: float) -> np.ndarray:
        return 0.5 * rabi / np.cosh((float(t) + t0) / tau) * SX

    def H_t(t: float) -> np.ndarray:
        return H_slow(t) + H_mu(t)

    arg = (rabi * tau) ** 2 - (sweep * tau) ** 2
    P = 1.0 - np.cos(np.pi / 2 * np.sqrt(complex(arg))).real ** 2 / np.cosh(
        np.pi * sweep * tau / 2
    ) ** 2
    return _two_level(
        "allen_eberly", H_t, T, float(P),
        {"spread": max(abs(sweep), rabi), "rabi": rabi, "tau": tau,
         "sweep": sweep, "swept": True},
        H_slow=H_slow, muw_hams=[H_mu],
    )


def landau_zener(*, rabi: float = 1.0, alpha: float = 2.0, windows: float = 60.0) -> Model:
    """Linear sweep at constant coupling.

    The textbook result is `P_stay = exp(-pi Om^2 / (2 alpha))`, the probability
    of remaining in the initial diabatic state; `analytic_transfer` holds its
    complement so the convention matches the other models. **This is an
    asymptote, not an oracle.** Measured against a
    converged propagator over a finite window it is off by `2.0e-03` to
    `5.8e-03` at the default `windows=60`, because the linear sweep never
    switches off and the population keeps oscillating as it approaches the limit.
    The error grows fast as the window narrows -- `7e-02` at `windows=30` -- so
    do not shrink it and then trust the formula. Compare at the `1e-2` level as a
    physics check; use `reference_propagator` for accuracy work.
    """
    T = 2.0 * windows / max(np.sqrt(alpha), 1.0)
    t0 = -T / 2

    def H_t(t: float) -> np.ndarray:
        s = float(t) + t0
        return 0.5 * alpha * s * SZ + 0.5 * rabi * SX

    # Stored as the transfer probability, matching every other model here.
    P_stay = float(np.exp(-np.pi * rabi**2 / (2 * alpha)))
    return _two_level(
        "landau_zener", H_t, T, 1.0 - P_stay,
        {"spread": alpha * T / 2, "rabi": rabi, "alpha": alpha,
         "asymptotic_only": True, "finite_window_error": 6e-3, "swept": True},
    )


def spa2_replica(*, rabi: float = 20.0, sigma: float = 0.08, detuning: float = 0.0,
                 depth: float = 6.0, centre: float = 0.65, T: float = 1.0) -> Model:
    """Gaussian envelope with a U-shaped sweep crossing resonance twice.

    Matched to SPA2's geometry rather than to a solvable form. The sweep
    `delta(t) = detuning + depth*(4*(t/T - 1/2)^2 - 1/2)` dips and returns, so it
    crosses resonance at `t/T = 1/2 +/- (1/2)sqrt(1 - 2*detuning/depth)` -- two
    crossings, as the real Stark ramp gives. The beam sits at `centre = 0.65`
    with `sigma = 0.08`, which puts the near crossing about `2.5 sigma` out in
    the Gaussian flank at zero detuning; the far one is `6 sigma` out and
    contributes nothing.

    That is the real `det=-1.0` situation: **partial** Landau-Zener transfer at a
    *weakly coupled* crossing. Defaults give transfer `0.68`-`0.82` across
    `detuning = -2.5..2.5` (the real cell sits at `0.69`), unsaturated and
    sensitive, because varying the detuning slides the crossing along the
    Gaussian tail and the coupling there changes exponentially. Saturated cells
    -- complete or absent transfer -- are insensitive by comparison, which is
    exactly why they converge cleanly and this one does not.

    No closed form, and none is needed: at `n = 2` a fine product propagator
    converges to `~1e-11`. Use `reference_propagator`.
    """

    def H_t(t: float) -> np.ndarray:
        s = float(t)
        delta = detuning + depth * (4.0 * ((s / T) - 0.5) ** 2 - 0.5)
        omega = rabi * np.exp(-((s - centre * T) ** 2) / (2.0 * (sigma * T) ** 2))
        return 0.5 * delta * SZ + 0.5 * omega * SX

    return _two_level(
        "spa2_replica", H_t, T, None,
        {"spread": depth, "rabi": rabi, "sigma": sigma, "detuning": detuning,
         "depth": depth, "centre": centre, "swept": True, "crossings": 2},
    )


SIGMA_PLUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SIGMA_MINUS = SIGMA_PLUS.conj().T


def rotating_coupling(
    *, detuning: float = 3.0e3, rabi: float = 4.0e2, beat: float = 2.8e3, T: float = 1.0
) -> Model:
    """A coupling that rotates at `beat` -- the multitone structure, exactly solved.

    Every other model here drives the single-tone path. The multitone loop is
    different in a way that matters: its `H_rot` carries components multiplied by
    `exp(-i*delta_omega*t)`, rotating at the inter-tone detuning, and a
    propagator that freezes that phase within a step rather than integrating it
    is wrong whenever `delta_omega * dt` is not small. Nothing could test that.

    This is the driven two-level problem,

        H(t) = (Delta/2) sz + (Omega/2) (exp(-i*w*t) s+ + exp(+i*w*t) s-)

    which is exactly the raising/lowering pair with a beat phase that the
    multitone loop assembles. Moving to the frame rotating at `w` makes it
    constant, so

        U(t) = exp(-i*w*t*sz/2) exp(-i[((Delta - w)/2) sz + (Omega/2) sx] t)

    Verified: a fine product converges to this at ratio `4.00` per refinement,
    i.e. exactly second order, at `2.283e-05` for `1.6e+06` steps.

    The defaults put `w * dt = 1.4` rad at `N_steps=2000`, deliberately far from
    small, so a frozen beat phase is measurably wrong rather than marginally so.

    `H_slow` is constant here, so the whole time dependence lives in the
    coupling; `beat_components` gives the `(upper, lower)` pair the multitone
    loop expects.

    **Keep `detuning` and `beat` the same sign.** What makes this a real test is
    that `detuning - beat` is small, so the drive is near resonance and actually
    moves population. Flipping one sign leaves the closed form perfectly valid
    but makes the term counter-rotating: `detuning - beat = -5800` against
    `rabi = 400` transfers `(rabi/rabi_eff)**2 = 0.005` of the population, and
    every scheme then looks catastrophically bad against dynamics that are not
    there. Check the transfer is non-trivial before reading an error.
    """

    def H_slow(t: float) -> np.ndarray:
        return 0.5 * detuning * SZ

    upper = 0.5 * rabi * SIGMA_PLUS
    lower = 0.5 * rabi * SIGMA_MINUS

    def components(t: float):
        return upper, lower

    def H_t(t: float) -> np.ndarray:
        s = float(t)
        return H_slow(s) + np.exp(-1j * beat * s) * upper + np.exp(1j * beat * s) * lower

    def U_exact(t: float) -> np.ndarray:
        s = float(t)
        return herm_expm(0.5 * beat * SZ, s) @ herm_expm(
            0.5 * (detuning - beat) * SZ + 0.5 * rabi * SX, s
        )

    model = Model(
        name="rotating_coupling",
        H_t=H_t,
        T=T,
        n=2,
        psi0=np.array([[1.0, 0.0]], dtype=complex),
        U_exact=U_exact,
        meta={"spread": abs(detuning), "commuting": False, "beat": beat,
              "rabi": rabi, "detuning": detuning, "multitone": True},
        H_slow=H_slow,
        muw_hams=[lambda t: np.exp(-1j * beat * float(t)) * upper
                  + np.exp(1j * beat * float(t)) * lower],
    )
    # What the multitone loop needs beyond the single-tone interface.
    model.beat_components = [components]
    model.beat_omega = beat
    return model


def reference_propagator(model: Model, *, steps: int = 400_000) -> np.ndarray:
    """Machine-precision `U(T)` by a fine midpoint product.

    For `n = 2` this reaches the floating-point floor (`~1e-11`) and *is* the
    oracle. For larger `n` it is still far finer than anything being tested, but
    prefer `model.U_exact` when it exists.
    """
    U = np.eye(model.n, dtype=complex)
    dt = model.T / steps
    for i in range(steps):
        U = herm_expm(model.H_t((i + 0.5) * dt), dt) @ U
    return U


# ---------------------------------------------------------------------------
# Driving the real evolution loops
# ---------------------------------------------------------------------------


class _FakeTrajectory:
    def __init__(self, T: float) -> None:
        self._T = float(T)

    def get_T(self) -> float:
        return self._T


class _FakeHamiltonian:
    def __init__(self, n: int) -> None:
        self.QN = list(range(n))


def stub_simulator(model: Model):
    """A `Simulator` that runs the shipped loops on an arbitrary `H(t)`.

    `Simulator` is a dataclass whose `__post_init__` only zeroes two attributes
    and validates nothing, so it constructs from `None`s. With
    `monitor_states=None` the loops touch only `hamiltonian.QN` (for its length),
    `init_state_vecs`, `psis` and `trajectory.get_T()`.

    `init_state_vecs` is overridden because the real one routes through
    `vector_to_state`, which raises unless `QN` holds centrex basis states. The
    override also lets a model pick its own initial state rather than inheriting
    whichever eigenvector happens to overlap most.
    """
    from state_prep import Simulator

    sim = object.__new__(Simulator)
    sim.trajectory = _FakeTrajectory(model.T)
    sim.electric_field = None
    sim.magnetic_field = None
    sim.hamiltonian = _FakeHamiltonian(model.n)
    sim.microwave_fields = None
    sim.initial_states_approx = []
    sim.initial_states = list(range(model.psi0.shape[0]))
    sim.psis = model.psi0.astype(complex).copy()

    psi0 = model.psi0.astype(complex).copy()

    def init_state_vecs(H_0, V_0=None):
        sim.psis = psi0.copy()
        sim.initial_states = list(range(psi0.shape[0]))

    sim.init_state_vecs = init_state_vecs
    return sim


REGIMES = {
    # delta*dt at N=2000, chosen from the table in the module docstring.
    "clean": 1e-3,
    "realistic": 7.65e4,
}


def spread_for_regime(regime: str, T: float, n_steps: int = 2000) -> float:
    """Spectral spread giving the regime's `delta*dt` at `n_steps`."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}; got {regime!r}")
    return REGIMES[regime] * (n_steps - 1) / T
