import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import centrex_tlf
import dill
from joblib import Parallel, delayed, effective_n_jobs
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.lapack import zheevd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .electric_fields import ElectricField
from .hamiltonians import Hamiltonian
from .magnetic_fields import MagneticField
from .microwaves import MicrowaveField, build_rotating_frame_shift
from .trajectory import Trajectory
from .utils import (
    LabelGapTracker,
    eigenstate_quantum_numbers,
    find_max_overlap_idx,
    reorder_evecs,
    select_eigenstate,
    vector_to_state,
)


@contextmanager
def limit_blas_threads(limit: Optional[int]) -> Iterator[None]:
    """Limit BLAS threads for the duration of a time-evolution loop.

    The Hamiltonians here are small (n = 64 for Js = [0,1,2,3], 100 for
    [0,1,2,3,4]). At that size OpenBLAS's internal threading costs more in
    fork/join synchronisation than it recovers, for every eigensolver backend
    tested, and the penalty grows with n.

    Parallelism belongs at the batch level instead, where each scan point is an
    independent unit of work. Running batch workers while BLAS is unpinned would
    also oversubscribe badly (workers x BLAS threads on the same cores).

    Scoped rather than set globally: BLAS threading is a genuine win for large
    matrices, so pinning is confined to this package's loops and restored on
    exit, leaving the rest of a notebook session untouched.

    `limit=None` disables the scoping and leaves BLAS configuration alone.
    """
    if limit is None:
        yield
        return

    with threadpool_limits(limits=limit, user_api="blas"):
        yield


def _sample_offset(time_sampling: str) -> float:
    """Fraction of the step at which `H` is evaluated.

    `"mid"` evaluates `H(t + dt/2)`, `"left"` evaluates `H(t)`. The step is
    exact for a frozen `H` either way, so this only changes which value is
    frozen, and it costs nothing: `h_slow_eval` is under `1%` of wall clock.

    **Against exact solutions, midpoint is second order and left-endpoint is
    first** -- measured as `2.00` and `1.00` per doubling on analytically
    solvable models (`benchmarks/analytic_models.py`, `tests/test_analytic.py`).

    On the real problem it does not look that way, and the reason is understood
    rather than mysterious. The observed order leaves its asymptotic value once
    `delta*dt` exceeds about `1` **and** `H(t)` fails to commute with itself at
    different times, because the neglected terms are time-ordering commutators
    carrying powers of `delta*dt`. Measured on the models: a commuting `H` holds
    order `2.00` at `delta*dt = 2e+02`, while a non-commuting one gives `-0.45`
    at `2e+01`. SPA2 runs at `delta*dt ~ 7.65e+04` with a non-commuting `H`, so
    no asymptotic power law is reachable there -- `delta*dt ~ 1` would need
    `N ~ 7.6e+08` steps.

    What that buys in practice is therefore a smaller error constant rather than
    a better rate. On a `scalar` model at production `delta*dt`, left-endpoint
    saturates near `0.4` with no convergence at all while midpoint holds order
    `2` and reaches `9.0e-03` -- a `45x` gap. On the SPA2 lineshape the measured
    saving was about `4x` fewer steps at `N_steps=1000` and `3.3x` at `2000`,
    with no measurable difference by `16000`; those figures come from a
    two-discretisation reference whose own uncertainty is `4.0e-04`, so treat
    them as indicative rather than tight.
    """
    if time_sampling == "mid":
        return 0.5
    if time_sampling == "left":
        return 0.0
    raise ValueError(
        f"time_sampling must be 'mid' or 'left'; got {time_sampling!r}"
    )


def field_variation_density(
    H_slow_t: Callable[[float], np.ndarray],
    muw_hams: Optional[List[Callable[[float], np.ndarray]]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the default step-density function, `||dH/dt||_2`.

    Returns a callable suitable for `build_time_grid(density=...)`: given an
    array of probe times it returns the spectral norm of a central difference of
    the total Hamiltonian, slow part plus every microwave field.

    This is the quantity to grade on. Three alternatives were measured and none
    beat it: the Magnus commutator `||[H, dH/dt]||` returns the same grid to
    within 5% of one step, because the dominant energy differences are the
    near-constant rotational splittings; the eigenbasis rotation rate `||dV/dt||`
    is no better; and grading on the microwave envelope alone is far worse,
    which is the result that shows the error really is dominated by where this
    quantity is large.
    """
    fields = list(muw_hams or [])

    def density(ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        if ts.size < 2:
            return np.ones(ts.size)
        lo_bound, hi_bound = float(ts[0]), float(ts[-1])
        h = 0.5 * float(ts[1] - ts[0])
        out = np.empty(ts.size, dtype=float)
        for i, t in enumerate(ts):
            lo = min(max(float(t) - h, lo_bound), hi_bound)
            hi = min(max(float(t) + h, lo_bound), hi_bound)
            d = H_slow_t(hi) - H_slow_t(lo)
            for H_mu_t in fields:
                d = d + (H_mu_t(hi) - H_mu_t(lo))
            out[i] = np.linalg.norm(d, 2) / (hi - lo)
        return out

    return density


MAGNUS_TAYLOR_TERMS = 8


def magnus_integral(delta: np.ndarray, dt: float, shift: float = 0.0) -> np.ndarray:
    """`(exp(i*(d_i - d_j + shift)*dt) - 1) / (i*(d_i - d_j + shift))`, elementwise.

    The first Magnus term for a coupling in the interaction picture of a diagonal
    `delta`. Built from an outer product of `n` exponentials rather than `n**2`
    of them, since `exp(i*(d_i - d_j)*dt)` factorises; the shift contributes a
    single scalar factor, so it stays `O(n**2)`.

    `shift` carries a coupling that itself rotates, which is what the multitone
    path needs. A field beating at `delta_omega` against the rotating-frame
    carrier contributes `exp(-i*delta_omega*(t + s))` to the integrand, and the
    `s` part merges into the oscillatory factor as
    `exp(i*((d_i - d_j) - delta_omega)*s)`. So the raising component takes
    `shift = -delta_omega` and the lowering component `shift = +delta_omega`,
    with the residual `exp(-i*delta_omega*t)` left outside as a constant.
    Freezing that phase instead -- evaluating it once at `t_sample` and treating
    the coupling as static -- is wrong whenever `delta_omega * dt` is not small.

    The small-argument branch is load-bearing: `delta` is nearly degenerate
    within a rotational manifold, so the quotient divides by ~0 there. `expm1`
    keeps the mid-range conditioned, where the outer-product form loses digits to
    cancellation.
    """
    e = np.exp(1j * delta * dt)
    return _magnus_integral_from_outer(
        np.outer(e, e.conj()), delta[:, None] - delta[None, :], dt, shift
    )


#: `"frozen"` freezes `H` at the sample point and exponentiates that frozen
#: matrix exactly; `"magnus"` integrates `H` across the step and exponentiates by
#: Taylor series. Both approximate the same time-ordered exponential -- neither is
#: the solution.
#:
#: **`"magnus"` is not uniformly better, and on some setups it is much worse.** It
#: truncates the second Magnus term, `Omega2 = -1/2 int int [H_I(t1), H_I(t2)]`,
#: which is its entire extra error over `"frozen"`. Where the oscillatory kernels
#: saturate -- gaps large enough that `K(x) -> 1/(i x)` rather than growing with
#: `dt` -- `||Omega2||` falls only *linearly* in `dt`, so refining the grid does
#: not reduce it relative to the term retained, and the scheme is first order.
#:
#: Measured: on SPA2 and the analytic models it matches `"frozen"` to four
#: significant figures. On the SPA1+SPA2 cascade it is first order and `~1e+05`
#: worse at matched `N_steps`; it is `1.35x` faster per run and still needs `~23x`
#: more wall clock to reach the same accuracy.
#:
#: `benchmarks/bench_magnus_commutator.py` computes `||Omega2||` directly for a
#: given setup -- no propagation, no reference -- so which case you are in is
#: answerable before spending anything. See "Performance Priority E" in
#: `IMPROVEMENTS.md`.
PROPAGATORS = ("frozen", "magnus")


def _normalize_propagator(propagator: str) -> str:
    """Validate a propagator name against `PROPAGATORS`."""
    if propagator not in PROPAGATORS:
        raise ValueError(
            f"propagator must be one of {PROPAGATORS}; got {propagator!r}"
        )
    return propagator


def _magnus_integral_from_outer(
    outer: np.ndarray, gaps: np.ndarray, dt: float, shift: float = 0.0
) -> np.ndarray:
    """`magnus_integral` with `outer(e, conj(e))` and the gap matrix precomputed.

    The multitone path needs three kernels per scan point -- one static, one per
    beat sense -- and they differ only in a scalar factor and a denominator. The
    `n**2` outer product is common to all of them, so building it once per scan
    point rather than once per kernel is most of the cost.
    """
    x = gaps + float(shift)
    xdt = x * dt
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = outer * np.exp(1j * float(shift) * dt) - 1.0
        out = numerator / (1j * x)

    # Three regimes, and the masks must not overlap. `mid` excludes `tiny`
    # explicitly: the diagonal has `x == 0`, so `expm1(0)/0` is a NaN that the
    # `tiny` branch would then overwrite -- correct by accident, and it emits a
    # divide warning on every step.
    tiny = np.abs(xdt) < 1e-8
    mid = (np.abs(xdt) < 1e-4) & ~tiny
    if mid.any():
        out[mid] = np.expm1(1j * xdt[mid]) / (1j * x[mid])
    if tiny.any():
        out[tiny] = dt * (1.0 + 0.5j * xdt[tiny])
    return out


def apply_magnus_taylor(A: np.ndarray, psis: np.ndarray) -> np.ndarray:
    """`psis @ expm(-i*A)^T`, by Taylor series applied to the state vectors.

    `psis` is `(S, n)` in the row-vector convention used throughout, so the
    propagator acts on the right transposed. Each term costs `n**2 * S`, never
    `n**3` -- which is the entire point, since it replaces an `n**3` eigensolve.

    Scaling and squaring is applied only when needed. At the production operating
    point `||A||` is about `0.038` rad and `k` is `0`, so this costs nothing; a
    fixed series diverges once `||A||` approaches `1`, and does so *silently* --
    measured against exact solutions, a coupling-to-spread ratio of `5e-05` gives
    `||A| ~ 1.9` and a 10x accuracy loss with no warning, and `5e-04` overflows
    to NaN.

    The bound uses the **Frobenius** norm, not the spectral norm. `||A||_2 <=
    ||A||_F`, so it is valid, and it costs `3 us` against `182 us` at `n=64` --
    the spectral norm is an SVD, and would consume `43%` of the `420 us`
    eigensolve this function exists to avoid. A guard that costs what it saves is
    not a guard.
    """
    norm = float(np.linalg.norm(A, "fro"))
    k = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    B = (-1j * A / (2**k)).T

    out = psis
    for _ in range(2**k):
        term = out
        acc = out.copy()
        for j in range(1, MAGNUS_TAYLOR_TERMS + 1):
            term = (term @ B) / j
            acc += term
        out = acc
    return out


def magnus_step_norms(
    muw_hams: List[Callable[[float], np.ndarray]],
    t_array: np.ndarray,
    *,
    coupling_scale: float = 1.0,
) -> np.ndarray:
    """`||A|| = ||H_mu(t_mid)|| * dt` per step, which governs the Magnus series.

    Needs no propagation, so it is cheap enough to assert before a run rather
    than discover afterwards.

    Worth checking whenever `step_density` is combined with `propagator="magnus"`.
    Grading stretches steps -- up to `50x` uniform on SPA2 -- and `||A||` scales
    with `dt`. Which way it moves is geometry: if the density peaks where the
    beam is, grading puts short steps there and `||A||` falls; on SPA2 the
    density peaks at the Stark ramp instead, the long steps land nearer the beam,
    and `max ||A||` rose from `0.044` to `0.32` against a guard that engages
    near `0.5`.
    """
    t_array = np.asarray(t_array, dtype=float)
    steps = np.diff(t_array)
    mids = 0.5 * (t_array[:-1] + t_array[1:])
    probe = np.linspace(t_array[0], t_array[-1], min(600, max(2, mids.size)))
    norms = np.array(
        [np.linalg.norm(sum(H(float(t)) for H in muw_hams), 2) for t in probe]
    )
    return np.interp(mids, probe, norms) * steps * float(coupling_scale)


def grading_ceiling(
    H_slow_t: Callable[[float], np.ndarray],
    muw_hams: Optional[List[Callable[[float], np.ndarray]]] = None,
    *,
    T: float,
    order: int = 1,
    probes: int = 400,
) -> float:
    """Upper bound on what `step_density` can buy: `N_uniform / N_optimal`.

    Needs no propagation -- it is arithmetic on the field profile -- so it is
    cheap enough to check before deciding whether to grade at all.

    The bound equals `1/f` for a trajectory active over a fraction `f` and static
    elsewhere (verified: `f=0.10` gives `10.02x`, `f=0.02` gives `50.60x`). Note
    it depends on how much of the run is **active**, not on how deep the quiet
    is: SPA2's density spans `63774x`, which sounds decisive, but its effective
    active fraction is `63%`, so the ceiling is only `1.58x`.

    **A ceiling above `1` is not enough to make grading worthwhile.** A uniform
    grid's leading error telescopes to a boundary term -- interior contributions
    cancel pairwise because every step is the same length -- and a non-uniform
    grid forfeits most of that. Measured by
    `benchmarks/bench_grid_cancellation.py` on an SPA2-like model: cancellation
    is `47.5x` on a uniform grid against `8.4x` graded, a `5.69x` loss, while
    equidistribution buys only `1.32x` of placement. Net, grading is `4.3x`
    *worse* -- and the decomposition predicts it (`5.69/1.32 = 4.3`).

    So the break-even is the cancellation ratio, not `1x`. Two real cases sit on
    the wrong side of it: SPA2 (`1.58x`) and the SPA1+SPA2 cascade (`1.581x`,
    measured `17-20x` worse than uniform). Grading paid `100-245x` on a pulse
    quiet for `97%` of its window, where the ceiling clears the threshold easily.

    **Grading harder does not help.** This ceiling is what the *optimal*
    placement achieves, since it derives from equidistribution being optimal;
    pushing past it lowers the placement gain while destroying more telescoping.
    Where `1/f` is genuinely large, the construction that banks it is a
    *piecewise-uniform* grid -- constant `dt` per block -- which keeps telescoping
    within each block and pays a boundary term per transition rather than per
    step. `step_density` does not build one.
    """
    if probes < 2:
        raise ValueError("probes must be >= 2")
    if order < 1:
        raise ValueError("order must be >= 1")

    ts = np.linspace(0.0, float(T), int(probes))
    density = np.asarray(field_variation_density(H_slow_t, muw_hams)(ts), dtype=float)
    if not np.all(np.isfinite(density)) or density.max() <= 0:
        raise ValueError("density must be finite and not identically zero")

    weighted = density ** (1.0 / (order + 1))
    numerator = float(T) ** order * float(np.trapezoid(density, ts))
    denominator = float(np.trapezoid(weighted, ts)) ** (order + 1)
    return float((numerator / denominator) ** (1.0 / order))


def build_time_grid(
    T: float,
    N_steps: int,
    *,
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    order: int = 1,
    max_ratio: float = 64.0,
    probes: int = 400,
) -> np.ndarray:
    """Integration times over `[0, T]`, uniform by default or graded by `density`.

    With `density=None` this returns `np.linspace(0, T, N_steps)` and nothing
    about a run changes. Otherwise steps are placed by equidistributing
    `density(t)**(1/(order+1))`, which concentrates them where the fields move
    and stretches them where the fields are quiet.

    The grid has exactly `N_steps` points either way: grading redistributes
    steps, it does not remove them. The saving is realised by the caller then
    passing a smaller `N_steps` for the same accuracy.

    `order` is the assumed global convergence order `p`, entering through the
    optimality condition `g**(1/(p+1)) * dt = const` for a local error density
    `g * dt**(p+1)`.

    It is a modelling assumption used to shape the grid, and deliberately not
    tied to the propagator's formal order. `time_sampling` now defaults to
    `"mid"`, which is second order against exact solutions, so `p=2` might look
    like the natural choice -- but at the production `delta*dt ~ 7.65e+04` no
    asymptotic power law is reachable, so neither `p` describes the observed
    error. `order=1` is kept because it produces the more aggressive grid and
    was the setting every measurement here used; changing it is an empirical
    question, not a corollary of the default flip.

    `max_ratio` caps a step at that multiple of the uniform step. The default is
    effectively no cap -- equidistribution on a realistic trajectory saturates
    well below it. A low cap is a mistake worth naming, since it silently
    removes most of the benefit: `exp(-i H dt)` is exact for constant `H`, so a
    quiet region tolerates a long step, and its eigenvectors are barely rotating
    there so adiabatic tracking is not at risk either.

    How much this can buy is bounded by the trajectory, not by the tuning. Local
    error goes as `g * dt**2`, so a region with `1/250` the density supports only
    `sqrt(250) ~ 16x` longer steps. On the SPA2 setup the ceiling is `1.58x` at
    `p=1` and `1.39x` at `p=2` -- arithmetic on a sampled `||dH/dt||` profile,
    so it is a bound on step placement and not a measured speedup.
    """
    if N_steps < 2:
        raise ValueError("N_steps must be >= 2")
    if density is None:
        return np.linspace(0, T, N_steps)
    if order < 1:
        raise ValueError("order must be >= 1")
    if max_ratio <= 1.0:
        raise ValueError("max_ratio must be > 1")
    if probes < 2:
        raise ValueError("probes must be >= 2")

    ts = np.linspace(0.0, float(T), int(probes))
    g = np.asarray(density(ts), dtype=float)
    if g.shape != ts.shape:
        raise ValueError(f"density must return one value per probe, got {g.shape}")
    if not np.all(np.isfinite(g)) or np.any(g < 0):
        raise ValueError("density must be finite and non-negative")
    if g.max() <= 0:
        return np.linspace(0, T, N_steps)

    weight = np.maximum(g, g.max() * 1e-12) ** (1.0 / (order + 1))

    # Flooring the weight caps the step, since a step is inversely proportional
    # to it. The floor has to be solved for rather than set directly: raising it
    # also raises the mean weight, which lengthens every step, so a floor of
    # `mean/max_ratio` computed from the *unfloored* mean overshoots the cap by
    # however much the mean moved. The fixed point of
    # `f = mean(max(w, f)) / max_ratio` makes the longest step exactly
    # `max_ratio` times the uniform one, and iterating converges monotonically.
    floor = float(np.trapezoid(weight, ts)) / float(T) / max_ratio
    for _ in range(64):
        mean_weight = float(np.trapezoid(np.maximum(weight, floor), ts)) / float(T)
        updated = mean_weight / max_ratio
        if abs(updated - floor) <= 1e-14 * updated:
            floor = updated
            break
        floor = updated
    weight = np.maximum(weight, floor)

    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(ts))]
    )
    cumulative /= cumulative[-1]
    grid = np.interp(np.linspace(0.0, 1.0, N_steps), cumulative, ts)
    grid[0], grid[-1] = 0.0, float(T)

    if not np.all(np.diff(grid) > 0):
        raise ValueError(
            "graded time grid is not strictly increasing: the density is too "
            "sharply peaked for N_steps={} at probes={}. Raise N_steps, lower "
            "probes, or smooth the density.".format(N_steps, probes)
        )
    return grid


@dataclass
class SimulationResult:
    """
    Class for storing results from simulations.

    t_array         : times at which results were returned (seconds)
    psis            : state vectors at each time when starting from initial states defined in
                      initial_states
    energies        : energies of all eigenstates of the hamiltonian at each time (2pi*Hz)
    probabilities   : for each state in initial staets, the probability of being found in
                      each eigenstate of the hamiltonian
    V_ref           : reference matrix of eigenstates of hamiltonian which tells what state each
                      index of energies and states corresponds to
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states: List[centrex_tlf.states.State]
    hamiltonian: Hamiltonian
    microwave_fields: Optional[List[MicrowaveField]]
    t_array: np.ndarray
    psis: Optional[np.ndarray]
    energies: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]
    probabilities_final: Optional[np.ndarray]
    V_ini: np.ndarray
    V_fin: np.ndarray
    monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None
    monitor_probabilities: Optional[np.ndarray] = None
    monitor_probabilities_final: Optional[np.ndarray] = None
    # Diagnostic from LabelGapTracker; see `unreliable_labels`.
    label_gaps: Optional[dict] = None
    time_grid: str = "uniform"
    time_sampling: str = "mid"

    def __post_init__(self):
        # Generate array of positions
        self.z_array = self.t_array * self.trajectory.Vini[2] + self.trajectory.Rini[2]

    def plot_state_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.UncoupledState,
        ax: Optional[plt.Axes] = None,
        position: bool = False,
        state_mapper: Optional[Callable] = None,
        tol: float = 1e-2,
    ) -> None:
        """
        Plots the probability of being found in a given adiabatically evolved eigenstate
        at different times.
        """
        if ax is None:
            fig, ax = plt.subplots()

        probs = self.get_state_probability(state, initial_state)

        label_state = state.remove_small_components(tol=tol)
        label_state.data = [(amp.real, stat) for amp, stat in label_state.data]
        label = (
            label_state.normalize()
            .remove_small_components(tol=tol)
            .state_string_custom(["J", "mJ", "m1", "m2"])
            if not state_mapper
            else state_mapper(state.normalize().remove_small_components(tol=tol))
        )
        if position:
            ax.plot(self.z_array / 1e-2, probs, label=label)
            ax.set_xlabel(r"Z-position / cm")
        else:
            ax.plot(self.t_array / 1e-6, probs, label=label)
            ax.set_xlabel(r"Time / $\mu$s")

    def plot_state_probabilities(
        self,
        states: List[centrex_tlf.states.UncoupledState],
        initial_state: centrex_tlf.states.UncoupledState,
        ax: Optional[plt.Axes] = None,
        position: bool = False,
        state_mapper: Optional[Callable] = None,
        tol: float = 1e-3,
    ) -> None:
        """
        Plots probabilities over time for states specified in the list states.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for state in states:
            self.plot_state_probability(
                state,
                initial_state,
                ax=ax,
                position=position,
                state_mapper=state_mapper,
                tol=tol,
            )

    def get_state_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
        ax=None,
    ):
        """
        Returns the probability of being found in given adiabatically evolved state
        for given initial state.
        """
        index_ini = self._resolve_initial_state_index(initial_state)

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )
        self._check_tracked_index(
            index_state, "get_state_probability", state.state_vector(self.hamiltonian.QN)
        )

        if self.probabilities is not None:
            return self.probabilities[:, index_ini, index_state]

        if self.probabilities_final is not None:
            # Return a 1-element array so existing code using `[-1]` keeps working.
            return np.array([self.probabilities_final[index_ini, index_state]])

        raise ValueError(
            "No probabilities stored on this SimulationResult. "
            "Re-run with store_probabilities=True or store_final_probabilities=True."
        )

    def _resolve_initial_state_index(
        self, initial_state: centrex_tlf.states.State
    ) -> int:
        """Resolve `initial_state` to an index in `self.initial_states`.

        In many workflows, users specify an *approximate* uncoupled state at t=0.
        Internally, the simulator maps that to the closest eigenstate and stores that
        mapped eigenstate in `self.initial_states`. This helper lets callers pass
        either representation.
        """
        try:
            return self.initial_states.index(initial_state)
        except ValueError:
            pass

        try:
            vec = initial_state.state_vector(self.hamiltonian.QN)
        except Exception as e:
            raise ValueError(
                "initial_state not found in result.initial_states and could not compute a state vector"
            ) from e

        overlaps = []
        for s in self.initial_states:
            try:
                s_vec = s.state_vector(self.hamiltonian.QN)
            except Exception:
                overlaps.append(-1.0)
                continue
            overlaps.append(float(np.abs(np.vdot(s_vec.conj(), vec)) ** 2))

        best = int(np.argmax(overlaps))
        if overlaps[best] < 0:
            raise ValueError(
                "Could not match initial_state to any stored initial state"
            )
        return best

    def get_monitor_probability(
        self,
        monitor_state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
    ) -> np.ndarray:
        """Return population of a monitored *adiabatically-tracked* state.

        The monitor state is mapped to the closest eigenstate at t=0 (via V_ini),
        and that eigenstate index is tracked through time using the library's
        eigenvector reordering.
        """
        if not self.monitor_states:
            raise ValueError(
                "No monitor_states present on this SimulationResult. "
                "Re-run with monitor_states=[...] and store_monitor_probabilities=True "
                "or store_final_monitor_probabilities=True."
            )

        try:
            mon_i = self.monitor_states.index(monitor_state)
        except ValueError as e:
            raise ValueError("monitor_state not found in result.monitor_states") from e

        index_ini = self._resolve_initial_state_index(initial_state)

        # The engine derives monitor_idx from V_ini the same way; recompute it so
        # the same t=0-index-on-t=T-data problem can be checked for here too.
        monitor_vector = monitor_state.state_vector(self.hamiltonian.QN)
        self._check_tracked_index(
            find_max_overlap_idx(monitor_vector, self.V_ini),
            "get_monitor_probability",
            monitor_vector,
        )

        if self.monitor_probabilities is not None:
            return self.monitor_probabilities[:, index_ini, mon_i]

        if self.monitor_probabilities_final is not None:
            return np.array([self.monitor_probabilities_final[index_ini, mon_i]])

        raise ValueError(
            "No monitor probabilities stored on this SimulationResult. "
            "Re-run with store_monitor_probabilities=True or store_final_monitor_probabilities=True."
        )

    def find_large_prob_states(
        self, initial_state: centrex_tlf.states.State, N: int = 5
    ) -> List[centrex_tlf.states.State]:
        """
        Returns the N states with the largest mean probabilities for given initial
        state.
        """
        index_ini = self._resolve_initial_state_index(initial_state)
        if self.probabilities is not None:
            score = np.mean(self.probabilities[:, index_ini, :], axis=0)
        elif self.probabilities_final is not None:
            score = self.probabilities_final[index_ini, :]
        else:
            raise ValueError(
                "No probabilities stored on this SimulationResult. "
                "Re-run with store_probabilities=True or store_final_probabilities=True."
            )

        index = np.argsort(-score)[:N]

        state_vecs = self.V_ini[:, index]
        states = []
        for i in range(state_vecs.shape[1]):
            states.append(vector_to_state(state_vecs[:, i], self.hamiltonian.QN))

        return states

    def plot_state_energy(
        self,
        state: centrex_tlf.states.UncoupledState,
        zero_state: Optional[centrex_tlf.states.UncoupledState] = None,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Plots the energy of state, using energy of zero_state as zero energy.
        """
        energies = self.get_state_energy(state)
        if zero_state:
            energies_zero = self.get_state_energy(zero_state)
            energies = energies - energies_zero

        if ax is None:
            fig, ax = plt.subplots()

        label = (
            state.remove_small_components(tol=0.1)
            .normalize()
            .state_string_custom(["J", "mJ", "m1", "m2"])
        )
        ax.plot(self.t_array / 1e-6, energies / (2 * np.pi * 1e3), label=label)
        ax.set_xlabel(r"Time / $\mu$s")
        ax.set_ylabel("Energy / kHz")
        return energies

    def plot_state_energies(
        self,
        states: List[centrex_tlf.states.UncoupledState],
        zero_state: centrex_tlf.states.UncoupledState,
        ax: plt.Axes = None,
    ) -> None:
        """
        Plots probabilities over time for states specified in the list states.
        """
        energies = []
        if ax is None:
            fig, ax = plt.subplots()
        for state in states:
            energies.append(self.plot_state_energy(state, zero_state=zero_state, ax=ax))

        return energies

    # A `get_state_energy_diabatic` used to be stubbed out here, backed by
    # `reorder_evecs(V, D, V_ref_ini)` -- matching each step against the t=0
    # eigenvectors rather than the previous step's. That is not diabatic
    # following, and it was removed rather than implemented: it is only
    # meaningful while the eigenvectors barely rotate, and here they rotate
    # completely. The SPA2 initial state is a Stark mixture at t=0 (F = 2.372
    # +- 4.899) and a pure F=2 state once the field is off, so overlaps with the
    # t=0 basis become meaningless well away from any crossing.
    #
    # The genuinely diabatic object is the propagated wavefunction, which is
    # already stored: project it onto the final eigenvectors and identify those
    # by quantum numbers with `population(...)`.

    def get_state_energy(self, state: centrex_tlf.states.UncoupledState) -> np.ndarray:
        """
        Gets the energy of state for all values in t_array.
        """

        if self.energies is None:
            raise ValueError(
                "No energies stored on this SimulationResult. Re-run with store_energies=True."
            )

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )

        return self.energies[:, index_state]

    def final_quantum_numbers(self) -> dict:
        """(J, F1, F, mF) and their spreads for every column of `V_fin`.

        `probabilities_final[..., k]` is the population in `V_fin[:, k]`, so this
        labels that axis by what the states physically are. See
        `state_prep.utils.eigenstate_quantum_numbers` for why the spreads matter.
        """
        return eigenstate_quantum_numbers(self.V_fin, self.hamiltonian.QN)

    def population(self, *, tolerance: float = 1e-3, **identity) -> np.ndarray:
        """Final population of the state with the given quantum numbers.

        Accepts any subset of `J`, `F1`, `F`, `mF`, e.g.
        ``result.population(J=1, F1=1.5, F=2, mF=0)``. Returns one value per
        initial state.

        Prefer this over `get_state_probability` when the answer matters: that
        method resolves the state through an index taken at t=0, which is only
        valid if the adiabatic label survived the whole trajectory. This reads
        the final eigenvectors directly and so cannot be fooled by a label swap
        at a crossing.
        """
        if self.probabilities_final is None:
            raise ValueError(
                "No final probabilities stored. Re-run with store_final_probabilities=True."
            )
        index = select_eigenstate(
            self.final_quantum_numbers(), identity, tolerance=tolerance
        )
        return self.probabilities_final[:, index]

    def unreliable_labels(self, max_gap_hz: Optional[float] = None) -> dict:
        """Tracked eigenstate labels that crossed another level during the trajectory.

        Returns ``{index: {"crossings", "gap_hz", "at_time_s"}}`` for every label
        whose energy rank changed at least once, optionally restricted to those
        whose closest crossing was below `max_gap_hz`.

        A label listed here is not necessarily wrong, but it cannot be assumed
        right: at a crossing the population follows its diabatic branch while the
        adiabatic label follows the other, so the index may no longer name the
        state you meant. Identify such states with `population(...)` instead,
        which reads the final eigenvectors rather than a carried index.

        On the SPA2 setup this lists most of the basis, because the Stark
        ramp-down folds the hyperfine structure together and produces crossings
        in bulk. That is the honest answer: adiabatic labels here are broadly
        unreliable, not exceptionally so.
        """
        if self.label_gaps is None:
            return {}

        crossings = self.label_gaps["crossings"]
        gaps = self.label_gaps["crossing_gap_hz"]
        selected = crossings > 0
        if max_gap_hz is not None:
            selected &= gaps < max_gap_hz

        return {
            int(index): {
                "crossings": int(crossings[index]),
                "gap_hz": float(gaps[index]),
                "at_time_s": float(self.label_gaps["crossing_time_s"][index]),
            }
            for index in np.flatnonzero(selected)
        }

    def _tracked_mF(self, vector: np.ndarray) -> float:
        """<mF> of a state vector. mF = mJ + m1 + m2 is exact in this basis."""
        mF = np.array([q.mJ + q.m1 + q.m2 for q in self.hamiltonian.QN], dtype=float)
        weights = np.abs(np.asarray(vector, dtype=complex)) ** 2
        return float(weights @ mF / weights.sum())

    def _check_tracked_index(
        self,
        index: int,
        what: str,
        reference_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Check a `V_ini`-derived index before it is used against final-time data.

        Two checks, strongest first. mF is exactly conserved here, so a tracked
        label whose final-time mF differs from the reference state's has
        definitely moved to a different state -- that is reported unconditionally.
        Otherwise, if the label crossed another level the answer merely *may* be
        wrong, which is reported once per result to avoid noise.
        """
        if reference_vector is not None and self.V_fin is not None:
            expected = self._tracked_mF(reference_vector)
            actual = self._tracked_mF(self.V_fin[:, index])
            if abs(expected - actual) > 1e-6:
                warnings.warn(
                    f"{what} resolves to tracked eigenstate {index}, which has "
                    f"mF={actual:+g} at the final time but the requested state has "
                    f"mF={expected:+g}. mF is exactly conserved, so the adiabatic label "
                    f"has definitely followed a different state and this value is wrong. "
                    f"Use population(...) with quantum numbers instead.",
                    UserWarning,
                    stacklevel=3,
                )
                return

        if getattr(self, "_crossing_warning_issued", False):
            return
        flagged = self.unreliable_labels()
        if int(index) not in flagged:
            return
        detail = flagged[int(index)]
        object.__setattr__(self, "_crossing_warning_issued", True)
        warnings.warn(
            f"{what} resolves to tracked eigenstate {index}, which changed energy rank "
            f"{detail['crossings']} time(s), passing within {detail['gap_hz']:.3g} Hz of "
            f"another level at t={detail['at_time_s']:.4g} s. At such a crossing the "
            f"population follows its diabatic branch while the adiabatic label follows "
            f"the other, so this index may not name the state you meant "
            f"({len(flagged)} of {self.label_gaps['crossings'].size} labels cross). "
            f"Identify states with population(...) and quantum numbers instead. "
            f"This warning is issued once per result.",
            UserWarning,
            stacklevel=3,
        )

    def _warn_if_unreliable(self, index: int, what: str) -> None:
        """Warn once per result when a tracked label is used and crossings occurred.

        Warns once rather than per call, and reports the population-wide count
        rather than only this label: crossings here are pervasive, so a per-label
        warning on every access would be noise.
        """
        if getattr(self, "_crossing_warning_issued", False):
            return
        flagged = self.unreliable_labels()
        if int(index) not in flagged:
            return
        detail = flagged[int(index)]
        object.__setattr__(self, "_crossing_warning_issued", True)
        warnings.warn(
            f"{what} resolves to tracked eigenstate {index}, which changed energy rank "
            f"{detail['crossings']} time(s), passing within {detail['gap_hz']:.3g} Hz of "
            f"another level at t={detail['at_time_s']:.4g} s. At such a crossing the "
            f"population follows its diabatic branch while the adiabatic label follows "
            f"the other, so this index may not name the state you meant "
            f"({len(flagged)} of {self.label_gaps['crossings'].size} labels cross). "
            f"Identify states with population(...) and quantum numbers instead. "
            f"This warning is issued once per result.",
            UserWarning,
            stacklevel=3,
        )

    def save_to_pickle(self, path: Path) -> None:
        """
        Saves the result to a pickle.
        """
        with open(path, "wb+") as f:
            dill.dump(self, f)


@dataclass
class MicrowaveScanResult:
    """Result from a microwave-parameter scan with shared slow Hamiltonian.

    This is intended for ensemble scans where the slow Hamiltonian H_slow(t)
    is identical for all scan points, and only microwave terms differ (detuning,
    power, background fields, etc.).
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states: List[centrex_tlf.states.State]
    hamiltonian: Hamiltonian
    t_array: np.ndarray
    psis_final: np.ndarray
    probabilities_final: Optional[np.ndarray]
    monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None
    monitor_probabilities_final: Optional[np.ndarray] = None
    V_ini: Optional[np.ndarray] = None
    V_fin: Optional[np.ndarray] = None
    # Diagnostic from LabelGapTracker; see `unreliable_labels`.
    label_gaps: Optional[dict] = None
    time_grid: str = "uniform"
    time_sampling: str = "mid"
    # Always one of `PROPAGATORS`.
    propagator: str = "frozen"

    @property
    def batch_size(self) -> int:
        """Number of scan points."""
        return int(self.psis_final.shape[0])

    def _initial_state_index(self, initial_state: centrex_tlf.states.State) -> int:
        """Resolve `initial_state` to an index into `self.initial_states`.

        Accepts either the approximate state supplied at setup or the tracked
        eigenstate the simulator mapped it to, matching `SimulationResult`.
        """
        for idx, state in enumerate(self.initial_states):
            if state is initial_state or state == initial_state:
                return idx
        if self.V_ini is None:
            raise ValueError(
                "initial_state not found in initial_states, and V_ini is unavailable "
                "for overlap-based matching."
            )
        target = find_max_overlap_idx(
            initial_state.state_vector(self.hamiltonian.QN), self.V_ini
        )
        for idx, state in enumerate(self.initial_states):
            if (
                find_max_overlap_idx(
                    state.state_vector(self.hamiltonian.QN), self.V_ini
                )
                == target
            ):
                return idx
        raise ValueError("Could not resolve initial_state to one of initial_states.")

    def get_state_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
    ) -> np.ndarray:
        """Final probability of `state` for `initial_state`, one value per scan point.

        `state` is resolved to the adiabatically tracked eigenstate with the
        largest overlap at t=0, the same convention `run_microwave_scan` uses for
        monitor states.
        """
        if self.probabilities_final is None:
            raise ValueError(
                "No probabilities stored. Re-run with store_final_probabilities=True."
            )
        if self.V_ini is None:
            raise ValueError("V_ini is required to resolve a state index.")

        idx_ini = self._initial_state_index(initial_state)
        idx_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )
        self._check_tracked_index(
            idx_state, "get_state_probability", state.state_vector(self.hamiltonian.QN)
        )
        return self.probabilities_final[:, idx_ini, idx_state]

    def get_monitor_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
    ) -> np.ndarray:
        """Final probability of a monitored state, one value per scan point."""
        if self.monitor_probabilities_final is None:
            raise ValueError(
                "No monitor probabilities stored. Re-run with monitor_states=[...] "
                "and store_final_monitor_probabilities=True."
            )
        if not self.monitor_states:
            raise ValueError("This result has no monitor_states.")

        idx_ini = self._initial_state_index(initial_state)
        for idx, monitored in enumerate(self.monitor_states):
            if monitored is state or monitored == state:
                # The engine derives monitor_idx from V_ini the same way; recompute
                # it so the same t=0-index-on-t=T-data problem is checked here too.
                if self.V_ini is not None:
                    monitor_vector = state.state_vector(self.hamiltonian.QN)
                    self._check_tracked_index(
                        find_max_overlap_idx(monitor_vector, self.V_ini),
                        "get_monitor_probability",
                        monitor_vector,
                    )
                return self.monitor_probabilities_final[:, idx_ini, idx]
        raise ValueError("state is not among monitor_states for this result.")

    def plot_state_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
        x: Optional[np.ndarray] = None,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        xlabel: str = "scan point",
    ) -> plt.Axes:
        """Plot the final probability of `state` across the scan.

        `x` is the scan axis (detunings, powers, ...). Defaults to the scan index,
        since the result object does not know which parameter was varied.
        """
        if ax is None:
            _, ax = plt.subplots()
        probs = self.get_state_probability(state, initial_state)
        ax.plot(np.arange(probs.size) if x is None else x, probs, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("final probability")
        return ax

    def to_polars(
        self,
        initial_state: Optional[centrex_tlf.states.State] = None,
        **columns: np.ndarray,
    ):
        """Return monitor populations as a polars DataFrame, one row per scan point.

        Extra keyword arguments are added as columns, which is how the scanned
        parameter gets in: ``result.to_polars(detuning_hz=detunings)``.
        """
        import polars as pl

        if self.monitor_probabilities_final is None:
            raise ValueError(
                "to_polars needs monitor probabilities. Re-run with monitor_states=[...]."
            )
        idx_ini = (
            0 if initial_state is None else self._initial_state_index(initial_state)
        )

        data: dict[str, np.ndarray] = {"scan_index": np.arange(self.batch_size)}
        for name, values in columns.items():
            values = np.asarray(values)
            if values.shape[0] != self.batch_size:
                raise ValueError(
                    f"column {name!r} has length {values.shape[0]}, expected "
                    f"{self.batch_size}"
                )
            data[name] = values
        for idx in range(self.monitor_probabilities_final.shape[2]):
            data[f"monitor_{idx}"] = self.monitor_probabilities_final[:, idx_ini, idx]
        return pl.DataFrame(data)

    def final_quantum_numbers(self) -> dict:
        """(J, F1, F, mF) and their spreads for every column of `V_fin`.

        `probabilities_final[:, :, k]` is the population in `V_fin[:, k]`, so this
        labels that axis by what the states physically are.
        """
        if self.V_fin is None:
            raise ValueError("V_fin is required to compute quantum numbers.")
        return eigenstate_quantum_numbers(self.V_fin, self.hamiltonian.QN)

    def population(
        self,
        initial_state: centrex_tlf.states.State,
        *,
        tolerance: float = 1e-3,
        **identity,
    ) -> np.ndarray:
        """Final population of the state with the given quantum numbers.

        Accepts any subset of `J`, `F1`, `F`, `mF`, e.g.
        ``result.population(ini, J=1, F1=1.5, F=2, mF=0)``. Returns one value per
        scan point.

        Prefer this over `get_state_probability`, which resolves the state
        through an index taken at t=0 and is therefore only valid if the
        adiabatic label survived the trajectory.
        """
        if self.probabilities_final is None:
            raise ValueError(
                "No probabilities stored. Re-run with store_final_probabilities=True."
            )
        idx_ini = self._initial_state_index(initial_state)
        index = select_eigenstate(
            self.final_quantum_numbers(), identity, tolerance=tolerance
        )
        return self.probabilities_final[:, idx_ini, index]

    def unreliable_labels(self, max_gap_hz: Optional[float] = None) -> dict:
        """Tracked eigenstate labels that crossed another level during the trajectory.

        Returns ``{index: {"crossings", "gap_hz", "at_time_s"}}`` for every label
        whose energy rank changed at least once, optionally restricted to those
        whose closest crossing was below `max_gap_hz`.

        A label listed here is not necessarily wrong, but it cannot be assumed
        right: at a crossing the population follows its diabatic branch while the
        adiabatic label follows the other, so the index may no longer name the
        state you meant. Identify such states with `population(...)` instead,
        which reads the final eigenvectors rather than a carried index.

        On the SPA2 setup this lists most of the basis, because the Stark
        ramp-down folds the hyperfine structure together and produces crossings
        in bulk. That is the honest answer: adiabatic labels here are broadly
        unreliable, not exceptionally so.
        """
        if self.label_gaps is None:
            return {}

        crossings = self.label_gaps["crossings"]
        gaps = self.label_gaps["crossing_gap_hz"]
        selected = crossings > 0
        if max_gap_hz is not None:
            selected &= gaps < max_gap_hz

        return {
            int(index): {
                "crossings": int(crossings[index]),
                "gap_hz": float(gaps[index]),
                "at_time_s": float(self.label_gaps["crossing_time_s"][index]),
            }
            for index in np.flatnonzero(selected)
        }

    def _tracked_mF(self, vector: np.ndarray) -> float:
        """<mF> of a state vector. mF = mJ + m1 + m2 is exact in this basis."""
        mF = np.array([q.mJ + q.m1 + q.m2 for q in self.hamiltonian.QN], dtype=float)
        weights = np.abs(np.asarray(vector, dtype=complex)) ** 2
        return float(weights @ mF / weights.sum())

    def _check_tracked_index(
        self,
        index: int,
        what: str,
        reference_vector: Optional[np.ndarray] = None,
    ) -> None:
        """Check a `V_ini`-derived index before it is used against final-time data.

        Two checks, strongest first. mF is exactly conserved here, so a tracked
        label whose final-time mF differs from the reference state's has
        definitely moved to a different state -- that is reported unconditionally.
        Otherwise, if the label crossed another level the answer merely *may* be
        wrong, which is reported once per result to avoid noise.
        """
        if reference_vector is not None and self.V_fin is not None:
            expected = self._tracked_mF(reference_vector)
            actual = self._tracked_mF(self.V_fin[:, index])
            if abs(expected - actual) > 1e-6:
                warnings.warn(
                    f"{what} resolves to tracked eigenstate {index}, which has "
                    f"mF={actual:+g} at the final time but the requested state has "
                    f"mF={expected:+g}. mF is exactly conserved, so the adiabatic label "
                    f"has definitely followed a different state and this value is wrong. "
                    f"Use population(...) with quantum numbers instead.",
                    UserWarning,
                    stacklevel=3,
                )
                return

        if getattr(self, "_crossing_warning_issued", False):
            return
        flagged = self.unreliable_labels()
        if int(index) not in flagged:
            return
        detail = flagged[int(index)]
        object.__setattr__(self, "_crossing_warning_issued", True)
        warnings.warn(
            f"{what} resolves to tracked eigenstate {index}, which changed energy rank "
            f"{detail['crossings']} time(s), passing within {detail['gap_hz']:.3g} Hz of "
            f"another level at t={detail['at_time_s']:.4g} s. At such a crossing the "
            f"population follows its diabatic branch while the adiabatic label follows "
            f"the other, so this index may not name the state you meant "
            f"({len(flagged)} of {self.label_gaps['crossings'].size} labels cross). "
            f"Identify states with population(...) and quantum numbers instead. "
            f"This warning is issued once per result.",
            UserWarning,
            stacklevel=3,
        )

    def _warn_if_unreliable(self, index: int, what: str) -> None:
        """Warn once per result when a tracked label is used and crossings occurred.

        Warns once rather than per call, and reports the population-wide count
        rather than only this label: crossings here are pervasive, so a per-label
        warning on every access would be noise.
        """
        if getattr(self, "_crossing_warning_issued", False):
            return
        flagged = self.unreliable_labels()
        if int(index) not in flagged:
            return
        detail = flagged[int(index)]
        object.__setattr__(self, "_crossing_warning_issued", True)
        warnings.warn(
            f"{what} resolves to tracked eigenstate {index}, which changed energy rank "
            f"{detail['crossings']} time(s), passing within {detail['gap_hz']:.3g} Hz of "
            f"another level at t={detail['at_time_s']:.4g} s. At such a crossing the "
            f"population follows its diabatic branch while the adiabatic label follows "
            f"the other, so this index may not name the state you meant "
            f"({len(flagged)} of {self.label_gaps['crossings'].size} labels cross). "
            f"Identify states with population(...) and quantum numbers instead. "
            f"This warning is issued once per result.",
            UserWarning,
            stacklevel=3,
        )

    def save_to_pickle(self, path: Path) -> None:
        """Serialise with dill, which handles the callables stored on fields."""
        with open(path, "wb") as f:
            dill.dump(self, f)


@dataclass
class Simulator:
    """
    Class used to run simulations and store data.
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states_approx: centrex_tlf.states.UncoupledState
    hamiltonian: Hamiltonian
    microwave_fields: Optional[List[MicrowaveField]] = None

    def __post_init__(self):
        self.psis = np.array([])
        self.initial_states = []

    def _resolve_step_density(self, step_density, H_slow_t, muw_hams):
        """Turn the public `step_density` argument into a density callable."""
        if step_density is None:
            return None
        if isinstance(step_density, str):
            if step_density != "auto":
                raise ValueError(
                    f"step_density must be None, 'auto', or a callable; got {step_density!r}"
                )
            return field_variation_density(H_slow_t, muw_hams)
        if callable(step_density):
            return step_density
        raise ValueError(
            "step_density must be None, 'auto', or a callable taking an array of "
            f"times and returning one density per time; got {type(step_density).__name__}"
        )

    def run(
        self,
        N_steps=int(1e4),
        store_every: int = 1,
        store_psis: bool = True,
        store_energies: bool = True,
        store_probabilities: bool = True,
        store_final_probabilities: bool = True,
        progress: bool = True,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None,
        store_monitor_probabilities: bool = False,
        store_final_monitor_probabilities: bool = True,
        eig_backend: str = "zheevd",
        blas_threads: Optional[int] = 1,
        step_density=None,
        step_density_order: int = 2,
        time_sampling: str = "mid",
    ):
        """
        Runs the simulation.

        Parameters
        ----------
        N_steps:
            Number of integration timesteps.
        store_every:
            Store results every N steps (1 = store all). Increasing this can
            significantly speed up simulations by reducing allocations and
            probability calculations.
        monitor_states:
            Optional list of states to monitor *as adiabatically-tracked eigenstates*.
            Each state is mapped to the closest eigenstate at t=0 (using max overlap
            with V_ini), and then that eigenstate index is tracked through time via the
            same eigenvector reordering used everywhere else.
        store_monitor_probabilities:
            If True, store populations for monitor_states at each stored time.
        store_final_monitor_probabilities:
            If True, store only the final populations for monitor_states.

        eig_backend:
            Eigen-solver backend used for diagonalizations.
            - "zheevd": use LAPACK zheevd (current default)
            - "numpy": use numpy.linalg.eigh

        blas_threads:
            BLAS threads to allow during time evolution. Defaults to 1, which is
            substantially faster here because the Hamiltonians are small enough
            that OpenBLAS's internal threading is pure overhead. The limit is
            scoped to this call and restored afterwards, so it does not affect
            other work in the same session. Pass None to leave BLAS
            configuration alone.

        step_density:
            Placement of the integration timesteps.
            - None (default): a uniform grid, identical to previous behaviour.
            - "auto": grade the grid by `||dH/dt||`, concentrating steps where
              the fields move.

              **On the real trajectories measured so far this makes accuracy
              worse**: `4.3x` on an SPA2-like model against closed-form
              solutions, and `17-20x` on the SPA1+SPA2 cascade. Equidistribution
              does reduce the summed local error (`1.32x`), but a uniform grid's
              leading error telescopes to a boundary term -- cancellation `47.5x`
              uniform against `8.4x` graded -- and a non-uniform grid forfeits
              most of it. `benchmarks/bench_grid_cancellation.py` measures both
              halves of that trade.

              It wins where a large fraction of the run is *static*: `100-245x`
              on a pulse that is quiet for `97%` of its window. The ceiling is
              `1/f` for active fraction `f`, and it must exceed the *cancellation
              ratio*, not `1x`, to pay. SPA2's ceiling is `1.58x` and the
              cascade's `1.581x`, which is why both lose. Grading more
              aggressively cannot rescue it -- see `grading_ceiling`.

              Estimate the ceiling before enabling it; no propagation needed:

                  d = field_variation_density(H_slow, muw_hams)(ts)
                  ceiling = T * trapezoid(d, ts) / trapezoid(sqrt(d), ts)**2

              Costs one extra pass of 400 Hamiltonian evaluations, `+0.92%` of
              wall clock at `N_steps=10000`, batch `6`, `n=64`. Fixed cost, so a
              larger share on shorter runs.
            - a callable `f(times) -> densities` for a custom monitor.
            See `build_time_grid`.

        step_density_order:
            Grading aggressiveness, entering as `density**(1/(order+1))`. Larger
            values grade less and approach a uniform grid, which is the
            `order -> infinity` limit. Ignored unless `step_density` is set.

            Defaults to `2`. Measured against closed-form solutions on a model
            matched to SPA2's geometry and level motion, `order=2` beat `order=1`
            by about `2.3x` at every step count, and the same ordering held on
            the earlier sweeps. The reason is that less aggressive grading gives
            up less of the error cancellation a uniform grid enjoys, which
            matters whenever the available gain is small.

            On a problem with a much larger ceiling -- a long static stretch, so
            a small active fraction -- the trade reverses and `order=1` may win.
            Measure rather than assume; the value is exposed precisely because it
            is problem-dependent.

        time_sampling:
            Where in each step `H` is evaluated. "mid" (default) uses
            `H(t + dt/2)`, "left" uses `H(t)` and reproduces results from before
            this option existed. Midpoint costs nothing per step and is second
            order against exact solutions where left-endpoint is first. At the
            production `delta*dt` no asymptotic order is reachable, so the gain
            shows up as a smaller error constant: about 4x fewer steps at
            `N_steps=1000`, 3.3x at 2000, and no measurable difference by 16000.
            See `_sample_offset`.
        """
        if store_every < 1:
            raise ValueError("store_every must be >= 1")
        # Calculate the total time for the simulation
        T = self.trajectory.get_T()

        # Generate Hamiltonian that has slow time-evolution included
        H_t = self.hamiltonian.get_H_t_func()

        # Generate Hamiltonians for microwaves
        muw_hams: Optional[List[Callable[[float], np.ndarray]]] = None
        if self.microwave_fields is not None:
            # Initialize matrix for shifting energies in rotating frame
            D_mu = np.zeros((len(self.hamiltonian.QN), len(self.hamiltonian.QN)))

            # Initialize container for Hamiltonians
            muw_hams = []

            # Container for frequencies of all microwaves
            omegas = []

            for microwave_field in self.microwave_fields:
                muw_hams.append(
                    microwave_field.get_H_t_func(
                        self.trajectory.R_t, self.hamiltonian.QN
                    )
                )
                # Background fields should always be at the same frequency as
                # as main field so don't do the shifting for them
                # Olivier: the background field can be at a different frequency, for
                # instance for the RC microwaves leaking through. Just check if the
                # frequency already is present in omegas
                if not np.any(
                    np.isclose((2 * np.pi * microwave_field.muW_freq), omegas)
                ):
                    omegas.append(2 * np.pi * microwave_field.muW_freq)
                    microwave_field.generate_D(self.hamiltonian.QN, omega=sum(omegas))
                    if np.any((D_mu != 0) & (microwave_field.D != 0)):
                        raise AssertionError(
                            "Rotating frame transform failed, coupling already present between same levels."
                        )
                    D_mu += microwave_field.D

            # Generate function that gives couplings due to all microwaves
            def H_mu_tot_t(t):
                H_mu_tot = muw_hams[0](t)
                if len(muw_hams) > 1:
                    for H_mu_t in muw_hams[1:]:
                        H_mu_tot = H_mu_tot + H_mu_t(t)
                return H_mu_tot

        # Generate integration time array
        t_array = build_time_grid(
            T,
            N_steps,
            density=self._resolve_step_density(step_density, H_t, muw_hams),
            order=step_density_order,
        )

        # Indices to store output (downsampled)
        save_idx = np.arange(0, N_steps, store_every, dtype=int)
        if save_idx.size == 0 or save_idx[0] != 0:
            save_idx = np.insert(save_idx, 0, 0)
        if save_idx[-1] != N_steps - 1:
            save_idx = np.append(save_idx, N_steps - 1)

        # Perform time-evolution. BLAS is pinned for the duration: these
        # Hamiltonians are small enough that its internal threading is overhead.
        with limit_blas_threads(blas_threads):
            if self.microwave_fields is None:
                (
                    psis_t,
                    energies,
                    probabilities,
                    probabilities_final,
                    monitor_probabilities,
                    monitor_probabilities_final,
                    V_ini,
                    V_fin,
                ) = self._time_evolve(
                    H_t,
                    t_array,
                    save_idx,
                    store_psis=store_psis,
                    store_energies=store_energies,
                    store_probabilities=store_probabilities,
                    store_final_probabilities=store_final_probabilities,
                    progress=progress,
                    monitor_states=monitor_states,
                    store_monitor_probabilities=store_monitor_probabilities,
                    store_final_monitor_probabilities=store_final_monitor_probabilities,
                    eig_backend=eig_backend,
                    time_sampling=time_sampling,
                )
            else:
                (
                    psis_t,
                    energies,
                    probabilities,
                    probabilities_final,
                    monitor_probabilities,
                    monitor_probabilities_final,
                    V_ini,
                    V_fin,
                ) = self._time_evolve_mu(
                    H_t,
                    H_mu_tot_t,
                    D_mu,
                    t_array,
                    save_idx,
                    store_psis=store_psis,
                    store_energies=store_energies,
                    store_probabilities=store_probabilities,
                    store_final_probabilities=store_final_probabilities,
                    progress=progress,
                    monitor_states=monitor_states,
                    store_monitor_probabilities=store_monitor_probabilities,
                    store_final_monitor_probabilities=store_final_monitor_probabilities,
                    eig_backend=eig_backend,
                    time_sampling=time_sampling,
                )

        # Generate a result object
        result = SimulationResult(
            trajectory=self.trajectory,
            electric_field=self.electric_field,
            magnetic_field=self.magnetic_field,
            initial_states=self.initial_states,
            hamiltonian=self.hamiltonian,
            microwave_fields=self.microwave_fields,
            t_array=t_array[save_idx],
            psis=psis_t,
            energies=energies,
            probabilities=probabilities,
            probabilities_final=probabilities_final,
            monitor_states=monitor_states,
            monitor_probabilities=monitor_probabilities,
            monitor_probabilities_final=monitor_probabilities_final,
            V_ini=V_ini,
            V_fin=V_fin,
            label_gaps=getattr(self, "_label_gaps", None),
            time_grid="uniform" if step_density is None else "graded",
            time_sampling=time_sampling,
        )

        return result

    def init_state_vecs(
        self, H_0: np.ndarray, V_0: Optional[np.ndarray] = None
    ) -> None:
        """
        Generates state vectors based on self.initial_states in the basis
        of self.hamiltonian
        """

        # Find the eigenstates of the Hamiltonian that most closely correspond to the
        # initial states
        initial_states = []
        psis = []
        V = V_0
        if V is None:
            _, V = np.linalg.eigh(H_0)
        for state in self.initial_states_approx:
            idx = find_max_overlap_idx(state.state_vector(self.hamiltonian.QN), V)
            psis.append(V[:, idx])
            initial_states.append(vector_to_state(V[:, idx], self.hamiltonian.QN))
        self.psis = np.array(psis)
        self.initial_states = initial_states

    def run_microwave_scan(
        self,
        *,
        detunings_hz: np.ndarray,
        intensity_prefactors: np.ndarray,
        N_steps: int = int(1e4),
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None,
        store_final_probabilities: bool = True,
        store_final_monitor_probabilities: bool = True,
        progress: bool = True,
        eig_backend: str = "zheevd",
        workers: int = 1,
        parallel_backend: str = "loky",
        allow_multitone_same_manifold: bool = False,
        blas_threads: Optional[int] = 1,
        step_density=None,
        step_density_order: int = 2,
        time_sampling: str = "mid",
        propagator: str = "frozen",
    ) -> MicrowaveScanResult:
        """Run a batched microwave scan reusing the slow diagonalization.

        This is optimized for scans where the *slow* Hamiltonian H_slow(t) is
        identical across scan points, and only microwave parameters vary.

        Microwave coupling shapes are taken from `self.microwave_fields` (same as
        `run()`), but you supply per-scan-point prefactors:

        - `detunings_hz`: detuning(s) in Hz, same shape as `intensity_prefactors`.
        - `intensity_prefactors`: dimensionless scaling of *intensity/power*.
          Internally, couplings scale as sqrt(intensity_prefactors).

        Shapes:
        - For M microwave fields, provide arrays shaped (B, M) for batch size B.
        - For M=1, 1D arrays of shape (B,) or scalars are accepted.
        - 1D arrays of shape (M,) imply B=1.

        Parallelism:
        - `workers=1` (the default) runs the serial shared-slow scan.
        - `workers>1` or `workers=-1` splits the batch into contiguous chunks and
          runs one serial shared-slow scan per chunk with joblib's loky backend.
          Each chunk repeats the shared slow diagonalization, but the chunks run
          concurrently, so that duplication costs CPU rather than wall time.
        - For production-sized scans this is a substantial win and is worth
          using. Output is bitwise identical to the serial path.
        - The win depends on BLAS being pinned (see `blas_threads`, on by
          default). With BLAS unpinned, workers x BLAS threads oversubscribe the
          cores and loky can end up slower than serial.
        - Loky loses badly on short scans, where process startup and pickling
          dominate the actual work. There is no automatic size-based fallback,
          so keep `workers=1` for short scans. The only automatic fallbacks are
          for a degenerate batch (batch == 1, or `effective_n_jobs(workers)`
          resolving to 1).
        - Run-to-run timing varies noticeably at high worker counts because of
          process startup. Benchmark a worker count on your own machine before
          committing to it: `benchmarks/bench_microwave_scan.py` sweeps workers,
          and `benchmarks/paired.py` compares variants with repeats.

        Multi-tone same-manifold fields:
        - By default, fields with different microwave frequencies that shift the
          same excited-J manifold are rejected.
        - `allow_multitone_same_manifold=True` keeps the first field for that
          manifold as the rotating-frame reference and applies a time-dependent
          beat phase to additional tones.

        Cascaded fields:
        - Fields that define distinct rotating-frame carriers must be ordered as
          a ladder: `J=0 -> 1`, then `J=1 -> 2`, and so on. A field's `Jg` must
          equal the preceding frame-defining field's `Je`.
        - A level in an upper rung carries the cumulative carrier and detuning
          shifts of every rung below it. Fields sharing a carrier define one
          frame and do not introduce another rung.

        BLAS threads:
        - `blas_threads` defaults to 1. The Hamiltonians are small (n = 64-100),
          so OpenBLAS's internal threading is overhead rather than speedup.
          Pinning does not change results at all.
        - The limit is scoped to the evolution loop and restored afterwards, so
          it does not affect other work in the same session.
        - Pass None to leave BLAS configuration untouched.

        Timestep placement:
        - `step_density=None` (default) uses a uniform grid, unchanged from
          previous behaviour.
        - `step_density="auto"` grades the grid by `||dH/dt||`, concentrating
          steps where the fields move and stretching them where the fields are
          quiet, so a given accuracy needs fewer steps. The grid depends only on
          the trajectory, the DC fields and the beam envelope -- never on
          detuning or prefactor -- so a single grid is shared across the whole
          batch and the shared slow diagonalization is preserved.
        - The gain is bounded by the trajectory: about `1.58x` on the SPA2
          setup. It is not a free lunch on accuracy either, and worst-case cells
          may not show it. Verify with `benchmarks/bench_step_grid.py` before
          relying on a reduced `N_steps`.
        - A callable is also accepted; under `workers>1` it must be
          dill-picklable, since loky ships it to each worker.
        - `step_density_order` tunes how aggressively the grid is graded; see
          `run` for why no value is recommended.

        One grid is built for the whole scan, before the loop, and shared by
        every scan point. That is required: a per-point grid would fork the batch
        and lose the shared slow diagonalization. Consequences:

        - **Detuning never affects the grid.** It enters through `D_mu`, which is
          not part of the density.
        - **Power affects it only through the slow-to-microwave ratio.** The
          density is `||d(H_slow + H_mu)/dt||`, a norm of a *sum*, so scaling the
          coupling changes the grid's shape rather than just its scale. On SPA2
          that ratio is about `3.3e+03` at the beam, and since coupling goes as
          `sqrt(intensity)`, parity needs an intensity prefactor near `1e+07`.
          Measured: prefactor `16` moves the grid by `0.56` of a uniform step.
        - The case to watch is a scan spanning a *regime change*, from
          microwave-negligible to microwave-dominant. One grid then suits neither
          end. Checkable without propagating anything, by comparing
          `field_variation_density` at the extreme prefactors.

        Timestep sampling:
        - `time_sampling="mid"` (default) evaluates `H(t + dt/2)`; `"left"`
          evaluates `H(t)` and reproduces results from before this option
          existed. Midpoint is free and reaches a given accuracy in about 4x
          fewer steps at `N_steps=1000` and 3.3x at 2000, converging to no
          measurable difference by 16000.

        Propagator:
        - `propagator="frozen"` (default) freezes `H` at the sample point and
          exponentiates that frozen matrix exactly, by eigendecomposition.
          `"magnus"` integrates `H` across the step instead and exponentiates by
          Taylor series, replacing the per-scan-point `O(n^3)` eigensolve with
          `O(n^2 S)` work, so it is faster and more so at larger batch.

          **Faster per run is not the same as cheaper.** `"magnus"` truncates the
          second Magnus term, and where the oscillatory kernels saturate that
          truncation falls only linearly in `dt`, making the scheme first order.
          On SPA2 and the analytic models it matches `"frozen"` to four
          significant figures. On the SPA1+SPA2 cascade it is `~1e+05` worse at
          matched `N_steps`, and `1.35x` faster per run buys `~23x` *more* wall
          clock to reach equal accuracy.

          Which case you are in is answerable without propagating anything:
          `benchmarks/bench_magnus_commutator.py` computes the truncated term
          directly. Do that before selecting `"magnus"` for a new setup.
        """
        # Validate here, not only in the loops, so a bad name is rejected before
        # any work is done rather than on the first timestep.
        propagator = _normalize_propagator(propagator)

        if N_steps < 2:
            raise ValueError("N_steps must be >= 2")

        if workers == 0 or workers < -1:
            raise ValueError("workers must be -1 or >= 1")
        if parallel_backend != "loky":
            raise ValueError("parallel_backend currently only supports 'loky'")

        if not self.microwave_fields:
            raise ValueError(
                "run_microwave_scan requires Simulator.microwave_fields (a list of MicrowaveField)."
            )

        n_fields = len(self.microwave_fields)

        def _as_batched(x: np.ndarray, name: str) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 0:
                if n_fields != 1:
                    raise ValueError(f"{name} is scalar but there are {n_fields} microwave fields")
                return arr.reshape(1, 1)
            if arr.ndim == 1:
                if n_fields == 1:
                    return arr.reshape(-1, 1)
                if arr.shape[0] != n_fields:
                    raise ValueError(
                        f"{name} must have shape (B,{n_fields}) or ({n_fields},); got {arr.shape}"
                    )
                return arr.reshape(1, n_fields)
            if arr.ndim == 2:
                if arr.shape[1] != n_fields:
                    raise ValueError(
                        f"{name} must have shape (B,{n_fields}); got {arr.shape}"
                    )
                return arr
            raise ValueError(f"{name} must be scalar, 1D, or 2D; got ndim={arr.ndim}")

        det_b = _as_batched(detunings_hz, "detunings_hz").astype(float, copy=False)
        inten_b = _as_batched(intensity_prefactors, "intensity_prefactors").astype(
            float, copy=False
        )
        if det_b.shape != inten_b.shape:
            raise ValueError(
                f"detunings_hz and intensity_prefactors must have identical shapes; got {det_b.shape} vs {inten_b.shape}"
            )

        if np.any(inten_b < 0):
            raise ValueError("intensity_prefactors must be >= 0")

        batch = int(det_b.shape[0])
        fields_by_je: dict[int, list[int]] = {}
        for idx, mw in enumerate(self.microwave_fields):
            fields_by_je.setdefault(mw.Je, []).append(idx)
        multitone_jes = [
            je
            for je, indices in fields_by_je.items()
            if len(
                {
                    round(float(self.microwave_fields[idx].muW_freq), 6)
                    for idx in indices
                }
            )
            > 1
        ]
        if multitone_jes and not allow_multitone_same_manifold:
            raise ValueError(
                "run_microwave_scan found different microwave frequencies on the same "
                f"excited-J manifold(s) {multitone_jes}. Use "
                "allow_multitone_same_manifold=True to enable the two-tone path."
            )
        if workers != 1 and batch > 1:
            return self._run_microwave_scan_parallel_loky(
                detunings_hz=det_b,
                intensity_prefactors=inten_b,
                N_steps=N_steps,
                monitor_states=monitor_states,
                store_final_probabilities=store_final_probabilities,
                store_final_monitor_probabilities=store_final_monitor_probabilities,
                progress=progress,
                eig_backend=eig_backend,
                workers=workers,
                allow_multitone_same_manifold=allow_multitone_same_manifold,
                blas_threads=blas_threads,
                step_density=step_density,
                step_density_order=step_density_order,
                time_sampling=time_sampling,
                propagator=propagator,
            )

        # Build per-microwave H_mu(t) functions (shared across batch)
        muw_hams = [
            mw.get_H_t_func(self.trajectory.R_t, self.hamiltonian.QN)
            for mw in self.microwave_fields
        ]

        # Build the rotating-frame shift, deduplicated by frequency. Shared with
        # the GPU path and the benchmarks so the convention cannot drift.
        D_mu_diag_batch, _, _ = build_rotating_frame_shift(
            self.microwave_fields,
            self.hamiltonian.QN,
            det_b,
            skip_repeated_manifolds=bool(multitone_jes),
        )

        # Coupling scaling: intensity/power prefactor -> E-field prefactor via sqrt
        coupling_scales = np.sqrt(inten_b)

        H_t = self.hamiltonian.get_H_t_func()
        T = self.trajectory.get_T()
        t_array = build_time_grid(
            T,
            N_steps,
            density=self._resolve_step_density(step_density, H_t, muw_hams),
            order=step_density_order,
        )

        if multitone_jes:
            component_hams = [
                mw.get_H_t_components_func(self.trajectory.R_t, self.hamiltonian.QN)
                for mw in self.microwave_fields
            ]
            ref_field_for_je = {
                je: indices[0]
                for je, indices in fields_by_je.items()
            }
            static_field_indices: list[int] = []
            beat_fields: list[tuple[int, np.ndarray]] = []
            for field_idx, mw in enumerate(self.microwave_fields):
                ref_idx = ref_field_for_je[mw.Je]
                ref = self.microwave_fields[ref_idx]
                if np.isclose(mw.muW_freq, ref.muW_freq):
                    static_field_indices.append(field_idx)
                    if not np.allclose(det_b[:, field_idx], det_b[:, ref_idx]):
                        raise ValueError(
                            "Microwave fields with the same reference frequency and "
                            "excited-J manifold must share detuning in detunings_hz."
                        )
                else:
                    delta_omega = 2 * np.pi * (
                        (float(mw.muW_freq) + det_b[:, field_idx])
                        - (float(ref.muW_freq) + det_b[:, ref_idx])
                    )
                    beat_fields.append((field_idx, delta_omega))

            with limit_blas_threads(blas_threads):
                (
                    psis_final,
                    probabilities_final,
                    monitor_probabilities_final,
                    V_ini,
                    V_fin,
                ) = self._time_evolve_mu_batched_shared_slow_multitone(
                    H_slow_t=H_t,
                    component_hams=component_hams,
                    coupling_scales=coupling_scales,
                    D_mu_diag_batch=D_mu_diag_batch,
                    t_array=t_array,
                    static_field_indices=static_field_indices,
                    beat_fields=beat_fields,
                    monitor_states=monitor_states,
                    store_final_probabilities=store_final_probabilities,
                    store_final_monitor_probabilities=store_final_monitor_probabilities,
                    progress=progress,
                    eig_backend=eig_backend,
                    time_sampling=time_sampling,
                    propagator=propagator,
                )
        else:
            with limit_blas_threads(blas_threads):
                (
                    psis_final,
                    probabilities_final,
                    monitor_probabilities_final,
                    V_ini,
                    V_fin,
                ) = self._time_evolve_mu_batched_shared_slow(
                    H_slow_t=H_t,
                    muw_hams=muw_hams,
                    coupling_scales=coupling_scales,
                    D_mu_diag_batch=D_mu_diag_batch,
                    t_array=t_array,
                    monitor_states=monitor_states,
                    store_final_probabilities=store_final_probabilities,
                    store_final_monitor_probabilities=store_final_monitor_probabilities,
                    progress=progress,
                    eig_backend=eig_backend,
                    time_sampling=time_sampling,
                    propagator=propagator,
                )

        return MicrowaveScanResult(
            trajectory=self.trajectory,
            electric_field=self.electric_field,
            magnetic_field=self.magnetic_field,
            initial_states=self.initial_states,
            hamiltonian=self.hamiltonian,
            t_array=t_array,
            psis_final=psis_final,
            probabilities_final=probabilities_final,
            monitor_states=monitor_states,
            monitor_probabilities_final=monitor_probabilities_final,
            V_ini=V_ini,
            V_fin=V_fin,
            label_gaps=getattr(self, "_label_gaps", None),
            time_grid="uniform" if step_density is None else "graded",
            time_sampling=time_sampling,
            propagator=propagator,
        )

    def _run_microwave_scan_parallel_loky(
        self,
        *,
        detunings_hz: np.ndarray,
        intensity_prefactors: np.ndarray,
        N_steps: int,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_final_probabilities: bool,
        store_final_monitor_probabilities: bool,
        progress: bool,
        eig_backend: str,
        workers: int,
        allow_multitone_same_manifold: bool,
        blas_threads: Optional[int] = 1,
        step_density=None,
        step_density_order: int = 2,
        time_sampling: str = "mid",
        propagator: str = "frozen",
    ) -> MicrowaveScanResult:
        propagator = _normalize_propagator(propagator)
        batch = int(detunings_hz.shape[0])
        n_jobs = min(effective_n_jobs(workers), batch)
        if n_jobs <= 1:
            return self.run_microwave_scan(
                detunings_hz=detunings_hz,
                intensity_prefactors=intensity_prefactors,
                N_steps=N_steps,
                monitor_states=monitor_states,
                store_final_probabilities=store_final_probabilities,
                store_final_monitor_probabilities=store_final_monitor_probabilities,
                progress=progress,
                eig_backend=eig_backend,
                workers=1,
                parallel_backend="loky",
                allow_multitone_same_manifold=allow_multitone_same_manifold,
                blas_threads=blas_threads,
                step_density=step_density,
                step_density_order=step_density_order,
                time_sampling=time_sampling,
                propagator=propagator,
            )

        chunk_indices = np.array_split(np.arange(batch), n_jobs)

        def _run_chunk(indices: np.ndarray) -> MicrowaveScanResult:
            return self.run_microwave_scan(
                detunings_hz=detunings_hz[indices],
                intensity_prefactors=intensity_prefactors[indices],
                N_steps=N_steps,
                monitor_states=monitor_states,
                store_final_probabilities=store_final_probabilities,
                store_final_monitor_probabilities=store_final_monitor_probabilities,
                progress=False,
                eig_backend=eig_backend,
                workers=1,
                parallel_backend="loky",
                allow_multitone_same_manifold=allow_multitone_same_manifold,
                blas_threads=blas_threads,
                step_density=step_density,
                step_density_order=step_density_order,
                time_sampling=time_sampling,
                propagator=propagator,
            )

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_chunk)(indices) for indices in chunk_indices if indices.size
        )
        first = results[0]

        self.psis = first.psis_final[0].copy()
        self.initial_states = first.initial_states

        probabilities_final = None
        if store_final_probabilities:
            probabilities_final = np.concatenate(
                [result.probabilities_final for result in results], axis=0
            )

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and first.monitor_probabilities_final is not None:
            monitor_probabilities_final = np.concatenate(
                [result.monitor_probabilities_final for result in results], axis=0
            )

        return MicrowaveScanResult(
            trajectory=self.trajectory,
            electric_field=self.electric_field,
            magnetic_field=self.magnetic_field,
            initial_states=first.initial_states,
            hamiltonian=self.hamiltonian,
            t_array=first.t_array,
            psis_final=np.concatenate([result.psis_final for result in results], axis=0),
            probabilities_final=probabilities_final,
            monitor_states=monitor_states,
            monitor_probabilities_final=monitor_probabilities_final,
            V_ini=first.V_ini,
            V_fin=first.V_fin,
            label_gaps=first.label_gaps,
            time_grid=first.time_grid,
            time_sampling=first.time_sampling,
            propagator=first.propagator,
        )

    def _time_evolve_mu_batched_shared_slow(
        self,
        *,
        H_slow_t: Callable[[float], np.ndarray],
        muw_hams: List[Callable[[float], np.ndarray]],
        coupling_scales: np.ndarray,
        D_mu_diag_batch: np.ndarray,
        t_array: np.ndarray,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_final_probabilities: bool,
        store_final_monitor_probabilities: bool,
        progress: bool,
        eig_backend: str,
        time_sampling: str = "mid",
        propagator: str = "frozen",
    ):
        """Batched microwave evolution with shared slow diagonalization.

        This is a CPU-only batched variant of `_time_evolve_mu` where H_slow(t)
        is identical across the batch and is diagonalized only once per timestep.
        """
        # Validate batch shapes
        if D_mu_diag_batch.ndim != 2:
            raise ValueError("D_mu_diag_batch must have shape (B,n)")
        batch = int(D_mu_diag_batch.shape[0])

        coupling_scales = np.asarray(coupling_scales)
        if coupling_scales.ndim != 2 or coupling_scales.shape[0] != batch:
            raise ValueError(
                "coupling_scales must have shape (B,M) matching D_mu_diag_batch batch size"
            )
        if coupling_scales.shape[1] != len(muw_hams):
            raise ValueError(
                f"coupling_scales has M={coupling_scales.shape[1]} but muw_hams has M={len(muw_hams)}"
            )

        # Calculate Hamiltonian at tini
        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        if D_mu_diag_batch.shape[1] != n:
            raise ValueError(
                f"D_mu_diag_batch has n={D_mu_diag_batch.shape[1]} but Hamiltonian has n={n}"
            )

        diag_idx = slice(None, None, n + 1)

        # Reference eigen-decomposition at t0 (reused for init + tracking)
        E_ref, V_ref = np.linalg.eigh(H_tini)
        index = np.argsort(E_ref)
        E_ref = E_ref[index]
        V_ref = V_ref[:, index]
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        # Initialize state vectors once, then replicate across batch
        self.init_state_vecs(H_tini, V_0=V_ref)
        psis_batch = np.repeat(self.psis[None, :, :], batch, axis=0)

        last_evecs = V_ref
        sample_offset = _sample_offset(time_sampling)
        # `"frozen"` freezes `H` at the sample point and exponentiates that
        # frozen matrix exactly, by eigendecomposition. `"magnus"` integrates `H`
        # across the step and exponentiates by Taylor series. Both are
        # second-order approximations of the same time-ordered exponential and
        # converge to the same answer; neither is the solution. Where `H` moves
        # appreciably within a step, integrating beats freezing.
        propagator = _normalize_propagator(propagator)
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + sample_offset * dt

            # Shared slow Hamiltonian diagonalization (once per timestep)
            H_slow_i = H_slow_t(t_sample)
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Track eigenvectors (for probabilities/monitor semantics)
            Es, evecs = D, V
            Es, evecs = reorder_evecs(evecs, Es, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)

            # Pre-rotate each microwave field into the slow-eigenbasis (shared across batch)
            Vh = V.conj().T
            H_mu_rot = [Vh @ H_mu_t(t_sample) @ V for H_mu_t in muw_hams]

            # The per-point propagator is U = A diag(ph) A^dagger with A = V @ V_rot,
            # which factors as V (V_rot diag(ph) V_rot^dagger) V^dagger. V is shared
            # across the batch, so rotate into the slow eigenbasis once here rather
            # than forming A (an n^3 matmul) for every scan point. Exact, not an
            # approximation.
            psis_slow = psis_batch @ V.conj()

            for b in range(batch):
                # Assemble rotating-frame Hamiltonian (in slow-eigenbasis)
                H_rot = (coupling_scales[b, 0] * H_mu_rot[0]).copy()
                for j in range(1, len(H_mu_rot)):
                    H_rot += coupling_scales[b, j] * H_mu_rot[j]

                if propagator == "magnus":
                    # Interaction picture with respect to the diagonal, which is
                    # handled exactly as phases. The coupling is small enough
                    # (`||A|| ~ 0.038` rad at the operating point) that a short
                    # Taylor series applied to the S state vectors converges to
                    # machine precision in `n**2 * S` work, replacing the `n**3`
                    # eigensolve below, and matches the frozen path to four
                    # significant figures on SPA2 and the analytic models. It does
                    # *not* match everywhere: the truncated second Magnus term can
                    # fall only linearly in `dt`, making this first order and far
                    # less accurate than `"frozen"` despite being faster per step.
                    # Check `bench_magnus_commutator.py` before selecting it; see
                    # `PROPAGATORS` above and Priority E in IMPROVEMENTS.md.
                    delta = D + D_mu_diag_batch[b]
                    A = H_rot * magnus_integral(delta, dt)
                    psis_slow[b] = apply_magnus_taylor(A, psis_slow[b]) * np.exp(
                        -1j * delta * dt
                    )[None, :]
                    continue

                # Add detunings + slow energies to diagonal
                H_rot.flat[diag_idx] += D
                H_rot.flat[diag_idx] += D_mu_diag_batch[b]

                # Diagonalize rotating-frame Hamiltonian (per scan point)
                if eig_backend == "zheevd":
                    D_rot, V_rot, info_rot = zheevd(H_rot)
                    if info_rot != 0:
                        D_rot, V_rot = np.linalg.eigh(H_rot)
                elif eig_backend == "numpy":
                    D_rot, V_rot = np.linalg.eigh(H_rot)
                else:
                    raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

                phases_rot = np.exp(-1j * D_rot * dt)
                tmp2 = psis_slow[b] @ V_rot.conj()
                tmp2 *= phases_rot[np.newaxis, :]
                psis_slow[b] = tmp2 @ V_rot.T

            # Rotate back out of the slow eigenbasis, once for the whole batch
            psis_batch = psis_slow @ V.T

            # Update reference for eigenvector tracking (shared)
            V_ref = evecs

        probabilities_final = None
        if store_final_probabilities:
            overlaps = psis_batch @ last_evecs.conj()
            probabilities_final = np.abs(overlaps) ** 2

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = psis_batch @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        self._label_gaps = gap_tracker.summary(self.trajectory.get_T())
        return psis_batch, probabilities_final, monitor_probabilities_final, V_ref_ini, V_ref

    def _time_evolve_mu_batched_shared_slow_multitone(
        self,
        *,
        H_slow_t: Callable[[float], np.ndarray],
        component_hams: List[Callable[[float], tuple[np.ndarray, np.ndarray]]],
        coupling_scales: np.ndarray,
        D_mu_diag_batch: np.ndarray,
        t_array: np.ndarray,
        static_field_indices: list[int],
        beat_fields: list[tuple[int, np.ndarray]],
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_final_probabilities: bool,
        store_final_monitor_probabilities: bool,
        progress: bool,
        eig_backend: str,
        time_sampling: str = "mid",
        propagator: str = "frozen",
    ):
        """Batched shared-slow evolution with beat phases for same-manifold tones."""
        # `"frozen"` freezes `H` at the sample point and exponentiates that
        # frozen matrix exactly, by eigendecomposition. `"magnus"` integrates `H`
        # across the step and exponentiates by Taylor series. Both are
        # second-order approximations of the same time-ordered exponential and
        # converge to the same answer; neither is the solution. Where `H` moves
        # appreciably within a step, integrating beats freezing.
        propagator = _normalize_propagator(propagator)
        if D_mu_diag_batch.ndim != 2:
            raise ValueError("D_mu_diag_batch must have shape (B,n)")
        batch = int(D_mu_diag_batch.shape[0])

        coupling_scales = np.asarray(coupling_scales)
        if coupling_scales.ndim != 2 or coupling_scales.shape[0] != batch:
            raise ValueError(
                "coupling_scales must have shape (B,M) matching D_mu_diag_batch batch size"
            )
        if coupling_scales.shape[1] != len(component_hams):
            raise ValueError(
                f"coupling_scales has M={coupling_scales.shape[1]} but component_hams has M={len(component_hams)}"
            )

        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        if D_mu_diag_batch.shape[1] != n:
            raise ValueError(
                f"D_mu_diag_batch has n={D_mu_diag_batch.shape[1]} but Hamiltonian has n={n}"
            )

        diag_idx = slice(None, None, n + 1)

        E_ref, V_ref = np.linalg.eigh(H_tini)
        index = np.argsort(E_ref)
        E_ref = E_ref[index]
        V_ref = V_ref[:, index]
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        self.init_state_vecs(H_tini, V_0=V_ref)
        psis_batch = np.repeat(self.psis[None, :, :], batch, axis=0)

        last_evecs = V_ref
        sample_offset = _sample_offset(time_sampling)
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + sample_offset * dt

            H_slow_i = H_slow_t(t_sample)
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            Es, evecs = D, V
            Es, evecs = reorder_evecs(evecs, Es, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)

            Vh = V.conj().T
            upper_rot: list[np.ndarray] = []
            lower_rot: list[np.ndarray] = []
            for component_t in component_hams:
                upper, lower = component_t(t_sample)
                upper_rot.append(Vh @ upper @ V)
                lower_rot.append(Vh @ lower @ V)

            beat_phases = [
                (field_idx, np.exp(-1j * delta_omega * t_sample))
                for field_idx, delta_omega in beat_fields
            ]
            # The exact propagator freezes the beat at `t_sample` (midpoint
            # sampling, second order). Magnus integrates it across the step
            # instead, and its kernel is anchored at the left endpoint.
            t0_step = float(t_array[i])

            # See the equivalent comment in _time_evolve_mu_batched_shared_slow.
            # The beat phases change the contents of H_rot but not its basis: the
            # component matrices are already rotated by Vh ... V, so V still
            # factors out of the propagator and can be applied once per timestep.
            psis_slow = psis_batch @ V.conj()

            for b in range(batch):
                H_rot = np.zeros((n, n), dtype=np.complex128)
                for field_idx in static_field_indices:
                    H_rot += coupling_scales[b, field_idx] * (
                        upper_rot[field_idx] + lower_rot[field_idx]
                    )
                for field_idx, phases in beat_phases:
                    phase = phases[b]
                    H_rot += coupling_scales[b, field_idx] * (
                        phase * upper_rot[field_idx]
                        + phase.conjugate() * lower_rot[field_idx]
                    )

                if propagator == "magnus":
                    # Each component needs its own oscillatory kernel. A beating
                    # field rotates at `delta_omega` *within* the step, so its
                    # raising part integrates against
                    # `exp(i*((d_i - d_j) - delta_omega)*s)` and its lowering part
                    # against `+delta_omega`. Reusing the static kernel with the
                    # beat phase frozen at `t_sample` would be wrong whenever
                    # `delta_omega * dt` is not small -- which is the regime this
                    # path exists for.
                    delta = D + D_mu_diag_batch[b]
                    e_delta = np.exp(1j * delta * dt)
                    outer = np.outer(e_delta, e_delta.conj())
                    gaps = delta[:, None] - delta[None, :]
                    A = np.zeros((n, n), dtype=np.complex128)
                    if static_field_indices:
                        static_kernel = _magnus_integral_from_outer(outer, gaps, dt)
                        for field_idx in static_field_indices:
                            A += (
                                coupling_scales[b, field_idx]
                                * (upper_rot[field_idx] + lower_rot[field_idx])
                                * static_kernel
                            )
                    for field_idx, delta_omega_col in beat_fields:
                        w = float(delta_omega_col[b])
                        # Anchored at the step's *left* endpoint, not `t_sample`.
                        # The kernel is `int_0^dt exp(i*(gap + shift)*s) ds` with
                        # `s` measured from `t0`, and the interaction picture it
                        # inverts is the `exp(-i*delta*dt)` applied below, also
                        # from `t0`. Factoring `exp(-i*w*(t0 + s))` therefore
                        # leaves `exp(-i*w*t0)`. Using the midpoint here mixes two
                        # anchors and leaves a spurious constant `exp(-i*w*dt/2)`.
                        #
                        # With a *single* coupled pair that stray phase is a gauge
                        # transformation -- `exp(-i*th)` on the raising part and
                        # its conjugate on the lowering part is `R^H (.) R` for
                        # diagonal `R` -- so it telescopes away and changes no
                        # population at all. It becomes physical as soon as a
                        # second field is present, because the static coupling
                        # does not receive the phase and the relative phase
                        # between them is observable. That is the production
                        # configuration (SPA2 + backgrounds), where it shifts
                        # populations by `5e-03` at `N_steps=4000`.
                        beat = np.exp(-1j * w * t0_step)
                        A += coupling_scales[b, field_idx] * (
                            beat * upper_rot[field_idx]
                            * _magnus_integral_from_outer(outer, gaps, dt, -w)
                            + beat.conjugate() * lower_rot[field_idx]
                            * _magnus_integral_from_outer(outer, gaps, dt, w)
                        )
                    psis_slow[b] = apply_magnus_taylor(A, psis_slow[b]) * np.exp(
                        -1j * delta * dt
                    )[None, :]
                    continue

                H_rot.flat[diag_idx] += D
                H_rot.flat[diag_idx] += D_mu_diag_batch[b]

                if eig_backend == "zheevd":
                    D_rot, V_rot, info_rot = zheevd(H_rot)
                    if info_rot != 0:
                        D_rot, V_rot = np.linalg.eigh(H_rot)
                elif eig_backend == "numpy":
                    D_rot, V_rot = np.linalg.eigh(H_rot)
                else:
                    raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

                phases_rot = np.exp(-1j * D_rot * dt)
                tmp2 = psis_slow[b] @ V_rot.conj()
                tmp2 *= phases_rot[np.newaxis, :]
                psis_slow[b] = tmp2 @ V_rot.T

            psis_batch = psis_slow @ V.T

            V_ref = evecs

        probabilities_final = None
        if store_final_probabilities:
            overlaps = psis_batch @ last_evecs.conj()
            probabilities_final = np.abs(overlaps) ** 2

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = psis_batch @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        self._label_gaps = gap_tracker.summary(self.trajectory.get_T())
        return psis_batch, probabilities_final, monitor_probabilities_final, V_ref_ini, V_ref

    def _time_evolve(
        self,
        H_slow: Callable,
        t_array: np.ndarray,
        save_idx: Optional[np.ndarray],
        *,
        store_psis: bool,
        store_energies: bool,
        store_probabilities: bool,
        store_final_probabilities: bool,
        progress: bool,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_monitor_probabilities: bool,
        store_final_monitor_probabilities: bool,
        eig_backend: str,
        time_sampling: str = "mid",
    ):
        """
        Time evolves the system using the Hamiltonian function H_t
        over the time period in t_array.
        """
        if save_idx is None:
            save_idx = np.arange(len(t_array), dtype=int)

        # Calculate Hamiltonian at tini
        H_tini = H_slow(t_array[0])

        # Reference eigen-decomposition at t0 (reused for init + storage)
        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        # Initialize state vectors
        self.init_state_vecs(H_tini, V_0=V_ref)

        # Initialize containers to store results (possibly downsampled)
        psis_t, energies, probabilities = self._init_results_containers(
            t_array[save_idx],
            H_tini,
            store_psis=store_psis,
            store_energies=store_energies,
            store_probabilities=store_probabilities,
            D_0=E_ref,
            V_0=V_ref,
        )

        # Initialize reference matrix of eigenvectors that is used to keep track
        # of adiabatic evolution of eigenstates

        # Loop over t_array to time-evolve
        monitor_probabilities = None
        if store_monitor_probabilities and (monitor_idx is not None):
            monitor_probabilities = np.zeros(
                (len(save_idx), len(self.initial_states), monitor_idx.size)
            )
            amps0 = self.psis @ V_ref_ini[:, monitor_idx].conj()
            monitor_probabilities[0, :, :] = np.abs(amps0) ** 2

        out_i = 0
        last_evecs = V_ref
        sample_offset = _sample_offset(time_sampling)
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            # Calculate the timestep
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + sample_offset * dt

            # Calculate Hamiltonian
            H_slow_i = H_slow(t_sample)

            # Diagonalize Hamiltonian
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Reorder eigenvectors and energies
            Es, evecs = reorder_evecs(V, D, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)

            # Apply propagator without forming U_dt:
            # For row-vector storage (each state is a row), the update is
            # psi <- psi @ V.conj() @ diag(exp(-i D dt)) @ V.T
            phases = np.exp(-1j * D * dt)
            tmp = self.psis @ V.conj()
            tmp *= phases[np.newaxis, :]
            self.psis = tmp @ V.T

            # Store results for this timestep if requested
            if (i + 1) == save_idx[out_i + 1]:
                out_i += 1
                if store_psis and psis_t is not None:
                    psis_t[out_i, :, :] = self.psis
                if store_energies and energies is not None:
                    energies[out_i, :] = Es
                if store_probabilities and probabilities is not None:
                    probabilities[out_i, :, :] = self.calculate_probabilities(
                        self.psis, evecs
                    )

                if monitor_probabilities is not None:
                    amps = self.psis @ evecs[:, monitor_idx].conj()
                    monitor_probabilities[out_i, :, :] = np.abs(amps) ** 2

            # Change V_ref
            V_ref = evecs

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = self.psis @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        probabilities_final = None
        if store_final_probabilities:
            probabilities_final = self.calculate_probabilities(self.psis, last_evecs)

        self._label_gaps = gap_tracker.summary(self.trajectory.get_T())
        return (
            psis_t,
            energies,
            probabilities,
            probabilities_final,
            monitor_probabilities,
            monitor_probabilities_final,
            V_ref_ini,
            V_ref,
        )

    def _time_evolve_mu(
        self,
        H_slow_t: Callable,
        H_mu_t: Callable,
        D_mu: np.ndarray,
        t_array: np.ndarray,
        save_idx: Optional[np.ndarray],
        *,
        store_psis: bool,
        store_energies: bool,
        store_probabilities: bool,
        store_final_probabilities: bool,
        progress: bool,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_monitor_probabilities: bool,
        store_final_monitor_probabilities: bool,
        eig_backend: str,
        time_sampling: str = "mid",
    ):
        """
        Time evolves the system using the Hamiltonian function H_t
        over the time period in t_array.
        """
        if save_idx is None:
            save_idx = np.arange(len(t_array), dtype=int)

        # Calculate Hamiltonian at tini
        H_tini = H_slow_t(t_array[0])

        # Reference eigen-decomposition at t0 (reused for init + storage)
        E_ref, V_ref = np.linalg.eigh(H_tini)
        index = np.argsort(E_ref)
        E_ref = E_ref[index]
        V_ref = V_ref[:, index]
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        # Initialize state vectors
        self.init_state_vecs(H_tini, V_0=V_ref)

        # Initialize containers to store results (possibly downsampled)
        psis_t, energies, probabilities = self._init_results_containers(
            t_array[save_idx],
            H_tini,
            store_psis=store_psis,
            store_energies=store_energies,
            store_probabilities=store_probabilities,
            D_0=E_ref,
            V_0=V_ref,
        )

        # Initialize reference matrix of eigenvectors that is used to keep track
        # of adiabatic evolution of eigenstates

        # Loop over t_array to time-evolve
        monitor_probabilities = None
        if store_monitor_probabilities and (monitor_idx is not None):
            monitor_probabilities = np.zeros(
                (len(save_idx), len(self.initial_states), monitor_idx.size)
            )
            amps0 = self.psis @ V_ref_ini[:, monitor_idx].conj()
            monitor_probabilities[0, :, :] = np.abs(amps0) ** 2

        out_i = 0
        last_evecs = V_ref
        sample_offset = _sample_offset(time_sampling)
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            # Calculate the timestep
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + sample_offset * dt

            # Calculate Hamiltonians
            H_slow_i = H_slow_t(t_sample)
            H_mu_i = H_mu_t(t_sample)

            # Diagonalize slow Hamiltonian and transfer to basis where it is
            # diagonal
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Sort the eigenvalues so they are in ascending order
            # index = np.argsort(D)
            # D = D[index]
            # V = V[:, index]

            # Build rotating-frame Hamiltonian in the slow-eigenbasis.
            # Since V diagonalizes H_slow_i, we have V^H H_slow_i V = diag(D),
            # so we only need to rotate the microwave part.
            H_rot = V.conj().T @ H_mu_i @ V
            H_rot = H_rot + D_mu
            H_rot[np.diag_indices_from(H_rot)] += D

            # Diagonalize the Hamiltonian in the rotating frame
            if eig_backend == "zheevd":
                D_rot, V_rot, info_rot = zheevd(H_rot)
                if info_rot != 0:
                    D_rot, V_rot = np.linalg.eigh(H_rot)
            elif eig_backend == "numpy":
                D_rot, V_rot = np.linalg.eigh(H_rot)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Reorder eigenvectors and energies
            Es, evecs = D, V
            Es, evecs = reorder_evecs(evecs, Es, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)

            # Compute the propagator
            # Combine the unitary matrices
            A = V @ V_rot

            # Apply propagator without forming U_dt (same trick as above)
            phases_rot = np.exp(-1j * D_rot * dt)
            tmp = self.psis @ A.conj()
            tmp *= phases_rot[np.newaxis, :]
            self.psis = tmp @ A.T

            # Store results for this timestep if requested
            if (i + 1) == save_idx[out_i + 1]:
                out_i += 1
                if store_psis and psis_t is not None:
                    psis_t[out_i, :, :] = self.psis
                if store_energies and energies is not None:
                    energies[out_i, :] = Es
                if store_probabilities and probabilities is not None:
                    probabilities[out_i, :, :] = self.calculate_probabilities(
                        self.psis, evecs
                    )

                if monitor_probabilities is not None:
                    amps = self.psis @ evecs[:, monitor_idx].conj()
                    monitor_probabilities[out_i, :, :] = np.abs(amps) ** 2

            # Change V_ref
            V_ref = evecs

        probabilities_final = None
        if store_final_probabilities:
            probabilities_final = self.calculate_probabilities(self.psis, last_evecs)

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = self.psis @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        self._label_gaps = gap_tracker.summary(self.trajectory.get_T())
        return (
            psis_t,
            energies,
            probabilities,
            probabilities_final,
            monitor_probabilities,
            monitor_probabilities_final,
            V_ref_ini,
            V_ref,
        )

    def _init_results_containers(
        self,
        t_array: np.ndarray,
        H_tini: np.ndarray,
        *,
        store_psis: bool,
        store_energies: bool,
        store_probabilities: bool,
        D_0: Optional[np.ndarray] = None,
        V_0: Optional[np.ndarray] = None,
    ):
        """
        Initializes containers for time evolution results based on array of times
        and Hamiltonian at initial time.
        """
        psis_t: Optional[np.ndarray] = None
        energies: Optional[np.ndarray] = None
        probabilities: Optional[np.ndarray] = None

        if store_psis:
            psis_t = np.zeros(
                (len(t_array), len(self.initial_states), len(self.hamiltonian.QN)),
                dtype="complex",
            )
            psis_t[0, :, :] = self.psis

        if store_energies:
            energies = np.zeros((len(t_array), len(self.hamiltonian.QN)))

        if store_probabilities:
            # probabilities has same shape as psis_t would have
            probabilities = np.zeros(
                (len(t_array), len(self.initial_states), len(self.hamiltonian.QN))
            )

        if store_energies or store_probabilities:
            D = D_0
            V = V_0
            if D is None or V is None:
                D, V = np.linalg.eigh(H_tini)

            if store_energies and energies is not None:
                energies[0, :] = D
            if store_probabilities and probabilities is not None:
                probabilities[0, :, :] = self.calculate_probabilities(self.psis, V)

        return psis_t, energies, probabilities

    def calculate_probabilities(self, psis: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Given state vectors as columns of psi, for each state vector, returns the
        probabilities of being in states stored as columns of V.
        """
        # overlaps_list = []
        # for psi in psis:
        #     overlaps_list.append(V.conj().T @ psi)
        # overlaps = np.array(overlaps_list)

        overlaps = psis @ V.conj()

        return np.abs(overlaps) ** 2
