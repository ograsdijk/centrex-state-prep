"""Midpoint sampling and a graded time grid, measured here with a metric that
does not work. **Both conclusions below were overturned; read this banner first.**

**Midpoint is not rejected -- it is the shipped default.** `time_sampling="mid"`
has been the default since `f75d57a`. Against closed-form solutions it is second
order (`2.00` measured) where left-endpoint is first (`1.00`), and on a commuting
model at production `delta*dt` left-endpoint saturates at `4e-01` with no
convergence while midpoint reaches `9.0e-03`. The "no systematic advantage" found
below came from comparing populations at fixed detunings near a steep lineshape,
where the comparison measures the slope rather than discretisation error.

**The graded-grid verdict is right, but not for the reason given.** Grading does
cut the summed local error; what defeats it on SPA2 is that a uniform grid's
leading error telescopes to a boundary term, worth `14-18x`, which grading gives
up. The ceiling is `1/f` for active fraction `f`, and SPA2 is `63%` active. It
does help where the active fraction is small.

**The reasoning about phase cancellation below is also wrong.** Midpoint does not
fail because the cancelled term is a per-eigenstate phase; it does not fail at
all. See "Performance Priority E" in `IMPROVEMENTS.md`, and score against
`benchmarks/analytic_models.py` rather than against a finer run of the same
scheme.

Original text follows, kept because the settling criterion and the label-stability
note are still sound.

Midpoint sampling and a graded time grid, both measured and both rejected.

Runtime is close to linear in `N_steps` (LAPACK is over 90% of the loop body),
so anything that cuts the step count at fixed accuracy converts almost 1:1 into
wall clock. Two candidates were tested here. Neither touches the propagator,
which stays an exact diagonalisation per step.

**Midpoint. Rejected.** All four evolution loops evaluate `H` at the left
endpoint `t_i`. The step is exact for a frozen `H`, so the only time-sampling
error is the time-ordering term -- `O(dt^2)` per step, `O(dt)` globally -- and
`H(t_i + dt/2)` costs the same and should have made it `O(dt^2)`. Measured on
partial-transfer cells, it does not: midpoint is sometimes better and sometimes
worse than left-endpoint with no systematic advantage, and both settle at the
same step count.

The reasoning that predicted a win was that the left-endpoint diagonal phase
error `delta'*dt^2/2` dominates and midpoint cancels it exactly. It does cancel
it, but that term is a per-eigenstate *phase*, and the observable is populations
in that same instantaneous eigenbasis, so cancelling it does not move the
answer. This is the fourth propagator change to fail here, after Strang, Lie and
Magnus, and it failed the same way: a plausible error-term argument that a
measurement refuses.

**Graded grid. Rejected.** The loops already read `dt = t_array[i+1] -
t_array[i]`, so a non-uniform grid needs no change to any loop body -- only to
the `np.linspace` that builds it. Steps are placed by equidistributing
`||dH/dt||^(1/(p+1))`. It helps materially below `N_steps=8000` (`4.1e-03`
against `2.2e-02` at 2000) but the advantage is gone by 16000, where it is
slightly worse, and both settle under `1e-03` at the same step count. The gain
is confined to the coarse regime nothing runs in.

**The trap, and why `report()` scores the way it does.** The error oscillates in
`N_steps` rather than falling as a power law: `delta*dt` is about `7.65e+04` rad
per step and the residual is quasi-periodic in `N`. Comparing variants at a
single step count therefore rewards whichever one happens to sit near a node,
which is exactly how a first pass at this benchmark produced a 2x that
evaporated. `report()` uses a settling criterion instead -- the smallest
`N_steps` beyond which *every* larger `N_steps` tested stays under tolerance.

**Accuracy is read as quantum-number populations**, never as elementwise
`probabilities_final`, which `IMPROVEMENTS.md` ("Label Swaps Are Crossings, Not
Degeneracy") shows is confounded by
adiabatic label swaps. The selected indices were verified stable across every
step count tested, so what the tables show is numerics rather than labelling.

Also measured, also a no-op: re-diagonalising at exactly `t=T` for the readout
basis (`--endpoint-basis`). After the loop `psis` is at `T` but `last_evecs` is
the basis at the last *sampled* time, which looked like an `O(dt)` readout error
capping everything else. It moves `V_fin` by `2.1e-04` and the populations by
`2.1e-08` -- quadratic, because the readout state is already an eigenstate, so
the first-order term vanishes by orthogonality.

Examples:

    python benchmarks/bench_step_grid.py --n-steps 2000 4000 8000 16000 32000         --reference-n-steps 128000 --output results/step_grid.json --csv
    python benchmarks/bench_step_grid.py --variants left-uniform left-graded         --n-steps 2000 4000 --reference-n-steps 32000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.linalg.lapack import zheevd

from common import (
    REPO_ROOT,
    build_spa2_setup,
    make_payload,
    parse_js,
    resolve_csv_path,
    result_row,
    write_results,
)
from state_prep import Simulator, scan_grid
from state_prep.utils import LabelGapTracker, find_max_overlap_idx, reorder_evecs

SHIPPED = Simulator._time_evolve_mu_batched_shared_slow

# The SPA2 initial state, J=1 triplet mF=0. Index into `spa2_initial_states()`.
INITIAL_IDX = 2

DEFAULT_DETUNINGS_MHZ = [-1.0, 0.0, 1.5]
DEFAULT_PREFACTORS = [1.0, 4.0]

VARIANTS = ("left-uniform", "mid-uniform", "left-graded", "mid-graded")


# --------------------------------------------------------------------------
# Graded time grid
# --------------------------------------------------------------------------


def align_eigenvectors(V0: np.ndarray, V1: np.ndarray) -> np.ndarray:
    """Permute and phase-fix the columns of `V1` onto `V0`.

    `eigh` fixes neither column order under near-degeneracy nor the arbitrary
    phase of each column, and both would show up as spurious rotation in a
    finite difference. Columns are matched by maximum overlap, then each is
    multiplied by the unit phase that makes `<v0|v1>` real and positive.
    """
    overlap = V0.conj().T @ V1
    order = np.argmax(np.abs(overlap), axis=0)
    # argmax per column can collide under exact degeneracy; fall back to
    # identity for any column whose partner was already claimed.
    seen: set[int] = set()
    perm = np.empty(V1.shape[1], dtype=int)
    for j in range(V1.shape[1]):
        k = int(order[j])
        perm[j] = k if k not in seen else j
        seen.add(perm[j])
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    V1 = V1[:, inverse]
    diag = np.einsum("ij,ij->j", V0.conj(), V1)
    phase = np.where(np.abs(diag) > 0, diag / np.maximum(np.abs(diag), 1e-300), 1.0)
    return V1 * phase.conj()[None, :]


MONITORS = ("dh", "comm", "nac", "mu")


def error_density(
    H_slow_t: Callable[[float], np.ndarray],
    muw_hams: list[Callable[[float], np.ndarray]],
    T: float,
    *,
    probes: int,
    monitor: str = "nac",
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a step-size indicator on a coarse grid spanning `[0, T]`.

    Which quantity to equidistribute is the whole design question, and the
    obvious choice is the wrong one. Writing the exact step as a Magnus series
    against the frozen-`H` step used by the loops:

        Omega_1 = -i (H dt + H' dt^2/2 + ...)     -> left-endpoint misses H' dt^2/2
        Omega_2 = (dt^3/12) [H, H'] + ...         -> the time-ordering term

    so `||H'||` is the leading local error density. But most of `||H'||` here is
    the *diagonal* -- GHz Stark shifts of the rotational levels -- and the loop
    propagates a frozen diagonal exactly, so that part is not an error at all.
    Measured on the SPA2 setup, the slow part outweighs the microwave part by
    about `3e+08` at the median, which makes `dh` grade almost entirely on the
    Stark ramp and barely on the beam crossing.

    The alternatives remove the diagonal in opposite ways. In the eigenbasis of
    `H`, `[H, H']_ij = (E_i - E_j) H'_ij`: purely off-diagonal, weighted *up* by
    the energy gap. The non-adiabatic coupling `H'_ij / (E_i - E_j)` is also
    purely off-diagonal but weighted *down* by it, and equals the rotation rate
    of the eigenbasis, which is what `reorder_evecs` has to track and where
    population actually moves between adiabatic states.

    - `dh`   -- `||dH/dt||_2`, slow plus microwaves. The naive choice.
    - `comm` -- `||[H, dH/dt]||_2`, the Magnus time-ordering term.
    - `nac`  -- `||dV/dt||_F`, eigenbasis rotation rate, by finite difference of
                phase-aligned eigenvectors. No gap denominator, so no
                regularisation is needed near degeneracies.
    - `mu`   -- `||dH_mu/dt||_2` alone, the beam envelope, as a control.
    """
    if monitor not in MONITORS:
        raise ValueError(f"monitor must be one of {MONITORS}")

    ts = np.linspace(0.0, T, probes)
    h = 0.5 * (ts[1] - ts[0])
    density = np.empty(probes)

    def H_total(t: float) -> np.ndarray:
        H = H_slow_t(t)
        for H_mu_t in muw_hams:
            H = H + H_mu_t(t)
        return H

    def H_mu_only(t: float) -> np.ndarray:
        H = muw_hams[0](t)
        for H_mu_t in muw_hams[1:]:
            H = H + H_mu_t(t)
        return H

    for i, t in enumerate(ts):
        lo = float(min(max(t - h, 0.0), T))
        hi = float(min(max(t + h, 0.0), T))
        span = hi - lo

        if monitor == "mu":
            density[i] = np.linalg.norm(H_mu_only(hi) - H_mu_only(lo), 2) / span
            continue

        H0, H1 = H_total(lo), H_total(hi)
        dH = (H1 - H0) / span

        if monitor == "dh":
            density[i] = np.linalg.norm(dH, 2)
        elif monitor == "comm":
            H_mid = H_total(float(t))
            density[i] = np.linalg.norm(H_mid @ dH - dH @ H_mid, 2)
        else:  # nac
            _, V0 = np.linalg.eigh(H0)
            _, V1 = np.linalg.eigh(H1)
            V1 = align_eigenvectors(V0, V1)
            density[i] = np.linalg.norm(V1 - V0, "fro") / span

    return ts, density


def graded_grid(
    ts: np.ndarray,
    density: np.ndarray,
    *,
    n_steps: int,
    order: int,
    max_ratio: float,
) -> np.ndarray:
    """Equidistribute `density**(1/(order+1))` over `n_steps` points.

    For a method of global order `p` with local error density `g`, the step
    count needed for a fixed error is minimised by making `g**(1/(p+1)) * dt`
    constant. `max_ratio` caps how much longer than the uniform step any step
    may become: adiabatic label tracking in `reorder_evecs` works by
    consecutive-step eigenvector overlap and degrades as steps grow.
    """
    T = float(ts[-1])
    weight = np.maximum(density, np.max(density) * 1e-12) ** (1.0 / (order + 1))

    # Floor the weight so no step exceeds max_ratio * uniform dt. A step is
    # inversely proportional to the weight, so a floor on the weight is a cap on
    # the step.
    mean_weight = np.trapezoid(weight, ts) / T
    weight = np.maximum(weight, mean_weight / max_ratio)

    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(ts))])
    cumulative /= cumulative[-1]
    grid = np.interp(np.linspace(0.0, 1.0, n_steps), cumulative, ts)
    grid[0], grid[-1] = 0.0, T
    return grid


# --------------------------------------------------------------------------
# Loop variants
# --------------------------------------------------------------------------


def make_loop(*, midpoint: bool, grade: dict[str, Any] | None, endpoint_basis: bool = False):
    """The shipped shared-slow loop, with time sampling and grid swapped out.

    Kept deliberately close to `Simulator._time_evolve_mu_batched_shared_slow`
    so the comparison isolates the two changes. Regrading happens here, on the
    `t_array` handed in, rather than in `run_microwave_scan`, so that Phase 1
    needs no change under `src/`.
    """

    def impl(
        self,
        *,
        H_slow_t,
        muw_hams,
        coupling_scales,
        D_mu_diag_batch,
        t_array,
        monitor_states,
        store_final_probabilities,
        store_final_monitor_probabilities,
        progress,
        eig_backend,
    ):
        batch = int(D_mu_diag_batch.shape[0])
        coupling_scales = np.asarray(coupling_scales)

        if grade is not None:
            t_array = graded_grid(
                grade["ts"],
                grade["density"],
                n_steps=len(t_array),
                order=2 if midpoint else 1,
                max_ratio=grade["max_ratio"],
            )

        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        diag_idx = slice(None, None, n + 1)

        E_ref, V_ref = np.linalg.eigh(H_tini)
        index = np.argsort(E_ref)
        V_ref = V_ref[:, index]
        V_ref_ini = V_ref

        monitor_idx = None
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

        def eig(M):
            if eig_backend == "zheevd":
                w, v, info = zheevd(M)
                if info != 0:
                    w, v = np.linalg.eigh(M)
                return w, v
            return np.linalg.eigh(M)

        last_evecs = V_ref
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        for i, t in enumerate(t_array[:-1]):
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + 0.5 * dt if midpoint else t

            D, V = eig(H_slow_t(t_sample))
            Es, evecs = reorder_evecs(V, D, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)

            Vh = V.conj().T
            H_mu_rot = [Vh @ H_mu_t(t_sample) @ V for H_mu_t in muw_hams]
            psis_slow = psis_batch @ V.conj()

            for b in range(batch):
                H_rot = (coupling_scales[b, 0] * H_mu_rot[0]).copy()
                for j in range(1, len(H_mu_rot)):
                    H_rot += coupling_scales[b, j] * H_mu_rot[j]
                H_rot.flat[diag_idx] += D
                H_rot.flat[diag_idx] += D_mu_diag_batch[b]

                D_rot, V_rot = eig(H_rot)
                tmp = psis_slow[b] @ V_rot.conj()
                tmp *= np.exp(-1j * D_rot * dt)[np.newaxis, :]
                psis_slow[b] = tmp @ V_rot.T

            psis_batch = psis_slow @ V.T
            V_ref = evecs

        if endpoint_basis:
            # `psis_batch` is at t_array[-1], but `last_evecs` is the eigenbasis
            # at the last *sampled* time -- one step earlier with left-endpoint
            # sampling, half a step earlier with midpoint. Projecting the state
            # onto a stale basis is an O(dt) error in the readout, which would
            # cap a second-order propagator at first order. One extra eigensolve
            # for the whole run fixes it.
            D_end, V_end = eig(H_slow_t(t_array[-1]))
            _, last_evecs = reorder_evecs(V_end, D_end, V_ref)
            # `V_fin` feeds `final_quantum_numbers()`, which `population()` reads
            # to pick an index into these probabilities. The two must be the same
            # basis or the labels name the wrong column.
            V_ref = last_evecs

        probs = None
        if store_final_probabilities:
            probs = np.abs(psis_batch @ last_evecs.conj()) ** 2
        mon = None
        if store_final_monitor_probabilities and monitor_idx is not None:
            mon = np.abs(psis_batch @ last_evecs[:, monitor_idx].conj()) ** 2

        self._label_gaps = gap_tracker.summary(self.trajectory.get_T())
        return psis_batch, probs, mon, V_ref_ini, V_ref

    return impl


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def make_grid(detunings_mhz, prefactors, n_fields: int):
    det = np.array([d * 1e6 for d in detunings_mhz for _ in prefactors], dtype=float)
    pref = np.array([p for _ in detunings_mhz for p in prefactors], dtype=float)
    det_b = np.column_stack([det] * n_fields)
    pref_b = np.column_stack([np.ones_like(pref)] + [pref] * (n_fields - 1))
    labels = [f"det={d:+.1f}MHz pref={p:g}" for d in detunings_mhz for p in prefactors]
    return det_b, pref_b, labels


def readout_identities(setup) -> tuple[dict, dict]:
    """Quantum numbers of the source and target states at readout.

    Reuses `scripts/analyze_spa2_bg_feature.readout_identity`, which propagates
    the DC fields with the microwaves off rather than trusting an adiabatic
    label. The Stark ramp-down turns both states from Stark mixtures into
    definite-F states, and which one is physics.
    """
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import analyze_spa2_bg_feature as bg

    from state_prep.approximate_states import J1_triplet_0, J2_triplet_0

    source = bg.readout_identity(setup, J1_triplet_0)
    target = bg.readout_identity(setup, J2_triplet_0)
    return source, target


def observables(result, setup, source_id, target_id) -> np.ndarray:
    """(remaining, transferred) per scan point, by quantum numbers."""
    ini = setup["initial_states"][INITIAL_IDX]
    keys = ("J", "F1", "F", "mF")
    remaining = result.population(ini, **{k: source_id[k] for k in keys})
    transferred = result.population(ini, **{k: target_id[k] for k in keys})
    return np.column_stack([remaining, transferred])


def run_scan(simulator, setup, *, det, pref, n_steps, eig_backend, workers):
    return simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        workers=workers,
        eig_backend=eig_backend,
    )


def timed_round(variants, impls, simulator, setup, *, det, pref, n_steps, args):
    """One timed run of every variant, interleaved.

    Interleaved rather than variant-at-a-time, per the protocol in
    `AGENTS.md:253-266`: running all of one variant then all of the next lets
    thermal drift masquerade as a difference between them.
    """
    times: dict[str, float] = {}
    results: dict[str, Any] = {}
    for variant in variants:
        Simulator._time_evolve_mu_batched_shared_slow = impls[variant]
        try:
            start = time.perf_counter()
            results[variant] = run_scan(
                simulator,
                setup,
                det=det,
                pref=pref,
                n_steps=n_steps,
                eig_backend=args.eig_backend,
                workers=args.workers,
            )
            times[variant] = time.perf_counter() - start
        finally:
            Simulator._time_evolve_mu_batched_shared_slow = SHIPPED
    return times, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--js", default="0,1,2,3")
    parser.add_argument("--n-steps", type=int, nargs="+", default=[1250, 2500, 5000, 10000])
    parser.add_argument("--reference-n-steps", type=int, default=128000)
    parser.add_argument("--reference-variant", default="left-graded", choices=list(VARIANTS),
                        help="scheme used for the reference. Defaults to a graded grid: "
                             "uniform at 128000 is measurably NOT converged (2.5e-04 from "
                             "the graded cluster), and referencing a scheme against a finer "
                             "run of itself flatters it through correlated error structure.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--detunings-mhz", type=float, nargs="+", default=DEFAULT_DETUNINGS_MHZ)
    parser.add_argument("--prefactors", type=float, nargs="+", default=DEFAULT_PREFACTORS)
    parser.add_argument("--max-ratio", type=float, default=64.0,
                        help="cap on a graded step, as a multiple of the uniform step. "
                             "The default is effectively uncapped: equidistribution on this "
                             "trajectory saturates at 47x and 256 gives the same grid as 64. "
                             "A low cap was tried first and is a mistake -- exp(-i H dt) is "
                             "exact for constant H, so quiet regions tolerate long steps, and "
                             "raising the cap from 4 to 64 costs no accuracy.")
    parser.add_argument("--monitor", default="nac", choices=list(MONITORS),
                        help="quantity the graded grid equidistributes; see error_density")
    parser.add_argument("--probes", type=int, default=400,
                        help="samples used to build the error density")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="worst-cell error a variant must settle under")
    parser.add_argument("--endpoint-basis", action="store_true",
                        help="re-diagonalise at t=T for the readout basis; measured "
                             "to move V_fin by 2.1e-04 and populations by 2.1e-08")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="loky workers; the monkeypatched loop must be dill-picklable above 1")
    parser.add_argument("--eig-backend", default="zheevd", choices=["zheevd", "numpy"])
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    js = parse_js(args.js)
    print(f"building setup js={js} ...", flush=True)
    setup = build_spa2_setup(js=js)
    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )

    print("resolving readout identities ...", flush=True)
    source_id, target_id = readout_identities(setup)
    keys = ("J", "F1", "F", "mF")
    print(f"  source {[source_id[k] for k in keys]}  pop={source_id['population']:.6f}")
    print(f"  target {[target_id[k] for k in keys]}  pop={target_id['population']:.6f}")

    n_fields = len(setup["microwave_fields"])
    det, pref, labels = make_grid(args.detunings_mhz, args.prefactors, n_fields)
    print(f"grid: {len(labels)} points, {n_fields} microwave fields", flush=True)

    # The error density is built once: it depends only on the fields, not on the
    # step count or the scan point.
    grade = None
    if any("graded" in v for v in args.variants):
        print("building error density ...", flush=True)
        T = setup["trajectory"].get_T()
        H_slow_t, muw_hams = probe_hamiltonians(simulator, setup)
        ts, density = error_density(
            H_slow_t, muw_hams, T, probes=args.probes, monitor=args.monitor
        )
        grade = {"ts": ts, "density": density, "max_ratio": args.max_ratio}
        print(f"  monitor={args.monitor}: max/median = {density.max() / np.median(density):.1f}")

    endpoint = args.endpoint_basis
    impls = {
        "left-uniform": make_loop(midpoint=False, grade=None, endpoint_basis=endpoint),
        "mid-uniform": make_loop(midpoint=True, grade=None, endpoint_basis=endpoint),
        "left-graded": make_loop(midpoint=False, grade=grade, endpoint_basis=endpoint),
        "mid-graded": make_loop(midpoint=True, grade=grade, endpoint_basis=endpoint),
    }

    # Reference: both uniform variants at the fine step count. Their gap bounds
    # how far the reference itself can be from converged, which is exactly what
    # `bench_convergence.py`'s header warns cannot be assumed away.
    print(f"reference at N_steps={args.reference_n_steps} ...", flush=True)
    # Two *different* discretisations, not one scheme at two step counts: the
    # gap between them is a usable bound on the reference error, whereas a
    # scheme compared to a finer version of itself understates its own error.
    ref_pair = (args.reference_variant,
                "left-uniform" if args.reference_variant != "left-uniform" else "left-graded")
    ref_times, ref_results = timed_round(
        ref_pair, impls, simulator, setup,
        det=det, pref=pref, n_steps=args.reference_n_steps, args=args,
    )
    references = {
        variant: observables(result, setup, source_id, target_id)
        for variant, result in ref_results.items()
    }
    for variant, elapsed in ref_times.items():
        print(f"  {variant}: {elapsed:.1f}s", flush=True)

    ref_gap = float(np.max(np.abs(references[ref_pair[0]] - references[ref_pair[1]])))
    reference = references[args.reference_variant]
    print(f"  reference: {args.reference_variant} at N={args.reference_n_steps}")
    print(f"  spread against {ref_pair[1]} at the same N: {ref_gap:.3e}")

    rows: list[dict[str, Any]] = []
    print()
    print(f"batch={len(labels)}, backend={args.eig_backend}, warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'variant':>13}{'N_steps':>9}{'mean':>10}{'std':>9}{'worst err':>13}{'worst cell':>22}")

    for n_steps in args.n_steps:
        collected: dict[str, list[float]] = {v: [] for v in args.variants}
        results: dict[str, Any] = {}
        for round_index in range(args.warmup + args.repeat):
            times, results = timed_round(
                args.variants, impls, simulator, setup,
                det=det, pref=pref, n_steps=n_steps, args=args,
            )
            if round_index >= args.warmup:
                for variant, elapsed in times.items():
                    collected[variant].append(elapsed)

        for variant in args.variants:
            times, result = collected[variant], results[variant]
            values = observables(result, setup, source_id, target_id)
            per_cell = np.abs(values - reference).max(axis=1)
            worst = int(np.argmax(per_cell))
            err = float(per_cell.max())
            mean_s, std_s = float(np.mean(times)), float(np.std(times))
            crossings = result.label_gaps["crossings"] if result.label_gaps else None
            print(
                f"{variant:>13}{n_steps:>9}{mean_s:>9.2f}s{std_s:>8.2f}s"
                f"{err:>13.3e}{labels[worst]:>22}"
            )
            rows.append(
                result_row(
                    variant,
                    n_steps=n_steps,
                    time_mean_s=mean_s,
                    time_std_s=std_s,
                    worst_abs_error=err,
                    worst_cell=labels[worst],
                    per_cell_error=[float(v) for v in per_cell],
                    labels_crossing=int((crossings > 0).sum()) if crossings is not None else None,
                )
            )

    report(rows, args, ref_gap)

    if args.output or args.csv is not None:
        payload = make_payload(
            benchmark="step_grid",
            config={
                "js": js,
                "n_steps": args.n_steps,
                "reference_n_steps": args.reference_n_steps,
                "reference_variant": args.reference_variant,
                "reference_uncertainty": ref_gap,
                "variants": args.variants,
                "detunings_mhz": args.detunings_mhz,
                "prefactors": args.prefactors,
                "max_ratio": args.max_ratio,
                "probes": args.probes,
                "monitor": args.monitor,
                "repeat": args.repeat,
                "warmup": args.warmup,
                "workers": args.workers,
                "eig_backend": args.eig_backend,
                "tolerance": args.tolerance,
                "endpoint_basis": args.endpoint_basis,
                "source_identity": {k: source_id[k] for k in keys},
                "target_identity": {k: target_id[k] for k in keys},
            },
            results=rows,
            include_gpu_env=False,
        )
        write_results(
            payload,
            output=args.output,
            csv_path=resolve_csv_path(args.output, args.csv, "step_grid"),
        )


def probe_hamiltonians(simulator, setup):
    """`H_slow(t)` and the microwave `H_mu(t)` callables, as the loop sees them.

    `run_microwave_scan` builds these internally and does not expose them, so
    they are rebuilt here from the same objects.
    """
    trajectory = setup["trajectory"]
    hamiltonian = setup["hamiltonian"]

    def H_slow_t(t: float) -> np.ndarray:
        return hamiltonian.H_R(trajectory.R_t(float(t)))

    muw_hams = [
        mf.get_H_t_func(trajectory.R_t, hamiltonian.QN)
        for mf in setup["microwave_fields"]
    ]
    return H_slow_t, muw_hams


def report(rows: list[dict[str, Any]], args, ref_gap: float) -> None:
    """Cost to *settle* below a tolerance, not cost at one step count.

    The error on this problem oscillates in `N_steps` rather than falling as a
    power law -- `delta*dt` is about `7.65e+04` rad per step, and the residual
    is quasi-periodic in `N`. So "the cheapest N_steps whose error is below the
    target" rewards a variant for happening to sit near a node, which is how a
    first pass at this benchmark produced a 2x that evaporated on a second look.

    The envelope criterion instead asks for the smallest `N_steps` beyond which
    the error *stays* under the tolerance at every larger `N_steps` tested. That
    is the number a caller can actually rely on.
    """
    print()
    if ref_gap > 0.1 * args.tolerance:
        print(f"WARNING: reference uncertainty {ref_gap:.3e} is not small against "
              f"the {args.tolerance:.1e} tolerance. Raise --reference-n-steps.")

    ordered = sorted({r["n_steps"] for r in rows})
    print(f"settling point at tolerance {args.tolerance:.1e} "
          f"(smallest N_steps beyond which every larger N stays under it)")
    print(f"{'variant':>13}{'settles at':>13}{'cost there':>13}{'speedup':>10}")

    baseline_cost = None
    for variant in args.variants:
        by_n = {r["n_steps"]: r for r in rows if r["case"] == variant}
        settled = None
        for n in ordered:
            if all(by_n[m]["worst_abs_error"] <= args.tolerance for m in ordered if m >= n):
                settled = n
                break
        if settled is None:
            print(f"{variant:>13}{'not in range':>13}{'-':>13}{'-':>10}")
            continue
        cost = by_n[settled]["time_mean_s"]
        if baseline_cost is None:
            baseline_cost = cost
        print(f"{variant:>13}{settled:>13}{cost:>12.2f}s{baseline_cost / cost:>9.2f}x")

    print()
    print("Gate: adopt a variant only if it settles at least 1.3x cheaper than")
    print("left-uniform, in the worst cell of the grid. A win that appears at one")
    print("step count but not at the next larger one is a node, not a gain.")


if __name__ == "__main__":
    main()
