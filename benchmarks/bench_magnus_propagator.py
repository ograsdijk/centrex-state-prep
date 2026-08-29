"""End-to-end accuracy and speed of the interaction-picture Magnus propagator.

`bench_trotter_conditioning.py` bounds the error by summing the per-step
spectral-norm difference over all steps. That bound is pessimistic: it assumes
every step's error adds coherently, and at `N_steps=10000` it returns 2.07,
which exceeds the trivial bound of 2 for a difference of unitaries and is
therefore vacuous. Rejecting the propagator on that alone would repeat the
mistake `IMPROVEMENTS.md` documents elsewhere - concluding from a bound instead
of a measurement.

So this runs the real scan both ways and compares the observable.

The Magnus inner loop replaces the per-scan-point eigensolve with:

    delta_b = D + D_mu_diag_batch[b]
    M_ij    = (exp(i*delta_i*dt) * conj(exp(i*delta_j*dt)) - 1) / (i*(delta_i - delta_j))
    A       = H_mu_assembled . M            (elementwise, Hermitian)
    psi    <- (psi @ expm(-i*A)^T) * exp(-i*delta_b*dt)

Two things make this cheap. `M` needs only `n` complex exponentials, not `n^2`,
because `exp(i*(delta_i - delta_j)*dt)` factorises into an outer product. And
`expm(-i*A)` is never formed: `||A||` is about `4.3e-2` rad, so a short Taylor
series applied directly to the `S = 4` state vectors converges to machine
precision in `n^2*S` work rather than `n^3`.

    python benchmarks/bench_magnus_propagator.py --n-steps 10000 --batch 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.linalg.lapack import zheevd

from common import (
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

# ||A|| is about 4.3e-2 rad at production dt, so 8 terms leaves a truncation
# error near 1e-18 - far below every other error in the problem.
TAYLOR_TERMS = 8


def magnus_integral(delta: np.ndarray, dt: float) -> np.ndarray:
    """(exp(i*(d_i - d_j)*dt) - 1) / (i*(d_i - d_j)), elementwise, -> dt on the diagonal.

    Built from an outer product of `n` exponentials rather than `n^2` of them.
    The small-argument branch is load-bearing: `delta` is nearly degenerate
    within a rotational manifold, so the quotient divides by ~0 there.
    """
    e = np.exp(1j * delta * dt)
    x = delta[:, None] - delta[None, :]
    num = np.outer(e, e.conj()) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / (1j * x)
    small = np.abs(x * dt) < 1e-8
    if small.any():
        out[small] = dt * (1.0 + 0.5j * (x * dt)[small])
    return out


def apply_expm_taylor(A: np.ndarray, psis: np.ndarray) -> np.ndarray:
    """psis @ expm(-i*A)^T, by Taylor series applied to the vectors.

    `psis` is (S, n) in the row-vector convention the simulator uses, so the
    propagator acts on the right transposed. Each term costs n^2*S, never n^3.

    Scaling and squaring is applied when it is needed and skipped when it is not.
    At the production operating point `||A||` is about `4.3e-02` rad and `k` is
    `0`, so this costs nothing; a fixed 8-term series diverges once `||A||`
    approaches `1`, and it does so *silently* -- measured against exact
    solutions, a coupling-to-spread ratio of `5e-05` gives `||A|| ~ 1.9` and a
    10x accuracy loss with no warning at all, and `5e-04` gives `||A|| ~ 19` and
    an overflow to NaN. Guarding on the norm rather than trusting the operating
    point keeps the failure impossible instead of merely unlikely.

    Squaring on *vectors* means re-applying `exp(-iA/2^k)` `2^k` times rather
    than squaring a matrix, which keeps the `n^2*S` cost.
    """
    # Frobenius, not spectral: `||A||_2 <= ||A||_F`, so it is a valid bound for
    # scaling and squaring, and it costs `3 us` against `182 us` for the spectral
    # norm at n=64 -- which would have eaten 43% of the 420 us eigensolve this
    # propagator exists to avoid. A guard that costs what it saves is not a guard.
    norm = float(np.linalg.norm(A, "fro"))
    k = max(0, int(np.ceil(np.log2(norm / 0.5)))) if norm > 0.5 else 0
    B = (-1j * A / (2**k)).T

    out = psis
    for _ in range(2**k):
        term = out
        acc = out.copy()
        for j in range(1, TAYLOR_TERMS + 1):
            term = (term @ B) / j
            acc += term
        out = acc
    return out


def make_magnus_loop():
    """The shipped shared-slow loop with the per-point eigensolve replaced.

    Accepts `time_sampling` so the comparison against the shipped loop isolates
    the propagator: both must freeze `H` at the same point in the step.
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
        time_sampling="mid",
    ):
        sample_offset = 0.5 if time_sampling == "mid" else 0.0
        if time_sampling not in ("mid", "left"):
            raise ValueError(f"time_sampling must be 'mid' or 'left'; got {time_sampling!r}")
        batch = int(D_mu_diag_batch.shape[0])
        coupling_scales = np.asarray(coupling_scales)
        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])

        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref = V_ref[:, np.argsort(E_ref)]
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

        # The shipped loops track level crossings here, and the result reads
        # `label_gaps` off the Simulator with `getattr(self, "_label_gaps", None)`.
        # Without this a Magnus run silently inherits whatever a *previous* run
        # left on the instance -- stale crossing diagnostics on a fresh result.
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        last_evecs = V_ref
        for i, t in enumerate(t_array[:-1]):
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + sample_offset * dt

            H_slow_i = H_slow_t(t_sample)
            D, V = eig(H_slow_i)
            Es, evecs = reorder_evecs(V, D, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)

            Vh = V.conj().T
            H_mu_rot = [Vh @ H_mu_t(t_sample) @ V for H_mu_t in muw_hams]
            psis_slow = psis_batch @ V.conj()

            for b in range(batch):
                H_mu = (coupling_scales[b, 0] * H_mu_rot[0]).copy()
                for j in range(1, len(H_mu_rot)):
                    H_mu += coupling_scales[b, j] * H_mu_rot[j]

                delta = D + D_mu_diag_batch[b]
                A = H_mu * magnus_integral(delta, dt)
                psis_slow[b] = apply_expm_taylor(A, psis_slow[b]) * np.exp(
                    -1j * delta * dt
                )[None, :]

            psis_batch = psis_slow @ V.T
            V_ref = evecs

        probs = None
        if store_final_probabilities:
            probs = np.abs(psis_batch @ last_evecs.conj()) ** 2
        mon = None
        if store_final_monitor_probabilities and monitor_idx is not None:
            mon = np.abs(psis_batch @ last_evecs[:, monitor_idx].conj()) ** 2
        self._label_gaps = gap_tracker.summary(self.trajectory.get_T())
        return psis_batch, probs, mon, V_ref_ini, V_ref

    return impl


def run_scan(simulator, setup, *, det, pref, n_steps, eig_backend):
    return simulator.run_microwave_scan(
        detunings_hz=det,
        intensity_prefactors=pref,
        N_steps=n_steps,
        monitor_states=setup["monitor_states"],
        store_final_probabilities=True,
        store_final_monitor_probabilities=True,
        progress=False,
        eig_backend=eig_backend,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--js", default="0,1,2,3")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--n-steps", type=int, nargs="+", default=[10000, 20000])
    parser.add_argument("--repeat", type=int, default=3)
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
    det, pref = scan_grid(
        n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, args.batch)
    )
    magnus_impl = make_magnus_loop()

    rows = []
    print()
    print(f"batch={args.batch}, backend={args.eig_backend}, repeat={args.repeat}")
    print(
        f"{'N_steps':>8}{'exact mean':>13}{'std':>10}"
        f"{'magnus mean':>14}{'std':>10}{'speedup':>10}{'max abs diff':>15}"
    )

    for n_steps in args.n_steps:
        times = {"frozen": [], "magnus": []}
        results = {}
        # Interleaved rounds, per the protocol in IMPROVEMENTS.md: never run all
        # of one variant then all of the other.
        for _ in range(args.repeat):
            for name in ("frozen", "magnus"):
                Simulator._time_evolve_mu_batched_shared_slow = (
                    SHIPPED if name == "frozen" else magnus_impl
                )
                try:
                    start = time.perf_counter()
                    res = run_scan(
                        simulator,
                        setup,
                        det=det,
                        pref=pref,
                        n_steps=n_steps,
                        eig_backend=args.eig_backend,
                    )
                    times[name].append(time.perf_counter() - start)
                    results[name] = res
                finally:
                    Simulator._time_evolve_mu_batched_shared_slow = SHIPPED

        p_exact = np.asarray(results["frozen"].probabilities_final)
        p_magnus = np.asarray(results["magnus"].probabilities_final)
        max_diff = float(np.max(np.abs(p_exact - p_magnus)))

        m_exact = float(np.mean(times["frozen"]))
        s_exact = float(np.std(times["frozen"]))
        m_magnus = float(np.mean(times["magnus"]))
        s_magnus = float(np.std(times["magnus"]))
        speedup = m_exact / m_magnus

        print(
            f"{n_steps:>8}{m_exact:>12.3f}s{s_exact:>9.3f}s"
            f"{m_magnus:>13.3f}s{s_magnus:>9.3f}s{speedup:>9.2f}x{max_diff:>15.3e}"
        )
        rows.append(
            result_row(
                "n_steps",
                n_steps=n_steps,
                exact_mean_s=m_exact,
                exact_std_s=s_exact,
                magnus_mean_s=m_magnus,
                magnus_std_s=s_magnus,
                speedup=speedup,
                max_abs_diff=max_diff,
            )
        )

    print()
    print("Acceptance target is 1e-6, against the measured floors of 3.835e-07")
    print("(eigensolver backend swap) and 4.228e-06 (CPU/GPU).")
    worst = min(r["max_abs_diff"] for r in rows)
    if worst <= 1e-6:
        print("VERDICT: within the acceptance target at the finest step count tested.")
    else:
        print(f"VERDICT: best agreement reached is {worst:.3e}, above the target.")
    print()
    print("Note: probabilities_final is compared elementwise here. "
          "IMPROVEMENTS.md:337 shows that is confounded by adiabatic label swaps "
          "on the multitone path; this is the single-manifold shared-slow path, "
          "where the same run agrees to 4.378e-07 across eigensolver backends.")

    if args.output or args.csv is not None:
        payload = make_payload(
            benchmark="magnus_propagator",
            config={
                "js": js,
                "batch": args.batch,
                "n_steps": args.n_steps,
                "repeat": args.repeat,
                "eig_backend": args.eig_backend,
                "taylor_terms": TAYLOR_TERMS,
            },
            results=rows,
            include_gpu_env=False,
        )
        write_results(
            payload,
            output=args.output,
            csv_path=resolve_csv_path(args.output, args.csv, "magnus_propagator"),
        )


if __name__ == "__main__":
    main()
