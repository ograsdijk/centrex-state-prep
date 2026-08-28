"""Rank propagators against exact solutions, in both regimes.

Every earlier ranking in this work was scored against a finer run of some
scheme, and the reference's own error kept turning out to be the size of the
thing being measured. Here the answer is known, so the error is the error.

Two regimes, and they answer different questions:

- `clean` (`delta*dt ~ 1e-03`): a **correctness** gate. Order is a meaningful
  quantity here, so `mid` must show `2` and `left` must show `1`. Anything else
  is a bug in the loop, not physics.
- `realistic` (`delta*dt ~ 7.65e+04`, matching SPA2): the **ranking**. Nothing
  attains its formal order here -- the time-ordering terms carry powers of
  `delta*dt` and no asymptotic power law exists -- so propagators are compared by
  error at matched wall clock and no order is reported.

**How the model is split so Magnus has a job.** `make_magnus_loop` replaces only
the per-scan-point eigensolve of `H_rot`; it still diagonalises `H_slow` exactly.
Handing it `muw_hams=[0]` would leave it propagating a pure diagonal phase and
measure nothing. So the model Hamiltonian is split

    H_slow(t) = H_model(t) - H_mu(t),      H_mu(t) = small, with a Gaussian envelope

which sums back to `H_model` exactly, so the closed form still applies, while
giving Magnus a genuine small coupling to integrate against a large diagonal --
the real problem's structure, where the ratio is about `5e-07`.

Batch entries are identical (`coupling_scales=1`, `D_mu_diag_batch=0`), so every
one has the same known answer while the per-scan-point cost still scales with
batch size. That keeps the wall-clock comparison honest at production batch
without giving up an exact solution.

    python benchmarks/bench_analytic.py --regime clean
    python benchmarks/bench_analytic.py --regime realistic --batch 25 --output results/analytic.json --csv
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

import analytic_models as am
from bench_magnus_propagator import apply_expm_taylor, make_magnus_loop
from common import make_payload, resolve_csv_path, result_row, write_results
from state_prep import Simulator, build_time_grid, field_variation_density

SHIPPED = Simulator._time_evolve_mu_batched_shared_slow
MODELS = ("rotating", "scalar", "engineered", "spa_like")
VARIANTS = ("left", "mid", "magnus", "strang", "lie")



def make_splitting_loop(kind: str):
    """Lie or Strang splitting of the rotating-frame Hamiltonian.

    `H_rot = diag(D + D_mu) + H_mu_rot` -- a large diagonal and a small dense
    coupling -- so splitting looks attractive: the diagonal exponentiates for
    free and only the small part needs expanding.

        Lie:     exp(-i diag dt) exp(-i H_mu dt)
        Strang:  exp(-i diag dt/2) exp(-i H_mu dt) exp(-i diag dt/2)

    Both were rejected in `IMPROVEMENTS.md` on a *propagator-norm bound* --
    accumulated error `1.39e+01` against a trivial bound of `2` -- with no
    observable ever run. This document's own repeated lesson is to conclude from
    a measurement rather than a bound, so they are implemented here to be scored
    against exact solutions like everything else.

    The mechanism the bound identified: Trotter error is governed by
    `delta*dt`, not by the coupling-to-diagonal ratio. At `delta*dt ~ 7.65e+04`
    the diagonal phase wraps about `12000` times per step, so `[diag, H_mu] dt^2`
    is enormous however small `H_mu` is.
    """
    if kind not in ("lie", "strang"):
        raise ValueError(f"kind must be 'lie' or 'strang'; got {kind!r}")

    def impl(self, *, H_slow_t, muw_hams, coupling_scales, D_mu_diag_batch,
             t_array, monitor_states, store_final_probabilities,
             store_final_monitor_probabilities, progress, eig_backend,
             time_sampling="mid"):
        offset = 0.5 if time_sampling == "mid" else 0.0
        batch = int(D_mu_diag_batch.shape[0])
        coupling_scales = np.asarray(coupling_scales)
        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref = V_ref[:, np.argsort(E_ref)]
        V_ref_ini = V_ref
        self.init_state_vecs(H_tini, V_0=V_ref)
        psis_batch = np.repeat(self.psis[None, :, :], batch, axis=0)

        last = V_ref
        for i, t in enumerate(t_array[:-1]):
            dt = t_array[i + 1] - t_array[i]
            ts = t + offset * dt
            D, V = np.linalg.eigh(H_slow_t(ts))
            last = V
            Vh = V.conj().T
            H_mu_rot = [Vh @ H(ts) @ V for H in muw_hams]
            psis_slow = psis_batch @ V.conj()

            for b in range(batch):
                H_mu = (coupling_scales[b, 0] * H_mu_rot[0]).copy()
                for j in range(1, len(H_mu_rot)):
                    H_mu += coupling_scales[b, j] * H_mu_rot[j]
                delta = D + D_mu_diag_batch[b]

                if kind == "lie":
                    psi = psis_slow[b] * np.exp(-1j * delta * dt)[None, :]
                    psis_slow[b] = apply_expm_taylor(H_mu * dt, psi)
                else:
                    half = np.exp(-0.5j * delta * dt)[None, :]
                    psi = psis_slow[b] * half
                    psi = apply_expm_taylor(H_mu * dt, psi)
                    psis_slow[b] = psi * half

            psis_batch = psis_slow @ V.T

        probs = None
        if store_final_probabilities:
            probs = np.abs(psis_batch @ last.conj()) ** 2
        return psis_batch, probs, None, V_ref_ini, last

    return impl


def split_model(model: am.Model, *, coupling: float, sigma: float = 0.12):
    """Split `H_model` into a large slow part and a small enveloped coupling.

    The split is exact by construction -- `H_slow + H_mu == H_model` -- so the
    closed form is untouched. Its only purpose is to give the interaction-picture
    Magnus propagator the structure it exists to exploit.
    """
    rng = np.random.default_rng(12345)
    base = am.hermitian(rng, model.n)
    base = base / np.linalg.norm(base, 2) * coupling
    centre, width = 0.5 * model.T, sigma * model.T

    def H_mu(t: float) -> np.ndarray:
        return base * float(np.exp(-((float(t) - centre) ** 2) / (2.0 * width**2)))

    def H_slow(t: float) -> np.ndarray:
        return model.H_t(t) - H_mu(t)

    return H_slow, [H_mu]


def run_variant(model, variant, n_steps, batch, eig_backend="zheevd", coupling=1.0,
                graded=False, order=1):
    """Propagate `model` with one variant; return (final states, seconds).

    `graded` builds the grid with the **shipped** `build_time_grid` and
    `field_variation_density`, so what gets verified is the path users get
    rather than a reimplementation of it. The density pass sits inside the timed
    region, which is the honest place for it: it is a real cost of the option.
    """
    if model.H_slow is not None and model.muw_hams:
        H_slow, muw = model.H_slow, model.muw_hams   # model supplies its own split
    else:
        H_slow, muw = split_model(model, coupling=coupling)
    sim = am.stub_simulator(model)
    if variant == "magnus":
        loop = make_magnus_loop()
    elif variant in ("strang", "lie"):
        loop = make_splitting_loop(variant)
    else:
        loop = SHIPPED
    sampling = "left" if variant == "left" else "mid"

    start = time.perf_counter()
    if graded:
        t_array = build_time_grid(
            model.T, n_steps,
            density=field_variation_density(H_slow, muw),
            order=order,
        )
    else:
        t_array = np.linspace(0.0, model.T, n_steps)
    psis, _, _, _, _ = loop(
        sim,
        H_slow_t=H_slow,
        muw_hams=muw,
        coupling_scales=np.ones((batch, 1)),
        D_mu_diag_batch=np.zeros((batch, model.n)),
        t_array=t_array,
        monitor_states=None,
        store_final_probabilities=False,
        store_final_monitor_probabilities=False,
        progress=False,
        eig_backend=eig_backend,
        time_sampling=sampling,
    )
    return psis[0], time.perf_counter() - start


def build(name, regime, n_steps_ref, n=6, T=1.0, profile="smooth"):
    """Construct a model at the spectral scale the regime asks for.

    Which model to pick matters more than it looks, because geometry decided
    every grading result in this work:

    - `rotating` -- flat density (`1.01x`). Cannot evaluate grading at all.
    - `scalar` -- commuting, so the problem collapses to a scalar quadrature and
      uniform grids gain Euler-Maclaurin cancellation. Cannot evaluate grading.
    - `engineered` -- closed form, non-flat density, non-commuting coupling, but
      the coupling sits *on* the density peak.
    - `spa_like` -- the same, with SPA2's geometry: density peaked at a Stark-like
      ramp, beam elsewhere, and a level pair crossing twice. **This is the one to
      use for anything meant to transfer to production.**

    `engineered` and `spa_like` are both built as `U = V exp(-i Phi)`, which is
    the Lie-split form, so neither may be used to evaluate splitting methods with
    its natural split.
    """
    spread = am.spread_for_regime(regime, T, n_steps_ref)
    if name == "rotating":
        return am.rotating(n=n, spread=spread, T=T)
    if name == "engineered":
        return am.engineered(n=n, spread=spread, T=T)
    if name == "spa_like":
        return am.spa_like(n=n, spread=spread, T=T)
    return am.scalar(n=n, spread=spread, T=T, profile=profile)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regime", default="clean", choices=["clean", "realistic", "both"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--n-steps", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--n", type=int, default=6, help="Hilbert-space dimension")
    parser.add_argument("--coupling", type=float, default=5e-7,
                        help="||H_mu|| as a FRACTION of the spectral spread. The real "
                             "problem runs at 5e-07 (2.5e+05 rad/s against 5.0e+11). "
                             "An absolute coupling is the wrong knob: it has to scale "
                             "with the spread, or Magnus is never stressed and simply "
                             "reports the shipped loop's error to four digits.")
    parser.add_argument("--profile", default="smooth", choices=["smooth", "peaked"],
                        help="time profile for the scalar model. 'peaked' gives "
                             "||dH/dt|| three orders of dynamic range, as the real "
                             "trajectory has; 'smooth' is nearly flat, where a graded "
                             "grid has nothing to exploit and can only lose.")
    parser.add_argument("--grids", nargs="+", default=["uniform"],
                        choices=["uniform", "graded1", "graded2"],
                        help="graded1/graded2 use the shipped build_time_grid with "
                             "order=1/2. Grading is already shipped, but its accuracy "
                             "claims were measured against a reference that was itself "
                             "a graded run; this checks it against truth.")
    parser.add_argument("--eig-backend", default="zheevd", choices=["zheevd", "numpy"])
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    regimes = ["clean", "realistic"] if args.regime == "both" else [args.regime]
    rows: list[dict[str, Any]] = []

    for regime in regimes:
        print()
        print(f"=== regime: {regime} ===", flush=True)
        for name in args.models:
            model = build(name, regime, args.n_steps[0], n=args.n, profile=args.profile)
            exact = model.exact_states()
            commuting = model.meta.get("commuting")
            print(f"\n{name}  (n={args.n}, batch={args.batch}, "
                  f"delta*dt@{args.n_steps[0]}={model.delta_dt(args.n_steps[0]):.2e}, "
                  f"commuting={commuting})")
            if model.H_slow is not None and model.muw_hams:
                own = np.linalg.norm(model.muw_hams[0](0.5 * model.T), 2)
                print(f"  model supplies its own split; ||H_mu|| at midpoint = "
                      f"{own:.3e}  (--coupling ignored)")
            else:
                print(f"  ||H_mu||/spread = {args.coupling:.1e}  ->  ||H_mu|| = "
                      f"{args.coupling * float(model.meta['spread']):.3e}")
            print(f"{'variant/grid':>16}{'N_steps':>9}{'seconds':>10}{'max err':>13}{'order':>8}")

            combos = [(v, g) for g in args.grids for v in args.variants]
            errors: dict[tuple, list[float]] = {c: [] for c in combos}
            for n_steps in args.n_steps:
                for variant, grid in combos:
                    psis, seconds = run_variant(
                        model, variant, n_steps, args.batch,
                        eig_backend=args.eig_backend,
                        coupling=args.coupling * float(model.meta["spread"]),
                        graded=grid != "uniform",
                        order=2 if grid == "graded2" else 1,
                    )
                    err = float(np.abs(psis - exact).max())
                    key = (variant, grid)
                    errors[key].append(err)
                    prev = errors[key][-2] if len(errors[key]) > 1 else None
                    order = np.log2(prev / err) if prev and err > 0 else float("nan")
                    label = variant if grid == "uniform" else f"{variant}/{grid}"
                    print(f"{label:>16}{n_steps:>9}{seconds:>10.2f}{err:>13.3e}"
                          f"{order:>8.2f}", flush=True)
                    rows.append(result_row(
                        f"{regime}:{name}:{variant}:{grid}", n_steps=n_steps,
                        seconds=seconds, max_error=err, order=float(order),
                        regime=regime, model=name, variant=variant, grid=grid,
                        delta_dt=model.delta_dt(n_steps), commuting=bool(commuting),
                    ))

            if regime == "clean":
                # The formal orders are only *provable* on `scalar`, where the
                # propagation reduces to a scalar quadrature of `F` -- midpoint
                # second order, left-endpoint first, exactly. On other models
                # left-endpoint can come out second order (its leading term is
                # `(dt/2)(H(T) - H(0))`, which some geometries suppress), so
                # asserting `1` there would be a false alarm rather than a check.
                provable = name == "scalar"
                verdict = []
                for variant, expected in (("mid", 2.0), ("left", 1.0), ("magnus", None)):
                    key = (variant, "uniform")
                    if key not in errors or len(errors[key]) < 2:
                        continue
                    obs = np.log2(errors[key][0] / errors[key][-1]) / (
                        len(errors[key]) - 1
                    )
                    if expected is None or not provable:
                        tag = ""
                    elif abs(obs - expected) < 0.3:
                        tag = "  OK"
                    else:
                        tag = f"  EXPECTED {expected}"
                    verdict.append(f"{variant}={obs:.2f}{tag}")
                label = "gate" if provable else "observed (no formal order provable)"
                print(f"    {label}: " + ", ".join(verdict))

    print()
    print("clean regime is a correctness gate: mid must be order 2, left order 1.")
    print("realistic regime is a ranking only -- compare error at matched wall")
    print("clock. No order exists there; see Priority E in IMPROVEMENTS.md.")

    if args.output or args.csv is not None:
        payload = make_payload(
            benchmark="analytic",
            config={
                "regimes": regimes, "models": args.models, "variants": args.variants,
                "n_steps": args.n_steps, "batch": args.batch, "n": args.n,
                "coupling_ratio": args.coupling, "grids": args.grids,
                "profile": args.profile,
                "eig_backend": args.eig_backend,
            },
            results=rows,
            include_gpu_env=False,
        )
        write_results(payload, output=args.output,
                      csv_path=resolve_csv_path(args.output, args.csv, "analytic"))


if __name__ == "__main__":
    main()
