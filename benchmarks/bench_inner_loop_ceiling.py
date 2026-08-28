"""What could a serial native port of the inner loop possibly win?

`IMPROVEMENTS.md` records that the per-scan-point inner loop is 93-96% of scan
runtime at production batch sizes, and that roughly 75% of the inner loop is the
LAPACK eigensolve. A Rust port would still call LAPACK for that eigensolve, so
the ceiling of a *serial* port is the remaining, Python-side share alone.

That 75% came from a profile at a different shape, so it is measured here
directly rather than reused. The batch-loop body is split four ways:

  full           the shipped loop body, exactly as
                 `_time_evolve_mu_batched_shared_slow` runs it
  eig_only       only the rotating-frame eigensolves, on the same matrices
  no_eig         the loop body with the eigensolve replaced by a cached result
  assemble_only  only the H_rot assembly

`1 - eig_only/full` is the Python-side share, and `no_eig/full` is an
independent estimate of the same quantity; both are reported so a disagreement
is visible rather than hidden. The serial-port ceiling is `full / eig_only`.

Inputs are captured from a real timestep of a real SPA2 scan rather than
synthesised, so the matrices carry production structure and conditioning.

    python benchmarks/bench_inner_loop_ceiling.py --js 0,1,2,3 --batch 25
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
from state_prep.simulator import limit_blas_threads
from state_prep.utils import find_max_overlap_idx, reorder_evecs

SHIPPED = Simulator._time_evolve_mu_batched_shared_slow


class _Captured(Exception):
    """Sentinel: inner-loop inputs captured, abandon the run."""


def make_capturing(store: dict, capture_step: int):
    """Run the shipped loop until `capture_step`, then snapshot its inputs.

    A copy of the shipped loop rather than instrumentation of it, for the same
    reason `bench_serial_fraction.py` keeps one: the production path must carry
    no benchmark overhead.
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
        batch = int(D_mu_diag_batch.shape[0])
        coupling_scales = np.asarray(coupling_scales)
        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        diag_idx = slice(None, None, n + 1)

        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref = V_ref[:, np.argsort(E_ref)]

        if monitor_states:
            [
                find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref)
                for s in monitor_states
            ]

        self.init_state_vecs(H_tini, V_0=V_ref)
        psis_batch = np.repeat(self.psis[None, :, :], batch, axis=0)

        def eig(M):
            if eig_backend == "zheevd":
                w, v, info = zheevd(M)
                if info != 0:
                    w, v = np.linalg.eigh(M)
                return w, v
            return np.linalg.eigh(M)

        for i, t in enumerate(t_array[:-1]):
            dt = t_array[i + 1] - t_array[i]
            H_slow_i = H_slow_t(t)
            D, V = eig(H_slow_i)
            _, evecs = reorder_evecs(V, D, V_ref)
            Vh = V.conj().T
            H_mu_rot = [Vh @ H_mu_t(t) @ V for H_mu_t in muw_hams]
            psis_slow = psis_batch @ V.conj()

            if i == capture_step:
                store.update(
                    H_mu_rot=[np.array(M, copy=True) for M in H_mu_rot],
                    D=np.array(D, copy=True),
                    D_mu_diag_batch=np.array(D_mu_diag_batch, copy=True),
                    coupling_scales=np.array(coupling_scales, copy=True),
                    psis_slow=np.array(psis_slow, copy=True),
                    dt=float(dt),
                    n=n,
                    batch=batch,
                    diag_idx=diag_idx,
                )
                raise _Captured

            for b in range(batch):
                H_rot = (coupling_scales[b, 0] * H_mu_rot[0]).copy()
                for j in range(1, len(H_mu_rot)):
                    H_rot += coupling_scales[b, j] * H_mu_rot[j]
                H_rot.flat[diag_idx] += D
                H_rot.flat[diag_idx] += D_mu_diag_batch[b]
                D_rot, V_rot = eig(H_rot)
                ph = np.exp(-1j * D_rot * dt)
                tmp = psis_slow[b] @ V_rot.conj()
                tmp *= ph[np.newaxis, :]
                psis_slow[b] = tmp @ V_rot.T

            psis_batch = psis_slow @ V.T
            V_ref = evecs

        raise AssertionError("capture_step is past the end of the trajectory")

    return impl


def capture_inputs(*, js, batch, n_steps, capture_step, eig_backend):
    setup = build_spa2_setup(js=js)
    simulator = Simulator(
        setup["trajectory"],
        setup["electric_field"],
        setup["magnetic_field"],
        setup["initial_states"],
        setup["hamiltonian"],
        setup["microwave_fields"],
    )
    det, pref = scan_grid(n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, batch))

    store: dict = {}
    Simulator._time_evolve_mu_batched_shared_slow = make_capturing(store, capture_step)
    try:
        simulator.run_microwave_scan(
            detunings_hz=det,
            intensity_prefactors=pref,
            N_steps=n_steps,
            monitor_states=setup["monitor_states"],
            store_final_probabilities=True,
            store_final_monitor_probabilities=True,
            progress=False,
            eig_backend=eig_backend,
        )
    except _Captured:
        pass
    finally:
        Simulator._time_evolve_mu_batched_shared_slow = SHIPPED
    if not store:
        raise RuntimeError("capture failed: the loop never reached capture_step")
    return store


def make_eig(eig_backend):
    if eig_backend == "zheevd":

        def eig(M):
            w, v, info = zheevd(M)
            if info != 0:
                w, v = np.linalg.eigh(M)
            return w, v

    else:

        def eig(M):
            return np.linalg.eigh(M)

    return eig


def assemble(inputs, b):
    """The H_rot assembly, exactly as the shipped loop performs it."""
    H_mu_rot = inputs["H_mu_rot"]
    cs = inputs["coupling_scales"]
    H_rot = (cs[b, 0] * H_mu_rot[0]).copy()
    for j in range(1, len(H_mu_rot)):
        H_rot += cs[b, j] * H_mu_rot[j]
    H_rot.flat[inputs["diag_idx"]] += inputs["D"]
    H_rot.flat[inputs["diag_idx"]] += inputs["D_mu_diag_batch"][b]
    return H_rot


def dump_binary(path: str, inputs: dict, reference: np.ndarray) -> None:
    """Flat little-endian dump, so a Rust binary can read it without a numpy crate.

    Layout, in order, no padding:

        i64   n, batch, S, M
        f64   dt
        c128  H_mu_rot        (M, n, n)   row-major
        f64   D               (n,)
        f64   D_mu_diag_batch (batch, n)
        f64   coupling_scales (batch, M)
        c128  psis_slow       (batch, S, n) row-major
        c128  psis_out        (batch, S, n) row-major, one loop body applied

    `psis_out` is the reference the Rust spike must reproduce. Both sides call
    the same LAPACK `evd` driver, so agreement should be bitwise or very near
    it; anything larger means the Rust binding is wrong, not that the numerics
    drifted.
    """
    H_mu_rot = np.ascontiguousarray(np.stack(inputs["H_mu_rot"]), dtype=np.complex128)
    psis = np.ascontiguousarray(inputs["psis_slow"], dtype=np.complex128)
    m = H_mu_rot.shape[0]
    n = inputs["n"]
    batch = inputs["batch"]
    s_states = psis.shape[1]
    with open(path, "wb") as f:
        np.array([n, batch, s_states, m], dtype="<i8").tofile(f)
        np.array([inputs["dt"]], dtype="<f8").tofile(f)
        H_mu_rot.astype("<c16").tofile(f)
        np.ascontiguousarray(inputs["D"], dtype="<f8").tofile(f)
        np.ascontiguousarray(inputs["D_mu_diag_batch"], dtype="<f8").tofile(f)
        np.ascontiguousarray(inputs["coupling_scales"], dtype="<f8").tofile(f)
        psis.astype("<c16").tofile(f)
        np.ascontiguousarray(reference, dtype="<c16").astype("<c16").tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--js", default="0,1,2,3", help="comma separated J values")
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--capture-step", type=int, default=100)
    parser.add_argument(
        "--inner-reps",
        type=int,
        default=20,
        help="batch-loop repetitions per timed call",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--eig-backend", default="zheevd", choices=["zheevd", "numpy"])
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    parser.add_argument(
        "--dump-bin",
        help="write the captured inputs as a flat little-endian binary blob for "
        "the Rust spike in rust/inner_loop_spike, then exit. Layout is "
        "documented in dump_binary().",
    )
    parser.add_argument(
        "--dump-npz",
        help="write the captured inner-loop inputs here and exit. Lets the same "
        "loop body be replayed on an interpreter that cannot import "
        "centrex_tlf, which is what the free-threading check needs.",
    )
    args = parser.parse_args()

    js = parse_js(args.js)
    print(f"building setup js={js} ...", flush=True)
    inputs = capture_inputs(
        js=js,
        batch=args.batch,
        n_steps=args.n_steps,
        capture_step=args.capture_step,
        eig_backend=args.eig_backend,
    )
    n = inputs["n"]
    batch = inputs["batch"]
    dt = inputs["dt"]
    psis_ref = inputs["psis_slow"]
    print(f"captured inner-loop inputs: n={n}, batch={batch}, S={psis_ref.shape[1]}")

    if args.dump_bin:
        ref_eig = make_eig(args.eig_backend)
        reference = psis_ref.copy()
        for b in range(batch):
            H_rot = assemble(inputs, b)
            D_rot, V_rot = ref_eig(H_rot)
            ph = np.exp(-1j * D_rot * dt)
            tmp = reference[b] @ V_rot.conj()
            tmp *= ph[np.newaxis, :]
            reference[b] = tmp @ V_rot.T
        dump_binary(args.dump_bin, inputs, reference)
        print(f"wrote {args.dump_bin}")
        return

    if args.dump_npz:
        np.savez(
            args.dump_npz,
            H_mu_rot=np.stack(inputs["H_mu_rot"]),
            D=inputs["D"],
            D_mu_diag_batch=inputs["D_mu_diag_batch"],
            coupling_scales=inputs["coupling_scales"],
            psis_slow=psis_ref,
            dt=np.array(dt),
        )
        print(f"wrote {args.dump_npz}")
        return

    eig = make_eig(args.eig_backend)
    reps = args.inner_reps

    # Pre-assembled matrices and their eigendecompositions, so `eig_only` and
    # `no_eig` operate on exactly the matrices the full loop would produce.
    H_rots = [assemble(inputs, b) for b in range(batch)]
    cached = [eig(H.copy()) for H in H_rots]

    def full():
        for _ in range(reps):
            psis_slow = psis_ref.copy()
            for b in range(batch):
                H_rot = assemble(inputs, b)
                D_rot, V_rot = eig(H_rot)
                ph = np.exp(-1j * D_rot * dt)
                tmp = psis_slow[b] @ V_rot.conj()
                tmp *= ph[np.newaxis, :]
                psis_slow[b] = tmp @ V_rot.T
        return psis_slow

    def no_eig():
        for _ in range(reps):
            psis_slow = psis_ref.copy()
            for b in range(batch):
                assemble(inputs, b)
                D_rot, V_rot = cached[b]
                ph = np.exp(-1j * D_rot * dt)
                tmp = psis_slow[b] @ V_rot.conj()
                tmp *= ph[np.newaxis, :]
                psis_slow[b] = tmp @ V_rot.T
        return psis_slow

    def eig_only():
        out = None
        for _ in range(reps):
            for b in range(batch):
                out = eig(H_rots[b].copy())
        return out

    def assemble_only():
        out = None
        for _ in range(reps):
            for b in range(batch):
                out = assemble(inputs, b)
        return out

    variants = {
        "full": full,
        "eig_only": eig_only,
        "no_eig": no_eig,
        "assemble_only": assemble_only,
    }

    stats = {}
    with limit_blas_threads(args.blas_threads):
        for _ in range(args.warmup):
            for fn in variants.values():
                fn()
        # Interleave rounds rather than running each variant to completion, so
        # thermal drift is shared across variants instead of being attributed to
        # whichever ran last. Same reasoning as benchmarks/paired.py.
        samples = {name: [] for name in variants}
        for _ in range(args.repeat):
            for name, fn in variants.items():
                start = time.perf_counter()
                fn()
                samples[name].append(time.perf_counter() - start)
        for name, times in samples.items():
            stats[name] = {
                "time_mean_s": float(np.mean(times)),
                "time_std_s": float(np.std(times)),
                "time_best_s": float(np.min(times)),
            }

    per_call = reps * batch
    print()
    print(
        f"n={n}, batch={batch}, backend={args.eig_backend}, "
        f"blas_threads={args.blas_threads}, repeat={args.repeat}, "
        f"{per_call} loop bodies per timed call"
    )
    print(f"{'variant':<16}{'mean':>12}{'std':>11}{'us/body':>11}")
    rows = []
    for name in variants:
        s = stats[name]
        print(
            f"{name:<16}{s['time_mean_s']:>11.4f}s{s['time_std_s']:>10.4f}s"
            f"{1e6 * s['time_mean_s'] / per_call:>10.1f}"
        )
        rows.append(
            result_row(
                "variant", variant=name, n=n, batch=batch, per_call_bodies=per_call, **s
            )
        )

    full_m = stats["full"]["time_mean_s"]
    eig_m = stats["eig_only"]["time_mean_s"]
    noeig_m = stats["no_eig"]["time_mean_s"]
    asm_m = stats["assemble_only"]["time_mean_s"]

    lapack_share = eig_m / full_m
    python_share_diff = 1.0 - lapack_share
    python_share_direct = noeig_m / full_m
    ceiling = full_m / eig_m

    print()
    print(f"{'LAPACK share (eig_only/full)':<38}{lapack_share:>8.1%}")
    print(f"{'Python-side share (1 - above)':<38}{python_share_diff:>8.1%}")
    print(
        f"{'Python-side share (no_eig/full)':<38}{python_share_direct:>8.1%}"
        "   independent estimate"
    )
    print(f"{'  of which H_rot assembly':<38}{asm_m / full_m:>8.1%}")
    print()
    print(f"Serial native-port ceiling (full/eig_only): {ceiling:.3f}x")
    print("  i.e. the best a serial port could do if every non-LAPACK operation")
    print("  in the batch-loop body became free.")

    disagreement = abs(python_share_direct - python_share_diff)
    if disagreement > 0.05:
        print()
        print(
            f"NOTE: the two Python-side estimates differ by {disagreement:.1%}. "
            "`no_eig` reads cached eigenvectors from memory rather than "
            "producing them in cache, so it is the looser of the two; treat "
            "`1 - eig_only/full` as primary."
        )

    rows.append(
        result_row(
            "summary",
            n=n,
            batch=batch,
            lapack_share=lapack_share,
            python_share_by_difference=python_share_diff,
            python_share_direct=python_share_direct,
            assemble_share=asm_m / full_m,
            serial_port_ceiling=ceiling,
        )
    )

    if args.output or args.csv is not None:
        payload = make_payload(
            benchmark="inner_loop_ceiling",
            config={
                "js": js,
                "batch": batch,
                "n": n,
                "n_steps": args.n_steps,
                "inner_reps": reps,
                "repeat": args.repeat,
                "eig_backend": args.eig_backend,
                "blas_threads": args.blas_threads,
            },
            results=rows,
            include_gpu_env=False,
        )
        write_results(
            payload,
            output=args.output,
            csv_path=resolve_csv_path(args.output, args.csv, "inner_loop_ceiling"),
        )


if __name__ == "__main__":
    main()
