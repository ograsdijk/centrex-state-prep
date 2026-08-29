"""Where does time go inside the shared-slow scan loop?

Threading the batch loop plateaued at about 2.6x on 8 threads, which by Amdahl
implies roughly 29% of runtime is serial - far above the ~7% intercept implied
by fitting runtime against batch size. That fit measured the intercept
indirectly and at a different BLAS setting, so this instruments the loop
directly instead.

The loop splits into work that is shared across the batch (done once per
timestep, and therefore serial with respect to batch parallelism) and work that
is per scan point (parallelisable):

  shared      H_slow_t evaluation, the slow eigensolve, reorder_evecs,
              rotating the microwave couplings, and the psis round trip
  per point   assembling H_rot, the rotating-frame eigensolve, propagation

Process parallelism (loky) repeats the shared part in every worker, so it pays
CPU rather than wall time for it and is not bounded by this fraction. Thread
parallelism is.

    python benchmarks/bench_serial_fraction.py --n-steps 2000 --batch 25
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
from scipy.linalg.lapack import zheevd
from tqdm import tqdm

from common import (
    build_spa2_setup,
    make_payload,
    resolve_csv_path,
    result_row,
    write_results,
)
from state_prep import Simulator, scan_grid
from state_prep.simulator import LabelGapTracker, _sample_offset
from state_prep.utils import find_max_overlap_idx, reorder_evecs

SHIPPED = Simulator._time_evolve_mu_batched_shared_slow

SHARED_SECTIONS = (
    "h_slow_eval",
    "slow_eigensolve",
    "reorder_evecs",
    "microwave_rotate",
    "psis_rotate_in",
    "psis_rotate_out",
)
PER_POINT_SECTIONS = ("batch_loop",)


def make_instrumented(timings: dict[str, float]):
    """A copy of the shipped loop with per-section timers.

    Kept as a separate implementation rather than instrumenting the shipped code,
    so the production path carries no timing overhead.
    """

    def impl(
        self, *, H_slow_t, muw_hams, coupling_scales, D_mu_diag_batch, t_array,
        monitor_states, store_final_probabilities, store_final_monitor_probabilities,
        progress, eig_backend, time_sampling="mid", propagator="frozen",
    ):
        # Refuse rather than silently ignore. This loop implements only the
        # frozen path; accepting `magnus` here would report the frozen split
        # under the Magnus label, and a benchmark-only copy that quietly drops
        # what the shipped loop does is exactly how the LabelGapTracker defect
        # got in.
        if propagator != "frozen":
            raise ValueError(
                f"bench_serial_fraction instruments the frozen propagator only; "
                f"got {propagator!r}"
            )
        sample_offset = _sample_offset(time_sampling)
        batch = int(D_mu_diag_batch.shape[0])
        coupling_scales = np.asarray(coupling_scales)
        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        diag_idx = slice(None, None, n + 1)

        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref = V_ref[:, np.argsort(E_ref)]
        V_ref_ini = V_ref

        monitor_idx = None
        if monitor_states:
            monitor_idx = np.array(
                [find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                 for s in monitor_states], dtype=int,
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
        # The shipped loop updates this every timestep, in the shared section.
        # Leaving it out understates the very fraction this benchmark measures.
        gap_tracker = LabelGapTracker(len(self.hamiltonian.QN))
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            dt = t_array[i + 1] - t_array[i]
            t_sample = t + sample_offset * dt

            t0 = time.perf_counter()
            H_slow_i = H_slow_t(t_sample)
            t1 = time.perf_counter()
            D, V = eig(H_slow_i)
            t2 = time.perf_counter()
            Es, evecs = reorder_evecs(V, D, V_ref)
            last_evecs = evecs
            gap_tracker.update(Es, t_sample)
            t3 = time.perf_counter()

            Vh = V.conj().T
            H_mu_rot = [Vh @ H_mu_t(t_sample) @ V for H_mu_t in muw_hams]
            t4 = time.perf_counter()

            psis_slow = psis_batch @ V.conj()
            t5 = time.perf_counter()

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
            t6 = time.perf_counter()

            psis_batch = psis_slow @ V.T
            t7 = time.perf_counter()

            timings["h_slow_eval"] += t1 - t0
            timings["slow_eigensolve"] += t2 - t1
            timings["reorder_evecs"] += t3 - t2
            timings["microwave_rotate"] += t4 - t3
            timings["psis_rotate_in"] += t5 - t4
            timings["batch_loop"] += t6 - t5
            timings["psis_rotate_out"] += t7 - t6

            V_ref = evecs

        probs = None
        if store_final_probabilities:
            probs = np.abs(psis_batch @ last_evecs.conj()) ** 2
        mon = None
        if store_final_monitor_probabilities and monitor_idx is not None:
            mon = np.abs(psis_batch @ last_evecs[:, monitor_idx].conj()) ** 2
        return psis_batch, probs, mon, V_ref_ini, V_ref

    return impl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--eig-backend", default="zheevd", choices=["zheevd", "numpy"])
    parser.add_argument("--output")
    parser.add_argument("--csv", nargs="?", const="", default=None)
    args = parser.parse_args()

    print("building setup ...", flush=True)
    setup = build_spa2_setup()
    simulator = Simulator(
        setup["trajectory"], setup["electric_field"], setup["magnetic_field"],
        setup["initial_states"], setup["hamiltonian"], setup["microwave_fields"],
    )
    det, pref = scan_grid(
        n_fields=2, detunings_hz=np.linspace(-2e6, 1e6, args.batch)
    )

    timings: dict[str, float] = defaultdict(float)
    Simulator._time_evolve_mu_batched_shared_slow = make_instrumented(timings)
    try:
        start = time.perf_counter()
        simulator.run_microwave_scan(
            detunings_hz=det,
            intensity_prefactors=pref,
            N_steps=args.n_steps,
            monitor_states=setup["monitor_states"],
            store_final_probabilities=True,
            store_final_monitor_probabilities=True,
            progress=False,
            eig_backend=args.eig_backend,
        )
        total = time.perf_counter() - start
    finally:
        Simulator._time_evolve_mu_batched_shared_slow = SHIPPED

    measured = sum(timings.values())
    shared = sum(timings[k] for k in SHARED_SECTIONS)
    per_point = sum(timings[k] for k in PER_POINT_SECTIONS)

    print()
    print(f"N_steps={args.n_steps}, batch={args.batch}, backend={args.eig_backend}")
    print(f"{'section':<20}{'time':>10}{'% measured':>13}{'kind':>12}")
    rows = []
    for name in SHARED_SECTIONS + PER_POINT_SECTIONS:
        kind = "shared" if name in SHARED_SECTIONS else "per point"
        share = timings[name] / measured if measured else float("nan")
        print(f"{name:<20}{timings[name]:>9.3f}s{share:>12.1%}{kind:>12}")
        rows.append(
            result_row("section", section=name, time_s=timings[name],
                       fraction=share, kind=kind)
        )

    serial_fraction = shared / measured if measured else float("nan")
    print()
    print(f"{'shared (serial)':<20}{shared:>9.3f}s{serial_fraction:>12.1%}")
    print(f"{'per scan point':<20}{per_point:>9.3f}s{per_point / measured:>12.1%}")
    print(f"{'instrumented total':<20}{measured:>9.3f}s")
    print(f"{'wall clock':<20}{total:>9.3f}s   "
          f"(unaccounted {1 - measured / total:.1%}: setup and result assembly)")

    print()
    for threads in (2, 4, 8, 16):
        ceiling = 1.0 / (serial_fraction + (1 - serial_fraction) / threads)
        print(f"  Amdahl ceiling for thread parallelism at {threads:>2} threads: {ceiling:.2f}x")
    print()
    print("Process parallelism is not bounded by this: each worker repeats the")
    print("shared part concurrently, paying CPU rather than wall time for it.")

    rows.append(
        result_row("summary", serial_fraction=serial_fraction,
                   shared_s=shared, per_point_s=per_point,
                   instrumented_total_s=measured, wall_s=total)
    )
    payload = make_payload(
        benchmark="serial_fraction",
        config={"n_steps": args.n_steps, "batch": args.batch,
                "eig_backend": args.eig_backend},
        results=rows,
        include_gpu_env=False,
    )
    write_results(
        payload,
        output=args.output,
        csv_path=resolve_csv_path(args.output, args.csv, "serial_fraction"),
    )


if __name__ == "__main__":
    main()
