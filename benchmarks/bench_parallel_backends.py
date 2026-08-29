"""Does restructuring the inner loop make threading beat loky?

Hypothesis: the current threaded variant loses to loky because it makes ~10
small NumPy calls per scan point. Each has Python-level dispatch that holds the
GIL, so ~1M GIL transitions per scan serialise the threads.

Fix: assemble all rotating-frame Hamiltonians vectorised, then have each thread
call np.linalg.eigh once on its own (chunk, n, n) sub-stack. NumPy's gufunc
loops the batch dimension in C with the GIL released for the whole call, cutting
GIL transitions by ~60x.

Two variants, differing in where the stack is materialised:
  batched : full (B,n,n) assembled once, threads take slices of it
  tiled   : each thread assembles only its own sub-stack (better locality)

Locality matters because the full stack is 1.6 MiB at n=64/B=25, past L2, while
a tile is ~200 KiB and L2-resident. The two effects (fewer GIL transitions vs
worse locality) oppose each other, which is why this needs measuring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scipy.linalg.lapack import zheevd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import build_spa2_setup, make_detunings  # noqa: E402
from paired import compare_paired, format_report  # noqa: E402

from state_prep.simulator import Simulator  # noqa: E402
from state_prep.simulator import _sample_offset
from state_prep.utils import find_max_overlap_idx, reorder_evecs  # noqa: E402

SHIPPED = Simulator._time_evolve_mu_batched_shared_slow

import argparse

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument("--n-steps", type=int, default=4000)
_parser.add_argument("--batch", type=int, default=25)
_parser.add_argument("--repeat", type=int, default=4)
_parser.add_argument("--warmup", type=int, default=1)
_parser.add_argument("--workers", type=int, default=8, help="loky worker count")
_parser.add_argument("--threads", type=int, default=8, help="thread count for the threaded variants")
_args = _parser.parse_args()

N_STEPS = _args.n_steps
BATCH = _args.batch
REPEAT = _args.repeat


def _preamble(self, H_slow_t, t_array, D_mu_diag_batch, monitor_states):
    batch = int(D_mu_diag_batch.shape[0])
    H_tini = H_slow_t(t_array[0])
    n = int(H_tini.shape[0])
    E_ref, V_ref = np.linalg.eigh(H_tini)
    V_ref = V_ref[:, np.argsort(E_ref)]
    monitor_idx = None
    if monitor_states:
        monitor_idx = np.array(
            [find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref)
             for s in monitor_states], dtype=int,
        )
    self.init_state_vecs(H_tini, V_0=V_ref)
    psis_batch = np.repeat(self.psis[None, :, :], batch, axis=0)
    return batch, n, V_ref, monitor_idx, psis_batch


def _finish(psis_batch, last_evecs, monitor_idx, store_probs, store_monitor):
    probs = np.abs(psis_batch @ last_evecs.conj()) ** 2 if store_probs else None
    mon = None
    if store_monitor and monitor_idx is not None:
        mon = np.abs(psis_batch @ last_evecs[:, monitor_idx].conj()) ** 2
    return probs, mon


def make_variant(n_threads: int, mode: str):
    """mode: 'naive' (per-point loop), 'batched' (full stack), 'tiled' (per-chunk stack)."""

    def impl(
        self, *, H_slow_t, muw_hams, coupling_scales, D_mu_diag_batch, t_array,
        monitor_states, store_final_probabilities, store_final_monitor_probabilities,
        progress, eig_backend, time_sampling="mid", propagator="frozen",
    ):
        # This loop implements the frozen propagator only. Refusing beats
        # silently ignoring: reporting a frozen result under the Magnus label is
        # how a benchmark-only copy misleads.
        if propagator != "frozen":
            raise ValueError(
                f"bench_parallel_backends instruments the frozen propagator only; got {propagator!r}"
            )
        coupling_scales = np.asarray(coupling_scales)
        batch, n, V_ref, monitor_idx, psis_batch = _preamble(
            self, H_slow_t, t_array, D_mu_diag_batch, monitor_states
        )
        V_ref_ini = V_ref
        diag_idx = slice(None, None, n + 1)
        didx = np.arange(n)

        def eig(M):
            if eig_backend == "zheevd":
                w, v, info = zheevd(M)
                if info != 0:
                    w, v = np.linalg.eigh(M)
                return w, v
            return np.linalg.eigh(M)

        bounds = np.linspace(0, batch, n_threads + 1).astype(int)
        chunks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

        last_evecs = V_ref
        sample_offset = _sample_offset(time_sampling)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
                dt = t_array[i + 1] - t_array[i]
                t_sample = t + sample_offset * dt

                D, V = eig(H_slow_t(t_sample))
                _, evecs = reorder_evecs(V, D, V_ref)
                last_evecs = evecs

                Vh = V.conj().T
                H_mu_rot = [Vh @ H_mu_t(t_sample) @ V for H_mu_t in muw_hams]
                psis_slow = psis_batch @ V.conj()

                shift = D[None, :] + D_mu_diag_batch  # (B,n)

                if mode == "naive":
                    def work(lo_hi):
                        lo, hi = lo_hi
                        for b in range(lo, hi):
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
                else:
                    H_mu_stack = np.stack(H_mu_rot)  # (M,n,n)

                    if mode == "batched":
                        # Materialise the whole (B,n,n) stack once.
                        H_all = np.einsum("bj,jnm->bnm", coupling_scales, H_mu_stack)
                        H_all[:, didx, didx] += shift

                    def work(lo_hi):
                        lo, hi = lo_hi
                        if mode == "batched":
                            H_c = H_all[lo:hi]
                        else:
                            # Assemble only this thread's tile: ~200 KiB, L2-resident.
                            H_c = np.einsum(
                                "bj,jnm->bnm", coupling_scales[lo:hi], H_mu_stack
                            )
                            H_c[:, didx, didx] += shift[lo:hi]

                        # One GIL-releasing gufunc call for the whole sub-stack.
                        w, v = np.linalg.eigh(H_c)
                        ph = np.exp(-1j * w * dt)  # (c,n)
                        tmp = psis_slow[lo:hi] @ v.conj()  # (c,S,n)
                        tmp *= ph[:, None, :]
                        psis_slow[lo:hi] = tmp @ v.transpose(0, 2, 1)

                if n_threads == 1:
                    work(chunks[0])
                else:
                    list(pool.map(work, chunks))

                psis_batch = psis_slow @ V.T
                V_ref = evecs

        probs, mon = _finish(
            psis_batch, last_evecs, monitor_idx,
            store_final_probabilities, store_final_monitor_probabilities,
        )
        return psis_batch, probs, mon, V_ref_ini, V_ref

    return impl


print("building setup ...", flush=True)
setup = build_spa2_setup()
sim = Simulator(
    setup["trajectory"], setup["electric_field"], setup["magnetic_field"],
    setup["initial_states"], setup["hamiltonian"], setup["microwave_fields"],
)
d = make_detunings(BATCH)
det = np.column_stack([d, d])
pref = np.ones((BATCH, 2))


def scan(*, impl, backend, workers):
    def run():
        Simulator._time_evolve_mu_batched_shared_slow = impl
        try:
            return sim.run_microwave_scan(
                detunings_hz=det, intensity_prefactors=pref, N_steps=N_STEPS,
                monitor_states=setup["monitor_states"],
                store_final_probabilities=True,
                store_final_monitor_probabilities=True,
                progress=False, eig_backend=backend, workers=workers,
            )
        finally:
            Simulator._time_evolve_mu_batched_shared_slow = SHIPPED
    return run


def max_diff(a, b):
    return float(np.max(np.abs(a.probabilities_final - b.probabilities_final)))


variants = {
    "serial_zheevd": scan(impl=SHIPPED, backend="zheevd", workers=1),
    "loky_zheevd": scan(impl=SHIPPED, backend="zheevd", workers=_args.workers),
    "thread_naive": scan(impl=make_variant(_args.threads, "naive"), backend="numpy", workers=1),
    "thread_batched": scan(impl=make_variant(_args.threads, "batched"), backend="numpy", workers=1),
    "thread_tiled": scan(impl=make_variant(_args.threads, "tiled"), backend="numpy", workers=1),
}

print(f"N_steps={N_STEPS}, batch={BATCH}, repeat={REPEAT}, loky workers={_args.workers}, threads={_args.threads}", flush=True)
result = compare_paired(
    variants, repeat=REPEAT, warmup=_args.warmup, check=max_diff, tolerance=1e-5, progress=True,
)
print(format_report(result))
