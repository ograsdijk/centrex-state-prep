"""Check 3: is a Strang split-step propagator numerically viable here?

Decides Performance Priority A before any of it is written. Rather than bounding
the commutator, this directly builds the real rotating-frame Hamiltonian at real
parameters and compares the exact one-step propagator against the split one.

H_rot = H_mu_rot(t) + diag(D + D_mu)
        ^ small, dense   ^ large, diagonal

Strang:  U ~ exp(-i*diag*dt/2) @ exp(-i*H_mu_rot*dt) @ exp(-i*diag*dt/2)
"""

from __future__ import annotations

import numpy as np

from common import build_spa2_setup  # noqa: E402

import argparse

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument("--samples", type=int, default=9, help="trajectory sample points")
_parser.add_argument("--n-steps", type=int, nargs="+", default=[10000, 20000, 40000])
_args = _parser.parse_args()

N_SAMPLES = _args.samples

print("building SPA2 setup ...", flush=True)
setup = build_spa2_setup()
trajectory = setup["trajectory"]
hamiltonian = setup["hamiltonian"]
microwave_fields = setup["microwave_fields"]
QN = hamiltonian.QN
n = len(QN)

H_t = hamiltonian.get_H_t_func()
muw_hams = [mw.get_H_t_func(trajectory.R_t, QN) for mw in microwave_fields]

# Rotating-frame diagonal shift, mirroring simulator.py with zero detuning.
D_mu_diag = np.zeros(n, dtype=float)
unique_omegas: list[float] = []
for mw in microwave_fields:
    omega = 2 * np.pi * mw.muW_freq
    if not any(np.isclose(omega, om) for om in unique_omegas):
        unique_omegas.append(omega)
        mw.generate_D(QN, omega=float(np.sum(unique_omegas)))
        D_mu_diag += np.diag(mw.D)

T = trajectory.get_T()


def herm_expm(H: np.ndarray, dt: float) -> np.ndarray:
    w, W = np.linalg.eigh(H)
    return (W * np.exp(-1j * w * dt)[None, :]) @ W.conj().T


def analyse(n_steps: int) -> dict:
    dt = T / n_steps
    ts = np.linspace(0, T, N_SAMPLES + 2)[1:-1]

    rows = []
    for t in ts:
        H_slow = H_t(t)
        D, V = np.linalg.eigh(H_slow)
        Vh = V.conj().T

        H_mu_rot = np.zeros((n, n), dtype=complex)
        for H_mu_t in muw_hams:
            H_mu_rot += Vh @ H_mu_t(t) @ V

        delta = D + D_mu_diag
        H_rot = H_mu_rot + np.diag(delta)

        U_exact = herm_expm(H_rot, dt)

        phase = np.exp(-0.5j * delta * dt)
        U_mid = herm_expm(H_mu_rot, dt)
        U_strang = phase[:, None] * U_mid * phase[None, :]

        phase_full = np.exp(-1j * delta * dt)
        U_lie = phase_full[:, None] * U_mid

        rows.append(
            {
                "t": t,
                "delta_spread": float(delta.max() - delta.min()),
                "norm_H_mu": float(np.linalg.norm(H_mu_rot, 2)),
                "err_strang": float(np.linalg.norm(U_exact - U_strang, 2)),
                "err_lie": float(np.linalg.norm(U_exact - U_lie, 2)),
            }
        )

    worst = max(r["err_strang"] for r in rows)
    worst_lie = max(r["err_lie"] for r in rows)
    return {
        "n_steps": n_steps,
        "dt": dt,
        "delta_spread": max(r["delta_spread"] for r in rows),
        "norm_H_mu": max(r["norm_H_mu"] for r in rows),
        "per_step_strang": worst,
        "per_step_lie": worst_lie,
        "accum_strang": worst * n_steps,
        "accum_lie": worst_lie * n_steps,
        "rows": rows,
    }


print(f"n = {n}, T = {T:.6e} s")
print()

out = {}
for n_steps in _args.n_steps:
    res = analyse(n_steps)
    out[n_steps] = res
    print(
        f"N_steps={n_steps:>6}  dt={res['dt']:.3e}s  "
        f"per-step Strang={res['per_step_strang']:.3e}  "
        f"x N = {res['accum_strang']:.3e}"
    )
    print(
        f"{'':>14}  {'':>16}  "
        f"per-step Lie   ={res['per_step_lie']:.3e}  "
        f"x N = {res['accum_lie']:.3e}"
    )

steps = sorted(out)
base = out[steps[0]]
print()
print(f"max |diag| spread      : {base['delta_spread']:.4e} rad/s")
print(f"max ||H_mu_rot||_2     : {base['norm_H_mu']:.4e} rad/s")
print(f"ratio (coupling/diag)  : {base['norm_H_mu'] / base['delta_spread']:.3e}")
print()
print("The coupling/diag ratio is not what governs the splitting error:")
print("delta * dt is, and it is enormous here.")
print(f"  max |delta| * dt at N_steps={steps[0]}: "
      f"{base['delta_spread'] * base['dt']:.3e} rad")

if len(steps) > 1:
    print()
    print("per-step error scaling when dt halves (expect ~8x for 3rd-order local error):")
    for lo, hi in zip(steps[:-1], steps[1:]):
        r = out[lo]["per_step_strang"] / out[hi]["per_step_strang"]
        print(f"  {lo} -> {hi} : {r:.2f}x")

print()
print(f"accumulated Strang error at N_steps={steps[0]}: {base['accum_strang']:.3e}")
print("For scale, a unitary propagator difference is bounded by 2, and the")
print("backend-agreement floor for this code is of order 1e-7 to 1e-6.")
if base["accum_strang"] < 1e-6:
    print("VERDICT: splitting error is at the existing numerical floor. Viable.")
elif base["accum_strang"] < 1e-3:
    print("VERDICT: above the noise floor but possibly tolerable. Judgement call.")
else:
    print("VERDICT: far too large at this dt. A Strang split is not usable here.")
