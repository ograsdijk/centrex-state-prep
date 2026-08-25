"""Check 3: is a Strang split-step propagator numerically viable here?

Decides Performance Priority A before any of it is written. Rather than bounding
the commutator, this directly builds the real rotating-frame Hamiltonian at real
parameters and compares the exact one-step propagator against the split one.

H_rot = H_mu_rot(t) + diag(D + D_mu)
        ^ small, dense   ^ large, diagonal

Strang:  U ~ exp(-i*diag*dt/2) @ exp(-i*H_mu_rot*dt) @ exp(-i*diag*dt/2)

The Strang and Lie variants were measured and rejected (IMPROVEMENTS.md
Priority A): delta*dt is about 7.65e+04 rad per step, so splitting the large
diagonal fails however small the coupling is.

This also checks the surviving variant identified there, which does not split
the diagonal at all. Move to the interaction picture with respect to `delta`,
where the propagator over one step is

    U = exp(-i*delta*dt) @ expm(-i * (H_mu_rot . M))

with `.` the elementwise (Hadamard) product and M the analytic oscillatory
integral

    M_ij = integral_0^dt exp(i*(delta_i - delta_j)*s) ds
         = (exp(i*(delta_i - delta_j)*dt) - 1) / (i*(delta_i - delta_j))

taken as `dt` when delta_i == delta_j. That is the first Magnus term; the
question this script answers is whether the neglected second term is small
enough. If it is, the per-scan-point n^3 eigensolve can be replaced by
n^2-scale work, which is the only remaining large lever on runtime.
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


def magnus_integral(delta: np.ndarray, dt: float) -> np.ndarray:
    """Elementwise integral of exp(i*(delta_i - delta_j)*s) over s in [0, dt].

    The small-argument branch is not optional: `delta` is nearly degenerate
    within a rotational manifold, so the naive quotient divides by ~0 there and
    the script would measure the guard rather than the physics.
    """
    x = delta[:, None] - delta[None, :]
    xdt = x * dt
    out = np.empty(x.shape, dtype=complex)
    small = np.abs(xdt) < 1e-8
    # expm1 keeps the difference accurate when the exponent is small but the
    # branch below has not yet taken over.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.expm1(1j * xdt) / (1j * x)
    out[small] = dt * (1.0 + 0.5j * xdt[small])
    return out


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

        # Interaction picture with respect to the diagonal, first Magnus term.
        A = H_mu_rot * magnus_integral(delta, dt)
        U_magnus = phase_full[:, None] * herm_expm(A, 1.0)
        norm_A = float(np.linalg.norm(A, 2))

        rows.append(
            {
                "t": t,
                "delta_spread": float(delta.max() - delta.min()),
                "norm_H_mu": float(np.linalg.norm(H_mu_rot, 2)),
                "err_strang": float(np.linalg.norm(U_exact - U_strang, 2)),
                "err_lie": float(np.linalg.norm(U_exact - U_lie, 2)),
                "err_magnus": float(np.linalg.norm(U_exact - U_magnus, 2)),
                "norm_magnus_arg": norm_A,
            }
        )

    worst = max(r["err_strang"] for r in rows)
    worst_lie = max(r["err_lie"] for r in rows)
    worst_magnus = max(r["err_magnus"] for r in rows)
    return {
        "n_steps": n_steps,
        "dt": dt,
        "delta_spread": max(r["delta_spread"] for r in rows),
        "norm_H_mu": max(r["norm_H_mu"] for r in rows),
        "per_step_strang": worst,
        "per_step_lie": worst_lie,
        "accum_strang": worst * n_steps,
        "accum_lie": worst_lie * n_steps,
        "per_step_magnus": worst_magnus,
        "accum_magnus": worst_magnus * n_steps,
        "norm_magnus_arg": max(r["norm_magnus_arg"] for r in rows),
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
    print(
        f"{'':>14}  {'':>16}  "
        f"per-step Magnus={res['per_step_magnus']:.3e}  "
        f"x N = {res['accum_magnus']:.3e}"
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

print()
print("--- interaction picture, first Magnus term ---")
print(f"max ||H_mu_rot . M||_2 at N_steps={steps[0]}: "
      f"{base['norm_magnus_arg']:.4e} rad")
print("  This is the argument that actually gets exponentiated. Unlike Strang,")
print("  it never contains the large diagonal, so it stays small as dt shrinks.")
if len(steps) > 1:
    print()
    print("per-step Magnus error scaling when dt halves "
          "(expect ~4x if the leading neglected term is the second Magnus term):")
    for lo, hi in zip(steps[:-1], steps[1:]):
        lo_e = out[lo]["per_step_magnus"]
        hi_e = out[hi]["per_step_magnus"]
        ratio = lo_e / hi_e if hi_e > 0 else float("inf")
        print(f"  {lo} -> {hi} : {ratio:.2f}x")
print()
print(f"accumulated Magnus error at N_steps={steps[0]}: "
      f"{base['accum_magnus']:.3e}")
if base["accum_magnus"] < 1e-6:
    print("VERDICT: at or below the 1e-6 acceptance target. The per-scan-point")
    print("  eigensolve can be replaced by n^2-scale work. Proceed.")
elif base["accum_magnus"] < 1e-3:
    print("VERDICT: above the 1e-6 target but not catastrophic. A second Magnus")
    print("  term, or a smaller dt, would need measuring before this is usable.")
else:
    print("VERDICT: too large. The first Magnus term alone is not sufficient.")
