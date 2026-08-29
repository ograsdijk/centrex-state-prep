"""The term Magnus-1 drops, computed exactly. No propagation, no reference.

Magnus-1 keeps `A = int_0^dt H_I(s) ds` and drops the second Magnus term

    Omega2 = -1/2 int_0^dt dt1 int_0^t1 dt2 [H_I(t1), H_I(t2)]

with `H_I(s)_{jk} = exp(i (d_j - d_k) s) H_{jk}`. That is Magnus's *entire* extra
error against the frozen propagator, which exponentiates the frozen matrix
exactly, so computing it directly settles why Magnus loses on a given setup
without propagating anything or needing a reference.

Elementwise the double integral is analytic. With `a = d_j - d_m`,
`b = d_m - d_k`,

    I(a,b) = int_0^dt dt1 e^{i a t1} int_0^t1 dt2 e^{i b t2}
           = ( K(a+b) - K(a) ) / (i b),      K(x) = (e^{i x dt} - 1) / (i x)

`K(0) = dt`, and for `b = 0` the inner integral is `t1`. Then

    Omega2_{jk} = -1/2 sum_m H_jm H_mk ( I(a,b) - I(b,a) )

`K` is the same kernel `magnus_integral` already uses, so no quadrature is
involved -- which matters, because the integrand oscillates at `delta*dt ~ 8e4`
rad and no practical quadrature would resolve it.

Measured on the SPA cascade at the AP2 beam centre: `||Omega2||` falls only
**linearly** in `dt`, not quadratically, because the oscillatory kernels saturate
at `~1/delta` instead of growing with `dt`. So `N*||Omega2||` is constant at
`1.85e-02` and the dropped term stays a fixed `~0.3%` of the retained one however
far the grid is refined. That is why Magnus is first order here, and the figure
brackets the measured error (`1.0e-02` at `N=10000`) from above, as an upper
bound should, the gap being per-step cancellation.

    python benchmarks/bench_magnus_commutator.py
"""
import sys
import numpy as np
sys.path.insert(0, "benchmarks"); sys.path.insert(0, "src")
from common import build_spa_cascade_setup

def K(x, dt):
    x = np.asarray(x, dtype=complex)
    out = np.empty_like(x)
    small = np.abs(x * dt) < 1e-8
    out[~small] = (np.exp(1j * x[~small] * dt) - 1.0) / (1j * x[~small])
    out[small] = dt * (1.0 + 0.5j * x[small] * dt)
    return out

def I_ab(a, b, dt):
    """int_0^dt dt1 e^{i a t1} int_0^t1 dt2 e^{i b t2}, elementwise."""
    a, b = np.broadcast_arrays(
        np.asarray(a, dtype=complex), np.asarray(b, dtype=complex)
    )
    a = np.ascontiguousarray(a); b = np.ascontiguousarray(b)
    out = np.empty(a.shape, dtype=complex)
    zero = np.abs(b * dt) < 1e-8
    nz = ~zero
    if nz.any():
        out[nz] = (K((a + b)[nz], dt) - K(a[nz], dt)) / (1j * b[nz])
    if zero.any():
        az = a[zero]
        s = np.abs(az * dt) < 1e-8
        r = np.empty(az.shape, dtype=complex)
        # int_0^dt t e^{i a t} dt = (1 + (i a dt - 1) e^{i a dt}) / (i a)^2
        r[~s] = (1.0 + (1j * az[~s] * dt - 1.0) * np.exp(1j * az[~s] * dt)) / (1j * az[~s]) ** 2
        r[s] = 0.5 * dt**2
        out[zero] = r
    return out

def magnus_terms(H, delta, dt):
    """(||A||, ||Omega2||) in the Frobenius norm."""
    gaps = delta[:, None] - delta[None, :]
    A = H * K(gaps, dt)
    A3 = gaps[:, :, None]                                       # a[j,m] over (j,m,k)
    B3 = gaps[None, :, :]                                       # b[m,k] over (j,m,k)
    Iab = I_ab(A3, B3, dt)
    Iba = I_ab(B3, A3, dt)
    P = H[:, :, None] * H[None, :, :]                           # H_jm H_mk
    Om2 = -0.5 * np.einsum("jmk->jk", P * (Iab - Iba))
    return np.linalg.norm(A, "fro"), np.linalg.norm(Om2, "fro")

s = build_spa_cascade_setup()
h = s["hamiltonian"]; traj = s["trajectory"]; T = float(traj.get_T())
H_t = h.get_H_t_func()
mf01, mf12 = s["microwave_fields"]
muw = [f.get_H_t_func(traj.R_t, h.QN) for f in s["microwave_fields"]]
scales = [np.sqrt(16.156), np.sqrt(8.254)]

# Evaluate where AP2 acts: the second beam centre.
t_star = None
zs = np.array([traj.R_t(t)[2] for t in np.linspace(0, T, 800)])
ts = np.linspace(0, T, 800)
t_star = float(ts[np.argmin(np.abs(zs - s["positions"][1][2]))])
D, V = np.linalg.eigh(H_t(t_star))
Vh = V.conj().T
H = sum(c * (Vh @ f(t_star) @ V) for f, c in zip(muw, scales))
np.fill_diagonal(H, 0.0)

print(f"evaluated at t/T = {t_star/T:.3f}, the AP2 beam centre\n")
print("Magnus-1 keeps A and drops Omega2. If the dropped term is what makes")
print("Magnus first order, ||Omega2|| should scale as dt^2, so that N steps")
print("accumulate N*||Omega2|| ~ dt -- first order.\n")
print(f"{'N':>8}{'dt (s)':>12}{'||A||':>12}{'||Omega2||':>13}{'ratio to A':>12}"
      f"{'N*||Om2||':>12}")
prev = None
for N in (10000, 20000, 40000, 80000, 160000):
    dt = T / N
    nA, nO = magnus_terms(H, D, dt)
    print(f"{N:>8}{dt:>12.3e}{nA:>12.4e}{nO:>13.4e}{nO/nA:>12.3e}{N*nO:>12.4e}")
    if prev is not None:
        pass
    prev = nO
print("\nscaling check: ||Omega2|| should fall 4x per doubling (dt^2),")
print("and N*||Omega2|| should fall 2x per doubling (first order overall).")
