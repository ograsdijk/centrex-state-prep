import pickle
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
from centrex_tlf.states import (
    CoupledBasisState,
    CoupledState,
    ElectronicState,
    State,
    UncoupledBasisState,
    UncoupledState,
    find_closest_vector_idx,
)
from scipy.optimize import linear_sum_assignment


def vector_to_state(
    state_vector: npt.NDArray[np.complex128],
    QN: Sequence[CoupledBasisState | UncoupledBasisState],
    E=None,
) -> UncoupledState | CoupledState:
    # Check if QN contains UncoupledBasisState or CoupledBasisState and return appropriate type
    if len(QN) > 0:
        if isinstance(QN[0], UncoupledBasisState):
            state = UncoupledState([])
        elif isinstance(QN[0], CoupledBasisState):
            state = CoupledState([])
        else:
            raise ValueError(
                "QN list must contain UncoupledBasisState or CoupledBasisState objects."
            )
    else:
        raise ValueError("QN list is empty, cannot determine state type.")

    # Get data in correct format for initializing state object
    for j, amp in enumerate(state_vector):
        state += amp * QN[j]

    return state


def matrix_to_states(
    V: npt.NDArray[np.complex128],
    QN: Sequence[CoupledBasisState | UncoupledBasisState],
    E=None,
):
    # Find dimensions of matrix
    matrix_dimensions = V.shape

    # Initialize a list for storing eigenstates
    eigenstates = []

    for i in range(0, matrix_dimensions[1]):
        # Find state vector
        state_vector = V[:, i]

        # Ensure that largest component has positive sign
        index = np.argmax(np.abs(state_vector))
        state_vector = state_vector * np.sign(state_vector[index])

        state = State()

        # Get data in correct format for initializing state object
        for j, amp in enumerate(state_vector):
            state += amp * QN[j]

        if E is not None:
            state.energy = E[i]

        # Store the state in the list
        eigenstates.append(state)

    # Return the list of states
    return eigenstates


def find_max_overlap_idx(state_vec: np.ndarray, V_matrix: np.ndarray) -> int:
    # Take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(V_matrix.conj().T @ state_vec)

    # Find index of state that has the largest overlap
    index = np.argmax(overlap_vectors)

    return index


def reorder_evecs(
    V_in: np.ndarray, E_in: np.ndarray, V_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorder eigenvectors so that overlaps with V_ref are maximized.
    """
    # Overlap matrix: O[j, k] = <V_ref_j | V_in_k>
    O = V_ref.conj().T @ V_in
    cost = -np.abs(O)

    rows, cols = linear_sum_assignment(cost)
    # rows is [0,1,...,N-1], cols gives matching
    p = cols

    V_out = V_in[:, p]
    E_out = E_in[p]

    # Phase alignment for continuity
    phases = O[rows, p]
    V_out *= (phases / np.abs(phases))[np.newaxis, :].conj()

    return E_out, V_out


# --- Identifying eigenstates by quantum numbers -----------------------------
#
# `reorder_evecs` above carries a label from one timestep to the next by overlap,
# which is *adiabatic* labelling. That is not always the physical mapping. In
# TlF the mF=0 levels of a manifold can cross exactly when B is perpendicular to
# E, and a small B parallel to E opens only a tiny gap, about 1620 Hz/G.
# Such a crossing is traversed diabatically, so the population stays on one
# branch while the adiabatic *label* follows the other.
#
# Measured for the SPA2 setup: the initial state meets its F=1 partner at
# t/T = 0.691 with a 3.2 Hz gap against a 657 Hz Fourier width, giving a
# Landau-Zener P_diabatic of 0.999997. With the microwaves off, F=2/mF=0 holds
# 0.99999 of the population at every N_steps from 2000 to 160000 while the
# eigenstate *index* of that state moves from 14 to 15 at 160000. The physics is
# stable; the label is not. Note the direction: a finer timestep resolves the
# crossing and so makes the adiabatic label *more* likely to be wrong.
#
# These helpers identify a state by what it is rather than by where it sits in
# an array, which is immune to that whole class of failure. Use them in
# preference to a tracked index whenever an observable depends on the answer.

def _coupled_transform(
    QN: Sequence[CoupledBasisState | UncoupledBasisState],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``U[c, u] = <coupled c|uncoupled u>``, and F and F1 per coupled index.

    F1 is needed because ``(J, F, mF)`` is not a complete label: for J=1 there
    are two F=1 levels, one built from F1=1/2 and one from F1=3/2.
    """
    decompositions = [q.transform_to_coupled() for q in QN]
    keys: dict[tuple, int] = {}
    for decomposition in decompositions:
        for _, coupled in decomposition.data:
            key = (coupled.electronic_state, coupled.J, coupled.F1, coupled.F, coupled.mF)
            keys.setdefault(key, len(keys))

    U = np.zeros((len(keys), len(QN)), dtype=complex)
    for u, decomposition in enumerate(decompositions):
        for amplitude, coupled in decomposition.data:
            key = (coupled.electronic_state, coupled.J, coupled.F1, coupled.F, coupled.mF)
            U[keys[key], u] += amplitude

    ordered = sorted(keys, key=keys.get)
    F_of_coupled = np.array([key[3] for key in ordered], dtype=float)
    F1_of_coupled = np.array([key[2] for key in ordered], dtype=float)
    return U, F_of_coupled, F1_of_coupled


def _spread(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Standard deviation of `values` under the per-column distribution `weights`."""
    mean = weights.T @ values
    return np.sqrt(np.maximum(weights.T @ values**2 - mean**2, 0.0))


def eigenstate_quantum_numbers(
    V: np.ndarray,
    QN: Sequence[CoupledBasisState | UncoupledBasisState],
) -> dict[str, np.ndarray]:
    """(J, F1, F, mF) expectation values and spreads for each column of ``V``.

    Every returned array has one entry per eigenvector. Each quantum number is
    accompanied by a ``*_spread``: a spread near zero means the quantum number is
    good for that state and the label is meaningful, while a large spread means
    the state is a mixture and the label is not. That distinction is load-bearing
    here -- in a strong Stark field F is badly mixed (the SPA2 "triplet" states
    have F = 2.372 +- 4.899), and only becomes good once the field has ramped
    off -- so callers should check the spread rather than assume.

    mF = mJ + m1 + m2 is exact in the uncoupled basis at all times, which makes
    it the most reliable of the four.

    Pass the *final-time* eigenvectors (``V_fin``) when identifying states that
    populations are reported against; ``probabilities_final[..., k]`` is the
    population in ``V_fin[:, k]``.
    """
    V = np.asarray(V, dtype=complex)
    if V.ndim == 1:
        V = V[:, np.newaxis]

    weights = np.abs(V) ** 2
    weights = weights / weights.sum(axis=0, keepdims=True)

    mF_uncoupled = np.array([q.mJ + q.m1 + q.m2 for q in QN], dtype=float)
    J_uncoupled = np.array([q.J for q in QN], dtype=float)

    U, F_of_coupled, F1_of_coupled = _coupled_transform(QN)
    coupled_weights = np.abs(U @ V) ** 2
    coupled_weights = coupled_weights / coupled_weights.sum(axis=0, keepdims=True)
    mean_FF1 = coupled_weights.T @ (F_of_coupled * (F_of_coupled + 1))

    return {
        "J": weights.T @ J_uncoupled,
        "J_spread": _spread(weights, J_uncoupled),
        "F1": coupled_weights.T @ F1_of_coupled,
        "F1_spread": _spread(coupled_weights, F1_of_coupled),
        "F": (-1 + np.sqrt(1 + 4 * mean_FF1)) / 2,
        "F_spread": _spread(coupled_weights, F_of_coupled),
        "mF": weights.T @ mF_uncoupled,
        "mF_spread": _spread(weights, mF_uncoupled),
    }


def select_eigenstate(
    table: dict[str, np.ndarray],
    identity: dict[str, float],
    *,
    tolerance: float = 1e-3,
) -> int:
    """Index of the single eigenstate in ``table`` carrying ``identity``'s quantum numbers.

    ``table`` comes from `eigenstate_quantum_numbers`; ``identity`` supplies any
    subset of ``J``, ``F1``, ``F`` and ``mF``. Unlike an index tracked from t=0,
    this does not depend on the adiabatic labelling having survived a crossing --
    it reads what the eigenvectors actually are.

    Raises:
        ValueError: if the quantum numbers do not pick out exactly one state,
            which means the label is incomplete (or the tolerance is wrong)
            rather than that the caller should guess.
    """
    keys = [key for key in ("J", "F1", "F", "mF") if key in identity]
    if not keys:
        raise ValueError("identity must specify at least one of J, F1, F, mF")

    match = np.ones(table["J"].shape, dtype=bool)
    for key in keys:
        match &= np.abs(table[key] - identity[key]) < tolerance
    matches = np.flatnonzero(match)

    if matches.size != 1:
        wanted = ", ".join(f"{key}={identity[key]:+.3f}" for key in keys)
        raise ValueError(
            f"expected exactly one eigenstate with {wanted}; found {matches.size} "
            f"at tolerance {tolerance}. Note that (J, F, mF) alone is not unique "
            f"for J=1, which has two F=1 levels -- specify F1 as well."
        )
    return int(matches[0])


class LabelGapTracker:
    """Counts level crossings seen by each adiabatically tracked label.

    `reorder_evecs` carries a label from step to step by overlap, which is the
    *adiabatic* convention. Where two levels cross, the population stays on its
    diabatic branch while the label follows the adiabatic one, and the
    disagreement is silent -- the reported number looks ordinary.

    A crossing is detected as a change in a label's energy *rank*, which is a
    discrete event and therefore robust. Thresholding on the bare gap instead
    does not work: every level in a Stark-ramped trajectory ends up nearly
    degenerate once the field is off, so a gap threshold flags essentially all
    of them. Neither does a Landau-Zener estimate built from finite differences
    of the nearest-neighbour gap -- which neighbour is nearest changes from step
    to step, so the implied sweep rate is noise-dominated.

    Measured on the SPA2 setup, this is not a rare event: 56 of 64 labels cross
    something, almost all between t/T = 0.90 and 0.96 as the Stark field
    collapses and the hyperfine structure folds together. Adiabatic labels in
    this system are broadly unreliable rather than exceptionally so, which is
    the argument for identifying states by quantum numbers
    (`eigenstate_quantum_numbers`) rather than by a carried index.

    Caveats: crossings recorded with gaps at or below roughly 1e-5 Hz here sit
    close to float64 resolution on a ~10 GHz absolute energy and may be jitter
    rather than physics; and the rank is only as meaningful as the tracking that
    produced it. This is a diagnostic, not an observable.

    Costs O(n log n) per timestep, negligible beside the O(n^3) eigensolve.
    """

    def __init__(self, n: int) -> None:
        self.crossings = np.zeros(n, dtype=int)
        self.min_gap = np.full(n, np.inf)
        self.at_time = np.full(n, np.nan)
        self.crossing_gap = np.full(n, np.inf)
        self.crossing_time = np.full(n, np.nan)
        self._prev_rank: np.ndarray | None = None

    @staticmethod
    def _nearest_gaps(energies: np.ndarray) -> np.ndarray:
        """Distance from each level to its nearest neighbour, in label order."""
        n = energies.size
        order = np.argsort(energies)
        d = np.diff(energies[order])
        nearest = np.empty(n)
        nearest[0] = d[0]
        nearest[-1] = d[-1]
        if n > 2:
            nearest[1:-1] = np.minimum(d[:-1], d[1:])
        gaps = np.empty(n)
        gaps[order] = nearest
        return gaps

    def update(self, energies: np.ndarray, t: float) -> None:
        """Fold one timestep's eigenvalues, in tracked-label order, into the statistics."""
        n = energies.size
        if n < 2:
            return

        gaps = self._nearest_gaps(energies)
        improved = gaps < self.min_gap
        self.min_gap[improved] = gaps[improved]
        self.at_time[improved] = t

        rank = np.empty(n, dtype=int)
        rank[np.argsort(energies)] = np.arange(n)
        if self._prev_rank is not None:
            moved = rank != self._prev_rank
            if moved.any():
                self.crossings += moved
                closer = moved & (gaps < self.crossing_gap)
                self.crossing_gap[closer] = gaps[closer]
                self.crossing_time[closer] = t
        self._prev_rank = rank

    def summary(self, transit_time_s: float) -> dict:
        """Diagnostic arrays, with gaps converted to Hz for reporting."""
        two_pi = 2 * np.pi
        return {
            "crossings": self.crossings,
            "min_gap_hz": self.min_gap / two_pi,
            "min_gap_time_s": self.at_time,
            "crossing_gap_hz": self.crossing_gap / two_pi,
            "crossing_time_s": self.crossing_time,
            "fourier_width_hz": 1.0 / float(transit_time_s),
        }


def make_hamiltonian(path, c1=126030.0, c2=17890.0, c3=700.0, c4=-13300.0):
    """
    Generates Hamiltonian based on a pickle file
    """
    with open(path, "rb") as f:
        hamiltonians = pickle.load(f)

        # Substitute values into hamiltonian
        variables = [
            sympy.symbols("Brot"),
            *sympy.symbols("c1 c2 c3 c4"),
            sympy.symbols("D_TlF"),
            *sympy.symbols("mu_J mu_Tl mu_F"),
        ]

        lambdified_hamiltonians = {
            H_name: sympy.lambdify(variables, H_matrix)
            for H_name, H_matrix in hamiltonians.items()
        }

        # Molecular constants

        # Values for rotational constant are from "Microwave Spectral tables: Diatomic molecules" by Lovas & Tiemann (1974).
        # Note that Brot differs from the one given by Ramsey by about 30 MHz.
        B_e = 6.689873e9
        alpha = 45.0843e6
        Brot = B_e - alpha / 2
        D_TlF = 4.2282 * 0.393430307 * 5.291772e-9 / 4.135667e-15  # [Hz/(V/cm)]
        mu_J = 35  # Hz/G
        mu_Tl = 1240.5  # Hz/G
        mu_F = 2003.63  # Hz/G

        H = {
            H_name: H_fn(Brot, c1, c2, c3, c4, D_TlF, mu_J, mu_Tl, mu_F)
            for H_name, H_fn in lambdified_hamiltonians.items()
        }

        Ham = (
            lambda E, B: 2
            * np.pi
            * (
                H["Hff"]
                + E[0] * H["HSx"]
                + E[1] * H["HSy"]
                + E[2] * H["HSz"]
                + B[0] * H["HZx"]
                + B[1] * H["HZy"]
                + B[2] * H["HZz"]
            )
        )

        return Ham


def make_QN(Jmin, Jmax, I1=1 / 2, I2=1 / 2):
    """
    Function that generates a list of quantum numbersfor TlF
    """
    QN = [
        UncoupledBasisState(
            J,
            mJ,
            I1,
            m1,
            I2,
            m2,
            Omega=0,
            P=(-1) ** int(J),
            electronic_state=ElectronicState.X,
        )
        for J in np.arange(Jmin, Jmax + 1)
        for mJ in np.arange(-J, J + 1)
        for m1 in np.arange(-I1, I1 + 1)
        for m2 in np.arange(-I2, I2 + 1)
    ]

    return QN


def calculate_transition_frequency(
    state1: UncoupledState,
    state2: UncoupledState,
    ham: npt.NDArray[np.complex128],
    QN: Sequence[UncoupledBasisState],
) -> float:
    D, V = np.linalg.eigh(ham)
    svec1 = state1.state_vector(QN)
    svec2 = state2.state_vector(QN)
    id1 = find_closest_vector_idx(svec1, V)
    id2 = find_closest_vector_idx(svec2, V)
    return (D[id2] - D[id1]).real / (2 * np.pi)
