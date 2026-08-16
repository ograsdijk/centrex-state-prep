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
