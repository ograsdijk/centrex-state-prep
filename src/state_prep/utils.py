import pickle
from typing import Sequence, Tuple

import numpy as np
import numpy.typing as npt
from centrex_tlf.states import (
    State,
    UncoupledBasisState,
    UncoupledState,
    find_closest_vector_idx,
)


def vector_to_state(state_vector, QN, E=None):
    state = State([])

    # Get data in correct format for initializing state object
    for j, amp in enumerate(state_vector):
        state += amp * QN[j]

    return state


def matrix_to_states(V, QN, E=None):
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
) -> Tuple[np.ndarray]:
    """
    Reorders the matrix of eigenvectors V_in (each column is an eigenvector) and
    corresponding energies E_in so that the eigenvector overlap between V_in and V_ref
    is maximised.
    """
    # Take dot product between each eigenvector in V and state_vec
    overlap_vectors = np.absolute(np.matmul(np.conj(V_in.T), V_ref))

    # Find which state has the largest overlap:
    index = np.argsort(np.argmax(overlap_vectors, axis=1))
    # Store energy and state
    E_out = E_in[index]
    V_out = V_in[:, index]

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
        UncoupledBasisState(J, mJ, I1, m1, I2, m2)
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
