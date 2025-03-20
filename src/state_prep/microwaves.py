from dataclasses import dataclass
from typing import Callable, List, Tuple

import centrex_tlf
import numpy as np
from centrex_tlf.constants import XConstants
from centrex_tlf.couplings.matrix_elements import calculate_ED_ME_mixed_state
from centrex_tlf.hamiltonian.wigner import threej_f
from centrex_tlf.states import State
from scipy import constants
from scipy.differentiate import derivative

from .intensity_profiles import Intensity


@dataclass
class Polarization:
    """
    Class for representing polarization of microwave fields.
    """

    p_R_main: Callable  # Main component of polarization
    k_vec: np.ndarray = None  # k-vector for field
    f_long: float = 1  # Factor that multiplies longitudinal component
    dir_long: np.ndarray = None
    basis_conversion: Tuple = (
        np.array([0, 0, 1]),
        -np.array([0, 1, 0]),
        np.array([1, 0, 0]),
    )

    def __post_init__(
        self,
    ):
        if self.k_vec is not None:
            # # Check that k-vector and polarization are orthogonal
            # err_msg = "k-vector and main polarization should be orthogonal: "
            # dot = np.abs(self.p_R_main(np.array((0, 0, 0))) @ self.k_vec)
            # assert dot < 1e-6, err_msg + str(dot)

            # Check that k-vector is normalized
            err_msg = "k-vector not normalized"
            assert np.sum(np.abs(self.k_vec) ** 2) == 1.0, err_msg

    def get_long_pol(self, R: np.ndarray, ip: Intensity, freq: float) -> np.ndarray:
        """
        Returns the longitudinal polarization component.

        inputs:
        R = position where longitudinal polarization is to be calculated
        ip = intensity profile of microwaves
        freq = Frequency of microwaves
        dir = direction of longitudinal polarization if other than k-vector
              (useful if using multiple bases)
        """
        # Take div of E_R along main polarization
        div = 0
        p_main_ini = self.p_R_main(R)

        # Convert polarization vector to XYZ basis using the provided basis conversion
        # vector
        p_main = np.array(
            [
                np.dot(self.basis_conversion[0], p_main_ini),
                np.dot(self.basis_conversion[1], p_main_ini),
                np.dot(self.basis_conversion[2], p_main_ini),
            ]
        )
        # print(p_main)

        # Calculate each component of divergence
        for i in range(3):
            unit_vec = np.zeros(R.shape)
            unit_vec[i] = 1
            # func = lambda x: (ip.E_R(R=(R + x * unit_vec)) / ip.E_R(R=ip.R0))

            dx = 1e-5
            deriv = (
                (ip.E_R(R=(R + dx / 2 * unit_vec)) - ip.E_R(R=(R - dx / 2 * unit_vec)))
                / ip.E_R(R=ip.R0)
                / dx
            )

            # print(deriv*p_main[i])

            div += deriv * p_main[i]

        # Calculate wavenumber for field
        k = 2 * np.pi * freq / constants.c

        # Scale polarization vector appropriately
        if k != 0:
            p_long = div / (1j * k)

        else:
            p_long = 0

        # Define the direction of the longitudinal polarization component
        # in the xyz-coordinates
        direction = self.k_vec if self.dir_long is None else self.dir_long

        return self.f_long * p_long * direction

    def p_R(self, R: np.ndarray, ip: Intensity = None, freq: float = 0) -> np.ndarray:
        """
        Calculate polarization vector at given point and return it
        """
        # If no spatial electric field, specified, ignore any longitudinal component
        # from spatial variation
        if (not ip) or self.k_vec is None:
            return self.p_R_main(R)

        else:
            # Calculate main component of polarization
            p_main = self.p_R_main(R)

            # Calculate longitudinal component
            p_long = self.get_long_pol(R, ip, freq)

            # Normalize polarization vector
            p = p_main + p_long
            p = p / np.sqrt(np.sum(np.abs(p) ** 2))

            return p


class MicrowaveField:
    """
    Class to represent microwave fields.
    """

    def __init__(
        self,
        Jg: int,
        Je: int,
        intensity: Intensity,
        polarization: Polarization,
        muW_freq: float,
        QN: List[centrex_tlf.states.State],
        background_field: bool = False,
    ) -> None:
        self.Jg = Jg  # J for ground state
        self.Je = Je  # J for excited state
        self.intensity = intensity  # Spatial dependence of microwave intensity
        self.polarization = polarization  # Polarization of microwave field
        self.muW_freq = muW_freq  # Frequency of microwaves
        self.generate_coupling_matrices(QN)  # Couplings for x,y,z-polarized microwaves
        self.QN = QN
        self.generate_D(QN)  # Matrix for shifting energies in rotating frame
        self.background_field = (
            background_field  # Flag to show if field is a background field
        )

    def get_H_t_func(
        self, R_t: Callable, QN: List[centrex_tlf.states.State]
    ) -> Callable:
        """
        Returns a function that gives Hamiltonian for the microwave field as a
        function of time.
        """
        # Find the electric field due to the microwaves as a function of time
        E_mu_t = lambda t: self.intensity.E_R(R_t(t))
        self.E_t = E_mu_t

        # Find polarization as a function of time:
        p_mu_t = lambda t: self.polarization.p_R(R_t(t), self.intensity, self.muW_freq)
        self.p_t = p_mu_t

        # Find upper and lower triangular parts of coupling matrices
        Hu_x, Hu_y, Hu_z = tuple([np.triu(H) for H in self.H_list])
        Hl_x, Hl_y, Hl_z = tuple([np.tril(H) for H in self.H_list])

        def H_t(t: float) -> np.ndarray:
            E = E_mu_t(t)
            p = p_mu_t(t)
            pd = p.conj()

            return (2 * np.pi * XConstants.D_TlF * E / 2) * (
                (p[0] * Hu_x + p[1] * Hu_y + p[2] * Hu_z)
                + (pd[0] * Hl_x + pd[1] * Hl_y + pd[2] * Hl_z)
            )

        return H_t

    def generate_coupling_matrices(self, QN: List[centrex_tlf.states.State]) -> None:
        """
        Generates list of coupling Hamiltonians for x,y,z polarized microwaves
        coupling Jg to Je in basis defined by QN.
        """
        Jg = self.Jg
        Je = self.Je

        # Loop over possible polarizations and generate coupling matrices
        H_list = []
        for i in range(0, 3):
            # Generate polarization vector
            pol_vec = np.array([0, 0, 0])
            pol_vec[i] = 1

            # Generate coupling matrix
            H = make_H_mu(Je, Jg, QN, pol_vec=pol_vec)

            # Remove small components
            H[np.abs(H) < 1e-3 * np.max(np.abs(H))] = 0

            # Check that matrix is Hermitian
            is_hermitian = np.allclose(H, H.conj().T)
            if not is_hermitian:
                print(
                    "Warning: Microwave coupling matrix {} is not Hermitian!".format(i)
                )

            # Append to list of coupling matrices
            H_list.append(H)

        self.H_list = H_list

    def generate_D(
        self, QN: List[centrex_tlf.states.State], omega: float = None
    ) -> None:
        """
        Generates a diagonal matrix that is used to shift energies in the rotating frame
        """
        Je = self.Je
        if not omega:
            omega = 2 * np.pi * self.muW_freq

        # Generate the shift matrix
        D = np.zeros((len(QN), len(QN)))
        for i in range(len(QN)):
            if QN[i].J == Je:
                D[i, i] = -omega

        self.D = D

    def calculate_microwave_power(
        self,
        state1: State,
        state2: State,
        Omega: float,
        R: np.ndarray,
    ) -> float:
        """
        Calculates the microwave power required to have Rabi rate Omega for the microwave
        transition between state1 and state2 at position R assuming polarization is
        fully along p_main.
        """

        # Calculate electric field magnitude at R for 1W of total power
        E = self.intensity.E_R(R, power=1.0)

        # Determine main polarization component of microwave field at given point
        pol_vec = self.polarization.p_R_main(R)

        # Calculate the angular part of the matrix element between the states
        ME = calculate_ED_ME_mixed_state(
            state1.transform_to_coupled(),
            state2.transform_to_coupled(),
            pol_vec=pol_vec,
        )

        # Calculate the Rabi rate for P = 1W (in Hz)
        Omega1W = 2 * np.pi * np.abs(ME) * XConstants.D_TlF * E / 2

        # Determine what power is required (Omega \propto sqrt(Power))
        power_req = (Omega / Omega1W) ** 2

        self.intensity.power = power_req

    def calculate_rabi_rate(
        self,
        state1: State,
        state2: State,
        power: float,
        R: np.ndarray,
    ) -> float:
        """
        Calculates the Rabi rate for given power for the microwave
        transition between state1 and state2 at position R assuming polarization is
        fully along p_main.

        returns:
        Omega = Rabi rate for given power in 2pi*Hz
        """

        # Calculate electric field magnitude at R
        E = self.intensity.E_R(R, power=power)

        # Determine main polarization component of microwave field at given point
        pol_vec = self.polarization.p_R_main(R)

        # Calculate the angular part of the matrix element between the states
        ME = calculate_ED_ME_mixed_state(
            state1.transform_to_coupled(),
            state2.transform_to_coupled(),
            pol_vec=pol_vec,
        )

        # Calculate the Rabi rate for P = 1W (in Hz)
        Omega = 2 * np.pi * np.abs(ME) * XConstants.D_TlF * E / 2

        return Omega

    def set_frequency(self, freq: float) -> None:
        """
        Sets the frequency for the microwave field.

        inputs:
        freq = microwave frequency in Hz
        """
        self.muW_freq = freq
        self.generate_D(self.QN)

    def set_position(self, R0: np.ndarray) -> None:
        """
        Sets the central position for the microwave intensity profile.
        """
        self.intensity.R0 = R0

    def set_power(self, power: float) -> None:
        """
        Sets the power for the microwaves.
        """
        self.intensity.power = power


def make_H_mu(J1, J2, QN, pol_vec=np.array((0, 0, 1))):
    """
    Function that generates Hamiltonian for microwave transitions between J1 and J2 (all hyperfine states) for given
    polarization of microwaves. Rotating wave approximation is applied implicitly by only taking the exp(+i*omega*t) part
    of the cos(omgega*t) into account

    inputs:
    J1 = J of the lower rotational state being coupled
    J2 = J of upper rotational state being coupled
    QN = Quantum numbers of each index (defines basis fo matrices and vectors)
    pol_vec = vector describing polarization

    returns:
    H_mu = Hamiltonian describing coupling between

    """
    # Figure out how many states there are in the system
    N_states = len(QN)

    # Initialize a Hamiltonian
    H_mu = np.zeros((N_states, N_states), dtype=complex)

    # Start looping over states and calculate microwave matrix elements between them
    for i in range(0, N_states):
        state1 = QN[i]

        for j in range(i, N_states):
            state2 = QN[j]

            # Check that the states have the correct values of J
            if (state1.J == J1 and state2.J == J2) or (
                state1.J == J2 and state2.J == J1
            ):
                # Calculate matrix element between the two states
                H_mu[i, j] = calculate_microwave_ME(
                    state1, state2, reduced=False, pol_vec=pol_vec
                )

    # Make H_mu hermitian
    H_mu = (H_mu + np.conj(H_mu.T)) - np.diag(np.diag(H_mu))

    # return the coupling matrix
    return H_mu


def calculate_microwave_ME(state1, state2, reduced=False, pol_vec=np.array((0, 0, 1))):
    """
    Function that evaluates the microwave matrix element between two states, state1 and state2, for a given polarization
    of the microwaves

    inputs:
    state1 = an UncoupledBasisState object
    state2 = an UncoupledBasisState object
    reduced = boolean that determines if the function returns reduced or full matrix element
    pol_vec = np.array describing the orientation of the microwave polarization in cartesian coordinates

    returns:
    Microwave matrix element between state 1 and state2
    """

    # Find quantum numbers for ground state
    J = float(state1.J)
    mJ = float(state1.mJ)
    I1 = float(state1.I1)
    m1 = float(state1.m1)
    I2 = float(state1.I2)
    m2 = float(state1.m2)

    # Find quantum numbers of excited state
    Jprime = float(state2.J)
    mJprime = float(state2.mJ)
    I1prime = float(state2.I1)
    m1prime = float(state2.m1)
    I2prime = float(state2.I2)
    m2prime = float(state2.m2)

    # Calculate reduced matrix element
    M_r = (
        threej_f(J, 1, Jprime, 0, 0, 0)
        * np.sqrt((2 * J + 1) * (2 * Jprime + 1))
        * float(I1 == I1prime and m1 == m1prime and I2 == I2prime and m2 == m2prime)
    )

    # If desired, return just the reduced matrix element
    if reduced:
        return float(M_r)
    else:
        p_vec = {}
        p_vec[1] = -1 / np.sqrt(2) * (pol_vec[0] + 1j * pol_vec[1])
        p_vec[0] = pol_vec[2]
        p_vec[-1] = +1 / np.sqrt(2) * (pol_vec[0] - 1j * pol_vec[1])

        prefactor = 0
        for p in range(-1, 2):
            prefactor += (
                (-1) ** (p - mJ) * p_vec[-p] * threej_f(J, 1, Jprime, -mJ, -p, mJprime)
            )

        return prefactor * M_r
