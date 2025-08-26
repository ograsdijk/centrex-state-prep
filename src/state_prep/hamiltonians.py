from dataclasses import dataclass
from typing import Callable, List

import centrex_tlf

from .electric_fields import ElectricField
from .magnetic_fields import MagneticField
from .trajectory import Trajectory
from .utils import make_hamiltonian, make_QN


class Hamiltonian:
    pass


@dataclass
class SlowHamiltonian(Hamiltonian):
    """
    Representation for the slowly evolving Hamiltonian which contains the
    internal Hamiltonian of the TlF molecule and the Stark and Zeeman Hamiltonians.
    """

    Js: List[int]
    trajectory: Trajectory
    electric_field: Callable
    magnetic_field: Callable
    basis: str = "uncoupled"

    def __post_init__(self):
        if self.basis == "uncoupled":
            self.QN = centrex_tlf.states.generate_uncoupled_states_ground(self.Js)
            H_dict = centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X(self.QN)
            self.H_EB = (
                centrex_tlf.hamiltonian.generate_uncoupled_hamiltonian_X_function(
                    H_dict
                )
            )
        else:
            NotImplementedError("Basis not implemented.")

        self.H_R = lambda R: self.H_EB(
            self.electric_field.E_R(R), self.magnetic_field.B_R(R)
        )

    def get_H_t_func(self) -> Callable:
        """
        Returns a function that gives the slow Hamiltonian as a function of time.
        """
        return lambda t: self.H_R(self.trajectory.R_t(t))


@dataclass
class SlowHamiltonianOld(Hamiltonian):
    """
    Representation for the slowly evolving Hamiltonian which contains the
    internal Hamiltonian of the TlF molecule and the Stark and Zeeman Hamiltonians.
    """

    Jmin: int
    Jmax: int
    trajectory: Trajectory
    electric_field: Callable
    magnetic_field: Callable
    path: str

    def __post_init__(self):
        self.QN = make_QN(self.Jmin, self.Jmax)
        self.H_EB = make_hamiltonian(self.path)
        self.H_R = lambda R: self.H_EB(
            self.electric_field.E_R(R), self.magnetic_field.B_R(R)
        )

    def get_H_t_func(self) -> Callable:
        """
        Returns a function that gives the slow Hamiltonian as a function of time.
        """
        return lambda t: self.H_R(self.trajectory.R_t(t))
