from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .core import StaticField
from .trajectory import Trajectory


@dataclass
class MagneticField(StaticField):
    """
    Class for representing magnetic fields.
    """

    def __init__(self, B_R: Callable = None, R_t: Callable = None):
        self.B_R = B_R
        self.R_to_r = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])

        # If molecule trajectory is provided, generate B as function of time
        if R_t is not None:
            self.B_t = self.get_B_t_func(R_t)
        else:
            self.B_t = None

    def __add__(self, other):
        """
        Addition of electric fields.
        """
        assert (
            type(other) != MagneticField
        ), f"Can't add {type(other)} and MagneticField"

        B_R = lambda R: self.B_R(R) + other.B_R(R)
        return MagneticField(B_R)

    def __sub__(self, other):
        """
        Subtraction of electric fields
        """
        return self + (-1) * other

    def __mul__(self, a):
        """
        Scalar multiplication of electric field.
        """
        B_R = lambda R: a * self.B_R(R)
        return MagneticField(B_R)

    def __neg__(self):
        """
        Negative of electric field
        """
        return -1 * self

    def get_B_R(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the value of the magnetic field at position R.
        """
        return self.B_R(R)

    def get_B_r(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the value of the magnetic field in the xyz basis at position R in XYZ
        coordinates.
        """
        return self.R_to_r @ self.get_B_R(R)

    def get_B_t_func(self, R_t: Callable) -> Callable:
        """
        Returns a function that gives the B-field experienced along
        a given trajectory over time.
        """
        return lambda t: self.get_B_R(R_t(t))

    def plot(self, trajectory: Trajectory, ax=None):
        """
        Plots the electric field along the given trajectory
        """
        B_t = self.get_B_t_func(trajectory.R_t)
        T = trajectory.get_T()

        t_array = np.linspace(0, T, 1000)

        Bs = np.array([B_t(t) for t in t_array])

        if not ax:
            fig, ax = plt.subplots()

        ax.plot(t_array / 1e-6, Bs[:, 0], label=r"B$_x$")
        ax.plot(t_array / 1e-6, Bs[:, 1], label=r"B$_y$")
        ax.plot(t_array / 1e-6, Bs[:, 2], label=r"B$_z$")
        ax.set_xlabel(r"Time / $\mu$s")
        ax.set_ylabel("Magnetic field / G")
        ax.set_title("Magnetic field experienced by molecule over time")
        ax.legend()
