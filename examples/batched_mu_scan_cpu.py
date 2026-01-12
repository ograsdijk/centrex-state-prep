from __future__ import annotations

import numpy as np

import centrex_tlf

from state_prep.electric_fields import ElectricField
from state_prep.hamiltonians import SlowHamiltonian
from state_prep.intensity_profiles import GaussianBeam
from state_prep.magnetic_fields import MagneticField
from state_prep.microwaves import MicrowaveField, Polarization
from state_prep.simulator import Simulator
from state_prep.trajectory import Trajectory


def main() -> None:
    # Shared slow Hamiltonian / trajectory
    trajectory = Trajectory(
        Rini=np.array([0.0, 0.0, 0.0]),
        Vini=np.array([0.0, 0.0, 200.0]),
        zfin=0.02,
    )

    electric_field = ElectricField(E_R=lambda R: np.array([0.0, 0.0, 0.0]))
    magnetic_field = MagneticField(B_R=lambda R: np.array([0.0, 0.0, 0.0]))

    slow = SlowHamiltonian(
        Js=[0, 1],
        trajectory=trajectory,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
    )

    # Define a microwave at a reference power P0.
    # For a scan in power only, we pass intensity_prefactors = P / P0.
    P0 = 1.0  # W
    muW_freq = 26.668e9  # Hz (example value)

    intensity = GaussianBeam(
        power=P0,
        sigma=0.004,
        R0=np.array([0.0, 0.0, 0.01]),
        k=np.array([0.0, 0.0, 1.0]),
        freq=muW_freq,
    )

    polarization = Polarization(
        p_R_main=lambda R: np.array([1.0, 0.0, 0.0]),
        k_vec=np.array([0.0, 0.0, 1.0]),
    )

    mw = MicrowaveField(
        Jg=0,
        Je=1,
        intensity=intensity,
        polarization=polarization,
        muW_freq=muW_freq,
        QN=slow.QN,
    )

    # Scan parameters
    powers = np.array([0.2, 0.5, 1.0, 2.0])  # W
    detunings_hz = np.array([-1e6, -0.5e6, 0.0, 0.5e6])  # Hz

    if powers.shape != detunings_hz.shape:
        raise ValueError("powers and detunings_hz must have the same shape in this simple example")

    # Scan inputs (shape can be (B,) since there's only 1 microwave field)
    intensity_prefactors = powers / P0

    # Initial state list (approximate), used to build psi0
    init = centrex_tlf.states.UncoupledState([(1.0, slow.QN[0])])

    sim = Simulator(
        trajectory=trajectory,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        initial_states_approx=[init],
        hamiltonian=slow,
        microwave_fields=[mw],
    )

    result = sim.run_microwave_scan(
        detunings_hz=detunings_hz,
        intensity_prefactors=intensity_prefactors,
        N_steps=800,
        progress=True,
        eig_backend="zheevd",
        store_final_probabilities=True,
    )

    # probabilities_final shape: (B, n_init, n_eigs)
    if result.probabilities_final is None:
        raise RuntimeError("Expected probabilities_final (set store_final_probabilities=True)")
    print("probabilities_final shape:", result.probabilities_final.shape)
    print(
        "sum check (first scan point):",
        float(np.sum(result.probabilities_final[0, 0, :])),
    )


if __name__ == "__main__":
    main()
