from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import centrex_tlf
import dill
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.lapack import zheevd
from tqdm import tqdm

from .electric_fields import ElectricField
from .hamiltonians import Hamiltonian
from .magnetic_fields import MagneticField
from .microwaves import MicrowaveField
from .trajectory import Trajectory
from .utils import find_max_overlap_idx, reorder_evecs, vector_to_state


@dataclass
class SimulationResult:
    """
    Class for storing results from simulations.

    t_array         : times at which results were returned (seconds)
    psis            : state vectors at each time when starting from initial states defined in
                      initial_states
    energies        : energies of all eigenstates of the hamiltonian at each time (2pi*Hz)
    probabilities   : for each state in initial staets, the probability of being found in
                      each eigenstate of the hamiltonian
    V_ref           : reference matrix of eigenstates of hamiltonian which tells what state each
                      index of energies and states corresponds to
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states: List[centrex_tlf.states.State]
    hamiltonian: Hamiltonian
    microwave_fields: Optional[List[MicrowaveField]]
    t_array: np.ndarray
    psis: Optional[np.ndarray]
    energies: Optional[np.ndarray]
    probabilities: Optional[np.ndarray]
    probabilities_final: Optional[np.ndarray]
    V_ini: np.ndarray
    V_fin: np.ndarray
    monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None
    monitor_probabilities: Optional[np.ndarray] = None
    monitor_probabilities_final: Optional[np.ndarray] = None

    def __post_init__(self):
        # Generate array of positions
        self.z_array = self.t_array * self.trajectory.Vini[2] + self.trajectory.Rini[2]

    def plot_state_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.UncoupledState,
        ax: Optional[plt.Axes] = None,
        position: bool = False,
        state_mapper: Optional[Callable] = None,
        tol: float = 1e-2,
    ) -> None:
        """
        Plots the probability of being found in a given adiabatically evolved eigenstate
        at different times.
        """
        if ax is None:
            fig, ax = plt.subplots()

        probs = self.get_state_probability(state, initial_state)

        label_state = state.remove_small_components(tol=tol)
        label_state.data = [(amp.real, stat) for amp, stat in label_state.data]
        label = (
            label_state.normalize()
            .remove_small_components(tol=tol)
            .state_string_custom(["J", "mJ", "m1", "m2"])
            if not state_mapper
            else state_mapper(state.normalize().remove_small_components(tol=tol))
        )
        if position:
            ax.plot(self.z_array / 1e-2, probs, label=label)
            ax.set_xlabel(r"Z-position / cm")
        else:
            ax.plot(self.t_array / 1e-6, probs, label=label)
            ax.set_xlabel(r"Time / $\mu$s")

    def plot_state_probabilities(
        self,
        states: List[centrex_tlf.states.UncoupledState],
        initial_state: centrex_tlf.states.UncoupledState,
        ax: Optional[plt.Axes] = None,
        position: bool = False,
        state_mapper: Optional[Callable] = None,
        tol: float = 1e-3,
    ) -> None:
        """
        Plots probabilities over time for states specified in the list states.
        """
        if ax is None:
            fig, ax = plt.subplots()
        for state in states:
            self.plot_state_probability(
                state,
                initial_state,
                ax=ax,
                position=position,
                state_mapper=state_mapper,
                tol=tol,
            )

    def get_state_probability(
        self,
        state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
        ax=None,
    ):
        """
        Returns the probability of being found in given adiabatically evolved state
        for given initial state.
        """
        index_ini = self._resolve_initial_state_index(initial_state)

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )

        if self.probabilities is not None:
            return self.probabilities[:, index_ini, index_state]

        if self.probabilities_final is not None:
            # Return a 1-element array so existing code using `[-1]` keeps working.
            return np.array([self.probabilities_final[index_ini, index_state]])

        raise ValueError(
            "No probabilities stored on this SimulationResult. "
            "Re-run with store_probabilities=True or store_final_probabilities=True."
        )

    def _resolve_initial_state_index(
        self, initial_state: centrex_tlf.states.State
    ) -> int:
        """Resolve `initial_state` to an index in `self.initial_states`.

        In many workflows, users specify an *approximate* uncoupled state at t=0.
        Internally, the simulator maps that to the closest eigenstate and stores that
        mapped eigenstate in `self.initial_states`. This helper lets callers pass
        either representation.
        """
        try:
            return self.initial_states.index(initial_state)
        except ValueError:
            pass

        try:
            vec = initial_state.state_vector(self.hamiltonian.QN)
        except Exception as e:
            raise ValueError(
                "initial_state not found in result.initial_states and could not compute a state vector"
            ) from e

        overlaps = []
        for s in self.initial_states:
            try:
                s_vec = s.state_vector(self.hamiltonian.QN)
            except Exception:
                overlaps.append(-1.0)
                continue
            overlaps.append(float(np.abs(np.vdot(s_vec.conj(), vec)) ** 2))

        best = int(np.argmax(overlaps))
        if overlaps[best] < 0:
            raise ValueError(
                "Could not match initial_state to any stored initial state"
            )
        return best

    def get_monitor_probability(
        self,
        monitor_state: centrex_tlf.states.UncoupledState,
        initial_state: centrex_tlf.states.State,
    ) -> np.ndarray:
        """Return population of a monitored *adiabatically-tracked* state.

        The monitor state is mapped to the closest eigenstate at t=0 (via V_ini),
        and that eigenstate index is tracked through time using the library's
        eigenvector reordering.
        """
        if not self.monitor_states:
            raise ValueError(
                "No monitor_states present on this SimulationResult. "
                "Re-run with monitor_states=[...] and store_monitor_probabilities=True "
                "or store_final_monitor_probabilities=True."
            )

        try:
            mon_i = self.monitor_states.index(monitor_state)
        except ValueError as e:
            raise ValueError("monitor_state not found in result.monitor_states") from e

        index_ini = self._resolve_initial_state_index(initial_state)

        if self.monitor_probabilities is not None:
            return self.monitor_probabilities[:, index_ini, mon_i]

        if self.monitor_probabilities_final is not None:
            return np.array([self.monitor_probabilities_final[index_ini, mon_i]])

        raise ValueError(
            "No monitor probabilities stored on this SimulationResult. "
            "Re-run with store_monitor_probabilities=True or store_final_monitor_probabilities=True."
        )

    def find_large_prob_states(
        self, initial_state: centrex_tlf.states.State, N: int = 5
    ) -> List[centrex_tlf.states.State]:
        """
        Returns the N states with the largest mean probabilities for given initial
        state.
        """
        index_ini = self._resolve_initial_state_index(initial_state)
        if self.probabilities is not None:
            score = np.mean(self.probabilities[:, index_ini, :], axis=0)
        elif self.probabilities_final is not None:
            score = self.probabilities_final[index_ini, :]
        else:
            raise ValueError(
                "No probabilities stored on this SimulationResult. "
                "Re-run with store_probabilities=True or store_final_probabilities=True."
            )

        index = np.argsort(-score)[:N]

        state_vecs = self.V_ini[:, index]
        states = []
        for i in range(state_vecs.shape[1]):
            states.append(vector_to_state(state_vecs[:, i], self.hamiltonian.QN))

        return states

    def plot_state_energy(
        self,
        state: centrex_tlf.states.UncoupledState,
        zero_state: Optional[centrex_tlf.states.UncoupledState] = None,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Plots the energy of state, using energy of zero_state as zero energy.
        """
        energies = self.get_state_energy(state)
        if zero_state:
            energies_zero = self.get_state_energy(zero_state)
            energies = energies - energies_zero

        if ax is None:
            fig, ax = plt.subplots()

        label = (
            state.remove_small_components(tol=0.1)
            .normalize()
            .state_string_custom(["J", "mJ", "m1", "m2"])
        )
        ax.plot(self.t_array / 1e-6, energies / (2 * np.pi * 1e3), label=label)
        ax.set_xlabel(r"Time / $\mu$s")
        ax.set_ylabel("Energy / kHz")
        return energies

    def plot_state_energies(
        self,
        states: List[centrex_tlf.states.UncoupledState],
        zero_state: centrex_tlf.states.UncoupledState,
        ax: plt.Axes = None,
    ) -> None:
        """
        Plots probabilities over time for states specified in the list states.
        """
        energies = []
        if ax is None:
            fig, ax = plt.subplots()
        for state in states:
            energies.append(self.plot_state_energy(state, zero_state=zero_state, ax=ax))

        return energies

    def get_state_energy(self, state: centrex_tlf.states.UncoupledState) -> np.ndarray:
        """
        Gets the energy of state for all values in t_array.
        """

        if self.energies is None:
            raise ValueError(
                "No energies stored on this SimulationResult. Re-run with store_energies=True."
            )

        index_state = find_max_overlap_idx(
            state.state_vector(self.hamiltonian.QN), self.V_ini
        )

        return self.energies[:, index_state]

    def get_state_energy_diabatic(
        self, state: centrex_tlf.states.UncoupledState
    ) -> np.array:
        """
        Gets the energy of state that is closes to provided state at each time step.

        Corresponds (somewhat) to diabatic following of eigenstates
        """
        raise NotImplementedError(
            "Diabatic energies are not stored on SimulationResult in this version."
        )

    def save_to_pickle(self, path: Path) -> None:
        """
        Saves the result to a pickle.
        """
        with open(path, "wb+") as f:
            dill.dump(self, f)


@dataclass
class MicrowaveScanResult:
    """Result from a microwave-parameter scan with shared slow Hamiltonian.

    This is intended for ensemble scans where the slow Hamiltonian H_slow(t)
    is identical for all scan points, and only microwave terms differ (detuning,
    power, background fields, etc.).
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states: List[centrex_tlf.states.State]
    hamiltonian: Hamiltonian
    t_array: np.ndarray
    psis_final: np.ndarray
    probabilities_final: Optional[np.ndarray]
    monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None
    monitor_probabilities_final: Optional[np.ndarray] = None
    V_ini: Optional[np.ndarray] = None
    V_fin: Optional[np.ndarray] = None


@dataclass
class Simulator:
    """
    Class used to run simulations and store data.
    """

    trajectory: Trajectory
    electric_field: ElectricField
    magnetic_field: MagneticField
    initial_states_approx: centrex_tlf.states.UncoupledState
    hamiltonian: Hamiltonian
    microwave_fields: Optional[List[MicrowaveField]] = None

    def __post_init__(self):
        self.psis = np.array([])
        self.initial_states = []

    def run(
        self,
        N_steps=int(1e4),
        store_every: int = 1,
        store_psis: bool = True,
        store_energies: bool = True,
        store_probabilities: bool = True,
        store_final_probabilities: bool = True,
        progress: bool = True,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None,
        store_monitor_probabilities: bool = False,
        store_final_monitor_probabilities: bool = True,
        eig_backend: str = "zheevd",
    ):
        """
        Runs the simulation.

        Parameters
        ----------
        N_steps:
            Number of integration timesteps.
        store_every:
            Store results every N steps (1 = store all). Increasing this can
            significantly speed up simulations by reducing allocations and
            probability calculations.
        monitor_states:
            Optional list of states to monitor *as adiabatically-tracked eigenstates*.
            Each state is mapped to the closest eigenstate at t=0 (using max overlap
            with V_ini), and then that eigenstate index is tracked through time via the
            same eigenvector reordering used everywhere else.
        store_monitor_probabilities:
            If True, store populations for monitor_states at each stored time.
        store_final_monitor_probabilities:
            If True, store only the final populations for monitor_states.

        eig_backend:
            Eigen-solver backend used for diagonalizations.
            - "zheevd": use LAPACK zheevd (current default)
            - "numpy": use numpy.linalg.eigh
        """
        if store_every < 1:
            raise ValueError("store_every must be >= 1")
        # Calculate the total time for the simulation
        T = self.trajectory.get_T()

        # Generate Hamiltonian that has slow time-evolution included
        H_t = self.hamiltonian.get_H_t_func()

        # Generate Hamiltonians for microwaves
        if self.microwave_fields is not None:
            # Initialize matrix for shifting energies in rotating frame
            D_mu = np.zeros((len(self.hamiltonian.QN), len(self.hamiltonian.QN)))

            # Initialize container for Hamiltonians
            muw_hams = []

            # Container for frequencies of all microwaves
            omegas = []

            for microwave_field in self.microwave_fields:
                muw_hams.append(
                    microwave_field.get_H_t_func(
                        self.trajectory.R_t, self.hamiltonian.QN
                    )
                )
                # Background fields should always be at the same frequency as
                # as main field so don't do the shifting for them
                # Olivier: the background field can be at a different frequency, for
                # instance for the RC microwaves leaking through. Just check if the
                # frequency already is present in omegas
                if not np.any(
                    np.isclose((2 * np.pi * microwave_field.muW_freq), omegas)
                ):
                    omegas.append(2 * np.pi * microwave_field.muW_freq)
                    microwave_field.generate_D(self.hamiltonian.QN, omega=sum(omegas))
                    if np.any((D_mu != 0) & (microwave_field.D != 0)):
                        raise AssertionError(
                            "Rotating frame transform failed, coupling already present between same levels."
                        )
                    D_mu += microwave_field.D

            # Generate function that gives couplings due to all microwaves
            def H_mu_tot_t(t):
                H_mu_tot = muw_hams[0](t)
                if len(muw_hams) > 1:
                    for H_mu_t in muw_hams[1:]:
                        H_mu_tot = H_mu_tot + H_mu_t(t)
                return H_mu_tot

        # Generate integration time array
        t_array = np.linspace(0, T, N_steps)

        # Indices to store output (downsampled)
        save_idx = np.arange(0, N_steps, store_every, dtype=int)
        if save_idx.size == 0 or save_idx[0] != 0:
            save_idx = np.insert(save_idx, 0, 0)
        if save_idx[-1] != N_steps - 1:
            save_idx = np.append(save_idx, N_steps - 1)

        # Perform time-evolution
        if self.microwave_fields is None:
            (
                psis_t,
                energies,
                probabilities,
                probabilities_final,
                monitor_probabilities,
                monitor_probabilities_final,
                V_ini,
                V_fin,
            ) = self._time_evolve(
                H_t,
                t_array,
                save_idx,
                store_psis=store_psis,
                store_energies=store_energies,
                store_probabilities=store_probabilities,
                store_final_probabilities=store_final_probabilities,
                progress=progress,
                monitor_states=monitor_states,
                store_monitor_probabilities=store_monitor_probabilities,
                store_final_monitor_probabilities=store_final_monitor_probabilities,
                eig_backend=eig_backend,
            )
        else:
            (
                psis_t,
                energies,
                probabilities,
                probabilities_final,
                monitor_probabilities,
                monitor_probabilities_final,
                V_ini,
                V_fin,
            ) = self._time_evolve_mu(
                H_t,
                H_mu_tot_t,
                D_mu,
                t_array,
                save_idx,
                store_psis=store_psis,
                store_energies=store_energies,
                store_probabilities=store_probabilities,
                store_final_probabilities=store_final_probabilities,
                progress=progress,
                monitor_states=monitor_states,
                store_monitor_probabilities=store_monitor_probabilities,
                store_final_monitor_probabilities=store_final_monitor_probabilities,
                eig_backend=eig_backend,
            )

        # Generate a result object
        result = SimulationResult(
            trajectory=self.trajectory,
            electric_field=self.electric_field,
            magnetic_field=self.magnetic_field,
            initial_states=self.initial_states,
            hamiltonian=self.hamiltonian,
            microwave_fields=self.microwave_fields,
            t_array=t_array[save_idx],
            psis=psis_t,
            energies=energies,
            probabilities=probabilities,
            probabilities_final=probabilities_final,
            monitor_states=monitor_states,
            monitor_probabilities=monitor_probabilities,
            monitor_probabilities_final=monitor_probabilities_final,
            V_ini=V_ini,
            V_fin=V_fin,
        )

        return result

    def init_state_vecs(
        self, H_0: np.ndarray, V_0: Optional[np.ndarray] = None
    ) -> None:
        """
        Generates state vectors based on self.initial_states in the basis
        of self.hamiltonian
        """

        # Find the eigenstates of the Hamiltonian that most closely correspond to the
        # initial states
        initial_states = []
        psis = []
        V = V_0
        if V is None:
            _, V = np.linalg.eigh(H_0)
        for state in self.initial_states_approx:
            idx = find_max_overlap_idx(state.state_vector(self.hamiltonian.QN), V)
            psis.append(V[:, idx])
            initial_states.append(vector_to_state(V[:, idx], self.hamiltonian.QN))
        self.psis = np.array(psis)
        self.initial_states = initial_states

    def run_microwave_scan(
        self,
        *,
        detunings_hz: np.ndarray,
        intensity_prefactors: np.ndarray,
        N_steps: int = int(1e4),
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]] = None,
        store_final_probabilities: bool = True,
        store_final_monitor_probabilities: bool = True,
        progress: bool = True,
        eig_backend: str = "zheevd",
    ) -> MicrowaveScanResult:
        """Run a batched microwave scan reusing the slow diagonalization.

        This is optimized for scans where the *slow* Hamiltonian H_slow(t) is
        identical across scan points, and only microwave parameters vary.

        Microwave coupling shapes are taken from `self.microwave_fields` (same as
        `run()`), but you supply per-scan-point prefactors:

        - `detunings_hz`: detuning(s) in Hz, same shape as `intensity_prefactors`.
        - `intensity_prefactors`: dimensionless scaling of *intensity/power*.
          Internally, couplings scale as sqrt(intensity_prefactors).

        Shapes:
        - For M microwave fields, provide arrays shaped (B, M) for batch size B.
        - For M=1, 1D arrays of shape (B,) or scalars are accepted.
        - 1D arrays of shape (M,) imply B=1.
        """
        if N_steps < 2:
            raise ValueError("N_steps must be >= 2")

        if not self.microwave_fields:
            raise ValueError(
                "run_microwave_scan requires Simulator.microwave_fields (a list of MicrowaveField)."
            )

        n_fields = len(self.microwave_fields)

        def _as_batched(x: np.ndarray, name: str) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 0:
                if n_fields != 1:
                    raise ValueError(f"{name} is scalar but there are {n_fields} microwave fields")
                return arr.reshape(1, 1)
            if arr.ndim == 1:
                if n_fields == 1:
                    return arr.reshape(-1, 1)
                if arr.shape[0] != n_fields:
                    raise ValueError(
                        f"{name} must have shape (B,{n_fields}) or ({n_fields},); got {arr.shape}"
                    )
                return arr.reshape(1, n_fields)
            if arr.ndim == 2:
                if arr.shape[1] != n_fields:
                    raise ValueError(
                        f"{name} must have shape (B,{n_fields}); got {arr.shape}"
                    )
                return arr
            raise ValueError(f"{name} must be scalar, 1D, or 2D; got ndim={arr.ndim}")

        det_b = _as_batched(detunings_hz, "detunings_hz").astype(float, copy=False)
        inten_b = _as_batched(intensity_prefactors, "intensity_prefactors").astype(
            float, copy=False
        )
        if det_b.shape != inten_b.shape:
            raise ValueError(
                f"detunings_hz and intensity_prefactors must have identical shapes; got {det_b.shape} vs {inten_b.shape}"
            )

        if np.any(inten_b < 0):
            raise ValueError("intensity_prefactors must be >= 0")

        batch = int(det_b.shape[0])

        # Build per-microwave H_mu(t) functions (shared across batch)
        muw_hams = [
            mw.get_H_t_func(self.trajectory.R_t, self.hamiltonian.QN)
            for mw in self.microwave_fields
        ]

        # Build base rotating-frame shift D_mu (deduplicated by *frequency*), like `run()`.
        # For per-field detunings, we require that fields with the same frequency
        # share the same detuning across the batch (otherwise the rotating-frame
        # definition would be ambiguous).
        n = len(self.hamiltonian.QN)
        D_mu_diag_base = np.zeros((n,), dtype=float)
        unique_omegas: List[float] = []
        group_diag_masks: List[np.ndarray] = []
        group_field_indices: List[List[int]] = []
        for field_i, mw in enumerate(self.microwave_fields):
            omega = 2 * np.pi * mw.muW_freq
            group = None
            for gi, om in enumerate(unique_omegas):
                if np.isclose(omega, om):
                    group = gi
                    break

            if group is None:
                unique_omegas.append(omega)
                omega_sum = float(np.sum(unique_omegas))
                mw.generate_D(self.hamiltonian.QN, omega=omega_sum)
                D_mu_diag_base += np.diag(mw.D)
                group_diag_masks.append((np.abs(np.diag(mw.D)) > 0).astype(float))
                group_field_indices.append([field_i])
                group = len(unique_omegas) - 1
            else:
                group_field_indices[group].append(field_i)

        # Validate per-field detunings within each frequency group, then build D_mu_batch
        detuning_groups = np.zeros((batch, len(unique_omegas)), dtype=float)
        for gi, idxs in enumerate(group_field_indices):
            base = det_b[:, idxs[0]]
            for j in idxs[1:]:
                if not np.allclose(det_b[:, j], base):
                    raise ValueError(
                        "Microwave fields with the same muW_freq must share the same detuning in detunings_hz. "
                        f"Frequency group {gi} has indices {idxs} with differing detunings."
                    )
            detuning_groups[:, gi] = base

        # Store detuning shifts as diagonals (B,n) instead of dense (B,n,n)
        D_mu_diag_batch = np.repeat(D_mu_diag_base[None, :], batch, axis=0)
        for gi, mask in enumerate(group_diag_masks):
            D_mu_diag_batch -= (2 * np.pi) * detuning_groups[:, gi, None] * mask[None, :]

        # Coupling scaling: intensity/power prefactor -> E-field prefactor via sqrt
        coupling_scales = np.sqrt(inten_b)

        H_t = self.hamiltonian.get_H_t_func()
        T = self.trajectory.get_T()
        t_array = np.linspace(0, T, N_steps)

        (
            psis_final,
            probabilities_final,
            monitor_probabilities_final,
            V_ini,
            V_fin,
        ) = self._time_evolve_mu_batched_shared_slow(
            H_slow_t=H_t,
            muw_hams=muw_hams,
            coupling_scales=coupling_scales,
            D_mu_diag_batch=D_mu_diag_batch,
            t_array=t_array,
            monitor_states=monitor_states,
            store_final_probabilities=store_final_probabilities,
            store_final_monitor_probabilities=store_final_monitor_probabilities,
            progress=progress,
            eig_backend=eig_backend,
        )

        return MicrowaveScanResult(
            trajectory=self.trajectory,
            electric_field=self.electric_field,
            magnetic_field=self.magnetic_field,
            initial_states=self.initial_states,
            hamiltonian=self.hamiltonian,
            t_array=t_array,
            psis_final=psis_final,
            probabilities_final=probabilities_final,
            monitor_states=monitor_states,
            monitor_probabilities_final=monitor_probabilities_final,
            V_ini=V_ini,
            V_fin=V_fin,
        )

    def _time_evolve_mu_batched_shared_slow(
        self,
        *,
        H_slow_t: Callable[[float], np.ndarray],
        muw_hams: List[Callable[[float], np.ndarray]],
        coupling_scales: np.ndarray,
        D_mu_diag_batch: np.ndarray,
        t_array: np.ndarray,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_final_probabilities: bool,
        store_final_monitor_probabilities: bool,
        progress: bool,
        eig_backend: str,
    ):
        """Batched microwave evolution with shared slow diagonalization.

        This is a CPU-only batched variant of `_time_evolve_mu` where H_slow(t)
        is identical across the batch and is diagonalized only once per timestep.
        """
        # Validate batch shapes
        if D_mu_diag_batch.ndim != 2:
            raise ValueError("D_mu_diag_batch must have shape (B,n)")
        batch = int(D_mu_diag_batch.shape[0])

        coupling_scales = np.asarray(coupling_scales)
        if coupling_scales.ndim != 2 or coupling_scales.shape[0] != batch:
            raise ValueError(
                "coupling_scales must have shape (B,M) matching D_mu_diag_batch batch size"
            )
        if coupling_scales.shape[1] != len(muw_hams):
            raise ValueError(
                f"coupling_scales has M={coupling_scales.shape[1]} but muw_hams has M={len(muw_hams)}"
            )

        # Calculate Hamiltonian at tini
        H_tini = H_slow_t(t_array[0])
        n = int(H_tini.shape[0])
        if D_mu_diag_batch.shape[1] != n:
            raise ValueError(
                f"D_mu_diag_batch has n={D_mu_diag_batch.shape[1]} but Hamiltonian has n={n}"
            )

        diag_idx = slice(None, None, n + 1)

        # Reference eigen-decomposition at t0 (reused for init + tracking)
        E_ref, V_ref = np.linalg.eigh(H_tini)
        index = np.argsort(E_ref)
        E_ref = E_ref[index]
        V_ref = V_ref[:, index]
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        # Initialize state vectors once, then replicate across batch
        self.init_state_vecs(H_tini, V_0=V_ref)
        psis_batch = np.repeat(self.psis[None, :, :], batch, axis=0)

        last_evecs = V_ref
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            dt = t_array[i + 1] - t_array[i]

            # Shared slow Hamiltonian diagonalization (once per timestep)
            H_slow_i = H_slow_t(t)
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Track eigenvectors (for probabilities/monitor semantics)
            Es, evecs = D, V
            Es, evecs = reorder_evecs(evecs, Es, V_ref)
            last_evecs = evecs

            # Pre-rotate each microwave field into the slow-eigenbasis (shared across batch)
            Vh = V.conj().T
            H_mu_rot = [Vh @ H_mu_t(t) @ V for H_mu_t in muw_hams]

            for b in range(batch):
                # Assemble rotating-frame Hamiltonian (in slow-eigenbasis)
                H_rot = (coupling_scales[b, 0] * H_mu_rot[0]).copy()
                for j in range(1, len(H_mu_rot)):
                    H_rot += coupling_scales[b, j] * H_mu_rot[j]

                # Add detunings + slow energies to diagonal
                H_rot.flat[diag_idx] += D
                H_rot.flat[diag_idx] += D_mu_diag_batch[b]

                # Diagonalize rotating-frame Hamiltonian (per scan point)
                if eig_backend == "zheevd":
                    D_rot, V_rot, info_rot = zheevd(H_rot)
                    if info_rot != 0:
                        D_rot, V_rot = np.linalg.eigh(H_rot)
                elif eig_backend == "numpy":
                    D_rot, V_rot = np.linalg.eigh(H_rot)
                else:
                    raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

                A = V @ V_rot

                phases_rot = np.exp(-1j * D_rot * dt)
                tmp2 = psis_batch[b] @ A.conj()
                tmp2 *= phases_rot[np.newaxis, :]
                psis_batch[b] = tmp2 @ A.T

            # Update reference for eigenvector tracking (shared)
            V_ref = evecs

        probabilities_final = None
        if store_final_probabilities:
            overlaps = psis_batch @ last_evecs.conj()
            probabilities_final = np.abs(overlaps) ** 2

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = psis_batch @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        return psis_batch, probabilities_final, monitor_probabilities_final, V_ref_ini, V_ref

    def _time_evolve(
        self,
        H_slow: Callable,
        t_array: np.ndarray,
        save_idx: Optional[np.ndarray],
        *,
        store_psis: bool,
        store_energies: bool,
        store_probabilities: bool,
        store_final_probabilities: bool,
        progress: bool,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_monitor_probabilities: bool,
        store_final_monitor_probabilities: bool,
        eig_backend: str,
    ):
        """
        Time evolves the system using the Hamiltonian function H_t
        over the time period in t_array.
        """
        if save_idx is None:
            save_idx = np.arange(len(t_array), dtype=int)

        # Calculate Hamiltonian at tini
        H_tini = H_slow(t_array[0])

        # Reference eigen-decomposition at t0 (reused for init + storage)
        E_ref, V_ref = np.linalg.eigh(H_tini)
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        # Initialize state vectors
        self.init_state_vecs(H_tini, V_0=V_ref)

        # Initialize containers to store results (possibly downsampled)
        psis_t, energies, probabilities = self._init_results_containers(
            t_array[save_idx],
            H_tini,
            store_psis=store_psis,
            store_energies=store_energies,
            store_probabilities=store_probabilities,
            D_0=E_ref,
            V_0=V_ref,
        )

        # Initialize reference matrix of eigenvectors that is used to keep track
        # of adiabatic evolution of eigenstates

        # Loop over t_array to time-evolve
        monitor_probabilities = None
        if store_monitor_probabilities and (monitor_idx is not None):
            monitor_probabilities = np.zeros(
                (len(save_idx), len(self.initial_states), monitor_idx.size)
            )
            amps0 = self.psis @ V_ref_ini[:, monitor_idx].conj()
            monitor_probabilities[0, :, :] = np.abs(amps0) ** 2

        out_i = 0
        last_evecs = V_ref
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            # Calculate the timestep
            dt = t_array[i + 1] - t_array[i]

            # Calculate Hamiltonian
            H_slow_i = H_slow(t)

            # Diagonalize Hamiltonian
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Reorder eigenvectors and energies
            Es, evecs = reorder_evecs(V, D, V_ref)
            # Es_diabatic, _ = reorder_evecs(V, D, V_ref_ini)
            last_evecs = evecs

            # Apply propagator without forming U_dt:
            # For row-vector storage (each state is a row), the update is
            # psi <- psi @ V.conj() @ diag(exp(-i D dt)) @ V.T
            phases = np.exp(-1j * D * dt)
            tmp = self.psis @ V.conj()
            tmp *= phases[np.newaxis, :]
            self.psis = tmp @ V.T

            # Store results for this timestep if requested
            if (i + 1) == save_idx[out_i + 1]:
                out_i += 1
                if store_psis and psis_t is not None:
                    psis_t[out_i, :, :] = self.psis
                if store_energies and energies is not None:
                    energies[out_i, :] = Es
                if store_probabilities and probabilities is not None:
                    probabilities[out_i, :, :] = self.calculate_probabilities(
                        self.psis, evecs
                    )

                if monitor_probabilities is not None:
                    amps = self.psis @ evecs[:, monitor_idx].conj()
                    monitor_probabilities[out_i, :, :] = np.abs(amps) ** 2

            # Change V_ref
            V_ref = evecs

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = self.psis @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        probabilities_final = None
        if store_final_probabilities:
            probabilities_final = self.calculate_probabilities(self.psis, last_evecs)

        return (
            psis_t,
            energies,
            probabilities,
            probabilities_final,
            monitor_probabilities,
            monitor_probabilities_final,
            V_ref_ini,
            V_ref,
        )

    def _time_evolve_mu(
        self,
        H_slow_t: Callable,
        H_mu_t: Callable,
        D_mu: np.ndarray,
        t_array: np.ndarray,
        save_idx: Optional[np.ndarray],
        *,
        store_psis: bool,
        store_energies: bool,
        store_probabilities: bool,
        store_final_probabilities: bool,
        progress: bool,
        monitor_states: Optional[List[centrex_tlf.states.UncoupledState]],
        store_monitor_probabilities: bool,
        store_final_monitor_probabilities: bool,
        eig_backend: str,
    ):
        """
        Time evolves the system using the Hamiltonian function H_t
        over the time period in t_array.
        """
        if save_idx is None:
            save_idx = np.arange(len(t_array), dtype=int)

        # Calculate Hamiltonian at tini
        H_tini = H_slow_t(t_array[0])

        # Reference eigen-decomposition at t0 (reused for init + storage)
        E_ref, V_ref = np.linalg.eigh(H_tini)
        index = np.argsort(E_ref)
        E_ref = E_ref[index]
        V_ref = V_ref[:, index]
        V_ref_ini = V_ref

        monitor_idx: Optional[np.ndarray] = None
        if monitor_states:
            monitor_idx = np.array(
                [
                    find_max_overlap_idx(s.state_vector(self.hamiltonian.QN), V_ref_ini)
                    for s in monitor_states
                ],
                dtype=int,
            )

        # Initialize state vectors
        self.init_state_vecs(H_tini, V_0=V_ref)

        # Initialize containers to store results (possibly downsampled)
        psis_t, energies, probabilities = self._init_results_containers(
            t_array[save_idx],
            H_tini,
            store_psis=store_psis,
            store_energies=store_energies,
            store_probabilities=store_probabilities,
            D_0=E_ref,
            V_0=V_ref,
        )

        # Initialize reference matrix of eigenvectors that is used to keep track
        # of adiabatic evolution of eigenstates

        # Loop over t_array to time-evolve
        monitor_probabilities = None
        if store_monitor_probabilities and (monitor_idx is not None):
            monitor_probabilities = np.zeros(
                (len(save_idx), len(self.initial_states), monitor_idx.size)
            )
            amps0 = self.psis @ V_ref_ini[:, monitor_idx].conj()
            monitor_probabilities[0, :, :] = np.abs(amps0) ** 2

        out_i = 0
        last_evecs = V_ref
        for i, t in enumerate(tqdm(t_array[:-1], disable=not progress)):
            # Calculate the timestep
            dt = t_array[i + 1] - t_array[i]

            # Calculate Hamiltonians
            H_slow_i = H_slow_t(t)
            H_mu_i = H_mu_t(t)

            # Diagonalize slow Hamiltonian and transfer to basis where it is
            # diagonal
            if eig_backend == "zheevd":
                D, V, info = zheevd(H_slow_i)
                if info != 0:
                    D, V = np.linalg.eigh(H_slow_i)
            elif eig_backend == "numpy":
                D, V = np.linalg.eigh(H_slow_i)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Sort the eigenvalues so they are in ascending order
            # index = np.argsort(D)
            # D = D[index]
            # V = V[:, index]

            # Build rotating-frame Hamiltonian in the slow-eigenbasis.
            # Since V diagonalizes H_slow_i, we have V^H H_slow_i V = diag(D),
            # so we only need to rotate the microwave part.
            H_rot = V.conj().T @ H_mu_i @ V
            H_rot = H_rot + D_mu
            H_rot[np.diag_indices_from(H_rot)] += D

            # Diagonalize the Hamiltonian in the rotating frame
            if eig_backend == "zheevd":
                D_rot, V_rot, info_rot = zheevd(H_rot)
                if info_rot != 0:
                    D_rot, V_rot = np.linalg.eigh(H_rot)
            elif eig_backend == "numpy":
                D_rot, V_rot = np.linalg.eigh(H_rot)
            else:
                raise ValueError("Unknown eig_backend. Expected 'zheevd' or 'numpy'.")

            # Reorder eigenvectors and energies
            Es, evecs = D, V
            Es, evecs = reorder_evecs(evecs, Es, V_ref)
            last_evecs = evecs

            # Compute the propagator
            # Combine the unitary matrices
            A = V @ V_rot

            # Apply propagator without forming U_dt (same trick as above)
            phases_rot = np.exp(-1j * D_rot * dt)
            tmp = self.psis @ A.conj()
            tmp *= phases_rot[np.newaxis, :]
            self.psis = tmp @ A.T

            # Store results for this timestep if requested
            if (i + 1) == save_idx[out_i + 1]:
                out_i += 1
                if store_psis and psis_t is not None:
                    psis_t[out_i, :, :] = self.psis
                if store_energies and energies is not None:
                    energies[out_i, :] = Es
                if store_probabilities and probabilities is not None:
                    probabilities[out_i, :, :] = self.calculate_probabilities(
                        self.psis, evecs
                    )

                if monitor_probabilities is not None:
                    amps = self.psis @ evecs[:, monitor_idx].conj()
                    monitor_probabilities[out_i, :, :] = np.abs(amps) ** 2

            # Change V_ref
            V_ref = evecs

        probabilities_final = None
        if store_final_probabilities:
            probabilities_final = self.calculate_probabilities(self.psis, last_evecs)

        monitor_probabilities_final = None
        if store_final_monitor_probabilities and (monitor_idx is not None):
            amps = self.psis @ last_evecs[:, monitor_idx].conj()
            monitor_probabilities_final = np.abs(amps) ** 2

        return (
            psis_t,
            energies,
            probabilities,
            probabilities_final,
            monitor_probabilities,
            monitor_probabilities_final,
            V_ref_ini,
            V_ref,
        )

    def _init_results_containers(
        self,
        t_array: np.ndarray,
        H_tini: np.ndarray,
        *,
        store_psis: bool,
        store_energies: bool,
        store_probabilities: bool,
        D_0: Optional[np.ndarray] = None,
        V_0: Optional[np.ndarray] = None,
    ):
        """
        Initializes containers for time evolution results based on array of times
        and Hamiltonian at initial time.
        """
        psis_t: Optional[np.ndarray] = None
        energies: Optional[np.ndarray] = None
        probabilities: Optional[np.ndarray] = None

        if store_psis:
            psis_t = np.zeros(
                (len(t_array), len(self.initial_states), len(self.hamiltonian.QN)),
                dtype="complex",
            )
            psis_t[0, :, :] = self.psis

        if store_energies:
            energies = np.zeros((len(t_array), len(self.hamiltonian.QN)))

        if store_probabilities:
            # probabilities has same shape as psis_t would have
            probabilities = np.zeros(
                (len(t_array), len(self.initial_states), len(self.hamiltonian.QN))
            )

        if store_energies or store_probabilities:
            D = D_0
            V = V_0
            if D is None or V is None:
                D, V = np.linalg.eigh(H_tini)

            if store_energies and energies is not None:
                energies[0, :] = D
            if store_probabilities and probabilities is not None:
                probabilities[0, :, :] = self.calculate_probabilities(self.psis, V)

        return psis_t, energies, probabilities

    def calculate_probabilities(self, psis: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Given state vectors as columns of psi, for each state vector, returns the
        probabilities of being in states stored as columns of V.
        """
        # overlaps_list = []
        # for psi in psis:
        #     overlaps_list.append(V.conj().T @ psi)
        # overlaps = np.array(overlaps_list)

        overlaps = psis @ V.conj()

        return np.abs(overlaps) ** 2
