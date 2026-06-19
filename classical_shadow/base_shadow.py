from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

from classical_shadow.utils import get_sampler


class BaseClassicalShadow(ABC):
    """
    Abstract base class for Classical Shadow implementations.

    Classical Shadows is an efficient quantum tomography technique that allows estimating expectation values of observables
    from a reduced number of random measurements of a quantum state.

    The general procedure consists of:

        1. Applying random unitaries to the quantum state.
        2. Measuring in the computational basis.
        3. Building a classical "shadow" from the measurement outcomes.

    Subclasses must implement:

        - :meth:`fit_shadow` : construction of the classical shadow.
        - :meth:`estimate_pauli_expectation_value` : estimation of a Pauli operator.

    Attributes:

        method: How the sampling should be performed "perfect", "noisy", or "real_hardware"
        n_snapshots (int): Number of snapshots (measurements) to perform.
        measures (np.ndarray): Array of recorded measurements after calling :meth:`fit_shadow`. Is None before that call.

    """

    def __init__(self, nb_snapshots: int, method: str = "perfect") -> None:
        """
        Initializes the parameters common to all Classical Shadow implementations.

        Args:

            num_snapshots (int | None): Number of snapshots to perform.
            method:      "perfect", "noisy", or "real_hardware".

        """
        self.sampler, self.pass_manager = get_sampler(method)

        if nb_snapshots < 0:
            raise ValueError("Number of snapshots must be greater than 0.")
        else:
            self.n_snapshots = nb_snapshots

        self.method = method
        self.measures = np.ndarray

    def _median_of_mean(self, scores: np.ndarray, nb_blocs: int = 20) -> complex:
        """
        Computes the Median of Means (MoM) estimator of an array of scores.

        The array is split into ``nb_blocs`` equally-sized blocks, the mean of
        each block is computed, and the median of those block means is returned.
        This estimator is more robust to outliers than a plain mean, and provides
        high-confidence bounds on the estimation error.

        Args:

            scores (np.ndarray): 1-D array of scalar scores to aggregate.
            nb_blocs (int): Number of blocks to split ``scores`` into. A higher
                value increases robustness but reduces the number of samples per
                block. Defaults to 20.

        Returns:
            complex: Median of the per-block means.
        """
        blocs = np.array_split(scores, nb_blocs)
        moyennes_blocs = [np.mean(bloc) for bloc in blocs]

        return np.median(moyennes_blocs)

    def _estimate_observable(self, observable: SparsePauliOp) -> complex:
        """
        Estimates the expectation value of a given observable from the
        already-constructed classical shadow.

        The observable is decomposed into a weighted sum of Pauli operators.
        The expectation value of each Pauli term is estimated separately via
        :meth:`_estimate_pauli_expectation_value`, then the contributions are
        summed taking the coefficients into account.

        Args:
            observable (SparsePauliOp): Quantum observable expressed as a
                sparse linear combination of Pauli operators.

        Returns:
            complex: Estimated expectation value of the observable.

        Raises:
            ValueError: If :meth:`fit_shadow` has not been called yet
                (i.e. ``self.measures`` is None).
            ValueError: If the number of qubits in the observable does not
                match the number of qubits in the shadow.
        """
        if self.measures is None:
            raise ValueError(
                "Call fit_shadow(QuantumCircuit) before the estimation to create the local classical shadow."
            )

        if self.measures.shape[1] != observable.num_qubits:
            raise ValueError(
                "The number of qubits in the observable is not the same as the number of qubits in the shadow."
            )

        estimation_value = 0
        for i, pauli in enumerate(observable.paulis):

            estimation_value += observable.coeffs[
                i
            ] * self._estimate_pauli_expectation_value(pauli)

        return estimation_value

    def _run_circuits(
        self,
        circuits: list[QuantumCircuit],
        batch_size: int = 300,
    ) -> np.ndarray:
        """
        Execute a list of circuits (1 shot each) and returns the bitstrings.

        Args:
            circuits:    List of QuantumCircuit, each with a classical register "meas".
            batch_size:  Amount max of circuits per job (limit IBM = ~300).

        Returns:
            np.ndarray of shape (N, num_qubits) containing the resulting bitstrings, ex: [["0","1","0"], ["1","1,","0"], [...]...]
        """

        if self.pass_manager is not None:
            circuits = self.pass_manager.run(circuits)

        all_bitstrings = []

        for i in tqdm(
            range(0, len(circuits), batch_size),
            f"Running circuit using {self.method} backend.",
        ):
            batch = circuits[i : i + batch_size]
            job = self.sampler.run(batch, shots=1)
            result = job.result()

            for j in range(len(batch)):
                bitstring = list(result[j].data.meas.get_bitstrings()[0])
                all_bitstrings.append(bitstring)

        return np.array(all_bitstrings).astype(int)

    @abstractmethod
    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ):
        """
        Builds the classical shadow from a quantum circuit.

        This method must be called before any observable estimation. It applies
        random unitaries, performs the measurements, and stores the results in
        ``self.measures`` as well as the qubit count in ``self.measures.shape[1]``.

        Args:

            quantum_state (QuantumCircuit): Quantum circuit representing the quantum state from which the shadow is constructed.
            observable (SparsePauliOp | None): Optional observable that may guide the choice of random unitaries in certain implementations
                (e.g. adaptive shadows). Defaults to None.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Fonction fit_shadow needs to be implemented.")

    @abstractmethod
    def _estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
        """
        Estimates the expectation value of a Pauli operator from the classical
        shadow.

        This method is called by :meth:`estimate_observable` for each Pauli
        term in the observable decomposition. It forms the core of the
        estimation computation and must be adapted to the sampling strategy of
        each implementation.

        Args:
            pauli (Pauli): Pauli operator whose expectation value is to be
                estimated (e.g. ``Pauli("XYZ")``).

        Returns:
            complex: Estimated expectation value of the Pauli operator.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Fonction estimate_observable not implemented.")

    @classmethod
    @abstractmethod
    def recover_shadow(cls, dir: str) -> "BaseClassicalShadow":
        """
        Restores the shadow state from a previously saved file.

        Loads the shadow's core attributes from the file located at ``dir``
        and returns a fully restored instance, ready for observable estimation
        without needing to re-run :meth:`fit_shadow`.

        Args:
            dir (str): Path to the file from which the shadow is loaded
                (e.g. ``"shadows/my_shadow.npz"``).

        Returns:
            BaseClassicalShadow: A new instance with its attributes restored.

        Raises:
            FileNotFoundError: If no file exists at ``dir``.
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Fonction recover_shadow not implemented.")

    @abstractmethod
    def save_shadow(self, dir: str) -> None:
        """
        Persists the shadow's core attributes to disk.

        Saves the minimum set of attributes required to fully restore the
        shadow via :meth:`recover_shadow`, without needing to re-run
        :meth:`fit_shadow`.

        Args:
            dir (str): Path to the output file
                (e.g. ``"shadows/my_shadow.npz"``). Parent directories must
                already exist.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Fonction save_shadow not implemented.")
