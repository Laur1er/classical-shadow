from abc import ABC, abstractmethod
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli


class BaseClassicalShadow(ABC):
    """
    OK
    """

    NUM_BLOC = 10

    def __init__(
        self,
        num_snapshots: int | None = None,
        error_margin: float | None = 0.05,
        precision: float | None = 0.05,
        num_active_qubits: int = 5,
        num_observable: int = 5,
    ) -> None:

        self.num_active_qubits = num_active_qubits
        self.num_observables = num_observable
        self.n_snapshots = num_snapshots

        if not (0 < error_margin < 1):
            raise ValueError(
                f"error_margin must stand between 0 et 1, given : {error_margin}"
            )
        self.error_margin = error_margin

        if not (0 < precision < 1):
            raise ValueError(
                f"precision must stand between 0 et 1, given : {precision}"
            )
        self.precision = precision

        self.measures: np.ndarray | None = None
        self.num_qubits: int | None = None

    def estimate_observable(self, observable: SparsePauliOp) -> complex:
        """
        Estimate an observable given. The shadow must already be computated using the method fit_shadow().
        """
        if self.measures is None:
            raise ValueError(
                "Call fit_shadow(QuantumCircuit) before the estimation to create the local classical shadow."
            )

        if self.num_qubits != observable.num_qubits:
            raise ValueError(
                "The number of qubits in the observable is not the same as the number of qubits in the shadow."
            )

        estimation_value = 0
        for i, pauli in enumerate(observable.paulis):

            estimation_value += observable.coeffs[
                i
            ] * self.estimate_pauli_expectation_value(pauli)

        return estimation_value

    @abstractmethod
    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ) -> bool:
        """
        Make the shadow.
        """
        raise NotImplementedError("Fonction fit_shadow not implemented. Implement it.")

    @abstractmethod
    def estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
        """
        Estimate an observable
        """
        raise NotImplementedError(
            "Fonction estimate_observable not implemented. Implement it."
        )
