import numpy as np

from qiskit.quantum_info import (
    PauliList,
    Pauli,
    Clifford,
    SparsePauliOp,
    random_clifford,
)
from qiskit.circuit import QuantumCircuit
from classical_shadow.base_shadow import BaseClassicalShadow

from utils import run_circuit_once


class GlobalClassicalShadow(BaseClassicalShadow):
    """
    This is an implementation of Global classical shadow using stabilizers.
    """

    def __init__(
        self,
        num_snapshots=None,
        error_margin=0.05,
        precision=0.05,
        num_active_qubits=5,
        num_observable=5,
    ):
        super().__init__(
            num_snapshots, error_margin, precision, num_active_qubits, num_observable
        )

        self.clifford_measures = list()

    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ) -> bool:
        """
        This does it
        """
        measures = []
        for _ in range(self.n_snapshots):

            cliff = random_clifford(self.num_qubits)
            self.clifford_measures.append(cliff)

            circuit = quantum_state.copy()
            circuit.compose(cliff.to_circuit(), inplace=True)
            circuit.measure_all()

            measures.append(list(run_circuit_once(circuit)))

        self.measures = np.array(measures)

        return True

    def estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
        """
        Ok
        """
        transformed_paulis = PauliList(
            [pauli.evolve(cliff) for cliff in self.clifford_measures]
        )
        phases = (-1j) ** transformed_paulis.phase

        eigenvalues = phases * (-1) ** np.mod(
            np.einsum("ij,ij->i", self.measures, transformed_paulis.z), 2
        )
        eigenvalues[transformed_paulis.x.any(axis=1)] = 0

        scores = (2**transformed_paulis.num_qubits + 1) * eigenvalues

        blocs = np.array_split(scores, self.NUM_BLOC)
        moyennes_blocs = [np.mean(b) for b in blocs]

        return np.median(moyennes_blocs)
