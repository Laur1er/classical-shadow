import numpy as np
from numpy import ndarray

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from classical_shadow.utils import run_circuit_once


class LocalClassicalShadow:
    """
    This is an implementation of local classical shadows.
    """

    # Attributs principaux de la classical shadow Local
    measures: ndarray
    measures_basis: ndarray

    # Attributs de precision
    n_snapshots: int

    PRECISION = 0.05
    FAILIURE_TOLARATED = 0.05

    def __init__(self):

        pass

    def collect_data(self, quantum_state: QuantumCircuit, n_snapshots) -> bool:
        """
        This fonction uses Shadow Local to produce the shadow of a quantum state.
        """
        # Choose pauli strings randomly
        bases = np.random.choice(
            ["X", "Y", "Z"], size=(n_snapshots, quantum_state.num_qubits)
        )
        pauli_strings = np.array(["".join(row) for row in bases])
        paulis = PauliList(pauli_strings)

        measures = []
        for pauli in paulis:

            circuit = quantum_state.copy()

            # Diagonalize the pauli
            where_y = np.nonzero(np.logical_and(pauli.x, pauli.z))[0]
            where_x = np.nonzero(pauli.x)[0]
            if len(where_y) > 0:
                circuit.sdg(where_y)
            if len(where_x) > 0:
                circuit.h(where_x)
            circuit.measure_all()

            # Simulate and store the result in measures
            measures.append(list(run_circuit_once(circuit)))

        self.measures_basis = bases
        self.measures = np.array(measures)

        return True

    def estimate_observable(self, observable: SparsePauliOp):
        """
        Estimate an observable given.
        """
        estimation_value = 0

        for i, pauli in enumerate(observable.paulis):
            estimation_value += observable.coeffs[
                i
            ] * self.estimate_pauli_expectation_value(pauli)

        return estimation_value

    def estimate_pauli_expectation_value(self, pauli: Pauli):
        """
        Fonction to predict expectation value of a single Pauli operator.
        """
        # Sortir les qubits actif et leur base
        pos_active_qubits = (
            pauli.num_qubits - 1 - np.nonzero(np.logical_or(pauli.x, pauli.z))[0]
        )
        base_active_qubits = np.array(list(pauli.to_label()))[pos_active_qubits]
        num_active_qubits = len(base_active_qubits)

        # Pour chaque snapshot, on doit verifier si les bases concordent.
        coresponding_snapshots = np.where(
            np.all(
                self.measures_basis[:, pos_active_qubits] == base_active_qubits, axis=1
            )
        )[0]

        # Calcule de la valeur propres
        scores = np.zeros(self.measures.shape[0])
        scores[coresponding_snapshots] += 3**num_active_qubits

        scores[coresponding_snapshots] *= (-1) ** np.mod(
            np.sum(
                self.measures[np.ix_(coresponding_snapshots, pos_active_qubits)].astype(
                    np.int32
                ),
                axis=1,
                dtype=np.int32,
            ),
            2,
        )

        # Median-of-Means
        # Comment calculer le nombre de blocs? ca doit dependre du nombre de snapshots
        num_blocs = 10
        blocs = np.array_split(scores, num_blocs)
        moyennes_blocs = [np.mean(b) for b in blocs]

        return np.median(moyennes_blocs)
