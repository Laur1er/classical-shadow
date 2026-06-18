import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from classical_shadow.base_shadow import BaseClassicalShadow


class LocalClassicalShadow(BaseClassicalShadow):
    """
    This is an implementation of local classical shadows.
    """

    def __init__(self, nb_snapshots: int, method: str = "perfect"):
        super().__init__(nb_snapshots, method)

        self.measures_basis = np.ndarray

    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ):
        """
        This fonction uses Shadow Local to produce the shadow of a quantum state.
        """

        if self.n_snapshots > 50000:
            yesno = input(
                f"You are about to make a shadow of {self.n_snapshots} snapshots, do you want to proceed? (Y/N)"
            )
            if "N" in yesno:
                raise TimeoutError("Change amount of shots.")

        bases = np.random.choice(
            ["X", "Y", "Z"], size=(self.n_snapshots, quantum_state.num_qubits)
        )
        pauli_strings = np.array(["".join(row) for row in bases])
        paulis = PauliList(pauli_strings)

        circuits = []
        for pauli in paulis:

            circuit = quantum_state.copy()

            where_y = np.nonzero(np.logical_and(pauli.x, pauli.z))[0]
            where_x = np.nonzero(pauli.x)[0]
            if len(where_y) > 0:
                circuit.sdg(where_y)
            if len(where_x) > 0:
                circuit.h(where_x)
            circuit.measure_all()

            circuits.append(circuit)

        self.measures_basis = bases
        self.measures = self._run_circuits(circuits)

    def _estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
        """
        Fonction to predict expectation value of a single Pauli operator using the classical shadow.
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
                    int
                ),
                axis=1,
                dtype=np.int32,
            ),
            2,
        )

        return self._median_of_mean(scores)

    def estimate_local_observable(self, observable: SparsePauliOp) -> complex:
        """
        Verify if the observable is local
        """

        return self._estimate_observable(observable)
