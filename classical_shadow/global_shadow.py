import numpy as np

from qiskit.quantum_info import (
    PauliList,
    Pauli,
    SparsePauliOp,
    random_clifford,
)
from qiskit.circuit import QuantumCircuit
from classical_shadow.base_shadow import BaseClassicalShadow

from classical_shadow.utils import run_circuit_once


class GlobalClassicalShadow(BaseClassicalShadow):
    """
    This is an implementation of Global classical shadow using stabilizers.
    """

    def __init__(self, num_snapshots):
        super().__init__(num_snapshots)

        self.measures_clifford = list()

    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ):
        """
        This does it
        """
        self.num_qubits = quantum_state.num_qubits

        measures = []
        for _ in range(self.n_snapshots):

            cliff = random_clifford(self.num_qubits)
            self.measures_clifford.append(cliff)

            circuit = quantum_state.copy()
            circuit.compose(cliff.to_circuit(), inplace=True)
            circuit.measure_all()

            measures.append(list(run_circuit_once(circuit)))

        self.measures = np.array(measures).astype(int)

    def _estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
        """
        Ok
        """
        transformed_paulis = PauliList(
            [pauli.evolve(cliff) for cliff in self.measures_clifford]
        )
        phases = (-1j) ** transformed_paulis.phase

        eigenvalues = phases * (-1) ** np.mod(
            np.einsum("ij,ij->i", self.measures, transformed_paulis.z), 2
        )
        eigenvalues[transformed_paulis.x.any(axis=1)] = 0

        scores = (2**transformed_paulis.num_qubits + 1) * eigenvalues

        return self._median_of_mean(scores)

    def estimate_global_observable(self, observable: SparsePauliOp) -> complex:
        """
        Verify if it is global
        """
        return self._estimate_observable(observable)

    ### Compute error margin

    def calculer_erreur_bootstrap(scores, K=10, num_resamples=500):
        N = len(scores)
        predictions_bootstrap = []

        for _ in range(num_resamples):

            scores_resampled = np.random.choice(scores, size=N, replace=True)

            # Calcul du MoM sur cet échantillon virtuel
            blocs = np.array_split(scores_resampled, K)
            moyennes_blocs = [np.mean(b) for b in blocs]
            predictions_bootstrap.append(np.median(moyennes_blocs))

        # Calcul de l'intervalle de confiance à 95%
        borne_inf = np.percentile(predictions_bootstrap, 2.5)
        borne_sup = np.percentile(predictions_bootstrap, 97.5)

        marge_erreur = (borne_sup - borne_inf) / 2
        return marge_erreur
