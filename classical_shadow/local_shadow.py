import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from classical_shadow.base_shadow import BaseClassicalShadow
from classical_shadow.utils import run_circuit_once


class LocalClassicalShadow(BaseClassicalShadow):
    """
    This is an implementation of local classical shadows.
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
        self.measures_basis: np.ndarray | None = None
        self.num_qubits: int | None = None

    @property
    def n_snapshots(self) -> int | None:
        return self._n_snapshots

    @n_snapshots.setter
    def n_snapshots(self, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("n_snapshots must be positive")
        self._n_snapshots = value

    def _compute_num_snapshots(self):
        """
        Computes the number of snapshot needed if number of active qubits (k) and observable (M) is 3 based on the formula:
            N = 3**(k+1)/ error_margin**2 * log2(2M/precision)
        """
        N = np.ceil(
            (3 ** (self.num_active_qubits + 1))
            * np.log2(2 * self.num_observables / self.precision)
            / self.error_margin**2
        )
        self.n_snapshots = int(N - (N % self.NUM_BLOC))

    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ) -> bool:
        """
        This fonction uses Shadow Local to produce the shadow of a quantum state.
        """
        if observable is not None:
            if isinstance(observable, SparsePauliOp):
                self.num_observables = len(observable.paulis)

                self.num_active_qubits = np.max(
                    np.sum(
                        np.logical_or(observable.paulis.x, observable.paulis.z), axis=-1
                    )
                )

        if self.n_snapshots is None:

            self._compute_num_snapshots()

        self.num_qubits = quantum_state.num_qubits

        if self.n_snapshots > 50000:
            yesno = input(
                f"You are about to make a shadow of {self.n_snapshots} snapshots, do you want to proceed? (Y/N)"
            )
            if "N" in yesno:
                return False

        bases = np.random.choice(
            ["X", "Y", "Z"], size=(self.n_snapshots, quantum_state.num_qubits)
        )
        pauli_strings = np.array(["".join(row) for row in bases])
        paulis = PauliList(pauli_strings)

        measures = []
        for pauli in paulis:

            circuit = quantum_state.copy()

            where_y = np.nonzero(np.logical_and(pauli.x, pauli.z))[0]
            where_x = np.nonzero(pauli.x)[0]
            if len(where_y) > 0:
                circuit.sdg(where_y)
            if len(where_x) > 0:
                circuit.h(where_x)
            circuit.measure_all()

            measures.append(list(run_circuit_once(circuit)))

        self.measures_basis = bases
        self.measures = np.array(measures)

        return True

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

    def estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
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

        # Median-of-Means
        # Comment calculer le nombre de blocs? ca doit dependre du nombre de snapshots
        blocs = np.array_split(scores, self.NUM_BLOC)
        moyennes_blocs = [np.mean(b) for b in blocs]

        return np.median(moyennes_blocs)
