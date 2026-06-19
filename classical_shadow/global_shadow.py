import numpy as np

from qiskit.quantum_info import (
    PauliList,
    Pauli,
    SparsePauliOp,
    random_clifford,
    Clifford,
)
from qiskit.circuit import QuantumCircuit
from classical_shadow.base_shadow import BaseClassicalShadow


class GlobalClassicalShadow(BaseClassicalShadow):
    """
    Global Classical Shadow implementation using random Clifford unitaries.

    In this scheme, a uniformly random global Clifford unitary is applied to
    the full quantum state at each snapshot before measuring in the
    computational basis. Because the unitary acts on all qubits simultaneously,
    this approach can efficiently estimate observables with arbitrary support,
    at the cost of a sample complexity that scales with the Frobenius norm of
    the observable rather than its locality.

    The expectation value of a Pauli operator is recovered by conjugating the
    Pauli through each snapshot's Clifford (via the stabilizer formalism) and
    comparing the resulting stabilizer eigenvalue with the observed bitstring.

    Attributes:
        measures_clifford (list[Clifford]): List of random Clifford operators
            applied at each snapshot, in the same order as ``self.measures``.
            Populated by :meth:`fit_shadow`.
    """

    def __init__(self, num_snapshots, method: str = "perfect"):
        """
        Initializes the Global Classical Shadow.

        Args:
            num_snapshots (int): Number of snapshots (random Clifford
                measurements) to perform when building the shadow.
            method (str): Simulation method passed to the parent class,
                controlling how circuits are executed (e.g. ``"perfect"`` for
                noiseless simulation). Defaults to ``"perfect"``.
        """
        super().__init__(num_snapshots, method)

        self.measures_clifford = list()

    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ):
        """
        Builds the global classical shadow of a quantum state.

        For each snapshot, a uniformly random Clifford unitary is sampled,
        appended to the quantum state circuit, and the resulting circuit is
        measured in the computational basis. The Clifford operators and the
        measurement bitstrings are stored in ``self.measures_clifford`` and
        ``self.measures`` respectively.

        Args:
            quantum_state (QuantumCircuit): Circuit representing the quantum
                state to shadow. Must not include measurements (they are added
                internally).
            observable (SparsePauliOp | None): Unused in this implementation.
                Present for compatibility with the base class interface.
                Defaults to None.
        """
        self.num_qubits = quantum_state.num_qubits

        circuits = []
        for _ in range(self.n_snapshots):

            cliff = random_clifford(self.num_qubits)
            self.measures_clifford.append(cliff)

            circuit = quantum_state.copy()
            circuit.compose(cliff.to_circuit(), inplace=True)
            circuit.measure_all()

            circuits.append(circuit)

        self.measures = self._run_circuits(circuits)

    def _estimate_pauli_expectation_value(self, pauli: Pauli) -> complex:
        """
        Estimates the expectation value of a single Pauli operator from the
        global classical shadow.

        For each snapshot, the Pauli is conjugated through the corresponding
        Clifford unitary via the stabilizer formalism (``pauli.evolve(cliff)``).
        If the resulting operator is a pure Z-type Pauli (no X component), its
        eigenvalue is read off the measurement bitstring using the phase and the
        parity of the measured bits. Snapshots that yield an X component after
        conjugation contribute zero. The per-snapshot scores are scaled by
        ``2^n + 1`` (where ``n`` is the number of qubits) to match the inverse
        channel of the global shadow, and the final estimate is obtained via
        Median of Means aggregation.

        Args:
            pauli (Pauli): Single Pauli operator whose expectation value is to
                be estimated (e.g. ``Pauli("XYZ")``).

        Returns:
            complex: Estimated expectation value of the Pauli operator.
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
        Estimates the expectation value of a global observable from the shadow.

        Unlike :meth:`~LocalClassicalShadow.estimate_local_observable`, no
        locality constraint is enforced here: the global Clifford scheme can
        handle observables with arbitrary qubit support.

        Args:
            observable (SparsePauliOp): Observable expressed as a sparse linear
                combination of Pauli operators.

        Returns:
            complex: Estimated expectation value of the observable.
        """
        return self._estimate_observable(observable)

    @classmethod
    def recover_shadow(cls, dir: str) -> "GlobalClassicalShadow":
        """
        Restores a global shadow from a previously saved ``.npz`` file.

        Instantiates a new :class:`GlobalClassicalShadow`, then loads
        ``measures`` and the symplectic tableau representation of each
        Clifford operator from ``dir``. The ``Clifford`` objects are
        reconstructed from their tableaux. ``num_qubits`` and ``n_snapshots``
        are inferred from the shape of ``measures``.

        Args:
            dir (str): Path to the ``.npz`` file produced by
                :meth:`save_shadow` (e.g. ``"shadows/global_shadow.npz"``).

        Returns:
            GlobalClassicalShadow: A new instance with its attributes restored,
                ready for observable estimation.

        Raises:
            FileNotFoundError: If no file exists at ``dir``.
        """
        from qiskit.quantum_info import Clifford

        instance = cls(num_snapshots=0)
        data = np.load(dir, allow_pickle=False)
        instance.measures = data["measures"]
        instance.num_qubits = instance.measures.shape[1]
        instance.n_snapshots = instance.measures.shape[0]

        tableaux = data["clifford_tableaux"]
        instance.measures_clifford = [
            Clifford(tableaux[i]) for i in range(len(tableaux))
        ]

        return instance

    def save_shadow(self, dir: str) -> None:
        """
        Persists the global shadow's core attributes to a compressed ``.npz``
        file.

        Saves ``measures`` (the measurement bitstrings) and the symplectic
        tableau representation of each Clifford unitary. The tableaux allow
        full reconstruction of the ``Clifford`` objects upon loading via
        :meth:`recover_shadow`, without relying on Python pickling.

        Args:
            dir (str): Destination path for the ``.npz`` file
                (e.g. ``"shadows/global_shadow.npz"``). NumPy appends
                ``.npz`` automatically if the extension is omitted.

        Raises:
            ValueError: If :meth:`fit_shadow` has not been called yet and
                ``self.measures`` or ``self.measures_clifford`` is empty.
        """
        if self.measures is None or not self.measures_clifford:
            raise ValueError(
                "Shadow has not been built yet. Call fit_shadow() before saving."
            )

        # Serialize each Clifford as its boolean symplectic tableau
        tableaux = np.array(
            [cliff.tableau.astype(bool) for cliff in self.measures_clifford]
        )
        np.savez_compressed(dir, measures=self.measures, clifford_tableaux=tableaux)
