import numpy as np

from qiskit.quantum_info import (
    PauliList,
    Pauli,
    SparsePauliOp,
    random_clifford,
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

    ### Compute error margin
