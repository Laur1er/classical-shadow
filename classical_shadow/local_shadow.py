import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from classical_shadow.base_shadow import BaseClassicalShadow


class LocalClassicalShadow(BaseClassicalShadow):
    """
    Local Classical Shadow implementation.

    In this scheme, random unitaries are drawn independently and uniformly
    from the single-qubit Pauli bases {X, Y, Z} for each qubit at each
    snapshot. This locality makes the estimator particularly efficient for
    observables that act non-trivially on a small number of qubits (local
    observables), as the sample complexity scales with the locality of the
    observable rather than the total number of qubits.

    Attributes:
        measures_basis (np.ndarray): 2-D array of shape
            ``(n_snapshots, num_qubits)`` storing the randomly chosen
            measurement basis ('X', 'Y', or 'Z') for each qubit at each
            snapshot. Populated by :meth:`fit_shadow`.
    """

    def __init__(self, nb_snapshots: int, method: str = "perfect"):
        """
        Initializes the Local Classical Shadow.

        Args:
            nb_snapshots (int): Number of snapshots (random measurements) to
                perform when building the shadow.
            method (str): Simulation method passed to the parent class,
                controlling how circuits are executed (e.g. ``"perfect"`` for
                noiseless simulation). Defaults to ``"perfect"``.
        """
        super().__init__(nb_snapshots, method)

        self.measures_basis = np.ndarray

    def fit_shadow(
        self, quantum_state: QuantumCircuit, observable: SparsePauliOp | None = None
    ):
        """
        Builds the local classical shadow of a quantum state.

        For each snapshot, a random single-qubit Pauli basis is independently
        sampled for every qubit. The state is then rotated into that basis and
        measured in the computational basis. The chosen bases and the resulting
        bitstrings are stored in ``self.measures_basis`` and ``self.measures``
        respectively.

        If the number of requested snapshots exceeds 50 000, the user is
        prompted for confirmation before proceeding.

        Args:
            quantum_state (QuantumCircuit): Circuit representing the quantum
                state to shadow. Must not include measurements (they are added
                internally).
            observable (SparsePauliOp | None): Unused in this implementation.
                Present for compatibility with the base class interface.
                Defaults to None.

        Raises:
            TimeoutError: If ``n_snapshots`` exceeds 50 000 and the user
                declines to proceed.
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
        Estimates the expectation value of a single Pauli operator from the
        local classical shadow.

        Only the snapshots whose randomly chosen basis matches the non-identity
        support of ``pauli`` on every active qubit are used. For those
        matching snapshots, the eigenvalue contribution is computed as
        ``3^k * (-1)^(sum of measurement bits)``, where ``k`` is the number of
        active (non-identity) qubits. The final estimate is obtained via the
        Median of Means aggregation over all snapshots.

        Args:
            pauli (Pauli): Single Pauli operator whose expectation value is to
                be estimated (e.g. ``Pauli("XYZ")``).

        Returns:
            complex: Estimated expectation value of the Pauli operator.
        """

        pos_active_qubits = (
            pauli.num_qubits - 1 - np.nonzero(np.logical_or(pauli.x, pauli.z))[0]
        )
        base_active_qubits = np.array(list(pauli.to_label()))[pos_active_qubits]
        num_active_qubits = len(base_active_qubits)

        coresponding_snapshots = np.where(
            np.all(
                self.measures_basis[:, pos_active_qubits] == base_active_qubits, axis=1
            )
        )[0]

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
        Estimates the expectation value of a local observable from the shadow.

        Before estimation, validates that the observable acts non-trivially on
        at most 4 qubits (i.e. has locality ≤ 4). If this condition is not
        met, a ``ValueError`` is raised.

        Args:
            observable (SparsePauliOp): Local observable expressed as a sparse
                linear combination of Pauli operators.

        Returns:
            complex: Estimated expectation value of the observable.

        Raises:
            ValueError: If any Pauli term in ``observable`` acts on more than
                4 qubits non-trivially.
        """
        if (
            np.sum(
                np.logical_or(observable.paulis.x, observable.paulis.z).astype(int),
                axis=-1,
            )
            > 4
        ):
            raise ValueError(f"Observable {observable} is not a local observable.")

        return self._estimate_observable(observable)

    @classmethod
    def recover_shadow(cls, dir: str) -> "LocalClassicalShadow":
        """
        Restores a local shadow from a previously saved ``.npz`` file.

        Instantiates a new :class:`LocalClassicalShadow`, then loads
        ``measures`` and ``measures_basis`` from the file at ``dir``.
        ``num_qubits`` and ``n_snapshots`` are inferred directly from the
        shape of ``measures``.

        Args:
            dir (str): Path to the ``.npz`` file produced by
                :meth:`save_shadow` (e.g. ``"shadows/local_shadow.npz"``).

        Returns:
            LocalClassicalShadow: A new instance with its attributes restored,
                ready for observable estimation.

        Raises:
            FileNotFoundError: If no file exists at ``dir``.
        """
        instance = cls(nb_snapshots=0)
        data = np.load(dir, allow_pickle=False)
        instance.measures = data["measures"]
        instance.measures_basis = data["measures_basis"]
        instance.num_qubits = instance.measures.shape[1]
        instance.n_snapshots = instance.measures.shape[0]
        return instance

    def save_shadow(self, dir: str) -> None:
        """
        Persists the local shadow's core attributes to a compressed ``.npz``
        file.

        Saves ``measures`` (the measurement bitstrings) and ``measures_basis``
        (the per-qubit Pauli bases) to ``dir``. These two arrays are the
        minimum required to fully restore the shadow via
        :meth:`recover_shadow`.

        Args:
            dir (str): Destination path for the ``.npz`` file
                (e.g. ``"shadows/local_shadow.npz"``). NumPy appends
                ``.npz`` automatically if the extension is omitted.

        Raises:
            ValueError: If :meth:`fit_shadow` has not been called yet and
                ``self.measures`` or ``self.measures_basis`` is None.
        """
        if self.measures is None or not isinstance(self.measures_basis, np.ndarray):
            raise ValueError(
                "Shadow has not been built yet. Call fit_shadow() before saving."
            )
        np.savez_compressed(
            dir, measures=self.measures, measures_basis=self.measures_basis
        )
