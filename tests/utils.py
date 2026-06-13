import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def classicaly_compute_estimation_value(
    observable: SparsePauliOp, state_vector: QuantumCircuit
) -> np.complex128:
    """
    Computes classicaly the estimation value by sandwitching the observable between the statevector.
    """
    statevector = Statevector(state_vector).data
    matrix_observable = observable.to_matrix()
    real_expect_value = np.einsum(
        "i,ij,j", statevector.conj(), matrix_observable, statevector
    )
    return real_expect_value
