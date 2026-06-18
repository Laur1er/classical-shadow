import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from classical_shadow.local_shadow import LocalClassicalShadow


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


# Voici le test afin de regarder si tout fonctionne avec le real_hardware
bell = QuantumCircuit(2)

bell.h(0)
bell.cx(0, 1)

shadow = LocalClassicalShadow(nb_snapshots=600, method="real_hardware")
shadow.fit_shadow(bell)


# Calculons les
observable = SparsePauliOp(["XX", "XZ", "IX"], [0.5j, 1.4, -0.5])
estimation_value = shadow.estimate_local_observable(observable)
real_estimation_value = classicaly_compute_estimation_value(observable, bell)

print(f"Classiquement: {real_estimation_value}")
print(f"Quantiquement: {estimation_value}")
