import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from classical_shadow.local_shadow import LocalClassicalShadow


def test_bell_state():

    observable = SparsePauliOp(["XI", "XX"], [1, 0.5])

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)

    # Classicaly
    statevector = Statevector(bell).data
    matrix_observable = observable.to_matrix()
    real_expect_value = np.einsum(
        "i,ij,j", statevector.conj(), matrix_observable, statevector
    )

    # With classical shadow
    shadow = LocalClassicalShadow()
    shadow.collect_data(bell, 10000)

    expect_value = shadow.estimate_observable(observable)

    assert np.allclose(real_expect_value, expect_value)
