import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector

from pauliarray import PauliArray


def test_bell_state():

    observable = PauliArray.from_labels(["XX"])

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)

    # Classicaly
    statevector = Statevector(bell).data
    matrix_observable = observable.to_matrices()
    real_expect_value = np.einsum(
        "i,ij,j", statevector.conj().T, matrix_observable, statevector
    )

    # With classical shadow
    shadow = ClassicalShadow(type="Local")
    shadow.from_circuit(bell)
    expect_value = shadow.estimate_observable(obs=observable)

    assert np.allclose(real_expect_value, expect_value)
