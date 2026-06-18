import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from classical_shadow.global_shadow import GlobalClassicalShadow
from tests.utils import classicaly_compute_estimation_value


@pytest.fixture(scope="module")
def global_Shadow():

    global_circuit = QuantumCircuit(8)
    global_circuit.h(0)

    for i in range(7):
        global_circuit.cx(i, i + 1)

    global_circuit.h([0, 3, 6])
    global_circuit.s([1, 2, 3])
    global_circuit.y([6, 7])

    shadow = GlobalClassicalShadow(num_snapshots=25000)
    shadow.fit_shadow(global_circuit)

    return global_circuit, shadow


@pytest.mark.parametrize(
    "observable",
    [
        SparsePauliOp(["IXXXZZYY", "ZZZZZYYY"], [0.3j, -1]),
        SparsePauliOp(["XIXXXXXX", "XXXXXXXX"], [1, 0.5]),
        SparsePauliOp(["ZIZZZZZZ", "ZZXXXXZX"], [0.7, -0.5j]),
        SparsePauliOp(["ZXYZXYZI", "ZIIYYYYZ"], [0.7, -0.5]),
    ],
)
def test_global_state(global_Shadow, observable: SparsePauliOp):

    global_circuit, shadow = global_Shadow
    expect_value = shadow.estimate_global_observable(observable)
    real_expect_value = classicaly_compute_estimation_value(observable, global_circuit)

    assert np.linalg.norm(real_expect_value - expect_value) < 0.1
