import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from classical_shadow.local_shadow import LocalClassicalShadow
from tests.utils import classicaly_compute_estimation_value


@pytest.fixture(scope="module")
def bell_shadow():

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)

    shadow = LocalClassicalShadow()
    shadow.collect_data(bell, 10000)
    return bell, shadow


@pytest.mark.parametrize(
    "observable",
    [
        SparsePauliOp(["IY", "YY"], [0.3, -1]),
        SparsePauliOp(["XI", "XX"], [1, 0.5]),
        SparsePauliOp(["ZI", "ZX"], [0.7, -0.5]),
        SparsePauliOp(["ZI", "ZZ"], [0.7, -0.5]),
    ],
)
def test_bell_state(bell_shadow, observable: SparsePauliOp):

    bell, shadow = bell_shadow
    expect_value = shadow.estimate_observable(observable)
    real_expect_value = classicaly_compute_estimation_value(observable, bell)

    assert np.abs(real_expect_value - expect_value) < 0.1


### Voir avec circuit generer aleatoirement


@pytest.fixture(scope="module")
def weird_state():

    state = QuantumCircuit(10)
    state.h([3, 6, 1, 3])
    state.cx(6, 2)
    state.ccx(9, 3, 1)
    state.sdg([2, 5, 6, 4])
    state.y([7, 8])

    shadow = LocalClassicalShadow()
    shadow.collect_data(state, 20000)

    return state, shadow


@pytest.mark.parametrize(
    "local_observable",
    [
        SparsePauliOp(["IYIIIIIIII", "IIIYIIYIII"], [0.3, -1]),
        SparsePauliOp(["IIIIIXIIII", "IIIIIIXIII"], [1, -2]),
        SparsePauliOp(["ZIIIIIIIII", "IIIZIXIIII"], [-0.7, -0.5]),
        SparsePauliOp(["IYIIZIIIII", "IIIZIIIIIZ"], [0.3, 0.75]),
    ],
)
def test_random_state(weird_state, local_observable: SparsePauliOp):

    state, shadow = weird_state
    expect_value = shadow.estimate_observable(local_observable)
    real_expect_value = classicaly_compute_estimation_value(local_observable, state)

    assert np.abs(real_expect_value - expect_value) < 0.1
