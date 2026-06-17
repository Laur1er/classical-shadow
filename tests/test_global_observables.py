import numpy as np
import pytest

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from classical_shadow.global_shadow import GlobalClassicalShadow
from tests.utils import classicaly_compute_estimation_value


@pytest.fixture(scope="module")
def GHZ_Shadow():

    ghz = QuantumCircuit(2)
    ghz.h(0)
    ghz.cx(0, 1)

    shadow = GlobalClassicalShadow(num_snapshots=10000)
    shadow.fit_shadow(ghz)
    return ghz, shadow
