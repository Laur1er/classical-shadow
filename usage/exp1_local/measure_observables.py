import numpy as np

from qiskit import qpy
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from classical_shadow.local_shadow import LocalClassicalShadow

# Recover shadows
perfect_shadow = LocalClassicalShadow.recover_shadow(
    "results/exp1_local/perfect_shadow.npz"
)
noisy_shadow = LocalClassicalShadow.recover_shadow(
    "results/exp1_local/noisy_shadow.npz"
)
real_hardware_shadow = LocalClassicalShadow.recover_shadow(
    "results/exp1_local/real_hardware_shadow.npz"
)

# Recover QuantumCircuit
with open("results/exp1_local/state.qpy", "rb") as handle:
    state = qpy.load(handle)[0]

statevector = Statevector(state).data

# Define local observables
num_qubits = state.num_qubits

simple_observable = SparsePauliOp(["XXIIII"], [1])
composed_X_observable = SparsePauliOp(["XIIXII", "XXXIII", "IXIXIX"], [0.5, 0.75, -0.5])
composed_Y_observable = SparsePauliOp(["YIIIIY", "YIIIYY", "IIYYII"], [0.1, -1, 0.5])
composed_Z_observable = SparsePauliOp(
    ["IIIZZI", "IIIIZZ", "IIZZIZ"], [0.15, -0.35, 0.75]
)
mixed_observable = SparsePauliOp(["XIZIYI", "XZIIIY", "ZIIZYI"], [0.5, -2, 0.5])
observables = [
    simple_observable,
    composed_X_observable,
    composed_Y_observable,
    composed_Z_observable,
    mixed_observable,
]

for i, observable in enumerate(observables):

    # Compute the observable expectation value classicaly
    real_expect_value = np.einsum(
        "i,ij,j", statevector.conj(), observable.to_matrix(), statevector
    )

    # Compute expectation values of observables using the different shadows.
    perfect_exp = perfect_shadow.estimate_local_observable(observable)
    noisy_exp = noisy_shadow.estimate_local_observable(observable)
    real_exp = real_hardware_shadow.estimate_local_observable(observable)

    print(f"For the observable {i+1} :{observable}")
    print(f"  Expectation value computed classicaly : {real_expect_value}")
    print(f"  1.  Expectation value computed with a perfect shadow : {perfect_exp}")
    print(f"  2.  Expectation value computed with a noisy shadow : {noisy_exp}")
    print(f"  3.  Expectation value computed with a real shadow : {real_exp}\n")
