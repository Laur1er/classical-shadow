import numpy as np

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp, Statevector

from classical_shadow.global_shadow import GlobalClassicalShadow

# Recover shadows
perfect_shadow = GlobalClassicalShadow.recover_shadow(
    "results/exp2_global/perfect_shadow.npz"
)
noisy_shadow = GlobalClassicalShadow.recover_shadow(
    "results/exp2_global/noisy_shadow.npz"
)
# real_hardware_shadow = GlobalClassicalShadow.recover_shadow(
#     "results/exp2_global/real_hardware_shadow.npz"
# )

# Recover Quantum State
with open("results/exp2_global/state.qpy", "rb") as handle:
    state = qpy.load(handle)[0]

statevector = Statevector(state).data

# Define observables. Because we want to compare with only the state that has been teleported, we have distinctions TP.
observable_X = SparsePauliOp(["XXXXIIIIIIII"], [1])
observable_X_TP = SparsePauliOp(["XXXX"], [1])

observables = [observable_X]
observables_TP = [observable_X_TP]

# Use shadow to measure observable
for i, (observable, observable_TP) in enumerate(zip(observables, observables_TP)):

    # Compute the observable expectation value classicaly
    real_expect_value = np.einsum(
        "i,ij,j", statevector.conj(), observable_TP.to_matrix(), statevector
    )

    # Compute expectation values of observables using the different shadows.
    perfect_exp = perfect_shadow.estimate_global_observable(observable)
    noisy_exp = noisy_shadow.estimate_global_observable(observable)
    # real_exp = real_hardware_shadow.estimate_global_observable(observable)

    print(f"For the observable {i+1} :{observable}")
    print(f"  Expectation value computed classicaly : {real_expect_value}")
    print(f"  1.  Expectation value computed with a perfect shadow : {perfect_exp}")
    print(f"  2.  Expectation value computed with a noisy shadow : {noisy_exp}")
    # print(f"  3.  Expectation value computed with a real shadow : {real_exp}\n")
