import numpy as np

from qiskit import qpy
from qiskit.circuit import QuantumCircuit

from classical_shadow.local_shadow import LocalClassicalShadow

# Recover shadows
perfect_shadow = LocalClassicalShadow.recover_shadow(
    "results/exp1_local/perfect_shadow"
)
noisy_shadow = LocalClassicalShadow.recover_shadow("results/exp1_local/noisy_shadow")
real_hardware_shadow = LocalClassicalShadow.recover_shadow(
    "results/exp1_local/real_hardware_shadow"
)


# Recover QuantumCircuit
with open("results/exp1_local/state.qpy", "rb") as handle:
    state = qpy.load(handle)
