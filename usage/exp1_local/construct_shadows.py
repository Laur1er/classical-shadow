import os
import numpy as np

from qiskit import qpy
from qiskit.circuit import QuantumCircuit

from classical_shadow.local_shadow import LocalClassicalShadow

# Create output directory
os.makedirs("results/exp1_local", exist_ok=True)

# Initialising and saving the QuantumCircuit
state = QuantumCircuit(6)

state.h([0, 2, 5])
state.cx([0, 2], [1, 3])
state.y(3)
state.rz(0.75 * np.pi, [0, 1])
state.x([1, 4])

with open("results/exp1_local/state.qpy", "wb") as file:
    qpy.dump(state, file)

print("Quantum circuit made successfuly\n")

# Initialising configurations
perfect_shadow = LocalClassicalShadow(nb_snapshots=15000, method="perfect")
noisy_shadow = LocalClassicalShadow(nb_snapshots=15000, method="noisy")
real_hardware_shadow = LocalClassicalShadow(nb_snapshots=15000, method="real_hardware")

print("Shadow configuration DONE.\n")

# Creation of the different shadows
perfect_shadow.fit_shadow(state)
noisy_shadow.fit_shadow(state)
real_hardware_shadow.fit_shadow(state)

print("Creation of shadows DONE.\n")

# Saving the shadows
perfect_shadow.save_shadow("results/exp1_local/perfect_shadow")
noisy_shadow.save_shadow("results/exp1_local/noisy_shadow")
real_hardware_shadow.save_shadow("results/exp1_local/real_hardware_shadow")

print("Shadows SAVED.\n")
