import os

from qiskit import qpy
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Create output directory
os.makedirs("results/exp2_global", exist_ok=True)

# 4 qubit GHZ, 8 Bell
init_reg = QuantumRegister(4)
bells = QuantumRegister(8)
adj_reg = ClassicalRegister(8)

state_teleportated = QuantumCircuit(init_reg, bells, adj_reg)

# Preparation GHZ sur les 4 premiers qubits.
state_teleportated.h(init_reg[0])
for id in range(3):
    state_teleportated.cx(init_reg[id], init_reg[id + 1])

# Teleportation
for i in range(4):

    # Créer la paire de Bell entre bell[0] et bell[1]
    state_teleportated.h(bells[i])
    state_teleportated.cx(bells[i], bells[4 + i])

    # Enchevêtrer init_reg avec bells[i]
    state_teleportated.cx(init_reg[i], bells[4 + i])
    state_teleportated.h(init_reg[i])

    # Mesures
    state_teleportated.measure(init_reg[i], adj_reg[i])
    state_teleportated.measure(bells[4 + i], adj_reg[4 + i])

    # Corrections classiquement conditionnelles
    with state_teleportated.if_test((adj_reg[4 + i], 1)):
        state_teleportated.x(bells[i])

    with state_teleportated.if_test((adj_reg[i], 1)):
        state_teleportated.z(bells[i])


# Prepare the actual state in order to compare
state = QuantumCircuit(4)

state.h(0)
for id in range(3):
    state.cx(id, id + 1)

with open("results/exp2_global/state.qpy", "wb") as file:
    qpy.dump([state, state_teleportated], file)


print("Quantum circuits made successfuly\n")
