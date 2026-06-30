from qiskit import qpy
from classical_shadow.global_shadow import GlobalClassicalShadow

# Recover QuantumCircuit
with open("results/exp2_global/state.qpy", "rb") as handle:
    state = qpy.load(handle)[1]

# Initialising configurations
perfect_shadow = GlobalClassicalShadow(nb_snapshots=15000, method="perfect")
noisy_shadow = GlobalClassicalShadow(nb_snapshots=15000, method="noisy")
# real_hardware_shadow = GlobalClassicalShadow(nb_snapshots=15000, method="real_hardware")

print("Shadow configuration DONE.\n")

# Perfect
perfect_shadow.fit_shadow(state)
perfect_shadow.save_shadow("results/exp2_global/perfect_shadow")
print("Creation of shadow using perfect backend DONE.\n")

# Noisy
noisy_shadow.fit_shadow(state)
noisy_shadow.save_shadow("results/exp2_global/noisy_shadow")
print("Creation of shadow using noisy backend DONE.\n")

# # Real hardware
# real_hardware_shadow.fit_shadow(state)
# real_hardware_shadow.save_shadow("results/exp2_global/real_hardware_shadow")
# print("Creation of shadow using real hardware backend DONE.\n")


print("Experiment #2: Global classical shadow of a quantum state -> DONE \n")
