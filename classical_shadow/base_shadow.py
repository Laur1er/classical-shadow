import numpy as np
from numpy import ndarray
from pauliarray import PauliArray

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from utils import run_circuit_once


class ClassicalShadow:
    """
    This is base for classical shadows
    """

    def __init__(self):
        pass


# Pour commencer, je vais seulement faire le protocole pour local
def classical_shadow_local_collect_data(
    quantum_state: QuantumCircuit, n_snapshots: int
):
    """
    This fonction uses Shadow Local to produce the shadow of a quantum state.
    """
    # Choose pauli strings randomly
    bases = np.random.choice(
        ["X", "Y", "Z"], size=(n_snapshots, quantum_state.num_qubits)
    )
    pauli_strings = np.array(["".join(row) for row in bases])
    paulis = PauliArray.from_labels(pauli_strings)

    measures = []
    for pauli in paulis:

        circuit = quantum_state.copy()

        # Diagonalize the pauli
        where_y = np.nonzero(np.logical_and(pauli.x_strings, pauli.z_strings))[0]
        where_x = np.nonzero(pauli.x_strings)[0]
        if len(where_y) > 0:
            circuit.sdg(where_y)
        if len(where_x) > 0:
            circuit.h(where_x)
        circuit.measure_all()

        # Simulate and store the result in measures
        measures.append(run_circuit_once(circuit))

    bit_strings = np.array(measures)

    return (paulis, bit_strings)


def reconstruct_local_shadow(paulis: PauliArray, bit_strings: ndarray):
    """
    Reconstruct the shadows
    """
    num_qubits = len(bit_strings[0])

    eigenstate_Z_0 = np.array([1, 0])
    eigenstate_Z_1 = np.array([0, 1])

    eigenstate_X_plus = (eigenstate_Z_0 + eigenstate_Z_1) / np.sqrt(2)
    eigenstate_X_minus = (eigenstate_Z_0 - eigenstate_Z_1) / np.sqrt(2)

    eigenstate_Y_plus_i = (eigenstate_Z_0 + (1j * eigenstate_Z_1)) / np.sqrt(2)
    eigenstate_Y_minus_i = (eigenstate_Z_0 - (1j * eigenstate_Z_1)) / np.sqrt(2)

    projector_Z_0 = 3 * np.outer(eigenstate_Z_0, eigenstate_Z_0) - np.eye(2)
    projector_Z_1 = 3 * np.outer(eigenstate_Z_1, eigenstate_Z_1) - np.eye(2)

    projector_X_plus = 3 * np.outer(eigenstate_X_plus, eigenstate_X_plus) - np.eye(2)
    projector_X_minus = 3 * np.outer(eigenstate_X_minus, eigenstate_X_minus) - np.eye(2)

    projector_Y_plus_i = 3 * np.outer(
        eigenstate_Y_plus_i, eigenstate_Y_plus_i
    ) - np.eye(2)
    projector_Y_minus_i = 3 * np.outer(
        eigenstate_Y_minus_i, eigenstate_Y_minus_i
    ) - np.eye(2)

    local_density_matrix = np.zeros(shape=(1,2**num_qubits, 2**num_qubits))

    for i, pauli in enumerate(paulis):
        


