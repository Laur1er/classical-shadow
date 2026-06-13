import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def run_circuit_once(circuit: QuantumCircuit):
    """
    Simulate the circuit for one single shot using AerSimulator.

    Params:
        circuit (QuantumCircuit): The state to be measured
    Returns:
        String: bitstring measured (little-endian)
    """
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1)
    counts = job.result().get_counts()

    return list(counts.keys())[0]
